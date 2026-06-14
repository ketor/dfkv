#include "rdma_transport.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>

#include "net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "rdma_verbs.h"   // RcEndpoint, QpInfo

namespace dfkv {

namespace {
void EncodePrefix(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                  uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(op);
  net::PutU64(p + 1, k.id);
  net::PutU32(p + 9, k.index);
  net::PutU32(p + 13, k.size);
  net::PutU64(p + 17, offset);
  net::PutU64(p + 25, length);
  net::PutU64(p + 33, payload_len);
}
}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
};

bool RdmaTransport::Available() {
  int n = 0;
  ibv_device** devs = ibv_get_device_list(&n);
  if (devs) ibv_free_device_list(devs);
  return n > 0;
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_msg_(max_msg), dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
}

RdmaTransport::~RdmaTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, cs] : pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c) { delete c; }  // RcEndpoint dtor tears down QP/MRs

RdmaTransport::Conn* RdmaTransport::Acquire(const std::string& node, bool* from_pool) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pool_.find(node);
    if (it != pool_.end() && !it->second.empty()) {
      Conn* c = it->second.back(); it->second.pop_back();
      *from_pool = true; return c;
    }
  }
  *from_pool = false;
  // Bootstrap the QP over a short-lived TCP connection to the node's member
  // address (control plane). The data plane then rides the named RDMA device.
  int fd = net::Dial(node, /*connect_ms=*/3000, /*io_ms=*/10000);
  if (fd < 0) return nullptr;

  auto* c = new Conn();
  if (!c->ep.Open(dev_name_.empty() ? nullptr : dev_name_.c_str(), max_msg_, 1)) {
    ::close(fd); delete c; return nullptr;
  }
  char mine[rdma::kQpInfoBytes], peer[rdma::kQpInfoBytes];
  rdma::SerializeQpInfo(c->ep.Local(), mine);
  // client sends its Qp info, then reads the server's; symmetric to the server.
  if (!net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !net::ReadAll(fd, peer, rdma::kQpInfoBytes)) {
    ::close(fd); delete c; return nullptr;
  }
  if (!c->ep.Connect(rdma::ParseQpInfo(peer))) { ::close(fd); delete c; return nullptr; }
  // Wait for the server's "ready" byte: it has posted its recvs, so our first
  // SEND won't hit RNR.
  char ready = 0;
  if (!net::ReadAll(fd, &ready, 1) || ready != 1) { ::close(fd); delete c; return nullptr; }
  ::close(fd);  // bootstrap done; QP is RTS
  return c;
}

void RdmaTransport::Release(const std::string& node, Conn* c) {
  std::lock_guard<std::mutex> lk(mu_);
  pool_[node].push_back(c);
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& k, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out) {
  if (kReqPrefix + payload_len > max_msg_) return Status::kInvalid;
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool from_pool = false;
    Conn* c = Acquire(node, &from_pool);
    if (!c) return Status::kIOError;
    rdma::RcEndpoint& ep = c->ep;

    EncodePrefix(ep.sbuf(0), op, k, offset, length, payload_len);
    if (payload_len) std::memcpy(ep.sbuf(0) + kReqPrefix, payload, payload_len);

    bool ok = ep.PostRecv(0) && ep.PostSend(0, kReqPrefix + payload_len);
    // Reap both the send and the recv completion (shared CQ; either order).
    uint32_t recv_bytes = 0;
    bool need_send = ok, need_recv = ok;
    while (ok && (need_send || need_recv)) {
      ibv_wc wc{};
      int g = ep.WaitComp(&wc, 1);
      if (g <= 0 || wc.status != IBV_WC_SUCCESS) { ok = false; break; }
      if (wc.opcode == IBV_WC_SEND) need_send = false;
      else if (wc.opcode == IBV_WC_RECV) { need_recv = false; recv_bytes = wc.byte_len; }
    }

    if (!ok) {
      Destroy(c);
      if (!from_pool) return Status::kIOError;
      continue;  // stale pooled conn -> retry fresh
    }
    if (recv_bytes < kRespPrefix) { Destroy(c); return Status::kIOError; }
    Status st = static_cast<Status>(static_cast<uint8_t>(ep.rbuf(0)[0]));
    uint64_t dlen = net::GetU64(ep.rbuf(0) + 1);
    if (out) {
      if (kRespPrefix + dlen > recv_bytes) { Destroy(c); return Status::kIOError; }
      out->assign(ep.rbuf(0) + kRespPrefix, dlen);
    }
    Release(node, c);
    return st;
  }
  return Status::kIOError;
}

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length, std::string* out) {
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key, bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

}  // namespace dfkv
