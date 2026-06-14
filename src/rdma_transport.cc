#include "rdma_transport.h"

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include <cstring>

#include "net_util.h"  // EncodePrefix-style codec (PutU64/GetU64)

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
  rdma_cm_id* id = nullptr;
  char* sbuf = nullptr;
  char* rbuf = nullptr;
  ibv_mr* smr = nullptr;
  ibv_mr* rmr = nullptr;
  size_t cap = 0;
};

bool RdmaTransport::Available() {
  int n = 0;
  ibv_device** devs = ibv_get_device_list(&n);
  if (devs) ibv_free_device_list(devs);
  return n > 0;
}

RdmaTransport::RdmaTransport(size_t max_msg) : max_msg_(max_msg) {}

RdmaTransport::~RdmaTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, cs] : pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c) {
  if (!c) return;
  if (c->smr) ibv_dereg_mr(c->smr);
  if (c->rmr) ibv_dereg_mr(c->rmr);
  // NOTE: do NOT call rdma_disconnect() here. With synchronous rdma_create_ep
  // endpoints (no CM event channel on either side) rdma_disconnect blocks
  // forever waiting for the peer's disconnect reply, which never comes because
  // the server thread sits in rdma_get_recv_comp. rdma_destroy_ep tears down the
  // QP locally; the peer observes the RC connection drop as an error completion
  // and cleans up its own endpoint. (Validated on hd03 InfiniBand, 2026-06-14.)
  if (c->id) rdma_destroy_ep(c->id);
  delete[] c->sbuf;
  delete[] c->rbuf;
  delete c;
}

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
  // parse host:port
  auto pos = node.rfind(':');
  if (pos == std::string::npos) return nullptr;
  std::string host = node.substr(0, pos), port = node.substr(pos + 1);

  rdma_addrinfo hints{}; hints.ai_port_space = RDMA_PS_TCP;
  rdma_addrinfo* res = nullptr;
  if (rdma_getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return nullptr;

  ibv_qp_init_attr attr{};
  attr.cap.max_send_wr = 4; attr.cap.max_recv_wr = 4;
  attr.cap.max_send_sge = 1; attr.cap.max_recv_sge = 1;
  attr.sq_sig_all = 1;
  rdma_cm_id* id = nullptr;
  if (rdma_create_ep(&id, res, nullptr, &attr) != 0) { rdma_freeaddrinfo(res); return nullptr; }
  rdma_freeaddrinfo(res);

  auto* c = new Conn();
  c->id = id; c->cap = max_msg_;
  c->sbuf = new char[c->cap]; c->rbuf = new char[c->cap];
  c->smr = rdma_reg_msgs(id, c->sbuf, c->cap);
  c->rmr = rdma_reg_msgs(id, c->rbuf, c->cap);
  if (!c->smr || !c->rmr) { Destroy(c); return nullptr; }
  if (rdma_connect(id, nullptr) != 0) { Destroy(c); return nullptr; }
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

    // build request in send buffer
    EncodePrefix(c->sbuf, op, k, offset, length, payload_len);
    if (payload_len) std::memcpy(c->sbuf + kReqPrefix, payload, payload_len);

    bool ok = true;
    ibv_wc wc{};
    // pre-post recv, then send, then reap completions
    if (rdma_post_recv(c->id, nullptr, c->rbuf, c->cap, c->rmr) != 0) ok = false;
    if (ok && rdma_post_send(c->id, nullptr, c->sbuf, kReqPrefix + payload_len,
                             c->smr, IBV_SEND_SIGNALED) != 0) ok = false;
    if (ok && rdma_get_send_comp(c->id, &wc) <= 0) ok = false;
    int rn = ok ? rdma_get_recv_comp(c->id, &wc) : -1;
    if (rn <= 0) ok = false;

    if (!ok) {
      Destroy(c);
      if (!from_pool) return Status::kIOError;
      continue;  // stale pooled conn -> retry fresh
    }
    // parse response: status(1) data_len(8) data...
    if (wc.byte_len < kRespPrefix) { Destroy(c); return Status::kIOError; }
    Status st = static_cast<Status>(static_cast<uint8_t>(c->rbuf[0]));
    uint64_t dlen = net::GetU64(c->rbuf + 1);
    if (out) {
      if (kRespPrefix + dlen > wc.byte_len) { Destroy(c); return Status::kIOError; }
      out->assign(c->rbuf + kRespPrefix, dlen);
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
