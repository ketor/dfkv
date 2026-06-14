#include "rdma_cm_server.h"

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include <cstring>
#include <string>
#include <vector>

#include "net_util.h"  // GetU64/GetU32/PutU64
#include "transport.h"  // kReqPrefix, kRespPrefix

namespace dfkv {

RdmaCmServer::RdmaCmServer(Handler handler, size_t max_msg)
    : handler_(std::move(handler)), max_msg_(max_msg) {}

RdmaCmServer::~RdmaCmServer() { Stop(); }

Status RdmaCmServer::Start(int port) {
  rdma_addrinfo hints{};
  hints.ai_flags = RAI_PASSIVE;
  hints.ai_port_space = RDMA_PS_TCP;
  rdma_addrinfo* res = nullptr;
  std::string p = std::to_string(port);
  if (rdma_getaddrinfo(nullptr, p.c_str(), &hints, &res) != 0) return Status::kIOError;
  rdma_cm_id* lid = nullptr;
  ibv_qp_init_attr attr{};
  attr.cap.max_send_wr = 4; attr.cap.max_recv_wr = 4;
  attr.cap.max_send_sge = 1; attr.cap.max_recv_sge = 1;
  attr.sq_sig_all = 1;
  if (rdma_create_ep(&lid, res, nullptr, &attr) != 0) { rdma_freeaddrinfo(res); return Status::kIOError; }
  rdma_freeaddrinfo(res);
  if (rdma_listen(lid, 128) != 0) { rdma_destroy_ep(lid); return Status::kIOError; }
  listen_id_ = lid;
  port_ = ntohs(rdma_get_src_port(lid));
  running_ = true;
  accept_thread_ = std::thread([this] { AcceptLoop(); });
  return Status::kOk;
}

void RdmaCmServer::Stop() {
  bool was = running_.exchange(false);
  if (listen_id_) {
    rdma_destroy_ep(static_cast<rdma_cm_id*>(listen_id_));  // unblocks rdma_get_request
    listen_id_ = nullptr;
  }
  if (was && accept_thread_.joinable()) accept_thread_.join();
}

void RdmaCmServer::AcceptLoop() {
  auto* lid = static_cast<rdma_cm_id*>(listen_id_);
  while (running_) {
    rdma_cm_id* cid = nullptr;
    if (rdma_get_request(lid, &cid) != 0) { if (!running_) break; continue; }
    std::thread([this, cid] { Serve(cid); }).detach();
  }
}

void RdmaCmServer::Serve(void* cm_id) {
  auto* id = static_cast<rdma_cm_id*>(cm_id);
  std::vector<char> rbuf(max_msg_), sbuf(max_msg_);
  ibv_mr* rmr = rdma_reg_msgs(id, rbuf.data(), rbuf.size());
  ibv_mr* smr = rdma_reg_msgs(id, sbuf.data(), sbuf.size());
  if (!rmr || !smr) { if (rmr) ibv_dereg_mr(rmr); if (smr) ibv_dereg_mr(smr); rdma_destroy_ep(id); return; }
  if (rdma_post_recv(id, nullptr, rbuf.data(), rbuf.size(), rmr) != 0 ||
      rdma_accept(id, nullptr) != 0) {
    ibv_dereg_mr(rmr); ibv_dereg_mr(smr); rdma_destroy_ep(id); return;
  }
  while (running_) {
    ibv_wc wc{};
    int rn = rdma_get_recv_comp(id, &wc);
    if (rn <= 0 || wc.byte_len < kReqPrefix) break;
    const char* req = rbuf.data();
    uint64_t payload_len = net::GetU64(req + 33);
    const char* payload = (payload_len && wc.byte_len >= kReqPrefix + payload_len)
                              ? req + kReqPrefix : nullptr;
    std::string data;
    Status st = handler_(static_cast<uint8_t>(req[0]), net::GetU64(req + 1),
                         net::GetU32(req + 9), net::GetU32(req + 13),
                         net::GetU64(req + 17), net::GetU64(req + 25),
                         payload, payload_len, &data);
    // re-arm recv for the next request before sending the reply
    if (rdma_post_recv(id, nullptr, rbuf.data(), rbuf.size(), rmr) != 0) break;
    if (kRespPrefix + data.size() > sbuf.size()) break;
    sbuf[0] = static_cast<char>(st);
    net::PutU64(sbuf.data() + 1, data.size());
    if (!data.empty()) std::memcpy(sbuf.data() + kRespPrefix, data.data(), data.size());
    if (rdma_post_send(id, nullptr, sbuf.data(), kRespPrefix + data.size(), smr,
                       IBV_SEND_SIGNALED) != 0) break;
    ibv_wc sc{};
    if (rdma_get_send_comp(id, &sc) <= 0) break;
  }
  ibv_dereg_mr(rmr); ibv_dereg_mr(smr);
  // No rdma_disconnect(): synchronous endpoints have no CM event loop to ack the
  // disconnect, so it would block. rdma_destroy_ep tears down locally; the client
  // sees the drop as an error completion. (See rdma_transport.cc Destroy note.)
  rdma_destroy_ep(id);
}

}  // namespace dfkv
