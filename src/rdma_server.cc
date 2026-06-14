#include "rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "net_util.h"     // ReadAll / WriteAll / Get*/Put*
#include "rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport.h"    // kReqPrefix, kRespPrefix

namespace dfkv {

RdmaServer::RdmaServer(Handler handler, size_t max_msg, const std::string& dev_name)
    : handler_(std::move(handler)), max_msg_(max_msg), dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
}

RdmaServer::~RdmaServer() { Stop(); }

Status RdmaServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);  // bootstrap reachable on any IP net
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 128) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] { AcceptLoop(); });
  return Status::kOk;
}

void RdmaServer::Stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);  // wake accept()
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // Per-connection serve threads are detached: they block in RDMA completion
  // waits and are reclaimed on process exit. We intentionally do not join them
  // (no clean interrupt for a blocked CQ wait), which keeps Stop() bounded.
}

void RdmaServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    std::thread([this, fd] { Serve(fd); }).detach();
  }
}

void RdmaServer::Serve(int boot_fd) {
  rdma::RcEndpoint ep;
  if (!ep.Open(dev_name_.empty() ? nullptr : dev_name_.c_str(), max_msg_, 1)) {
    ::close(boot_fd); return;
  }
  // QP bootstrap: read client's info, send ours (symmetric to the client).
  char peer[rdma::kQpInfoBytes], mine[rdma::kQpInfoBytes];
  rdma::SerializeQpInfo(ep.Local(), mine);
  if (!net::ReadAll(boot_fd, peer, rdma::kQpInfoBytes) ||
      !net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes)) {
    ::close(boot_fd); return;
  }
  if (!ep.Connect(rdma::ParseQpInfo(peer))) { ::close(boot_fd); return; }
  if (!ep.PostRecv(0)) { ::close(boot_fd); return; }
  // Tell the client we are ready (recv posted) so its first SEND won't hit RNR.
  char ready = 1;
  bool ok = net::WriteAll(boot_fd, &ready, 1);
  ::close(boot_fd);  // bootstrap done
  if (!ok) return;

  // Serve loop: one request in flight (depth 1). RC two-sided ping-pong.
  while (running_) {
    ibv_wc wc{};
    int g = ep.WaitComp(&wc, 1);
    if (g <= 0 || wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_RECV) break;
    if (wc.byte_len < kReqPrefix) break;

    const char* req = ep.rbuf(0);
    uint64_t payload_len = net::GetU64(req + 33);
    const char* payload = (payload_len && wc.byte_len >= kReqPrefix + payload_len)
                              ? req + kReqPrefix : nullptr;
    std::string data;
    Status st = handler_(static_cast<uint8_t>(req[0]), net::GetU64(req + 1),
                         net::GetU32(req + 9), net::GetU32(req + 13),
                         net::GetU64(req + 17), net::GetU64(req + 25),
                         payload, payload_len, &data);

    // Re-arm recv (request consumed) before sending so the client's next SEND
    // always finds a posted receive — no RNR.
    if (!ep.PostRecv(0)) break;
    if (kRespPrefix + data.size() > ep.cap()) break;
    char* sb = ep.sbuf(0);
    sb[0] = static_cast<char>(st);
    net::PutU64(sb + 1, data.size());
    if (!data.empty()) std::memcpy(sb + kRespPrefix, data.data(), data.size());
    if (!ep.PostSend(0, kRespPrefix + data.size())) break;

    // Reap the send completion (a re-armed recv may also be pending; with a
    // synchronous client it won't fire until it gets this reply).
    bool sent = false, fail = false;
    while (!sent && !fail) {
      ibv_wc sc{};
      int s = ep.WaitComp(&sc, 1);
      if (s <= 0 || sc.status != IBV_WC_SUCCESS) fail = true;
      else if (sc.opcode == IBV_WC_SEND) sent = true;
    }
    if (fail) break;
  }
  // ep dtor tears down the QP; the peer observes the drop as an error completion.
}

}  // namespace dfkv
