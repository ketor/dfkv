/* RDMA cache-node listener (RC two-sided via librdmacm). Built only when
 * DFKV_WITH_RDMA is defined. Reuses the wire frames + a request handler so the
 * cache logic (DiskCacheGroup + metrics) is shared with the TCP server.
 * COMPILE-verified; runtime verification pending RDMA hardware. */
#ifndef DFKV_RDMA_SERVER_H_
#define DFKV_RDMA_SERVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "kv_store.h"  // Status

namespace dfkv {

class RdmaServer {
 public:
  using Handler = std::function<Status(
      uint8_t op, uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, const char* payload, uint64_t payload_len,
      std::string* out_data)>;

  explicit RdmaServer(Handler handler, size_t max_msg = (8u << 20));
  ~RdmaServer();

  Status Start(int port);
  void Stop();
  int port() const { return port_; }

 private:
  void AcceptLoop();
  void Serve(void* cm_id);  // void* = rdma_cm_id*

  Handler handler_;
  size_t max_msg_;
  void* listen_id_ = nullptr;  // rdma_cm_id*
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
};

}  // namespace dfkv

#endif  // DFKV_RDMA_SERVER_H_
