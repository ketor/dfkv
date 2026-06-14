/* RDMA client transport (RC, two-sided SEND/RECV via librdmacm). Built only when
 * DFKV_WITH_RDMA is defined. Mirrors the TCP wire frames so the server logic is
 * shared. Connection-pooled per node. Verified to COMPILE against libibverbs/
 * librdmacm; runtime verification pending RDMA hardware. */
#ifndef DFKV_RDMA_TRANSPORT_H_
#define DFKV_RDMA_TRANSPORT_H_

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport.h"

namespace dfkv {

class RdmaTransport : public Transport {
 public:
  static bool Available();  // true if at least one RDMA device is present

  explicit RdmaTransport(size_t max_msg = (8u << 20));
  ~RdmaTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;

 private:
  struct Conn;
  Conn* Acquire(const std::string& node, bool* from_pool);
  void Release(const std::string& node, Conn* c);
  void Destroy(Conn* c);
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out);

  std::mutex mu_;
  std::unordered_map<std::string, std::vector<Conn*>> pool_;
  size_t max_msg_;
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
