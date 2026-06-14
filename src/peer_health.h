#ifndef DFKV_PEER_HEALTH_H_
#define DFKV_PEER_HEALTH_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dfkv {

// Layer-1 fast avoidance: a peer (keyed by "ip:port") that fails a transport IO
// is marked unhealthy for cooldown_ms; while unhealthy the client short-circuits
// requests to it (miss) WITHOUT touching the ring. A served response (hit or
// logical miss) clears it. Thread-safe: BatchGet/Put route from many threads.
class PeerHealth {
 public:
  explicit PeerHealth(uint64_t cooldown_ms = 2000) : cooldown_ms_(cooldown_ms) {}

  bool Healthy(const std::string& peer, uint64_t now_ms) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = until_.find(peer);
    return it == until_.end() || it->second <= now_ms;
  }
  void MarkBad(const std::string& peer, uint64_t now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    until_[peer] = now_ms + cooldown_ms_;
  }
  void MarkGood(const std::string& peer) {
    std::lock_guard<std::mutex> lk(mu_);
    until_.erase(peer);
  }

 private:
  mutable std::mutex mu_;
  uint64_t cooldown_ms_;
  std::unordered_map<std::string, uint64_t> until_;  // peer -> unhealthy-until (ms)
};

}  // namespace dfkv

#endif  // DFKV_PEER_HEALTH_H_
