#include "transport_factory.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "tcp_transport.h"

#ifdef DFKV_WITH_RDMA
#include "rdma_transport.h"
#endif

namespace dfkv {

namespace {
bool EnvTruthy(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0 &&
         std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
}
}  // namespace

std::unique_ptr<Transport> MakeClientTransport(std::string* reason) {
#ifdef DFKV_WITH_RDMA
  if (EnvTruthy("DFKV_RDMA")) {
    if (RdmaTransport::Available()) {
      if (reason) *reason = "rdma";
      return std::make_unique<RdmaTransport>();
    }
    if (reason) *reason = "tcp(rdma-requested-but-no-device)";
    return std::make_unique<TcpTransport>();
  }
  if (reason) *reason = "tcp(rdma-not-requested)";
#else
  if (reason) *reason = EnvTruthy("DFKV_RDMA") ? "tcp(rdma-not-built)" : "tcp";
#endif
  return std::make_unique<TcpTransport>();
}

}  // namespace dfkv
