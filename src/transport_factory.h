/* Client transport selection: RDMA when built with DFKV_WITH_RDMA, requested
 * (env DFKV_RDMA=1), and an RDMA device is usable; otherwise TCP. Always returns
 * a working transport (TCP fallback), so a cluster runs with or without RDMA. */
#ifndef DFKV_TRANSPORT_FACTORY_H_
#define DFKV_TRANSPORT_FACTORY_H_

#include <memory>

#include "transport.h"

namespace dfkv {

// reason (optional) receives "rdma" or "tcp(...)" describing what was chosen.
std::unique_ptr<Transport> MakeClientTransport(std::string* reason = nullptr);

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_FACTORY_H_
