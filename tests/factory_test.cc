// TDD R13 — transport factory: always returns a working transport (TCP fallback
// when RDMA is not built / not requested / no device).
#include "transport_factory.h"
#include "kv_node_server.h"
#include "key_map.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

TEST(Factory, ReturnsWorkingTransportWithTcpFallback) {
  auto dir = fs::temp_directory_path() / "dfkv_factory";
  fs::remove_all(dir); fs::create_directories(dir);
  KvNodeServer srv(dir.string(), 1ull << 30);
  ASSERT_EQ(srv.Start(0), Status::kOk);
  std::string addr = "127.0.0.1:" + std::to_string(srv.port());

  std::string reason;
  auto t = MakeClientTransport(&reason);
  ASSERT_NE(t, nullptr);
  EXPECT_NE(reason.find("tcp"), std::string::npos) << reason;  // no DFKV_RDMA env here

  std::string v(128, 'f');
  ASSERT_EQ(t->Cache(addr, ToBlockKey("k"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t->Range(addr, ToBlockKey("k"), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  srv.Stop();
}
