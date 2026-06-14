/* dfkv_bench — throughput / latency benchmark for a dfkv cluster.
 *
 *   dfkv_bench --members "n=ip:port,..." [--size BYTES] [--count N]
 *              [--threads T] [--op put|get|both]
 *
 * Transport is chosen by the same env switch as the client: set DFKV_RDMA=1 to
 * use RDMA (build must be -DDFKV_WITH_RDMA=ON and a device present), else TCP.
 * Each op uses a unique key ("dfkv-bench-<i>") so the write-once cache never
 * collides across runs of different sizes. Reports aggregate GB/s and per-op
 * p50/p99/max latency for the PUT and GET phases. No GPU needed. */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "kv_client.h"
#include "transport_factory.h"
#include "value_header.h"

using namespace dfkv;  // NOLINT
using Clock = std::chrono::steady_clock;

static std::vector<std::pair<std::string, std::string>> ParseMembers(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

static double Pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  size_t k = static_cast<size_t>(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Run `count` ops across `threads`, collecting per-op latency in ms. fn(i) does
// one op and returns true on success. Returns wall seconds; fills lat + fails.
static double RunPhase(size_t count, size_t threads,
                       const std::function<bool(size_t)>& fn,
                       std::vector<double>* lat, std::atomic<size_t>* fails) {
  lat->assign(count, 0.0);
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
      auto t0 = Clock::now();
      bool ok = fn(i);
      auto t1 = Clock::now();
      (*lat)[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (!ok) fails->fetch_add(1);
    }
  };
  auto start = Clock::now();
  std::vector<std::thread> ts;
  for (size_t w = 0; w < threads; ++w) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static void Report(const char* phase, size_t count, size_t size, size_t threads,
                   double secs, std::vector<double>& lat, size_t fails) {
  double total_gb = double(count) * double(size) / 1e9;
  double opss = secs > 0 ? count / secs : 0;
  std::printf("%-5s n=%zu size=%zu threads=%zu | %.3fs  %.2f GB/s  %.0f ops/s  "
              "lat ms p50=%.3f p99=%.3f max=%.3f  fails=%zu\n",
              phase, count, size, threads, secs, total_gb / secs, opss,
              Pct(lat, 0.50), Pct(lat, 0.99), Pct(lat, 1.0), fails);
}

int main(int argc, char** argv) {
  std::string members, op = "both";
  size_t size = 2752512, count = 2000, threads = 8;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--members")) members = argv[i + 1];
    else if (!std::strcmp(argv[i], "--size")) size = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--count")) count = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--threads")) threads = std::stoull(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--op")) op = argv[i + 1];
  }
  auto mem = ParseMembers(members);
  if (mem.empty()) { std::fprintf(stderr, "need --members name=ip:port,...\n"); return 2; }

  std::string reason;
  MakeClientTransport(&reason);  // just to report the resolved transport
  std::printf("dfkv_bench transport=%s members=%zu\n", reason.c_str(), mem.size());

  // GLM-5.1 / MLA geometry (matches dfkv_smoke / dfkvctl defaults)
  ValueHeader hdr = ValueHeader::Make(0x51, 64, 0x46384534u, ValueHeader::kFlagIsMla,
                                      8, 0, 78, 1, 576);
  KVClient c(mem, hdr);

  std::string val(size, '\0');
  for (size_t i = 0; i < size; ++i) val[i] = static_cast<char>(i & 0xFF);
  auto key = [](size_t i) { return "dfkv-bench-" + std::to_string(i); };

  std::vector<double> lat;
  std::atomic<size_t> fails{0};

  if (op == "put" || op == "both") {
    fails = 0;
    double s = RunPhase(count, threads,
        [&](size_t i) { return c.Put(key(i), val.data(), val.size()); }, &lat, &fails);
    Report("PUT", count, size, threads, s, lat, fails.load());
  }
  if (op == "get" || op == "both") {
    fails = 0;
    double s = RunPhase(count, threads, [&](size_t i) {
      std::string out(size, '\0');
      return c.Get(key(i), &out[0], out.size());
    }, &lat, &fails);
    Report("GET", count, size, threads, s, lat, fails.load());
  }
  return 0;
}
