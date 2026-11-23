// The M3 measurement harness (docs/design/log-structured-engine.md §13):
// the slab engine and the log engine compared on the same disk, through the
// same DiskCacheGroup the node uses, with direct I/O:
//   - effective capacity (committed payload / configured capacity, and the
//     object count) at steady state;
//   - write throughput and PUT latency while the store is first filled and,
//     separately, at steady state (every segment / extent reused);
//   - read throughput and latency (whole values, O_DIRECT, random keys);
//   - REMOVE latency, and GET latency on other threads during a REMOVE burst;
//   - the share of PUTs answered kCacheFull, evictions, and (log engine)
//     write amplification: relocated bytes per client byte;
//   - eviction quality: hit ratio of a Zipf-skewed GET-then-PUT-on-miss
//     workload over twice the capacity;
//   - restart time after a clean close and after a kill, with the fraction
//     of objects recovered.
//
// It runs only on a test ring: the directory named by DFKV_LSE_BENCH_DIR or
// kRingDir, which must exist already. Everywhere else, CI included, every
// case is skipped. Each case writes only inside that directory; it removes
// what any case left there when it starts (a case cut off by a timeout cannot
// clean up) and what it wrote when it ends. A case refuses to start, and
// stops writing, when the disk has less than kMinFreeBytes free; each case
// writes at most kMaxCaseBytes (checked against its plan before it starts,
// and counted as it goes), and no store is larger than kCapacity. Run one
// case at a time. The node's test runner shows the output of failing tests
// only, so on a ring each case ends with one non-fatal failure carrying its
// report ("LSE-BENCH").
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cache/disk_cache_group.h"

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// The ring directory, or empty: then every case is skipped. Only an exact,
// explicitly given path: DFKV_LSE_BENCH_DIR, or kRingDir, the directory
// assigned to this harness on the development node. Nothing is looked for
// or created.
constexpr const char* kRingDir = "/mnt/disk1/dfkv-test/lse-m3/lse-bench";
std::string BenchRoot() {
  if (const char* d = std::getenv("DFKV_LSE_BENCH_DIR")) return d;
  std::error_code ec;
  if (fs::is_directory(kRingDir, ec)) return kRingDir;
  return "";
}

// Limits on what a case may do to its disk.
constexpr uint64_t kCapacity = 4ull << 30;  // per store: 4 GiB (fits 600 s)
constexpr uint64_t kMaxStoreBytes = 300ull << 30;
static_assert(kCapacity <= kMaxStoreBytes, "a store stays far below the disk");
constexpr uint64_t kMaxCaseBytes = 50'000'000'000ull;  // written per case
constexpr uint64_t kMinFreeBytes = 400'000'000'000ull;

uint64_t FreeBytes(const std::string& root) {
  std::error_code ec;
  const auto sp = fs::space(root, ec);
  return ec ? 0 : sp.available;
}

// Removes what earlier cases left in the ring (only the names the cases
// below create).
std::string RemoveLeftovers(const std::string& root) {
  std::string removed;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root, ec)) {
    const std::string name = e.path().filename().string();
    bool ours = name.size() > 6 && name.compare(name.size() - 6, 6, ".count") == 0;
    for (const char* prefix : {"slab-", "slabwarm-", "log-", "zipf-", "markers-"})
      ours = ours || name.rfind(prefix, 0) == 0;
    if (!ours) continue;
    std::error_code ec2;
    fs::remove_all(e.path(), ec2);
    removed += " " + name;
  }
  return removed;
}

// The start of every case: leftovers removed, then the free-space floor,
// and the case's planned writes within the bound. Empty: skip (no ring).
std::string StartCase(uint64_t planned_bytes) {
  const std::string root = BenchRoot();
  if (root.empty()) return root;
  const std::string removed = RemoveLeftovers(root);
  if (!removed.empty()) std::fprintf(stderr, "LSE-BENCH-PROGRESS removed%s\n", removed.c_str());
  const uint64_t free_bytes = FreeBytes(root);
  EXPECT_GE(free_bytes, kMinFreeBytes)
      << "refusing to run: " << free_bytes / 1000000000 << " GB free in " << root;
  EXPECT_LE(planned_bytes, kMaxCaseBytes)
      << "the case would write " << planned_bytes / 1000000000 << " GB";
  if (free_bytes < kMinFreeBytes || planned_bytes > kMaxCaseBytes) return "";
  return root;
}

struct Aligned {
  explicit Aligned(size_t n) : size(n) {
    void* p = nullptr;
    if (::posix_memalign(&p, 4096, n) != 0) p = nullptr;
    data.reset(static_cast<char*>(p));
  }
  struct Free {
    void operator()(char* p) const { std::free(p); }
  };
  std::unique_ptr<char, Free> data;
  size_t size;
};

BlockKey Key(uint64_t id) { return BlockKey{id * 0x9e3779b97f4a7c15ULL + 1, id, 0}; }

// Latency samples (microseconds) with percentiles.
class Lat {
 public:
  void Add(double us) {
    std::lock_guard<std::mutex> lk(mu_);
    v_.push_back(us);
  }
  double P(double q) {
    std::lock_guard<std::mutex> lk(mu_);
    if (v_.empty()) return 0;
    std::sort(v_.begin(), v_.end());
    return v_[std::min(v_.size() - 1, static_cast<size_t>(q * v_.size()))];
  }
  size_t N() {
    std::lock_guard<std::mutex> lk(mu_);
    return v_.size();
  }

 private:
  std::mutex mu_;
  std::vector<double> v_;
};

// Report lines, also printed at once with the seconds since the case began:
// a case cut off by the test timeout still shows how far it got.
struct Report {
  std::ostringstream out;
  void Line(const std::string& phase, const std::string& kv) {
    out << "LSE-BENCH " << tag << " " << phase << " " << kv << "\n";
    std::fprintf(stderr, "LSE-BENCH-PROGRESS t=%.1f %s %s %s\n",
                 std::chrono::duration<double>(Clock::now() - t0).count(),
                 tag.c_str(), phase.c_str(), kv.c_str());
  }
  std::string tag;
  Clock::time_point t0 = Clock::now();
};

uint64_t EngineMetric(DiskCacheGroup& g, const char* name) {
  std::vector<StoreEngine::EngineMetric> m;
  g.DiskEngineMetrics(0, &m);
  for (const auto& e : m)
    if (std::strcmp(e.name, name) == 0) return e.value;
  return 0;
}

class Bench {
 public:
  Bench(std::string engine, size_t value_len, uint64_t capacity, std::string dir)
      : root_(fs::path(dir).parent_path().string()),
        engine_(std::move(engine)),
        len_(value_len),
        capacity_(capacity),
        dir_(std::move(dir)) {
    ::setenv("DFKV_SLAB_WRITE", "direct", 1);  // the deployment I/O mode
    ::setenv("DFKV_LSE_EXPERIMENTAL", "1", 1);
    // "slabwarm": slab as deployed, keeping its table after an unclean exit.
    if (engine_ == "slabwarm") ::setenv("DFKV_SLAB_WARM_RESTART", "auto", 1);
    else ::unsetenv("DFKV_SLAB_WARM_RESTART");
    rep_.tag = "engine=" + engine_ + " size=" + std::to_string(len_);
  }
  ~Bench() {
    group_.reset();
    std::error_code ec;
    fs::remove_all(dir_, ec);  // everything this case wrote
  }

  bool Open() {
    DiskCacheGroup::Options o;
    o.cache_dirs = {dir_};
    o.capacity_bytes = capacity_;
    o.engine = engine_ == "slabwarm" ? "slab" : engine_;
    const auto t0 = Clock::now();
    group_ = std::make_unique<DiskCacheGroup>(o);
    open_secs_ = Seconds(t0, Clock::now());
    return group_->Healthy();
  }
  void Close() { group_.reset(); }
  DiskCacheGroup& G() { return *group_; }
  double OpenSecs() const { return open_secs_; }
  Report& R() { return rep_; }
  size_t Len() const { return len_; }

  struct WriteStats {
    double secs = 0;
    uint64_t bytes = 0, puts = 0, ok = 0, full = 0, other = 0;
  };
  // PUTs of keys [first, first + n) by `threads` threads, values written
  // from aligned buffers (the transport's zero-copy path). Stops early (a
  // failure) past the case's write bound or below the free-space floor.
  WriteStats Write(uint64_t first, uint64_t n, int threads, Lat* lat) {
    WriteStats ws;
    if (FreeBytes(root_) < kMinFreeBytes) {
      ADD_FAILURE() << "stopped: less than the free-space floor in " << root_;
      return ws;
    }
    std::atomic<uint64_t> next{first}, ok{0}, full{0}, other{0};
    std::atomic<bool> over{false};
    const auto t0 = Clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
      th.emplace_back([&] {
        Aligned buf(((len_ + 4095) / 4096) * 4096 + 4096);
        std::memset(buf.data.get(), 0x5a, buf.size);
        for (;;) {
          const uint64_t id = next.fetch_add(1);
          if (id >= first + n) break;
          if (written_.fetch_add(len_) + len_ > kMaxCaseBytes) {
            over = true;
            break;
          }
          std::memcpy(buf.data.get(), &id, std::min<size_t>(8, len_));
          const auto a = Clock::now();
          const Status st = group_->CacheDirect(Key(id), buf.data.get(), len_, buf.size);
          lat->Add(Seconds(a, Clock::now()) * 1e6);
          if (st == Status::kOk) ++ok;
          else if (st == Status::kCacheFull) ++full;
          else ++other;
        }
      });
    // Progress every 20 s, for a phase cut off by the test timeout.
    std::atomic<bool> done{false};
    std::thread progress([&] {
      for (int i = 1; !done; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (i % 200 == 0)
          std::fprintf(stderr, "LSE-BENCH-PROGRESS %s write %.0fs puts=%llu/%llu ok=%llu full=%llu\n",
                       rep_.tag.c_str(), Seconds(t0, Clock::now()),
                       static_cast<unsigned long long>(std::min(next.load(), first + n) - first),
                       static_cast<unsigned long long>(n),
                       static_cast<unsigned long long>(ok.load()),
                       static_cast<unsigned long long>(full.load()));
      }
    });
    for (auto& x : th) x.join();
    done = true;
    progress.join();
    if (over) ADD_FAILURE() << "stopped: the case reached its write bound";
    ws.secs = Seconds(t0, Clock::now());
    ws.puts = n;
    ws.ok = ok;
    ws.full = full;
    ws.other = other;
    ws.bytes = ok * len_;
    return ws;
  }

  struct ReadStats {
    double secs = 0;
    uint64_t gets = 0, hits = 0, bytes = 0, bad = 0;
  };
  // Whole-value GETs of random keys in [lo, hi), O_DIRECT into aligned
  // buffers; each served value's first bytes are checked. Latencies are
  // those of hits only.
  ReadStats Read(uint64_t lo, uint64_t hi, uint64_t n, int threads, Lat* lat,
                 const std::function<uint64_t(std::mt19937_64&)>& pick = nullptr) {
    ReadStats rs;
    std::atomic<int64_t> left{static_cast<int64_t>(n)};  // signed: goes below 0
    std::atomic<uint64_t> hits{0}, bad{0};
    const auto t0 = Clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
      th.emplace_back([&, t] {
        std::mt19937_64 rng(1000 + t);
        Aligned buf(((len_ + 4095) / 4096) * 4096 + 8192);
        while (left.fetch_sub(1) > 0) {
          const uint64_t id = pick ? pick(rng) : lo + rng() % (hi - lo);
          const char* data = nullptr;
          size_t got = 0;
          const auto a = Clock::now();
          const Status st = group_->RangeDirect(Key(id), 0, 0, buf.data.get(),
                                                buf.size, &data, &got);
          if (st != Status::kOk) continue;  // a miss: not in the latencies
          lat->Add(Seconds(a, Clock::now()) * 1e6);
          ++hits;
          uint64_t seen = 0;
          std::memcpy(&seen, data, std::min<size_t>(8, got));
          if (got != len_ || (len_ >= 8 && seen != id)) ++bad;
        }
      });
    for (auto& x : th) x.join();
    rs.secs = Seconds(t0, Clock::now());
    rs.gets = n;
    rs.hits = hits;
    rs.bad = bad;
    rs.bytes = hits * len_;
    return rs;
  }

  // Counts a write outside Write() against the bound; false past it.
  bool Count(uint64_t bytes) { return written_.fetch_add(bytes) + bytes <= kMaxCaseBytes; }
  uint64_t Written() const { return written_.load(); }

 private:
  std::string root_;
  std::atomic<uint64_t> written_{0};  // client bytes this case wrote
  std::string engine_;
  size_t len_;
  uint64_t capacity_;
  std::string dir_;
  std::unique_ptr<DiskCacheGroup> group_;
  double open_secs_ = 0;
  Report rep_;
};

std::string Fmt(const char* f, double a) {
  char b[64];
  std::snprintf(b, sizeof(b), f, a);
  return b;
}

std::string WriteLine(const Bench::WriteStats& w, Lat& lat) {
  return "puts=" + std::to_string(w.puts) + " ok=" + std::to_string(w.ok) +
         " cache_full=" + std::to_string(w.full) + " other=" + std::to_string(w.other) +
         " GBps=" + Fmt("%.3f", w.bytes / 1e9 / std::max(w.secs, 1e-9)) +
         " p50us=" + Fmt("%.0f", lat.P(0.5)) + " p99us=" + Fmt("%.0f", lat.P(0.99));
}
std::string ReadLine(const Bench::ReadStats& r, Lat& lat) {
  return "gets=" + std::to_string(r.gets) + " hits=" + std::to_string(r.hits) +
         " bad=" + std::to_string(r.bad) +
         " GBps=" + Fmt("%.3f", r.bytes / 1e9 / std::max(r.secs, 1e-9)) +
         " p50us=" + Fmt("%.0f", lat.P(0.5)) + " p99us=" + Fmt("%.0f", lat.P(0.99));
}

struct Case {
  const char* engine;
  size_t len;
};

class LseBench : public ::testing::TestWithParam<Case> {};

// Removes what an earlier, interrupted run left, and reports the free
// space (every case does the same when it starts).
TEST(LseBenchSetup, RemoveLeftovers) {
  const std::string root = BenchRoot();
  if (root.empty()) GTEST_SKIP() << "no test ring given";
  const std::string removed = RemoveLeftovers(root);
  ADD_FAILURE() << "report (expected on a ring; not a failure):\nLSE-BENCH setup free_gb="
                << FreeBytes(root) / 1000000000 << " root=" << root
                << " removed=" << (removed.empty() ? " none" : removed);
}

constexpr int kThreads = 8;
// Planned client bytes of a CapacityThroughputRemoveRestart case: a first
// fill, two steady capacities, a quarter with one writer, an eighth by the
// killed child; a quarter more for relocations (DFKV_LSE_RELOCATE_PCT).
constexpr uint64_t kCapacityCasePlan = (kCapacity + 2 * kCapacity + kCapacity / 4 +
                                        kCapacity / 8) * 5 / 4;
// EvictionHitRatio: at most every GET of its two passes misses and PUTs.
constexpr uint64_t kZipfCasePlan = 2 * 3 * kCapacity * 5 / 4;

// Capacity, throughput by phase, REMOVE latency, kCacheFull, write
// amplification, restarts: one engine, one object size.
TEST_P(LseBench, CapacityThroughputRemoveRestart) {
  if (BenchRoot().empty()) GTEST_SKIP() << "no test ring given";
  const std::string root = StartCase(kCapacityCasePlan);
  if (root.empty()) return;
  const Case c = GetParam();
  const std::string dir =
      root + "/" + c.engine + "-" + std::to_string(c.len);
  Bench b(c.engine, c.len, kCapacity, dir);
  ASSERT_TRUE(b.Open());
  auto& r = b.R();
  const uint64_t fill = kCapacity / c.len;  // objects in one capacity's worth
  uint64_t next = 0;

  r.Line("open", "fill_objects=" + std::to_string(kCapacity / c.len));
  // First fill: one capacity's worth into a store never written.
  {
    Lat lat;
    const auto w = b.Write(next, fill, kThreads, &lat);
    next += fill;
    r.Line("first_fill", WriteLine(w, lat));
  }
  // Steady state: two more capacities, every extent / segment reused.
  uint64_t client_bytes = 0;
  const uint64_t relocated_before = EngineMetric(b.G(), "dfkv_lse_relocated_bytes_total");
  const char* kWaits[] = {"dfkv_lse_inline_reclaims_total",
                          "dfkv_lse_inline_reclaim_microseconds_total",
                          "dfkv_lse_sealer_queue_waits_total",
                          "dfkv_lse_sealer_queue_wait_microseconds_total",
                          "dfkv_lse_cleaner_waits_total",
                          "dfkv_lse_cleaner_wait_microseconds_total"};
  uint64_t waits_before[6];
  for (int i = 0; i < 6; ++i) waits_before[i] = EngineMetric(b.G(), kWaits[i]);
  auto waits = [&] {
    static const char* kShort[] = {"inline_reclaims", "inline_reclaim_us", "sealer_waits",
                                   "sealer_wait_us", "cleaner_waits", "cleaner_wait_us"};
    std::string out;
    for (int i = 0; i < 6; ++i)
      out += std::string(" ") + kShort[i] + "=" +
             std::to_string(EngineMetric(b.G(), kWaits[i]) - waits_before[i]);
    return out;
  };
  {
    Lat lat;
    const auto w = b.Write(next, 2 * fill, kThreads, &lat);
    next += 2 * fill;
    client_bytes = w.bytes;
    r.Line("steady_write", WriteLine(w, lat));
    r.Line("steady_state", "kcachefull_pct=" +
                               Fmt("%.4f", 100.0 * w.full / std::max<uint64_t>(w.puts, 1)) +
                               " evictions=" + std::to_string(b.G().Evictions()) +
                               waits());
  }
  {
    Lat lat;
    const auto w = b.Write(next, fill / 4, 1, &lat);  // one writer
    next += fill / 4;
    client_bytes += w.bytes;
    r.Line("steady_write_1t", WriteLine(w, lat));
  }
  const uint64_t relocated =
      EngineMetric(b.G(), "dfkv_lse_relocated_bytes_total") - relocated_before;
  const size_t count = b.G().Count();
  r.Line("capacity",
         "objects=" + std::to_string(count) + " effective_pct=" +
             Fmt("%.2f", 100.0 * count * c.len / kCapacity) + " used_bytes=" +
             std::to_string(b.G().UsedBytes()) + " write_amp_pct=" +
             Fmt("%.2f", 100.0 * relocated / std::max<uint64_t>(client_bytes, 1)));
  // Reads: whole values of random recently written keys (O_DIRECT).
  {
    Lat lat;
    const uint64_t lo = next > 2 * count ? next - 2 * count : 0;
    const auto rd = b.Read(lo, next, std::min<uint64_t>(4 * count, 20000), kThreads, &lat);
    r.Line("read", ReadLine(rd, lat) + " (hit latencies)");
    EXPECT_EQ(rd.bad, 0u);
  }
  // REMOVE latency, and GET latency (hits) on other threads during the
  // burst: a quarter of the stored objects, at most 2000, each removed once.
  {
    Lat rm, get;
    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
      readers.emplace_back([&, t] {
        std::mt19937_64 rng(77 + t);
        Aligned buf(((c.len + 4095) / 4096) * 4096 + 8192);
        while (!stop) {
          const uint64_t id = next - 1 - rng() % std::max<uint64_t>(count, 1);
          const char* data = nullptr;
          size_t got = 0;
          const auto a = Clock::now();
          if (b.G().RangeDirect(Key(id), 0, 0, buf.data.get(), buf.size, &data, &got) ==
              Status::kOk)
            get.Add(Seconds(a, Clock::now()) * 1e6);
        }
      });
    std::vector<uint64_t> stored;
    for (uint64_t id = next > 3 * count ? next - 3 * count : 0; id < next; ++id)
      if (b.G().IsCached(Key(id))) stored.push_back(id);
    std::mt19937_64 rng(5);
    std::shuffle(stored.begin(), stored.end(), rng);
    stored.resize(std::min<size_t>(stored.size() / 4, 2000));
    for (uint64_t id : stored) {
      const auto a = Clock::now();
      b.G().Remove(Key(id));
      rm.Add(Seconds(a, Clock::now()) * 1e6);
    }
    stop = true;
    for (auto& x : readers) x.join();
    r.Line("remove", "n=" + std::to_string(rm.N()) + " p50us=" + Fmt("%.0f", rm.P(0.5)) +
                         " p99us=" + Fmt("%.0f", rm.P(0.99)) + " get_during_p99us=" +
                         Fmt("%.0f", get.P(0.99)));
  }
  // Restart after a clean close.
  const size_t before = b.G().Count();
  b.Close();
  ASSERT_TRUE(b.Open());
  r.Line("restart_clean", "open_secs=" + Fmt("%.3f", b.OpenSecs()) + " recovered_pct=" +
                              Fmt("%.2f", 100.0 * b.G().Count() / std::max<size_t>(before, 1)));
  // Restart after a kill: a child writes into the store and exits without
  // closing it (no destructors), as a kill -9 leaves it.
  b.Close();
  const uint64_t extra = std::max<uint64_t>(fill / 8, 1);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    Bench k(c.engine, c.len, kCapacity, dir);
    if (!k.Count(b.Written()) || !k.Open()) ::_exit(2);
    Lat lat;
    k.Write(next, extra, kThreads, &lat);
    const size_t n = k.G().Count();
    FILE* f = std::fopen((dir + ".count").c_str(), "w");
    if (f) {
      std::fprintf(f, "%zu\n", n);
      std::fclose(f);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  size_t at_kill = 0;
  if (FILE* f = std::fopen((dir + ".count").c_str(), "r")) {
    if (std::fscanf(f, "%zu", &at_kill) != 1) at_kill = 0;
    std::fclose(f);
  }
  std::remove((dir + ".count").c_str());
  ASSERT_TRUE(b.Open());
  r.Line("restart_kill", "open_secs=" + Fmt("%.3f", b.OpenSecs()) + " recovered_pct=" +
                             Fmt("%.2f", 100.0 * b.G().Count() / std::max<size_t>(at_kill, 1)));
  r.Line("footprint", "client_gb=" + Fmt("%.1f", (b.Written() + extra * c.len) / 1e9) +
                          " free_gb_end=" + std::to_string(FreeBytes(root) / 1000000000));
  ADD_FAILURE() << "report (expected on a ring; not a failure):\n" << r.out.str();
}

// Eviction quality: a Zipf-skewed key popularity over twice the capacity;
// each GET that misses PUTs the key. Hit ratio after a warm-up.
TEST_P(LseBench, EvictionHitRatio) {
  if (BenchRoot().empty()) GTEST_SKIP() << "no test ring given";
  const std::string root = StartCase(kZipfCasePlan);
  if (root.empty()) return;
  const Case c = GetParam();
  const std::string dir = root + "/zipf-" + std::string(c.engine) + "-" + std::to_string(c.len);
  Bench b(c.engine, c.len, kCapacity, dir);
  ASSERT_TRUE(b.Open());
  const uint64_t keys = 2 * kCapacity / c.len;
  // Zipf(0.9) over `keys` by inverse CDF on a precomputed table.
  std::vector<double> cdf(keys);
  double sum = 0;
  for (uint64_t i = 0; i < keys; ++i) sum += 1.0 / std::pow(double(i + 1), 0.9);
  double acc = 0;
  for (uint64_t i = 0; i < keys; ++i) {
    acc += 1.0 / std::pow(double(i + 1), 0.9) / sum;
    cdf[i] = acc;
  }
  auto pick = [&](std::mt19937_64& rng) {
    const double u = std::uniform_real_distribution<double>(0, 1)(rng);
    return static_cast<uint64_t>(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
  };
  std::atomic<uint64_t> gets{0}, hits{0};
  auto run = [&](uint64_t n, bool count) {
    std::atomic<int64_t> left{static_cast<int64_t>(n)};  // signed: goes below 0
    std::vector<std::thread> th;
    for (int t = 0; t < kThreads; ++t)
      th.emplace_back([&, t] {
        std::mt19937_64 rng(31 + t + (count ? 100 : 0));
        Aligned buf(((c.len + 4095) / 4096) * 4096 + 8192);
        std::memset(buf.data.get(), 0x33, buf.size);
        while (left.fetch_sub(1) > 0) {
          const uint64_t id = pick(rng);
          const char* data = nullptr;
          size_t got = 0;
          const Status st = b.G().RangeDirect(Key(id), 0, 0, buf.data.get(), buf.size,
                                              &data, &got);
          if (count) ++gets;
          if (st == Status::kOk) {
            if (count) ++hits;
            continue;
          }
          std::memcpy(buf.data.get(), &id, std::min<size_t>(8, c.len));
          if (!b.Count(c.len)) break;
          b.G().CacheDirect(Key(id), buf.data.get(), c.len, buf.size);
        }
      });
    for (auto& x : th) x.join();
  };
  const uint64_t ops = 3 * kCapacity / c.len;
  run(ops, false);  // warm-up: fills the store and starts eviction
  const uint64_t relocated_before = EngineMetric(b.G(), "dfkv_lse_relocated_bytes_total");
  const uint64_t written_before = b.Written();
  run(ops, true);
  const uint64_t relocated =
      EngineMetric(b.G(), "dfkv_lse_relocated_bytes_total") - relocated_before;
  EXPECT_LE(b.Written(), kMaxCaseBytes) << "stopped: the case reached its write bound";
  b.R().Line("eviction_zipf0.9_2x",
             "gets=" + std::to_string(gets.load()) + " hit_pct=" +
                 Fmt("%.2f", 100.0 * hits / std::max<uint64_t>(gets, 1)) + " evictions=" +
                 std::to_string(b.G().Evictions()) + " client_gb=" +
                 Fmt("%.1f", b.Written() / 1e9) + " measured_client_gb=" +
                 Fmt("%.2f", (b.Written() - written_before) / 1e9) + " relocated_gb=" +
                 Fmt("%.2f", relocated / 1e9) + " write_amp_pct=" +
                 Fmt("%.2f", 100.0 * relocated /
                                 std::max<uint64_t>(b.Written() - written_before, 1)));
  ADD_FAILURE() << "report (expected on a ring; not a failure):\n" << b.R().out.str();
}

INSTANTIATE_TEST_SUITE_P(
    Engines, LseBench,
    ::testing::Values(Case{"slab", 1069548}, Case{"log", 1069548},     // 1.02 MiB
                      Case{"slab", 2873098}, Case{"log", 2873098},     // 2.74 MiB
                      Case{"slabwarm", 2873098},  // slab as deployed (warm restart)
                      Case{"slab", 27158118}, Case{"log", 27158118}),  // 25.9 MiB
    [](const ::testing::TestParamInfo<Case>& info) {
      return std::string(info.param.engine) + "_" + std::to_string(info.param.len);
    });

// 1-byte markers: compared by object count (a slab slot per marker against
// a 4 KiB record).
TEST(LseBenchMarkers, ObjectCountForOneByteValues) {
  if (BenchRoot().empty()) GTEST_SKIP() << "no test ring given";
  // 100,000 records of a 4 KiB block each at most, per engine.
  const std::string root = StartCase(2 * 100000ull * 4096);
  if (root.empty()) return;
  std::ostringstream out;
  for (const char* engine : {"slab", "log"}) {
    const std::string dir = root + "/markers-" + engine;
    Bench b(engine, 1, kCapacity, dir);  // 4 GiB: the slab's markers are few
    ASSERT_TRUE(b.Open());
    Lat lat;
    const auto w = b.Write(0, 100000, kThreads, &lat);
    out << "LSE-BENCH engine=" << engine << " size=1 markers objects="
        << b.G().Count() << " capacity_bytes=" << (4ull << 30) << " "
        << WriteLine(w, lat) << "\n";
  }
  ADD_FAILURE() << "report (expected on a ring; not a failure):\n" << out.str();
}

}  // namespace
