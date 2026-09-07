// Fan-out client benchmark against an EXTERNAL cache server. Client objects
// remain alive throughout active and idle sampling; server RSS never includes
// the benchmark's payloads. Every requested PUT is attempted exactly once by
// this tool (transport retry policy remains in effect); readback checks bytes.
#include "cache/rdma_server.h"
#include "client/key_map.h"
#include "client/kv_client.h"
#include "transport/rdma_transport.h"
#include "utils/http_client.h"
#include "utils/net_util.h"
#include "utils/prom_parse.h"

#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
size_t Size(const char* text) {
  const std::string input(text);
  size_t value = 0;
  const auto parsed = std::from_chars(input.data(), input.data()+input.size(), value);
  if (parsed.ec != std::errc() || parsed.ptr == input.data())
    throw std::runtime_error("invalid nonnegative integer: " + input);
  const std::string suffix(parsed.ptr, input.data()+input.size());
  size_t factor = 1;
  if (suffix == "k" || suffix == "KiB") factor = 1ull<<10;
  else if (suffix == "m" || suffix == "MiB") factor = 1ull<<20;
  else if (suffix == "g" || suffix == "GiB") factor = 1ull<<30;
  else if (!suffix.empty()) throw std::runtime_error("invalid size suffix: " + input);
  if (value > std::numeric_limits<size_t>::max()/factor)
    throw std::runtime_error("size overflow");
  return value*factor;
}
std::string Scrape(const std::string& endpoint) {
  const int fd = dfkv::net::Dial(endpoint, 2000, 2000);
  if (fd < 0) throw std::runtime_error("cannot connect to metrics endpoint");
  const std::string req = "GET /metrics HTTP/1.0\r\nHost: " + endpoint +
                          "\r\nConnection: close\r\n\r\n";
  std::string response;
  if (!dfkv::net::WriteAll(fd, req.data(), req.size())) {
    ::close(fd); throw std::runtime_error("metrics request failed");
  }
  char buf[16384];
  ssize_t n;
  while ((n=::recv(fd, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, static_cast<size_t>(n));
    if (response.size() > (8u<<20)) break;
  }
  ::close(fd);
  int status = 0; long content = 0; size_t head = 0;
  if (n < 0 || !dfkv::ParseResponseHead(response, &status, &content, &head) ||
      status != 200) throw std::runtime_error("invalid metrics response");
  return response.substr(head);
}
uint64_t Metric(const std::string& body, const std::string& name) {
  uint64_t value = 0;
  if (!dfkv::PromMetricValue(body, name, &value))
    throw std::runtime_error("required metric missing: " + name);
  return value;
}
uint64_t Rss(size_t pid) {
  std::ifstream in("/proc/"+std::to_string(pid)+"/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:",0)==0) return std::stoull(line.substr(6))*1024;
  }
  throw std::runtime_error("server PID has no readable VmRSS");
}
struct Worker {
  std::unique_ptr<dfkv::RdmaTransport> transport;
  std::unique_ptr<dfkv::KVClient> client;
  std::vector<char> value;
  std::vector<char> readback;
};
}
int main(int argc, char** argv) {
  try {
    std::string member, metrics, key_seed = "leasebench-"+std::to_string(getpid());
    size_t clients=16, bytes=32ull<<20, ops=8, idle_ms=2000, sample_ms=50,
           pid=0, sg_segs=1;
    for (int i=1; i<argc; ++i) {
      const std::string flag(argv[i]);
      if (flag=="--help") {
        std::cout << "dfkvleasebench --member IP:RDMA_PORT --metrics IP:PORT --server-pid PID "
          "[--clients N --obj-size 32MiB --ops N --sg-segs N --idle-ms N --sample-ms N --key-seed S]\n"
          "Configure DFKV_RDMA_DEV/MAX_BLOCK_BYTES/INLINE_PUT_MAX_BYTES in the environment.\n";
        return 0;
      }
      if (++i==argc) throw std::runtime_error("missing value for "+flag);
      if (flag=="--member") member=argv[i];
      else if (flag=="--metrics") metrics=argv[i];
      else if (flag=="--key-seed") key_seed=argv[i];
      else if (flag=="--clients") clients=Size(argv[i]);
      else if (flag=="--obj-size") bytes=Size(argv[i]);
      else if (flag=="--ops") ops=Size(argv[i]);
      else if (flag=="--sg-segs") sg_segs=Size(argv[i]);
      else if (flag=="--server-pid") pid=Size(argv[i]);
      else if (flag=="--idle-ms") idle_ms=Size(argv[i]);
      else if (flag=="--sample-ms") sample_ms=Size(argv[i]);
      else throw std::runtime_error("unknown option: "+flag);
    }
    if (member.empty() || metrics.empty() || !pid || !clients || clients>1024 ||
        !bytes || !ops || !sg_segs || sg_segs>bytes || !sample_ms)
      throw std::runtime_error("invalid benchmark geometry; see --help");
    if (!dfkv::RdmaTransport::Available()) throw std::runtime_error("no RDMA device");
    std::vector<Worker> workers(clients);
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv;
    size_t initialized=0;
    bool start=false;
    std::atomic<size_t> done{0}, failures{0}, corrupt{0};
    std::array<std::atomic<uint64_t>, 16> statuses{};
    std::atomic<uint64_t> delivered{0};
    for (size_t c=0; c<clients; ++c) {
      threads.emplace_back([&, c] {
        bool announced=false;
        try {
          auto& w=workers[c];
          w.transport=std::make_unique<dfkv::RdmaTransport>();
          w.client=std::make_unique<dfkv::KVClient>(
              std::vector<std::pair<std::string,std::string>>{{"bench",member}},
              "leasebench",w.transport.get());
          w.value.resize(bytes); w.readback.resize(bytes);
          for (size_t i=0;i<bytes;++i) w.value[i]=static_cast<char>((i*131+c*17)&255);
          { std::unique_lock<std::mutex> lk(mu); ++initialized; announced=true;
            cv.notify_all(); cv.wait(lk,[&]{return start;}); }
          for (size_t o=0;o<ops;++o) {
            const auto key=dfkv::ToBlockKey("leasebench",key_seed+"/"+std::to_string(c)+"/"+std::to_string(o));
            dfkv::Status status;
            if (sg_segs==1) {
              status=w.transport->CacheFrom(member,{{key,w.value.data(),bytes}})[0];
            } else {
              dfkv::CacheSrcMulti source; source.key=key;
              for(size_t s=0;s<sg_segs;++s) {
                const size_t begin=s*bytes/sg_segs,end=(s+1)*bytes/sg_segs;
                source.payloads.emplace_back(w.value.data()+begin,end-begin);
              }
              status=w.transport->CacheFromMulti(member,{source})[0];
            }
            const size_t index=static_cast<size_t>(status);
            statuses[std::min(index,statuses.size()-1)].fetch_add(1);
            if(status!=dfkv::Status::kOk) { ++failures; continue; }
            delivered.fetch_add(bytes);
            std::vector<uint64_t> lengths;
            std::fill(w.readback.begin(),w.readback.end(),0);
            const auto got=w.transport->RangeInto(member,{key},{{w.readback.data(),bytes}},&lengths);
            if(got[0]!=dfkv::Status::kOk || lengths.size()!=1 || lengths[0]!=bytes ||
               std::memcmp(w.value.data(),w.readback.data(),bytes)!=0) ++corrupt;
          }
        } catch(const std::exception& e) {
          ++failures;
          std::lock_guard<std::mutex> lk(mu);
          std::cerr << "worker " << c << ": " << e.what() << '\n';
          if(!announced) {++initialized; cv.notify_all();}
        }
        ++done;
      });
    }
    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk,[&]{return initialized==clients;}); }
    const auto begin=Clock::now();
    uint64_t peak_rss=0,peak_committed=0,peak_used=0;
    size_t samples=0;
    bool sampling_failed=false;
    auto sample=[&](const char* phase) {
      const auto body=Scrape(metrics);
      const auto rss=Rss(pid);
      const auto committed=Metric(body,"dfkv_rdma_recv_segment_bytes");
      const auto used=Metric(body,"dfkv_rdma_recv_segment_used_bytes");
      const auto connections=Metric(body,"dfkv_rdma_active_conns");
      peak_rss=std::max(peak_rss,rss); peak_committed=std::max(peak_committed,committed);
      peak_used=std::max(peak_used,used); ++samples;
      std::cout << "{\"phase\":\""<<phase<<"\",\"ms\":"
        <<std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-begin).count()
        <<",\"server_rss_bytes\":"<<rss<<",\"committed_bytes\":"<<committed
        <<",\"used_bytes\":"<<used<<",\"connections\":"<<connections<<"}\n";
    };
    try { sample("before"); } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';sampling_failed=true;}
    {std::lock_guard<std::mutex> lk(mu); start=true; cv.notify_all();}
    while(done.load()<clients) {
      try {sample("active");} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';sampling_failed=true;}
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
    }
    for(auto& t:threads)t.join();
    const double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    // Worker owns transport/client beyond thread exit: connections remain live
    // through the idle window, allowing MR/chunk reclamation to be observed.
    const auto idle_end=Clock::now()+std::chrono::milliseconds(idle_ms);
    do {
      try {sample("idle_clients_alive");} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';sampling_failed=true;}
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
    } while(Clock::now()<idle_end);
    std::cout << "{\"summary\":true,\"clients\":"<<clients<<",\"operations\":"<<clients*ops
      <<",\"put_failures\":"<<failures<<",\"readback_failures\":"<<corrupt
      <<",\"put_bytes\":"<<delivered<<",\"workload_wall_seconds\":"<<seconds
      <<",\"peak_server_rss_bytes\":"<<peak_rss<<",\"peak_committed_bytes\":"<<peak_committed
      <<",\"peak_used_bytes\":"<<peak_used<<",\"samples\":"<<samples<<",\"statuses\":[";
    for(size_t i=0;i<statuses.size();++i)std::cout<<(i?",":"")<<statuses[i];
    std::cout << "]}\n";
    return failures || corrupt || sampling_failed ? 1 : 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
