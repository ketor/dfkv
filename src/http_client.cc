#include "http_client.h"

#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "net_util.h"

namespace dfkv {

std::string BuildPostRequest(const std::string& host_port, const std::string& path,
                             const std::string& body) {
  std::string r;
  r += "POST " + path + " HTTP/1.1\r\n";
  r += "Host: " + host_port + "\r\n";
  r += "Content-Type: application/json\r\n";
  r += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  r += "Connection: close\r\n";
  r += "\r\n";
  r += body;
  return r;
}

bool ParseResponseHead(const std::string& raw, int* status, long* content_length,
                       size_t* head_end) {
  size_t end = raw.find("\r\n\r\n");
  if (end == std::string::npos) return false;
  *head_end = end + 4;
  size_t sp = raw.find(' ');
  if (sp == std::string::npos) return false;
  *status = std::atoi(raw.c_str() + sp + 1);
  *content_length = -1;
  size_t pos = 0;
  while (pos < end) {
    size_t eol = raw.find("\r\n", pos);
    if (eol == std::string::npos || eol > end) eol = end;
    std::string line = raw.substr(pos, eol - pos);
    if (line.size() > 15) {
      std::string name = line.substr(0, 15);
      for (char& ch : name) ch = char(tolower((unsigned char)ch));
      if (name == "content-length:")
        *content_length = std::atol(line.c_str() + 15);
    }
    pos = eol + 2;
  }
  return true;
}

bool TcpHttpTransport::Post(const std::string& path, const std::string& body,
                            HttpResponse* out) {
  int fd = net::Dial(addr_, timeout_ms_, timeout_ms_);
  if (fd < 0) return false;
  std::string req = BuildPostRequest(addr_, path, body);
  if (!net::WriteAll(fd, req.data(), req.size())) { ::close(fd); return false; }

  std::string buf;
  char chunk[4096];
  int status = 0; long clen = -1; size_t head_end = 0;
  while (!ParseResponseHead(buf, &status, &clen, &head_end)) {
    ssize_t n = ::read(fd, chunk, sizeof(chunk));
    if (n <= 0) { ::close(fd); return false; }
    buf.append(chunk, n);
  }
  if (clen >= 0) {
    while (buf.size() - head_end < (size_t)clen) {
      ssize_t n = ::read(fd, chunk, sizeof(chunk));
      if (n <= 0) break;
      buf.append(chunk, n);
    }
    out->body = buf.substr(head_end, clen);
  } else {
    ssize_t n;
    while ((n = ::read(fd, chunk, sizeof(chunk))) > 0) buf.append(chunk, n);
    out->body = buf.substr(head_end);
  }
  out->status = status;
  ::close(fd);
  return true;
}

}  // namespace dfkv
