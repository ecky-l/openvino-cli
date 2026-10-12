#include "util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace ovgenai {

std::string env_or(const std::string& key, const std::string& def,
                   bool treat_empty_as_missing) {
  const char* v = std::getenv(key.c_str());
  if (v == nullptr) return def;
  if (treat_empty_as_missing && v[0] == '\0') return def;
  return std::string(v);
}

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool env_flag_on(const std::string& key) {
  // Matches _STUB_ON = ("1", "true", "yes", "on"). Unset => false.
  const char* v = std::getenv(key.c_str());
  if (v == nullptr) return false;
  std::string s = lower(std::string(v));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

// --- base64 (RFC 4648, standard alphabet; decode also accepts URL-safe) ------
namespace {
const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64val(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;  // '-' = URL-safe
  if (c == '/' || c == '_') return 63;  // '_' = URL-safe
  return -1;
}
}  // namespace

std::string base64_encode(const std::string& in) {
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  size_t i = 0;
  const size_t n = in.size();
  while (i + 3 <= n) {
    unsigned int triple = (static_cast<unsigned char>(in[i]) << 16) |
                          (static_cast<unsigned char>(in[i + 1]) << 8) |
                          (static_cast<unsigned char>(in[i + 2]));
    out.push_back(kB64[(triple >> 18) & 0x3F]);
    out.push_back(kB64[(triple >> 12) & 0x3F]);
    out.push_back(kB64[(triple >> 6) & 0x3F]);
    out.push_back(kB64[triple & 0x3F]);
    i += 3;
  }
  size_t rem = n - i;
  if (rem == 1) {
    unsigned int b = static_cast<unsigned char>(in[i]);
    out.push_back(kB64[(b >> 2) & 0x3F]);
    out.push_back(kB64[(b << 4) & 0x3F]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    unsigned int b = (static_cast<unsigned char>(in[i]) << 8) |
                     (static_cast<unsigned char>(in[i + 1]));
    out.push_back(kB64[(b >> 10) & 0x3F]);
    out.push_back(kB64[(b >> 4) & 0x3F]);
    out.push_back(kB64[(b << 2) & 0x3F]);
    out.push_back('=');
  }
  return out;
}

std::string base64_decode(const std::string& in) {
  std::string out;
  out.reserve(in.size() * 3 / 4);
  int val = 0, bits = -8;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
      if (c == '=') {  // padding terminates the data
        break;
      }
      continue;  // skip whitespace
    }
    int d = b64val(c);
    if (d < 0) continue;  // skip other non-alphabet bytes (be permissive on decode)
    val = (val << 6) | d;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

}  // namespace ovgenai
