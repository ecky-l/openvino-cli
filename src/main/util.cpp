#include "util.hpp"

#include "config.hpp"                  // config::ConfigError (thrown on a model not in the cache)

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

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

// --- HuggingFace cache resolution (the `hf://` model-scheme helpers) ----------
// See util.hpp for the contract.  This touches nothing but the filesystem + the
// three cache env vars -- it never downloads, and returns a plain model path
// verbatim before any of it runs.

// Expand a leading "~" / "~/" to $HOME; anything else (the normal case -- the HF
// env vars are absolute) passes through untouched.  A "~user" is out of scope.
static std::string expandTilde(const std::string& s) {
  if (s == "~" || s.rfind("~/", 0) == 0) {
    const char* h = std::getenv("HOME");
    if (h == nullptr || h[0] == '\0') return s;  // no $HOME: leave the literal; a later step fails loudly
    return std::string(h) + s.substr(1);         // drop the leading "~"
  }
  return s;
}

// Is `seg` a HuggingFace repo-type marker -- the leading segment huggingface_hub
// absorbs into the "models--" / "datasets--" / "spaces--" dir prefix?  Case-
// insensitive; the singulars ("model"/"dataset"/"space") are the example's form.
static bool isRepoTypeSeg(const std::string& seg) {
  std::string s = seg;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s == "model"   || s == "models"   ||
         s == "dataset" || s == "datasets" ||
         s == "space"   || s == "spaces";
}

std::string hfHubDir() {
  // $HF_HUB_CACHE wins outright; else $HF_HOME/hub; else $HOME/.cache/huggingface/hub
  // -- huggingface_hub's own fall-through, expanded the same way.
  const std::string hub = env_or("HF_HUB_CACHE", "", true);
  if (!hub.empty()) return expandTilde(hub);
  const std::string home = env_or("HF_HOME", "", true);
  if (!home.empty()) return expandTilde(home) + "/hub";
  const std::string h = env_or("HOME", "", true);
  return (h.empty() ? std::string("~") : expandTilde(h)) + "/.cache/huggingface/hub";
}

std::string resolveModelPathIn(const std::string& model, const std::string& hubDir) {
  constexpr const char* kScheme = "hf://";
  if (model.rfind(kScheme, 0) != 0) return model;  // not a `hf://` reference: verbatim, no I/O
  const std::string name = model.substr(std::string(kScheme).size());  // ref for `hf download <name>`

  // The repo dir is `models--` + the repo id, its path components (separated by
  // '/') written as '--'.  A leading repo-type segment is absorbed by the constant
  // `models--` prefix (the example's constant "first part"), so
  // `model/tiny-random/qwen3.5` names `.../models--tiny-random--qwen3.5` exactly
  // as huggingface_hub lays it out; a plain `org/name` reference (no type segment)
  // keeps its full id.
  std::vector<std::string> parts;
  std::string cur;
  std::istringstream ps(name);
  while (std::getline(ps, cur, '/')) parts.push_back(cur);
  if (parts.size() > 1 && isRepoTypeSeg(parts.front()))
    parts.erase(parts.begin());
  std::string repoId;
  for (const auto& p : parts)                       // skip empties so "a//b" can't double-dash
    if (!p.empty())
      repoId += (repoId.empty() ? "" : "--") + p;
  const std::string repoDir = hubDir + "/models--" + repoId;

  // A model not (fully) in the cache is reported -- not downloaded -- with the
  // exact fetch command the operator needs: `<name>` is the reference minus
  // "hf://" (per the request), so the message names the model on its own.
  auto requireInCache = [&]() -> std::string {
    return "model '" + model + "' is not in the HuggingFace cache (" + repoDir +
           "); it must be downloaded first with:  hf download " + name;
  };

  // `<repo-dir>/refs/main` holds the committed snapshot name -- a git-like ref,
  // one per line as "<rev> <hash>" (a plain tree is a bare <hash>).  The last
  // whitespace-delimited token of the file is therefore the snapshot hash that
  // names the dir under <repo-dir>/snapshots/.
  std::ifstream refIs(repoDir + "/refs/main");
  if (!refIs) throw config::ConfigError(requireInCache());   // repo not in the cache
  std::string ref;
  std::string tok;
  while (refIs >> tok) ref = tok;                          // the final token = the snapshot hash
  if (ref.empty()) throw config::ConfigError(requireInCache());   // refs/main was empty

  const std::string snapshot = repoDir + "/snapshots/" + ref;
  namespace fs = std::filesystem;
  if (!fs::exists(snapshot) || !fs::is_directory(snapshot))
    throw config::ConfigError(requireInCache());   // the ref names a missing snapshot
  return snapshot;
}

std::string resolveModelPath(const std::string& model) {
  // Only an "hf://" reference touches the cache (env look-up + reads); a plain
  // model path returns before any of that, so the common case does no I/O.
  if (model.rfind("hf://", 0) != 0) return model;
  return resolveModelPathIn(model, hfHubDir());
}

}  // namespace ovgenai
