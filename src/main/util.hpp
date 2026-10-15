// Small self-contained helpers (no third-party deps beyond the C++ stdlib).
// Kept deliberately tiny: base64 + env access, independent of the config layer,
// so earlier phases can use them before the config code exists.
#ifndef OVGENAI_UTIL_HPP
#define OVGENAI_UTIL_HPP

#include <optional>
#include <string>
#include <vector>

namespace ovgenai {

// getenv(key), or `def` when the key is unset or, when `treat_empty_as_missing`
// is true, when the value is an empty string (an empty value counts as missing).
std::string env_or(const std::string& key, const std::string& def,
                   bool treat_empty_as_missing = true);

// True if the env var is set to one of these (case-insensitive 1/true/yes/on).
bool env_flag_on(const std::string& key);

// RFC 4648 standard base64 (A-Za-z0-9+/ with '=' padding). Decode tolerates
// both the standard alphabet and '-'/'_' (URL-safe) so a future raw switch
// does not break; encode always emits standard base64.
std::string base64_encode(const std::string& in);
std::string base64_decode(const std::string& in);

// --- Reaching a model by its HuggingFace name: the `hf://` scheme -------------
//
// A `--model` value beginning "hf://" names a repo already downloaded into the
// HuggingFace cache (e.g. `hf://model/tiny-random/qwen3.5`), so the operator
// can name what they fetched instead of pasting the cache's opaque
// `.../snapshots/<git-hash>` path.  Resolution mirrors huggingface_hub's
// `try_to_load_from_cache` / `repo_id_to_path`:
//   * cache root -- $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub;
//   * repo dir   -- the repo id's '/' written as '--' and prefixed "models--";
//                   a leading repo-type segment -- model / datasets / space -- is
//                   absorbed by that "models--" prefix (it is the constant "first
//                   part" of the example), so `model/tiny-random/qwen3.5` folds to
//                   `models--tiny-random--qwen3.5`, the dir huggingface_hub creates;
//   * the ref    -- `<repo-dir>/refs/main` holds the committed snapshot name
//                   (a git-like ref: `<rev> <hash>`); the returned path is
//                   `<repo-dir>/snapshots/<hash>`, the concrete model dir.
// A value without the "hf://" scheme (a plain model path) is returned verbatim,
// so a model named by an absolute / relative path is entirely unaffected.
//
// If the named model is not (fully) present in the cache -- no repo dir, a
// missing / empty `refs/main`, or a ref pointing at an absent snapshot -- it is
// NOT downloaded; instead a config::ConfigError is thrown, telling the operator
// the model does not exist and that it must be fetched with
// `hf download <name>` (`<name>` = the reference with "hf://" stripped).  ovgenai
// never reaches out to the network -- a missing model is a usage error the
// operator resolves.

// The cache root ("hub" dir), the $HF_HUB_CACHE / $HF_HOME/hub /
// ~/.cache/huggingface/hub fall-through above.
std::string hfHubDir();

// Resolve `model` against an *explicit* cache root (bypassing the env look-up);
// a non-`hf://` value is returned verbatim.  This is the seam the tests drive
// (a scratch hub, not the real cache).  Throws config::ConfigError when the
// `hf://` reference is not (fully) in the cache (see above).
std::string resolveModelPathIn(const std::string& model, const std::string& hubDir);

// Resolve a `--model` value: `hf://<name>` is folded to its cache snapshot dir
// (above, using hfHubDir for the cache root); any other value is returned
// verbatim.  Throws config::ConfigError (a usage error, reported like a bad
// --model) when the `hf://` reference is not (fully) in the cache.
std::string resolveModelPath(const std::string& model);

}  // namespace ovgenai

#endif  // OVGENAI_UTIL_HPP
