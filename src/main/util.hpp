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

}  // namespace ovgenai

#endif  // OVGENAI_UTIL_HPP
