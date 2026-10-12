// Shared config *plumbing* for ovcli -- the pieces that BOTH the model-load
// config and the per-request config depend on, declared once here and defined
// in config.cpp.
//
// It deliberately owns NO parameters: each boost option section (the actual
// flag definitions, shown under their own "Model load parameters" / "Model
// request parameters" header in --help) is defined by, and lives in, the class
// that describes it --
//   * model_load_config.{hpp,cpp}    -- ModelLoadConfig + loadOptionsDesc / usage / parser / its JSON
//   * model_request_config.{hpp,cpp} -- ModelRequestConfig + requestOptionsDesc / usage / parser / its JSON
// config.* keeps only what neither side owns on its own:
//   * ConfigError              -- the shared exception type;
//   * the argv / boost / number / JSON-object coercions shared by both parsers
//     (preProcess + rephraseBoost / inSet / joinList / parseSize /
//     requireJsonObject / compactJsonOf);
//   * showConfigJson           -- the one place that legitimately composes both
//     configs (the `show-config` subcommand's JSON);
//   * combinedConfigUsage      -- renders the combined --help view: each section
//     exactly as the class that owns it renders it, so the help text and the
//     parser cannot drift.
// (The three fields ovcli deliberately does not expose -- context_window /
// worker_max_respawns / worker_line_limit -- are documented at the top of
// model_load_config.hpp, beside the load flag map they would have sat in.)
#ifndef OVCLI_CONFIG_HPP
#define OVCLI_CONFIG_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <rapidjson/document.h>

namespace po = boost::program_options;

namespace ovcli {
namespace config {

// The three concrete config structs; their full definitions live in
// model_load_config.hpp / model_scheduler_config.hpp / model_request_config.hpp.
// Forward-declared here so showConfigJson (below) can take them by const
// reference without an include cycle -- config.cpp pulls in all three concrete
// headers to define showConfigJson and to render combinedConfigUsage.
struct ModelLoadConfig;
struct ModelSchedulerConfig;
struct ModelRequestConfig;
struct ModelRuntimeConfig;

// --- Help-column geometry, single-sourced here so every section's `--help`
//     agrees.  boost::program_options constructs options_description with
//     (caption, line_length = 80, min_description_length = 40);  with only a
//     caption that default is used,  and -- given the long runtime flags such as
//     --performance-hint-num-requests --  an 80 / 40 default cramps the
//     description (right) column down to ~40 characters.  These two wider knobs
//     are passed to every *OptionsDesc() (and the combined root):  a wider
//     overall line length gives the description column real room,  while the
//     flag-name (first) column is auto-sized by boost to its longest entry and
//     is left untouched.
inline constexpr unsigned kHelpLineLength           = 120;  // total help line width
inline constexpr unsigned kHelpMinDescriptionLength = 56;   // the description column floor

// Thrown when the operator's config arguments are missing, unknown, or invalid.
// Its message points at the offending flag; callers print it to stderr and exit
// non-zero -- just enough to be actionable. Defined in config.cpp; both
// classes' parsers throw it.
class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// --- Parse helpers shared by BOTH the model-load and the per-request parsers --
// Declared here (in ovcli::config) and defined in config.cpp, so both
// model_load_config.cpp and model_request_config.cpp can link to them; each
// also carries its own module-local helpers for the flags only it parses.

// True iff `v` is an element of the set `s`.
bool inSet(const std::string& v, const std::vector<std::string>& s);

// Join `v` with `sep` between consecutive elements.
std::string joinList(const std::vector<std::string>& v, const std::string& sep);

// Re-serialize any parsed/constructed JSON value compactly (used to canonicalize
// the free-form JSON blobs that travel between argv and JSON).
std::string compactJsonOf(rapidjson::Value& v);

// A JSON string that must be a JSON *object*, re-emitted compactly; an
// absent / malformed / non-object value is a ConfigError mentioning `flag`.
std::string requireJsonObject(const std::string& s, const char* flag);

// Parse `s` as a non-negative base-10 integer (sized for a size_t field); a
// malformed or negative value is a ConfigError mentioning `flag`.
long long parseSize(const std::string& s, const char* flag);

// Result of classifying `args` into options / values / positionals before the
// boost parse (see config.cpp for the preprocessor that drops `--` and maps a
// lone positional onto --model when no explicit --model is present).
struct PreProcessed {
  std::vector<std::string> args;         // args with the `--` terminator removed
  std::size_t positionalCount = 0;       // positionals seen (before/after `--`)
  bool explicitModelPath = false;        // an explicit --model / -m was present
};

PreProcessed preProcess(const std::vector<std::string>& args);

// Turn a boost parser error into a ConfigError in ovcli's own vocabulary
// ("unknown option" / "requires a value") so callers see consistent text.
ConfigError rephraseBoost(const po::error& e);

// The composite JSON for the `show-config` subcommand: a single object with
// four top-level keys -- "load_config" (modelLoadConfigToJson),
// "runtime_config" (modelRuntimeConfigToJson),  "scheduler_config"
// (modelSchedulerConfigToJson) and "request_config" (modelRequestConfigToJson).
std::string showConfigJson(const ModelLoadConfig& loadConfig,
                           const ModelRuntimeConfig& runtimeConfig,
                           const ModelSchedulerConfig& schedulerConfig,
                           const ModelRequestConfig& requestConfig);

// The combined --help view: renders the load, then runtime, then scheduler, then
// request sections (each exactly as its owning class renders it), so `ovcli
// <cmd> --help` shows all four parameter groups.  Owns no parameters -- it only
// composes the three-plus-one.
std::string combinedConfigUsage();

}  // namespace config
}  // namespace ovcli

#endif  // OVCLI_CONFIG_HPP
