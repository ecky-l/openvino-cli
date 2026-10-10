// The single model one ovcli process loads.
//
// A supervisor may also push this down a pipe via a `LOAD` protocol message
// carrying a `load_config` JSON string. ovcli is decoupled from that: it hosts
// exactly ONE model, so its load config is a single-model view of the same knobs
// and arrives from the command line (for the long-lived `worker`) and the
// process's own defaults (for the one-shot `run`), and can round-trip as JSON so
// the struct is a self-describing answer to "which model, what settings".
//
// Fields ovcli deliberately does not expose, and why:
//   * worker_max_respawns -- a *supervisor-side* respawn budget. ovcli IS the
//     worker; whoever respawns it owns that budget.
//   * context_window      -- an OpenAI-/v1_models *advertising* knob the
//     HTTP server reports; ovcli has no such face, so nothing to advertise.
//   * worker_line_limit   -- a *transport* knob (max bytes per line of the
//     stdin/stdout protocol loop); only `ovcli worker` has such a loop.
// The load flag map + its single-letter alias rules live below
// (loadOptionsDesc / usage); the argv/boost/number/JSON coercions both sections
// share live in config.hpp.
#ifndef OVCLI_MODEL_LOAD_CONFIG_HPP
#define OVCLI_MODEL_LOAD_CONFIG_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // for po::options_description in loadOptionsDesc
#include "config.hpp"                  // ConfigError + the parse helpers both sections share

namespace ovcli {
namespace config {

// The load knobs this struct carries, in declaration order (see the note above
// for the three fields deliberately left out).
struct ModelLoadConfig {
  // --- required: there is exactly one model ---
  std::string model;  // dir of the OpenVINO IR model; the model's identity
  std::string model_type;  // one of kModelTypes (defined in model_load_config.cpp)
  std::string engine;      // one of kEngines (defined in model_load_config.cpp)
  std::string device;      // OpenVINO device string, e.g. "CPU", "GPU", "NPU"

  // --- optional load knobs ---
  std::optional<std::string> cache_dir;               // OpenVINO CACHE_DIR path
  std::optional<std::string> draft_model;        // speculative-decoding draft
  std::optional<std::string> draft_device = std::string("CPU");  // default "CPU"

  // The KV-cache / batching knobs (the former 10 scheduler_config fields) now
  // live in ModelSchedulerConfig (model_scheduler_config.hpp) -- the pipeline
  // gets them through that class instead of off the load config.

  // --- free-form JSON blobs (authored; carried verbatim) ---
  // "{}" means "the empty object" for the two that default to {}.
  std::string runtime_config_json = "{}";     // OpenVINO runtime properties

  bool operator==(const ModelLoadConfig&) const = default;

  // --- The "Model load parameters" boost section this class owns ------------
  //
  // The single source of truth for the load flags (the four required + the
  // optional load / free-JSON knobs, each with a single-letter short where its
  // first letter is not claimed by a longer-name prefix: see the note at the top
  // and loadOptionsDesc()).  The KV-cache / batching knobs used to sit here; they
  // now have their own class (ModelSchedulerConfig), whose section this one splices
  // alongside.  Feeds both usage() (its contribution to the shared --help) and
  // parseModelLoadConfig (spliced alongside the request + scheduler sections the
  // other classes own).
  static po::options_description loadOptionsDesc();

  // This class's slice of the --help listing (its "Model load parameters"
  // section only); the combined three-section view is config.cpp's
  // combinedConfigUsage.
  static std::string usage();
};

// Parse + validate the load-model flags from `args` (the argv after the
// subcommand name) via boost::program_options, returning the one ModelLoadConfig
// both `ovcli worker` and `ovcli run` share. Parsed over all three sections (our
// own plus the request + scheduler sections the sibling classes own, so the
// flags a caller also passes are "known", not "unknown option") but only the
// load fields are read. Throws ConfigError on a missing / unknown / invalid flag.
ModelLoadConfig parseModelLoadConfig(const std::vector<std::string>& args);

// Parse a load_config JSON object into a ModelLoadConfig. ovcli deliberately
// *tolerates* unknown top-level keys -- it consumes the single model it uses and
// drops the rest (a strict schema would reject them outright), since the config
// no longer travels inside a fixed schema. Throws ConfigError on a malformed /
// non-object value or a missing required field.
ModelLoadConfig modelLoadConfigFromJson(const std::string& json);

// Serialize a config back to compact JSON using only the field set ovcli uses
// (context_window / worker_max_respawns / worker_line_limit are absent).
// Round-trips through modelLoadConfigFromJson.
std::string modelLoadConfigToJson(const ModelLoadConfig& config);

// A single-line human summary of the resolved load config (for logging what
// resolved).
std::string describeConfig(const ModelLoadConfig& config);

}  // namespace config
}  // namespace ovcli

#endif  // OVCLI_MODEL_LOAD_CONFIG_HPP
