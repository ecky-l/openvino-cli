// ovcli's scheduler config: the KV-cache / batching knobs that shape how a
// loaded pipeline schedules and caches.  It is the first-class sibling of
// ModelLoadConfig / ModelRequestConfig (see those headers): one source of truth
// that
//   * owns the "Scheduler Options" boost section (schedulerOptionsDesc / usage),
//     shown as its own header in `ovcli <cmd> --help`,
//   * parses them from the CLI (parseModelSchedulerConfig) and from a JSON
//     object (modelSchedulerConfigFromJson), and serialises them (…ToJson),
//   * is the C++ holder for those knobs: every field is std::optional so
//     nullopt means "leave the engine default";  toSchedulerConfig() then builds
//     the ov::genai::SchedulerConfig an LLMPipeline / VLMPipeline consumes,
//     forwarding only the truthy fields (has_value() and, for size/bool,
//     nonzero / true), so an explicit 0 or false leaves the engine default alone.
//
// It absorbed the former GenAISchedulerConfig from the engine-config layer: that
// typed analog now lives here, in ovcli::config, so the one set of scheduler
// fields is owned by exactly one class.  ModelLoadConfig no longer carries them
// -- the pipeline gets them through this class instead.  (The 10 fields are
// kept in *alphabetical order*,  so the struct,  the --help listing,  and the
// JSON round-trip all read the same way top-to-bottom.)
#ifndef OVCLI_MODEL_SCHEDULER_CONFIG_HPP
#define OVCLI_MODEL_SCHEDULER_CONFIG_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>               // for po::options_description in schedulerOptionsDesc
#include <openvino/genai/scheduler_config.hpp>     // for toSchedulerConfig() -> the engine's typed config

#include "config.hpp"  // ConfigError + the parse helpers all sections share

namespace ovcli {
namespace config {

// The 10 scheduler knobs, each std::optional so "nullopt" means "leave the
// engine default",  in *alphabetical field order* (the flags / the --help /
// the JSON keys all follow the same order).  The comment to the right of each
// is the knob it is,  echoing the matching --help line.
struct ModelSchedulerConfig {
  std::optional<std::size_t> cache_interval_multiplier;  // linear-attention checkpoint interval x for prefix caching (interval = KV block size * x; default 8, auto when unset; 0 only when prefix caching is off)
  std::optional<std::size_t> cache_size;                // total KV-cache size in GB; 0 = dynamic allocation (when both this & num_kv_blocks are set, num_kv_blocks wins)
  std::optional<bool>        dynamic_split_fuse;         // split prompt/generate into different scheduling phases (default true; off -> prompts longer than the batch error)
  std::optional<bool>        enable_prefix_caching;      // keep prior KV-caches in memory for reuse (default false; on for the LLMPipeline client scenario; more RAM)
  std::optional<std::size_t> max_num_batched_tokens;    // max tokens in a single batch step (default 256; unbounded in the LLMPipeline client scenario)
  std::optional<std::size_t> max_num_seqs;              // max scheduled sequences = batch size on the generate phase (default 256)
  std::optional<std::size_t> num_kv_blocks;            // total KV-cache blocks available to the scheduler (default 0 = dynamic)
  std::optional<std::size_t> num_linear_attention_blocks; // total linear-attention (state/SSM) blocks (default 0 = auto from max_num_seqs when present)
  std::optional<bool>        use_cache_eviction;         // cap per-sequence KV usage (memory saved at a quality cost; default false)
  std::optional<bool>        use_sparse_attention;        // apply block-wise sparse attention to the prefill stage (default false)

  bool operator==(const ModelSchedulerConfig&) const = default;

  // --- The "Scheduler Options" boost section this class owns ---------------
  //
  // The 10 KV-cache / batching flags, in alphabetical-flag order, long-form only
  // (the scheduler names start with c/m/n/u,  all of which are claimed as
  // single-letter prefixes by the load / request sections,  so the schedulers
  // keep long form).  Single source of truth for both usage() (its contribution
  // to the shared --help) and parseModelSchedulerConfig (spliced alongside the
  // load + request sections the other classes own).
  static po::options_description schedulerOptionsDesc();

  // This class's slice of the --help listing (its "Scheduler Options" section
  // only); the combined four-section view is config.cpp's combinedConfigUsage.
  static std::string usage();

  // True when at least one field is non-nullopt (i.e. the block is "authored").
  [[nodiscard]] bool hasAnySet() const;

  // Build the ov::genai::SchedulerConfig the pipeline consumes, forwarding only
  // the truthy fields of this config (has_value() and, for size/bool, nonzero /
  // true) so an explicit 0 or false leaves the engine default untouched.
  ov::genai::SchedulerConfig toSchedulerConfig() const;
};

// Parse + build a ModelSchedulerConfig from `args` (the argv after the
// subcommand name) via boost::program_options.  Parsed over the full option set
// (our section plus the load / request sections the other classes own, so the
// flags a caller also passes are "known", not "unknown option") but only the
// scheduler fields are read.  Throws ConfigError on an unknown / invalid flag.
ModelSchedulerConfig parseModelSchedulerConfig(const std::vector<std::string>& args);

// Parse a scheduler_config JSON object into a ModelSchedulerConfig: the flat
// form with the 10 fields at the top level -- exactly what `show-config` emits
// under the "scheduler_config" key and what a `LOAD` message nests under its
// load_config.  Unknown keys are silently ignored (matching the tolerant readers
// in the sibling configs); a key with the wrong JSON type throws ConfigError.
ModelSchedulerConfig modelSchedulerConfigFromJson(const std::string& json);

// Serialise a config to a compact scheduler_config JSON object (only the set
// fields; all-unset -> the empty object).  Round-trips through
// modelSchedulerConfigFromJson.
std::string modelSchedulerConfigToJson(const ModelSchedulerConfig& config);

}  // namespace config
}  // namespace ovcli

#endif  // OVCLI_MODEL_SCHEDULER_CONFIG_HPP
