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
// -- the pipeline gets them through this class instead.
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

// The scheduler knobs, each std::optional so "nullopt" means "leave the engine
// default".
// The same field set that used to hang off ModelLoadConfig.
struct ModelSchedulerConfig {
  std::optional<std::size_t> max_num_batched_tokens;
  std::optional<std::size_t> num_kv_blocks;
  std::optional<std::size_t> cache_size;
  std::optional<std::size_t> num_linear_attention_blocks;
  std::optional<std::size_t> cache_interval_multiplier;
  std::optional<bool>        dynamic_split_fuse;
  std::optional<std::size_t> max_num_seqs;
  std::optional<bool>        enable_prefix_caching;
  std::optional<bool>        use_cache_eviction;
  std::optional<bool>        use_sparse_attention;

  bool operator==(const ModelSchedulerConfig&) const = default;

  // --- The "Scheduler Options" boost section this class owns ---------------
  //
  // The 10 KV-cache / batching flags, each with a single-letter short only where
  // its first letter is unclaimed by a longer-name prefix (see
  // schedulerOptionsDesc() for the exact rules).  Single source of truth for
  // both usage() (its contribution to the shared --help) and
  // parseModelSchedulerConfig (spliced alongside the load + request sections
  // the other classes own).
  static po::options_description schedulerOptionsDesc();

  // This class's slice of the --help listing (its "Scheduler Options" section
  // only); the combined three-section view is config.cpp's combinedConfigUsage.
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
