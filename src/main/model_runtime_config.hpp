// ovcli's runtime config: the knobs that shape how a loaded pipeline is compiled
// and run on a device.  It is the first-class sibling of ModelLoadConfig /
// ModelSchedulerConfig / ModelRequestConfig (see those headers);  it absorbs the
// free-form `--runtime-config` JSON blob that used to hang off ModelLoadConfig.
//
// The parameter set is the full set of *settable* (read-write) OpenVINO Runtime
// API properties -- the ones an operator can actually name for a pipeline --
// taken verbatim from OpenVINO's own properties header
// (src/inference/include/openvino/runtime/properties.hpp;  the hint / device /
// log / streams namespaces),  plus ATTENTION_BACKEND,  the one genai-specific
// runtime property GenAIPipelineConfig keys its SDPA-vs-paged gate on.
// Read-only properties a user can't set (AVAILABLE_DEVICES / NETWORK_NAME /
// *NUMBER_OF_INFER_REQUESTS / SUPPORTED_PROPERTIES / ...) are deliberately
// excluded.  Two moves about the device / cache knobs are worth noting:
//   * CACHE_DIR  (ov::cache_dir)  used to sit on ModelLoadConfig,  but it is
//     itself an OpenVINO Runtime property,  so it belongs HERE,  in this class.
//   * DEVICE_ID  (ov::device_id:  id 0 = first device,  1 = second,  ...) is only
//     the *instance* filter on a plugin's available_devices --  but the device
//     itself is the load config's --device,  the positional argument OpenArc /
//     the worker pass to the LLMPipeline / VLMPipeline / ...  constructor;  it is
//     never set as a property,  so DEVICE_ID stays excluded,  overlapping the
//     load config's primary device selection.
//
// Each parameter is identified by ONE name -- its C++ (lower_snake) field -- and
// every other surface is a deterministic transform of it:
//   * the OpenVINO Runtime property key (UPPER_SNAKE,  e.g. KV_CACHE_PRECISION)
//     is the field upper-cased;  toProperties() writes the value under it into
//     the ov::AnyMap handed to the LLMPipeline / VLMPipeline / ASRPipeline  ...
//     constructor's `properties` argument;
//   * the "runtime_config" JSON key  (show-config + a LOAD message) is the field
//     name,  and `...ToJson` / `...FromJson` round-trip on it;
//   * the CLI flag (lowercase,  "trailing -,  _ substituted by -",  e.g.
//     --kv-cache-precision) is the field dashed;  the "Runtime config options"
//     section is long-form only (with ~28 of them a single letter is no longer a
//     sensible alias).
//   (So `num_requests` is NOT a field:  the canonical PERFORMANCE_HINT_NUM_REQUESTS
//    maps to the field `performance_hint_num_requests`,  keeping the field ==
//    lowercased-property invariant.  The few multi-word values that are
//    themselves the whole property,  e.g. execution_mode_hint for
//    EXECUTION_MODE_HINT,  likewise keep the field == lowercased-property form.)
//
// Every field is std::optional so nullopt (or an empty string for the string
// fields) means "not set":  such a field is dropped from the JSON and from the
// emitted ov::AnyMap,  so an unmentioned runtime config is the empty object and
// the engine's own default is left to the engine,  while a field an operator
// names (including 0 / false,  which are valid explicit values) is forwarded.
#ifndef OVCLI_MODEL_RUNTIME_CONFIG_HPP
#define OVCLI_MODEL_RUNTIME_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <boost/program_options.hpp>        // for po::options_description in runtimeOptionsDesc
#include <openvino/core/any.hpp>            // for toProperties() -> the ov::AnyMap a pipeline consumes

#include "config.hpp"  // ConfigError + the parse / number / JSON helpers all sections share

namespace ovcli {
namespace config {

// The runtime knobs, in *alphabetical field order* (the field below IS the
// lower-cased OpenVINO property name,  so reading the field names top to bottom
// reads the properties top to bottom),  each std::optional.  The comment to the
// right of each is a one-line echo of the --help text in model_runtime_config.cpp
// (runtimeOptionsDesc):  the property key toProperties() emits it under,  with a
// short paraphrase of the value OpenVINO accepts (a hint / enum field is a token;
// a *precision is an element type name like f16 / f32;  the numbers and booleans
// are their value).
struct ModelRuntimeConfig {
  // The 28 fields in *alphabetical order by field name* (the field is also the
  // lower-cased property name,  so top-to-bottom is also the property order);
  // std types are interleaved,  not grouped,  to keep the alphabet strict.
  std::optional<double>       activations_scale_factor;    // ACTIVATIONS_SCALE_FACTOR : factor easing f16 activation overflow
  std::optional<std::string> attention_backend;           // ATTENTION_BACKEND        : PA (paged, continuous batching) / SDPA (default)
  std::optional<std::size_t> auto_batch_timeout;          // AUTO_BATCH_TIMEOUT        : ms to await more inputs before flushing a batch
  std::optional<std::string> cache_dir;                   // CACHE_DIR                : plugin compile/blob cache dir (empty disables caching)
  std::optional<std::string> cache_mode;                  // CACHE_MODE               : OPTIMIZE_SIZE (small files) / OPTIMIZE_SPEED (default)
  std::optional<std::int32_t> compilation_num_threads;   // COMPILATION_NUM_THREADS  : max compile threads; negative / 0 = all cores
  std::optional<bool>        enable_cpu_pinning;          // ENABLE_CPU_PINNING         : bind threads to cores (win/linux; off on macOS)
  std::optional<bool>        enable_cpu_reservation;       // ENABLE_CPU_RESERVATION     : reserve cores from other plugins (off by default)
  std::optional<bool>        enable_hyper_threading;       // ENABLE_HYPER_THREADING     : use CPU hyper-threads
  std::optional<bool>        enable_mmap;                  // ENABLE_MMAP                : read the model via mmap (default on)
  std::optional<bool>        enable_weightless;            // ENABLE_WEIGHTLESS          : weightless cache (topology cached without its weights)
  std::optional<std::string> execution_mode_hint;         // EXECUTION_MODE_HINT        : PERFORMANCE (speed>accuracy) / ACCURACY (default)
  std::optional<bool>        force_tbb_terminate;          // FORCE_TBB_TERMINATE        : force-terminate TBB on core destruction
  std::optional<std::int32_t> inference_num_threads;     // INFERENCE_NUM_THREADS      : max inference threads; negative / 0 = all cores
  std::optional<std::string> inference_precision_hint;    // INFERENCE_PRECISION_HINT   : inference element type, e.g. f16 / f32 / bf16
  std::optional<std::size_t> key_cache_group_size;        // KEY_CACHE_GROUP_SIZE       : key-cache sub-channel quant group size; 0 = whole channel
  std::optional<std::string> key_cache_precision;         // KEY_CACHE_PRECISION        : key-cache element type (u4 / i8 / u8 / f16 / bf16 / f32)
  std::optional<std::string> kv_cache_precision;          // KV_CACHE_PRECISION         : umbrella KV-cache element type (bf16 / f16 / f32 / ...)
  std::optional<std::string> log_level;                   // LOG_LEVEL                  : LOG_NONE / LOG_ERROR / LOG_WARNING / LOG_INFO / LOG_DEBUG / LOG_TRACE
  std::optional<std::string> model_priority;              // MODEL_PRIORITY             : LOW / MEDIUM (default) / HIGH / DEFAULT
  std::optional<std::size_t> num_streams;                 // NUM_STREAMS                : executor streams (logical partitions)
  std::optional<std::string> performance_hint;            // PERFORMANCE_HINT           : LATENCY / THROUGHPUT / CUMULATIVE_THROUGHPUT
  std::optional<std::size_t> performance_hint_num_requests; // PERFORMANCE_HINT_NUM_REQUESTS : in-flight requests the performance hint plans for
  std::optional<std::string> scheduling_core_type;        // SCHEDULING_CORE_TYPE       : ANY_CORE / PCORE_ONLY / ECORE_ONLY
  std::optional<std::size_t> value_cache_group_size;      // VALUE_CACHE_GROUP_SIZE     : value-cache sub-channel quant group size; 0 = whole channel
  std::optional<std::string> value_cache_precision;       // VALUE_CACHE_PRECISION      : value-cache element type (u4 / u8 / f16 / bf16 / f32)
  std::optional<std::string> weights_path;                // WEIGHTS_PATH               : weights file for a weightless cache (OPTIMIZE_SIZE)
  std::optional<std::string> workload_type;               // WORKLOAD_TYPE              : NPU only: DEFAULT / EFFICIENT (execution priority)

  bool operator==(const ModelRuntimeConfig&) const = default;

  // --- The "Runtime config options" boost section this class owns -----------
  //
  // One long-form flag per runtime property (the field dashed,  e.g.
  // --kv-cache-precision),  in alphabetical-flag order.  Single source of truth
  // for both usage() (its contribution to the shared --help) and
  // parseModelRuntimeConfig (spliced alongside the load / scheduler / request
  // sections the other classes own,  so a caller's other flags are "known",
  // not "unknown option").
  static po::options_description runtimeOptionsDesc();

  // This class's slice of the --help listing (its "Runtime config options"
  // section only);  the combined four-section view is config.cpp's
  // combinedConfigUsage.
  static std::string usage();

  // True when at least one field is set (a string field counts only when
  // non-empty,  so "unset" is uniform across value kinds);  an unmentioned
  // runtime config is all-unset -> hasAnySet() == false -> the engine defaults.
  [[nodiscard]] bool hasAnySet() const;

  // Build the ov::AnyMap the LLMPipeline / VLMPipeline / ASRPipeline  ...
  // constructor consumes in its `properties` argument:  for every set field,
  // its value under the field's Upper-Case property name -- strings as
  // ov::Any(std::string),  bools as ov::Any(bool),  ints as ov::Any(int64_t),
  // and the one float as ov::Any(float).  Unset (empty / nullopt) fields are
  // omitted,  leaving the engine's own default in place.
  ov::AnyMap toProperties() const;
};

// Parse + build a ModelRuntimeConfig from `args` (the argv after the subcommand
// name) via boost::program_options.  Parsed over the full option set (our section
// plus the load / scheduler / request sections the other classes own,  so the
// flags a caller also passes are "known",  not "unknown option") but only the
// runtime fields are read.  Throws ConfigError on an unknown / invalid flag
// (e.g. a value for a hint / enum field that is not one OpenVINO accepts).
ModelRuntimeConfig parseModelRuntimeConfig(const std::vector<std::string>& args);

// Parse a "runtime_config" JSON object into a ModelRuntimeConfig: the flat form
// with the field names (lower_snake) at the top level -- exactly what
// `show-config` emits under the "runtime_config" key and what a `LOAD` message
// may nest under its load_config.  Unknown keys are silently ignored (the
// tolerant readers in the sibling configs' spirit);  a key with the wrong JSON
// type throws ConfigError.
ModelRuntimeConfig modelRuntimeConfigFromJson(const std::string& json);

// Serialise a config to a compact "runtime_config" JSON object emitting only the
// set fields (an all-unset config -> the empty object "{}").  Round-trips
// through modelRuntimeConfigFromJson.
std::string modelRuntimeConfigToJson(const ModelRuntimeConfig& config);

}  // namespace config
}  // namespace ovcli

#endif  // OVCLI_MODEL_RUNTIME_CONFIG_HPP
