// ovgenai's runtime config: the "Runtime config options" boost section this class
// owns (runtimeOptionsDesc / usage), its parser (validated against the load /
// scheduler / request sections the siblings own so their flags are "known",
// reading only the runtime fields), its JSON round-trip, and toProperties()
// (which builds the ov::AnyMap a pipeline is constructed from).
//
// The OpenVINO Runtime properties are the settable (read-write) ones from
// OpenVINO's properties header (see the .hpp for the full provenance); the field
// is the lower-Cased property name, so the C++ field  ==  the JSON key  ==  the
// lower-cased UPPER_SNAKE the value lands under as a runtime property  ==  the
// (dashed) CLI flag,  all four reading off one name.
//
// Runtime-only helpers live in this file's anonymous namespace (parseBool /
// parseDouble / parseSignedInt + the string reader set + the enum validator);
// the argv / boost / number / JSON-object coercions the sections share come from
// config.hpp;  the load + scheduler + request section descriptions are included
// so the parser splices them alongside our own (so a caller's other flags are
// "known",  not "unknown option").
#include "model_runtime_config.hpp"
#include "config.hpp"
#include "util.hpp"                    // resolveModelPath — the model-dir default for cache_dir
#include "model_load_config.hpp"       // the load section, spliced into our parse
#include "model_scheduler_config.hpp"  // the scheduler section, spliced into our parse
#include "model_request_config.hpp"    // the request section, spliced into our parse

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>    // `po` alias is provided by config.hpp

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovgenai {
namespace config {
namespace {

// --- CLI value coercion (each parser carries its own, like the siblings) -----

std::string toLower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// "true"/"1"/"on"/"yes" (case-insensitive) -> true;  "false"/"0"/"off"/"no" ->
// false;  anything else is a ConfigError.  Mirrors the scheduler's reader.
bool parseBool(const std::string& s, const char* flag) {
  const std::string lower = toLower(s);
  if (lower == "true"  || lower == "1" || lower == "on"  || lower == "yes") return true;
  if (lower == "false" || lower == "0" || lower == "off" || lower == "no")  return false;
  throw ConfigError(std::string("invalid boolean for ") + flag +
                    ": expected true/false/1/0/on/off/yes/no, got '" + s + "'");
}

// A base-10 double (a runtime float,  e.g. ACTIVATIONS_SCALE_FACTOR);  a value
// with trailing garbage is a ConfigError.  Mirrors the request config's reader.
double parseDouble(const std::string& s, const char* flag) {
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    return v;
  } catch (...) {
    throw ConfigError(std::string("invalid number for ") + flag + ": '" + s + "'");
  }
}

// A base-10 signed integer (the two *num_threads runtime knobs may be negative;
//  0 = "all cores");  a malformed / out-of-32-bit value is a ConfigError.
int32_t parseSignedInt(const std::string& s, const char* flag) {
  try {
    size_t pos = 0;
    long long v = std::stoll(s, &pos, 10);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    if (v < (long long)std::numeric_limits<int32_t>::min() ||
        v > (long long)std::numeric_limits<int32_t>::max())
      throw std::out_of_range("out of 32-bit range");
    return static_cast<int32_t>(v);
  } catch (...) {
    throw ConfigError(std::string("invalid integer for ") + flag + ": '" + s + "'");
  }
}

// --- The closed-value (hint / enum) fields:  the UPPER-SNAKE a value must be
//     (OpenVINO's own operator>> forms),  one vector per such field (matching
//     the field name to the left in the parser). ----------------------------
// ATTENTION_BACKEND's accepted tokens come from openvino_genai (utils.cpp:
// PA_BACKEND == "PA",  SDPA_BACKEND == "SDPA") -- the two backends the GenAI
// pipeline actually selects between;  everything else is from ov::properties.
const std::vector<std::string> kAttentionBackend = {"PA", "SDPA"};
const std::vector<std::string> kCacheMode        = {"OPTIMIZE_SIZE", "OPTIMIZE_SPEED"};
const std::vector<std::string> kExecutionMode    = {"PERFORMANCE", "ACCURACY"};
const std::vector<std::string> kLogLevel         = {"LOG_NONE", "LOG_ERROR", "LOG_WARNING",
                                                   "LOG_INFO", "LOG_DEBUG", "LOG_TRACE"};
const std::vector<std::string> kModelPriority    = {"LOW", "MEDIUM", "HIGH", "DEFAULT"};
const std::vector<std::string> kPerformanceMode  = {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"};
const std::vector<std::string> kSchedulingCore   = {"ANY_CORE", "PCORE_ONLY", "ECORE_ONLY"};

// Set a closed-value field from its CLI string: an empty value clears it to
// nullopt;  otherwise the value is matched case-insensitively against `allowed`
// and the *canonical* (the matched,  UPPER_SNAKE) form is stored,  so the value
// the engine parses is the one OpenVINO expects regardless of how the operator
// typed it;  an unknown value is a ConfigError naming the accepted set.
void setEnumFromCli(std::optional<std::string>& field, const char* flag,
                    const std::string& raw, const std::vector<std::string>& allowed) {
  if (raw.empty()) { field = std::nullopt; return; }
  const std::string want = toLower(raw);
  for (const auto& a : allowed)
    if (toLower(a) == want) { field = a; return; }
  throw ConfigError(std::string("invalid value '") + raw + "' for " + flag +
                    ": must be one of " + joinList(allowed, ", "));
}

// Set a free-form string field from its CLI string: empty clears to nullopt,
// any other value is stored verbatim (the value is open-ended: a precision,
// a path, a weights path,  ...).
void setStrFromCli(std::optional<std::string>& field, const std::string& raw) {
  if (raw.empty()) field = std::nullopt; else field = raw;
}

// --- JSON readers over a flat "runtime_config" object (the tolerant reader) ---
// A present JSON null leaves the field at nullopt (engine default);  a present
// value of the wrong JSON type is a ConfigError (like the scheduler reader).

void readOptStr(rapidjson::Value& v, const char* name, std::optional<std::string>& out) {
  if (v.IsNull()) return;                                // null -> leave at nullopt
  if (!v.IsString())
    throw ConfigError(std::string("runtime_config.") + name + ": expected a string");
  const std::string s(v.GetString());
  if (!s.empty()) out = s;                              // empty == not set
}

void readOptBool(rapidjson::Value& v, const char* name, std::optional<bool>& out) {
  if (v.IsNull()) return;
  if (!v.IsBool())
    throw ConfigError(std::string("runtime_config.") + name +
                      ": expected a boolean (got a " +
                      (v.IsString() ? "string" : v.IsNumber() ? "number"
                       : v.IsNull() ? "null" : "non-boolean") + ")");
  out = v.GetBool();
}

void readOptSize(rapidjson::Value& v, const char* name, std::optional<std::size_t>& out) {
  if (v.IsNull()) return;
  if (!v.IsInt() && !v.IsInt64() && !v.IsUint())
    throw ConfigError(std::string("runtime_config.") + name +
                      ": expected an integer (got a " +
                      (v.IsString() ? "string" : v.IsNumber() ? "non-integer number"
                       : v.IsNull() ? "null" : "non-numeric") + ")");
  if (v.GetInt64() < 0)
    throw ConfigError(std::string("runtime_config.") + name + ": must be non-negative");
  out = static_cast<std::size_t>(v.GetUint64());        // safe: already checked >= 0
}

void readOptInt32(rapidjson::Value& v, const char* name, std::optional<int32_t>& out) {
  if (v.IsNull()) return;
  if (!v.IsNumber())
    throw ConfigError(std::string("runtime_config.") + name +
                      ": expected a number (got a " +
                      (v.IsString() ? "string" : v.IsNull() ? "null" : "non-number") + ")");
  out = static_cast<int32_t>(v.GetInt64());
}

void readOptDouble(rapidjson::Value& v, const char* name, std::optional<double>& out) {
  if (v.IsNull()) return;
  if (!v.IsNumber())
    throw ConfigError(std::string("runtime_config.") + name +
                      ": expected a number (got a " +
                      (v.IsString() ? "string" : v.IsNull() ? "null" : "non-number") + ")");
  out = v.GetDouble();
}

}  // close the anonymous namespace of runtime-only helpers

// ---------------------------------------------------------------------------
// The "Runtime config options" boost section this class owns:  one long-form
// flag per runtime property (the field dashed -- lowercase,  " _ substituted by
// -,  trailing --",  per the user's rule,  e.g. --kv-cache-precision),  in
// alphabetical-flag order.  Long-form only (with ~28 of them a single letter is
// no longer a sensible alias,  unlike the smaller load / scheduler sections).
// Single source of truth for both usage() and parseModelRuntimeConfig.
// Each help string is a concise paraphrase of the matching OpenVINO property's
// doc comment (ov runtime properties.hpp;  genai utils.cpp for ATTENTION_BACKEND).
// ---------------------------------------------------------------------------
po::options_description ModelRuntimeConfig::runtimeOptionsDesc() {
  po::options_description s(
      "Runtime config options (OpenVINO Runtime properties; empty/default -- leave the engine default)",
      kHelpLineLength, kHelpMinDescriptionLength);
  s.add_options()
    ("activations-scale-factor", po::value<std::string>(),
        "Scales down activations to guard against overflow when the inference precision is f16 (a float factor)")
    ("attention-backend", po::value<std::string>(),
        "Attention (KV-cache) backend: PA = PagedAttention (the continuous-batching backend); SDPA = Scaled-Dot-Product (the default)")
    ("auto-batch-timeout", po::value<std::string>(),
        "Auto-batching timeout in milliseconds: how long to wait for more input before dispatching a collected batch")
    ("cache-dir", po::value<std::string>(),
        "Directory the plugins cache compiled blobs/models into; a path -- empty or unset disables caching (may go stale after an OpenVINO update)")
    ("cache-mode", po::value<std::string>(),
        "Cache mode: OPTIMIZE_SIZE (smaller cache files) or OPTIMIZE_SPEED (faster load, larger files; the default)")
    ("compilation-num-threads", po::value<std::string>(),
        "Maximum number of threads for compiling the model; may be negative, 0 = all cores")
    ("enable-cpu-pinning", po::value<std::string>(),
        "Whether to bind each inference thread to a core (true/false; Windows/Linux only, off on macOS -- the engine picks a default if unset)")
    ("enable-cpu-reservation", po::value<std::string>(),
        "Whether to reserve CPU cores so other plugins/compiled models will not use them (true/false; off by default; Windows/Linux only)")
    ("enable-hyper-threading", po::value<std::string>(),
        "Whether to use CPU hyper-threading (true/false; the engine chooses a default if unset)")
    ("enable-mmap", po::value<std::string>(),
        "Whether to read the model via mmap() (true/false; on by default; supported only by the IR frontend for now)")
    ("enable-weightless", po::value<std::string>(),
        "Whether to use weightless caching -- cache the model topology separately from its weights (true/false)")
    ("execution-mode-hint", po::value<std::string>(),
        "Device-agnostic execution hint: PERFORMANCE (optimises for speed, may cost accuracy) or ACCURACY (optimises for accuracy, the default)")
    ("force-tbb-terminate", po::value<std::string>(),
        "Whether to force-terminate TBB when the core is destroyed (true terminates it; false does no extra TBB work on destruction)")
    ("inference-num-threads", po::value<std::string>(),
        "Maximum number of threads for inference tasks; may be negative, 0 = all cores")
    ("inference-precision-hint", po::value<std::string>(),
        "Element type to use for inference, e.g. f16 / f32 / bf16 -- every device turns this hint into its own device-specific settings")
    ("key-cache-group-size", po::value<std::string>(),
        "Sub-channel quantization group size for the key cache; 0 (the default) treats each whole channel as one group")
    ("key-cache-precision", po::value<std::string>(),
        "Element type for KV-cache keys, overriding the umbrella KV_CACHE_PRECISION for the key side (u4 / i8 / u8 / f16 / bf16 / f32)")
    ("kv-cache-precision", po::value<std::string>(),
        "Umbrella KV-cache precision: sets BOTH key and value at once (unless a per-side value is given) -- "
        "valid on both sides: u4 / u8 / f16 / bf16 / f32; default u8 on x86_64/ARM64, f16 otherwise "
        "(i8 is a key-side-only value, not a valid umbrella).  Overridable per side by "
        "--key-cache-precision / --value-cache-precision")
    ("log-level", po::value<std::string>(),
        "Logging level: LOG_NONE / LOG_ERROR / LOG_WARNING / LOG_INFO / LOG_DEBUG / LOG_TRACE")
    ("model-priority", po::value<std::string>(),
        "Model priority hint: LOW / MEDIUM (the default) / HIGH / DEFAULT -- a higher priority is served more-performant bounded resources first")
    ("num-streams", po::value<std::string>(),
        "Number of executor streams (logical partitions) for inference")
    ("performance-hint", po::value<std::string>(),
        "Device-agnostic performance hint: LATENCY / THROUGHPUT / CUMULATIVE_THROUGHPUT -- every device turns it into device-specific settings")
    ("performance-hint-num-requests", po::value<std::string>(),
        "Number of inference requests expected in flight, which backs the performance hint (e.g. number of cameras / input sources)")
    ("scheduling-core-type", po::value<std::string>(),
        "CPU core type for inference: ANY_CORE / PCORE_ONLY (performance cores) / ECORE_ONLY (efficient cores) -- a p-core-only platform still uses its performance cores")
    ("value-cache-group-size", po::value<std::string>(),
        "Sub-channel quantization group size for the value cache; 0 (the default) treats each whole channel as one group")
    ("value-cache-precision", po::value<std::string>(),
        "Element type for KV-cache values, overriding the umbrella KV_CACHE_PRECISION for the value side (u4 / u8 / f16 / bf16 / f32)")
    ("weights-path", po::value<std::string>(),
        "Path to a file holding the model's weights; used for weightless caching, i.e. when CACHE_MODE is OPTIMIZE_SIZE")
    ("workload-type", po::value<std::string>(),
        "NPU execution priority: DEFAULT or EFFICIENT -- NPU only");
  return s;
}

std::string ModelRuntimeConfig::usage() {
  std::ostringstream os;
  os << ModelRuntimeConfig::runtimeOptionsDesc();
  return os.str();
}

bool ModelRuntimeConfig::hasAnySet() const {
  // A string field counts only when non-empty (so "not set" is uniform across
  // value kinds);  an int/bool/double counts whenever has_value().
  return activations_scale_factor.has_value() ||
         !attention_backend.value_or("").empty()  ||
         auto_batch_timeout.has_value()        ||
         !cache_dir.value_or("").empty()       ||
         !cache_mode.value_or("").empty()    ||
         compilation_num_threads.has_value()   ||
         enable_cpu_pinning.has_value()        ||
         enable_cpu_reservation.has_value()    ||
         enable_hyper_threading.has_value()    ||
         enable_mmap.has_value()              ||
         enable_weightless.has_value()         ||
         !execution_mode_hint.value_or("").empty()  ||
         force_tbb_terminate.has_value()        ||
         inference_num_threads.has_value()      ||
         !inference_precision_hint.value_or("").empty()  ||
         key_cache_group_size.has_value()       ||
         !key_cache_precision.value_or("").empty()    ||
         !kv_cache_precision.value_or("").empty()     ||
         !log_level.value_or("").empty()       ||
         !model_priority.value_or("").empty()  ||
         num_streams.has_value()               ||
         !performance_hint.value_or("").empty()  ||
         performance_hint_num_requests.has_value() ||
         !scheduling_core_type.value_or("").empty()    ||
         value_cache_group_size.has_value()       ||
         !value_cache_precision.value_or("").empty()   ||
         !weights_path.value_or("").empty()    ||
         !workload_type.value_or("").empty();
}

// ---------------------------------------------------------------------------
// toProperties() — the ov::AnyMap the LLMPipeline / VLMPipeline / ASRPipeline
// ...  constructor consumes in its `properties` argument.  For every set field,
// the value under its UPPER_SNAKE property name:  a string as ov::Any(str),
// a bool as ov::Any(bool),  an int as ov::Any(int64_t),  the one float as
// ov::Any(float).  This is the same Any-valued form the free-form runtime_config
// JSON the runtime layer used to fold in,  so the pipeline construction is
// unchanged -- only the *source* of those values is now this typed class.
// ---------------------------------------------------------------------------
ov::AnyMap ModelRuntimeConfig::toProperties() const {
  ov::AnyMap m;
  auto addStr = [&](const char* key, const std::optional<std::string>& v) {
    if (v.has_value() && !v->empty()) m[key] = ov::Any(*v);
  };
  auto addBool = [&](const char* key, const std::optional<bool>& v) {
    if (v.has_value()) m[key] = ov::Any(*v);  // false is a valid, explicit value
  };
  auto addInt = [&](const char* key, std::optional<std::size_t> v) {
    if (v.has_value()) m[key] = ov::Any(static_cast<int64_t>(*v));
  };
  auto addInt32 = [&](const char* key, std::optional<int32_t> v) {
    if (v.has_value()) m[key] = ov::Any(static_cast<int64_t>(*v));
  };
  auto addFloat = [&](const char* key, std::optional<double> d) {
    if (d.has_value()) m[key] = ov::Any(static_cast<float>(*d));
  };
  // (the float,  then alphabetically,  matching the struct)
  addFloat("ACTIVATIONS_SCALE_FACTOR",   activations_scale_factor);
  addStr("ATTENTION_BACKEND",        attention_backend);
  addInt("AUTO_BATCH_TIMEOUT",       auto_batch_timeout);
  addStr("CACHE_DIR",                cache_dir);
  addStr("CACHE_MODE",               cache_mode);
  addInt32("COMPILATION_NUM_THREADS", compilation_num_threads);
  addBool("ENABLE_CPU_PINNING",      enable_cpu_pinning);
  addBool("ENABLE_CPU_RESERVATION",  enable_cpu_reservation);
  addBool("ENABLE_HYPER_THREADING",  enable_hyper_threading);
  addBool("ENABLE_MMAP",             enable_mmap);
  addBool("ENABLE_WEIGHTLESS",       enable_weightless);
  addStr("EXECUTION_MODE_HINT",      execution_mode_hint);
  addBool("FORCE_TBB_TERMINATE",     force_tbb_terminate);
  addInt32("INFERENCE_NUM_THREADS",  inference_num_threads);
  addStr("INFERENCE_PRECISION_HINT", inference_precision_hint);
  addInt("KEY_CACHE_GROUP_SIZE",     key_cache_group_size);
  addStr("KEY_CACHE_PRECISION",      key_cache_precision);
  addStr("KV_CACHE_PRECISION",       kv_cache_precision);
  addStr("LOG_LEVEL",                log_level);
  addStr("MODEL_PRIORITY",           model_priority);
  addInt("NUM_STREAMS",              num_streams);
  addStr("PERFORMANCE_HINT",         performance_hint);
  addInt("PERFORMANCE_HINT_NUM_REQUESTS", performance_hint_num_requests);
  addStr("SCHEDULING_CORE_TYPE",     scheduling_core_type);
  addInt("VALUE_CACHE_GROUP_SIZE",   value_cache_group_size);
  addStr("VALUE_CACHE_PRECISION",    value_cache_precision);
  addStr("WEIGHTS_PATH",             weights_path);
  addStr("WORKLOAD_TYPE",            workload_type);
  return m;
}

// ---------------------------------------------------------------------------
// parseModelRuntimeConfig — read only the runtime fields,  parsed over the full
// option set (this class's runtime section plus the load / scheduler / request
// sections the siblings own) so a caller's other flags are "known".
// ---------------------------------------------------------------------------
ModelRuntimeConfig parseModelRuntimeConfig(const std::vector<std::string>& args) {
  const PreProcessed pp = preProcess(args);
  if (pp.positionalCount > 1) {
    throw ConfigError("only one positional (the model path) may be given; found "
                       + std::to_string(pp.positionalCount)
                       + ". ovgenai loads exactly one model -- set it with --model");
  }
  std::vector<std::string> cleanArgs = pp.args;
  cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                  cleanArgs.end());

  // Parse over the full argv (our runtime section plus the load / scheduler /
  // request sections the other classes own), so a caller's other flags are
  // "known" (not "unknown option") while we read only the runtime fields.
  po::options_description desc;
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelLoadConfig::loadOptionsDesc());
  desc.add(ModelSchedulerConfig::schedulerOptionsDesc());
  desc.add(ModelRequestConfig::requestOptionsDesc());
  desc.add_options()("help", po::bool_switch(), "Show this help message");

  po::positional_options_description posDesc;
  posDesc.add("model", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(cleanArgs)
                 .options(desc)
                 .positional(posDesc)
                 .style(po::command_line_style::unix_style)
                 .run(), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    throw rephraseBoost(e);
  }

  ModelRuntimeConfig cfg;  // every field at nullopt
  // float
  if (vm.count("activations-scale-factor"))
    cfg.activations_scale_factor = parseDouble(
        vm["activations-scale-factor"].as<std::string>(), "--activations-scale-factor");
  // hint / enum (validated;  canonical form stored)
  if (vm.count("attention-backend"))
    setEnumFromCli(cfg.attention_backend, "--attention-backend",
                   vm["attention-backend"].as<std::string>(), kAttentionBackend);
  if (vm.count("cache-mode"))
    setEnumFromCli(cfg.cache_mode, "--cache-mode",
                   vm["cache-mode"].as<std::string>(), kCacheMode);
  if (vm.count("execution-mode-hint"))
    setEnumFromCli(cfg.execution_mode_hint, "--execution-mode-hint",
                   vm["execution-mode-hint"].as<std::string>(), kExecutionMode);
  if (vm.count("inference-precision-hint"))
    setStrFromCli(cfg.inference_precision_hint,
                  vm["inference-precision-hint"].as<std::string>());
  if (vm.count("kv-cache-precision"))
    setStrFromCli(cfg.kv_cache_precision, vm["kv-cache-precision"].as<std::string>());
  if (vm.count("key-cache-precision"))
    setStrFromCli(cfg.key_cache_precision, vm["key-cache-precision"].as<std::string>());
  if (vm.count("value-cache-precision"))
    setStrFromCli(cfg.value_cache_precision, vm["value-cache-precision"].as<std::string>());
  if (vm.count("log-level"))
    setEnumFromCli(cfg.log_level, "--log-level",
                   vm["log-level"].as<std::string>(), kLogLevel);
  if (vm.count("model-priority"))
    setEnumFromCli(cfg.model_priority, "--model-priority",
                   vm["model-priority"].as<std::string>(), kModelPriority);
  if (vm.count("performance-hint"))
    setEnumFromCli(cfg.performance_hint, "--performance-hint",
                   vm["performance-hint"].as<std::string>(), kPerformanceMode);
  if (vm.count("scheduling-core-type"))
    setEnumFromCli(cfg.scheduling_core_type, "--scheduling-core-type",
                   vm["scheduling-core-type"].as<std::string>(), kSchedulingCore);
  // int (size_t)
  if (vm.count("auto-batch-timeout"))
    cfg.auto_batch_timeout = static_cast<std::size_t>(
        parseSize(vm["auto-batch-timeout"].as<std::string>(), "--auto-batch-timeout"));
  if (vm.count("key-cache-group-size"))
    cfg.key_cache_group_size = static_cast<std::size_t>(
        parseSize(vm["key-cache-group-size"].as<std::string>(), "--key-cache-group-size"));
  if (vm.count("num-streams"))
    cfg.num_streams = static_cast<std::size_t>(
        parseSize(vm["num-streams"].as<std::string>(), "--num-streams"));
  if (vm.count("performance-hint-num-requests"))
    cfg.performance_hint_num_requests = static_cast<std::size_t>(
        parseSize(vm["performance-hint-num-requests"].as<std::string>(),
                  "--performance-hint-num-requests"));
  if (vm.count("value-cache-group-size"))
    cfg.value_cache_group_size = static_cast<std::size_t>(
        parseSize(vm["value-cache-group-size"].as<std::string>(), "--value-cache-group-size"));
  // int (signed;  may be negative)
  if (vm.count("compilation-num-threads"))
    cfg.compilation_num_threads = parseSignedInt(
        vm["compilation-num-threads"].as<std::string>(), "--compilation-num-threads");
  if (vm.count("inference-num-threads"))
    cfg.inference_num_threads = parseSignedInt(
        vm["inference-num-threads"].as<std::string>(), "--inference-num-threads");
  // string
  if (vm.count("cache-dir"))
    setStrFromCli(cfg.cache_dir, vm["cache-dir"].as<std::string>());
  if (vm.count("weights-path"))
    setStrFromCli(cfg.weights_path, vm["weights-path"].as<std::string>());
  if (vm.count("workload-type"))
    setStrFromCli(cfg.workload_type, vm["workload-type"].as<std::string>());
  // bool
  auto readBool = [&](const char* flag, std::optional<bool>& out) {
    if (!vm.count(flag)) return;
    const std::string s = vm[flag].as<std::string>();
    if (s.empty()) out = std::nullopt; else out = parseBool(s, flag);
  };
  readBool("enable-cpu-pinning",     cfg.enable_cpu_pinning);
  readBool("enable-cpu-reservation", cfg.enable_cpu_reservation);
  readBool("enable-hyper-threading", cfg.enable_hyper_threading);
  readBool("enable-mmap",            cfg.enable_mmap);
  readBool("enable-weightless",      cfg.enable_weightless);
  readBool("force-tbb-terminate",    cfg.force_tbb_terminate);

  // The one model-dependent runtime default: when --cache-dir is NOT given but
  // --model is, default the plugin compile/cache dir to a subdirectory
  // "model_cache" within the model's own dir --
  //    --cache-dir (unset)  ->  resolveModelPath(--model) + "/model_cache"
  // so the model's local cache sits co-located with the model itself.  The model
  // is resolved FIRST (an `hf://...` --model folds to its real `.../snapshots/<hash>`
  // dir, so no literal `hf://.../model_cache` path ever appears); a plain path
  // resolves verbatim.  The result lands in this runtime config's `cache_dir`, so
  // it flows out via toProperties()/toJson() as the CACHE_DIR property, and --
  // threaded to the pipeline constructor (GenAIPipelineConfig::toPipelineProperties
  // seeds its AnyMap from runtime_.toProperties(), which the VLMPipeline /
  // LLMPipeline constructor receives as the `properties` arg passed on to
  // ov::Core::compile_model()) -- through to the model at load time.
  //
  // "not given" is detected here via vm.count, not via the field: an explicit
  // --cache-dir (including an empty "" that means "disable caching") and an absent
  // one both land at nullopt after the reads above and are otherwise indistinguishable,
  // so a default keying off the field would wrongly overwrite an operator's
  // explicit --cache-dir "" -- the one state the operator can opt out of caching with.
  if (!vm.count("cache-dir") && vm.count("model")) {
    const std::string model = vm["model"].as<std::string>();
    if (!model.empty())
      cfg.cache_dir = ovgenai::resolveModelPath(model) + "/model_cache";
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// modelRuntimeConfigFromJson — parse a flat "runtime_config" object (the keys are
// the field names,  lower_snake;  exactly what show-config / a LOAD message
// carry).  Null/absent leaves a field at nullopt;  a present value of the wrong
// JSON type is a ConfigError;  unknown keys are silently ignored (the tolerant
// reader's spirit,  matching the sibling configs).
// ---------------------------------------------------------------------------
ModelRuntimeConfig modelRuntimeConfigFromJson(const std::string& json) {
  rapidjson::Document doc;
  if (doc.Parse<0>(json.c_str()).HasParseError())
    throw ConfigError("runtime_config: not valid JSON at position " +
                        std::to_string(doc.GetErrorOffset()));
  if (!doc.IsObject()) throw ConfigError("runtime_config: expected a JSON object");

  ModelRuntimeConfig cfg;  // every field at nullopt
  // Route each member to the reader matching its value kind,  alphabetically.
  for (auto kv = doc.MemberBegin(); kv != doc.MemberEnd(); ++kv) {
    const std::string k = kv->name.GetString();
    rapidjson::Value& v = kv->value;
    if (k == "activations_scale_factor")       readOptDouble(v, "activations_scale_factor", cfg.activations_scale_factor);
    else if (k == "attention_backend")         readOptStr(v, "attention_backend", cfg.attention_backend);
    else if (k == "auto_batch_timeout")        readOptSize(v, "auto_batch_timeout", cfg.auto_batch_timeout);
    else if (k == "cache_dir")                 readOptStr(v, "cache_dir", cfg.cache_dir);
    else if (k == "cache_mode")                readOptStr(v, "cache_mode", cfg.cache_mode);
    else if (k == "compilation_num_threads")   readOptInt32(v, "compilation_num_threads", cfg.compilation_num_threads);
    else if (k == "enable_cpu_pinning")         readOptBool(v, "enable_cpu_pinning", cfg.enable_cpu_pinning);
    else if (k == "enable_cpu_reservation")     readOptBool(v, "enable_cpu_reservation", cfg.enable_cpu_reservation);
    else if (k == "enable_hyper_threading")     readOptBool(v, "enable_hyper_threading", cfg.enable_hyper_threading);
    else if (k == "enable_mmap")                readOptBool(v, "enable_mmap", cfg.enable_mmap);
    else if (k == "enable_weightless")          readOptBool(v, "enable_weightless", cfg.enable_weightless);
    else if (k == "execution_mode_hint")        readOptStr(v, "execution_mode_hint", cfg.execution_mode_hint);
    else if (k == "force_tbb_terminate")        readOptBool(v, "force_tbb_terminate", cfg.force_tbb_terminate);
    else if (k == "inference_num_threads")      readOptInt32(v, "inference_num_threads", cfg.inference_num_threads);
    else if (k == "inference_precision_hint")   readOptStr(v, "inference_precision_hint", cfg.inference_precision_hint);
    else if (k == "key_cache_group_size")       readOptSize(v, "key_cache_group_size", cfg.key_cache_group_size);
    else if (k == "key_cache_precision")        readOptStr(v, "key_cache_precision", cfg.key_cache_precision);
    else if (k == "kv_cache_precision")         readOptStr(v, "kv_cache_precision", cfg.kv_cache_precision);
    else if (k == "log_level")                  readOptStr(v, "log_level", cfg.log_level);
    else if (k == "model_priority")             readOptStr(v, "model_priority", cfg.model_priority);
    else if (k == "num_streams")                readOptSize(v, "num_streams", cfg.num_streams);
    else if (k == "performance_hint")           readOptStr(v, "performance_hint", cfg.performance_hint);
    else if (k == "performance_hint_num_requests") readOptSize(v, "performance_hint_num_requests", cfg.performance_hint_num_requests);
    else if (k == "scheduling_core_type")       readOptStr(v, "scheduling_core_type", cfg.scheduling_core_type);
    else if (k == "value_cache_group_size")     readOptSize(v, "value_cache_group_size", cfg.value_cache_group_size);
    else if (k == "value_cache_precision")      readOptStr(v, "value_cache_precision", cfg.value_cache_precision);
    else if (k == "weights_path")               readOptStr(v, "weights_path", cfg.weights_path);
    else if (k == "workload_type")              readOptStr(v, "workload_type", cfg.workload_type);
    // else: an unknown key -- silently ignored (the tolerant reader).
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// modelRuntimeConfigToJson — a compact "runtime_config" object emitting only the
// set fields (an all-unset config -> the empty object "{}");  round-trips
// through modelRuntimeConfigFromJson.
// ---------------------------------------------------------------------------
std::string modelRuntimeConfigToJson(const ModelRuntimeConfig& c) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  auto addStr = [&](const char* key, const std::optional<std::string>& s) {
    if (s.has_value() && !s->empty()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v(s->c_str(), s->size(), a);
      out.AddMember(k, v, a);
    }
  };
  auto addBool = [&](const char* key, std::optional<bool> b) {
    if (b.has_value()) {                       // false is a real value: forwarded
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetBool(*b);
      out.AddMember(k, v, a);
    }
  };
  auto addInt = [&](const char* key, std::optional<std::size_t> n) {
    if (n.has_value()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetUint64(*n);
      out.AddMember(k, v, a);
    }
  };
  auto addInt32 = [&](const char* key, std::optional<int32_t> n) {
    if (n.has_value()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetInt64(*n);
      out.AddMember(k, v, a);
    }
  };
  auto addDouble = [&](const char* key, std::optional<double> d) {
    if (d.has_value()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetDouble(*d);
      out.AddMember(k, v, a);
    }
  };
  addDouble("activations_scale_factor",       c.activations_scale_factor);
  addStr("attention_backend",                 c.attention_backend);
  addInt("auto_batch_timeout",                c.auto_batch_timeout);
  addStr("cache_dir",                        c.cache_dir);
  addStr("cache_mode",                      c.cache_mode);
  addInt32("compilation_num_threads",         c.compilation_num_threads);
  addBool("enable_cpu_pinning",               c.enable_cpu_pinning);
  addBool("enable_cpu_reservation",           c.enable_cpu_reservation);
  addBool("enable_hyper_threading",           c.enable_hyper_threading);
  addBool("enable_mmap",                      c.enable_mmap);
  addBool("enable_weightless",                c.enable_weightless);
  addStr("execution_mode_hint",               c.execution_mode_hint);
  addBool("force_tbb_terminate",              c.force_tbb_terminate);
  addInt32("inference_num_threads",           c.inference_num_threads);
  addStr("inference_precision_hint",          c.inference_precision_hint);
  addInt("key_cache_group_size",              c.key_cache_group_size);
  addStr("key_cache_precision",               c.key_cache_precision);
  addStr("kv_cache_precision",                c.kv_cache_precision);
  addStr("log_level",                         c.log_level);
  addStr("model_priority",                    c.model_priority);
  addInt("num_streams",                       c.num_streams);
  addStr("performance_hint",                  c.performance_hint);
  addInt("performance_hint_num_requests",     c.performance_hint_num_requests);
  addStr("scheduling_core_type",              c.scheduling_core_type);
  addInt("value_cache_group_size",            c.value_cache_group_size);
  addStr("value_cache_precision",             c.value_cache_precision);
  addStr("weights_path",                      c.weights_path);
  addStr("workload_type",                     c.workload_type);

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

}  // namespace config
}  // namespace ovgenai
