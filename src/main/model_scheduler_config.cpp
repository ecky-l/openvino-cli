// ovgenai's scheduler config: the "Scheduler Options" boost section this class
// owns (schedulerOptionsDesc / usage), its parser (validated against the load /
// request sections the siblings own so their flags are "known", reading only the
// 10 scheduler fields), and its JSON round-trip.  The engine-config layer's
// typed scheduler analog moved in here too: hasAnySet() and toSchedulerConfig()
// (which builds the ov::genai::SchedulerConfig a pipeline consumes), absorbed
// from the former GenAISchedulerConfig in the engine-config layer.
//
// Scheduler-only helpers live in this file's anonymous namespace (parseBool +
// the two flat JSON readers); the argv / boost / number coercions all three
// sections share come from config.hpp; the load + request section descriptions
// are included so the parser splices them alongside our own.  Every parallel
// structure (the options,  the parser,  the JSON reader / writer,  hasAnySet,
// toSchedulerConfig) lists the 10 fields in the same *alphabetical* order so it
// reads top-to-bottom the way the header does.
#include "model_scheduler_config.hpp"
#include "config.hpp"
#include "model_load_config.hpp"       // the load section, spliced into our parse
#include "model_runtime_config.hpp"    // the runtime section, spliced into our parse
#include "model_request_config.hpp"    // the request section, spliced into our parse

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // `po` alias is provided by config.hpp

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovgenai {
namespace config {
namespace {

// Parse a boolean from the CLI: "true"/"1"/"on"/"yes" (case-insensitive) -> true;
// "false"/"0"/"off"/"no" -> false; anything else is a ConfigError.  Shared with
// the model-load parser for its flag family.
bool parseBool(const std::string& s, const char* flag) {
  std::string lower;
  for (char c : s) lower += static_cast<char>(std::tolower(c));
  if (lower == "true"  || lower == "1" || lower == "on"  || lower == "yes") return true;
  if (lower == "false" || lower == "0" || lower == "off" || lower == "no")  return false;
  throw ConfigError(std::string("invalid boolean for ") + flag +
                    ": expected true/false/1/0/on/off/yes/no, got '" + s + "'");
}

// --- JSON readers over the flat scheduler_config object ---------------------
// A recognised key is read as the matching type from SchedulerConfigSchema
// (int/bool); a null (Python None) leaves the field at nullopt; a present value
// of the wrong JSON type is a ConfigError.  Unknown keys fall out of the loop
// unread -- silently ignored (matching the sibling configs' tolerance).
std::optional<std::size_t> readOptionalSize(rapidjson::Value& v, const char* name) {
  if (!v.IsInt() && !v.IsInt64())
    throw ConfigError(std::string("scheduler_config.") + name +
                      ": expected an integer (got a " +
                      (v.IsString() ? "string" : v.IsNumber() ? "non-integer number"
                       : v.IsNull() ? "null" : "non-numeric") + ")");
  if (v.GetInt64() < 0)
    throw ConfigError(std::string("scheduler_config.") + name + ": must be non-negative");
  return static_cast<std::size_t>(v.GetUint64());  // safe: already checked >= 0
}

std::optional<bool> readOptionalBool(rapidjson::Value& v, const char* name) {
  if (!v.IsBool())
    throw ConfigError(std::string("scheduler_config.") + name +
                      ": expected a boolean (got a " +
                      (v.IsString() ? "string" : v.IsNumber() ? "number"
                       : v.IsNull() ? "null" : "non-boolean") + ")");
  return v.GetBool();
}

}  // close the anonymous namespace of scheduler-only helpers

// --- The "Scheduler Options" boost section this class owns -------------------
//
// The 10 KV-cache / batching flags, in alphabetical-flag order, long-form only.
// Single source of truth for both usage() (its contribution to the shared
// --help) and parseModelSchedulerConfig.  Each description below is a concise
// paraphrase of the matching ov::genai::SchedulerConfig field's doc comment.
po::options_description ModelSchedulerConfig::schedulerOptionsDesc() {
  po::options_description s(
      "Scheduler Options (KV-cache / batching knobs the pipeline is built with; nullopt = engine default)",
      kHelpLineLength, kHelpMinDescriptionLength);
  s.add_options()
    ("cache-interval-multiplier", po::value<std::string>(),
        "Multiplier for the linear-attention checkpoint interval used by prefix caching "
        "(the internal interval = KV cache block size x this; default 8, auto-derived from the model when unset). "
        "A larger value uses less memory at the cost of coarser prefix-cache reuse; "
        "0 is valid only when prefix caching is disabled, and is ignored for models without linear-attention cache inputs.")
    ("cache-size", po::value<std::string>(),
        "Total size of the KV cache in GB; 0 turns on dynamic (auto) cache allocation. "
        "If both --cache-size and --num-kv-blocks are set, --num-kv-blocks takes precedence.")
    ("dynamic-split-fuse", po::value<std::string>(),
        "Split prompt and generate into different scheduling phases (default true). "
        "Allows a prompt longer than the batch to be processed in parts; if disabled, "
        "any prompt longer than the batch size errors out.")
    ("enable-prefix-caching", po::value<std::string>(),
        "Keep previously-computed KV-caches in memory for reuse across sequences (default false). "
        "On by default in the LLMPipeline 'client' scenario; when on, more RAM is used "
        "(bounded by cache-size / num-kv-blocks); when off, a finished sequence's cache is released.")
    ("max-num-batched-tokens", po::value<std::string>(),
        "Maximum number of tokens in a single batch step (default 256; unbounded in the "
        "LLMPipeline 'client' scenario). Counts total batched tokens, unlike --max-num-seqs, which counts sequences.")
    ("max-num-seqs", po::value<std::string>(),
        "Maximum number of scheduled sequences -- effectively the batch size on the generate phase "
        "(default 256); on the prompt phase the scheduled-token count is usually larger.")
    ("num-kv-blocks", po::value<std::string>(),
        "Total number of KV-cache blocks available to the scheduler (default 0 = dynamic). "
        "If both --num-kv-blocks and --cache-size are set, this one takes precedence.")
    ("num-linear-attention-blocks", po::value<std::string>(),
        "Total number of linear-attention (state/SSM) blocks available to the scheduler "
        "(default 0 = auto-derived from --max-num-seqs when linear-attention layers are present; "
        "each block holds the full state for one sequence).")
    ("use-cache-eviction", po::value<std::string>(),
        "Enable cache eviction for all sequences this pipeline processes (default false). "
        "Caps per-sequence KV-cache usage by a user-configurable value -- memory savings at the cost of generation quality.")
    ("use-sparse-attention", po::value<std::string>(),
        "Apply block-wise sparse attention to the prefill (prompt) stage (default false).");
  return s;
}

// This class's slice of the --help listing (its "Scheduler Options" section
// only); the combined four-section view is config.cpp's combinedConfigUsage.
std::string ModelSchedulerConfig::usage() {
  std::ostringstream os;
  os << ModelSchedulerConfig::schedulerOptionsDesc();
  return os.str();
}

// True when at least one field is non-nullopt (corresponds to Python's
// `sched_config_dict = model_dump(exclude_unset=True); if not dict:`).  Listed
// in the same alphabetical order as the struct.
bool ModelSchedulerConfig::hasAnySet() const {
  return cache_interval_multiplier.has_value() || cache_size.has_value()  ||
         dynamic_split_fuse.has_value()        || enable_prefix_caching.has_value()  ||
         max_num_batched_tokens.has_value()    || max_num_seqs.has_value()  ||
         num_kv_blocks.has_value()             || num_linear_attention_blocks.has_value()  ||
         use_cache_eviction.has_value()        || use_sparse_attention.has_value();
}

// Build the ov::genai::SchedulerConfig the pipeline consumes:
//   result = a default-constructed SchedulerConfig, and then, for every field,
//   forward it only when it is "truthy".  "truthy" means has_value() && *val is
//   truthy: a size_t forwards only when nonzero (0 is the engine-default
//   sentinel), and a bool forwards only when true (a false value leaves the
//   engine default untouched).  The one exception is cache_interval_multiplier,
//   which the engine keeps as an optional and forward on has_value() alone (0 is
//   a valid value when prefix caching is off; the engine's own validate()
//   enforces that).  Fields are forwarded in alphabetical order.
ov::genai::SchedulerConfig ModelSchedulerConfig::toSchedulerConfig() const {
  ov::genai::SchedulerConfig result;   // every field starts at the engine default
  if (cache_interval_multiplier.has_value())
    result.cache_interval_multiplier = *cache_interval_multiplier;   // has_value() only (0 valid when prefix off)
  if (cache_size.has_value()                && *cache_size != 0)
    result.cache_size             = *cache_size;
  if (dynamic_split_fuse.has_value()        && *dynamic_split_fuse)
    result.dynamic_split_fuse    = *dynamic_split_fuse;
  if (enable_prefix_caching.has_value()     && *enable_prefix_caching)
    result.enable_prefix_caching = *enable_prefix_caching;
  if (max_num_batched_tokens.has_value()    && *max_num_batched_tokens != 0)
    result.max_num_batched_tokens = *max_num_batched_tokens;
  if (max_num_seqs.has_value()             && *max_num_seqs != 0)
    result.max_num_seqs          = *max_num_seqs;
  if (num_kv_blocks.has_value()            && *num_kv_blocks != 0)
    result.num_kv_blocks          = *num_kv_blocks;
  if (num_linear_attention_blocks.has_value() && *num_linear_attention_blocks != 0)
    result.num_linear_attention_blocks = *num_linear_attention_blocks;
  if (use_cache_eviction.has_value()        && *use_cache_eviction)
    result.use_cache_eviction    = *use_cache_eviction;
  if (use_sparse_attention.has_value()      && *use_sparse_attention)
    result.use_sparse_attention  = *use_sparse_attention;
  return result;
}

// --- Public: parseModelSchedulerConfig -------------------------------------
//
// Reads only the 10 scheduler fields (in alphabetical order).  Parsed over the
// full argv (this class's section plus the load + request sections the other
// classes own) so boost doesn't reject the sibling flags that this parser
// simply doesn't read.
ModelSchedulerConfig parseModelSchedulerConfig(const std::vector<std::string>& args) {
  const PreProcessed pp = preProcess(args);
  if (pp.positionalCount > 1) {
    throw ConfigError("only one positional (the model path) may be given; found "
                       + std::to_string(pp.positionalCount)
                       + ". ovgenai loads exactly one model -- set it with --model");
  }
  std::vector<std::string> cleanArgs = pp.args;
  cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                  cleanArgs.end());

  // Parse over the full argv: our (scheduler) section plus the load + request
  // sections the other classes own, so a caller's load / request flags are
  // "known" (not "unknown option") while we read only the scheduler fields.
  po::options_description desc;
  desc.add(ModelSchedulerConfig::schedulerOptionsDesc());
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelLoadConfig::loadOptionsDesc());
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

  // Read only the 10 scheduler fields (alphabetical); the load / request fields
  // present in the same vm (for the other parsers) are simply not accessed.
  ModelSchedulerConfig cfg;  // all 10 optionals at nullopt
  if (vm.count("cache-interval-multiplier"))
    cfg.cache_interval_multiplier = static_cast<std::size_t>(
        parseSize(vm["cache-interval-multiplier"].as<std::string>(), "--cache-interval-multiplier"));
  if (vm.count("cache-size"))
    cfg.cache_size = static_cast<std::size_t>(
        parseSize(vm["cache-size"].as<std::string>(), "--cache-size"));
  if (vm.count("dynamic-split-fuse"))
    cfg.dynamic_split_fuse = parseBool(vm["dynamic-split-fuse"].as<std::string>(),
                                        "--dynamic-split-fuse");
  if (vm.count("enable-prefix-caching"))
    cfg.enable_prefix_caching = parseBool(vm["enable-prefix-caching"].as<std::string>(),
                                          "--enable-prefix-caching");
  if (vm.count("max-num-batched-tokens"))
    cfg.max_num_batched_tokens = static_cast<std::size_t>(
        parseSize(vm["max-num-batched-tokens"].as<std::string>(), "--max-num-batched-tokens"));
  if (vm.count("max-num-seqs"))
    cfg.max_num_seqs = static_cast<std::size_t>(
        parseSize(vm["max-num-seqs"].as<std::string>(), "--max-num-seqs"));
  if (vm.count("num-kv-blocks"))
    cfg.num_kv_blocks = static_cast<std::size_t>(
        parseSize(vm["num-kv-blocks"].as<std::string>(), "--num-kv-blocks"));
  if (vm.count("num-linear-attention-blocks"))
    cfg.num_linear_attention_blocks = static_cast<std::size_t>(
        parseSize(vm["num-linear-attention-blocks"].as<std::string>(), "--num-linear-attention-blocks"));
  if (vm.count("use-cache-eviction"))
    cfg.use_cache_eviction = parseBool(vm["use-cache-eviction"].as<std::string>(),
                                        "--use-cache-eviction");
  if (vm.count("use-sparse-attention"))
    cfg.use_sparse_attention = parseBool(vm["use-sparse-attention"].as<std::string>(),
                                          "--use-sparse-attention");
  return cfg;
}

// --- Public: modelSchedulerConfigFromJson ----------------------------------
//
// Parse a (flat) scheduler_config object.  Each recognised key is read as the
// matching SchedulerConfigSchema type;  an outright type mismatch throws
// ConfigError,  a JSON null leaves the field at nullopt,  and unknown keys are
// silently ignored (matching the sibling configs' tolerance).  Keys are read in
// the same alphabetical order as the struct (the if/else-if chain is
// order-independent in effect, but alphabetical keeps it readable).
ModelSchedulerConfig modelSchedulerConfigFromJson(const std::string& json) {
  rapidjson::Document doc;
  if (doc.Parse<0>(json.c_str()).HasParseError())
    throw ConfigError("scheduler_config: not valid JSON at position " +
                        std::to_string(doc.GetErrorOffset()));
  if (!doc.IsObject()) throw ConfigError("scheduler_config: expected a JSON object");

  ModelSchedulerConfig sc;  // all 10 fields at nullopt
  for (auto kv = doc.MemberBegin(); kv != doc.MemberEnd(); ++kv) {
    const std::string k = kv->name.GetString();
    if (kv->value.IsNull())  continue;   // JSON null -> Python None -> leave at nullopt
    if (k == "cache_interval_multiplier")
      sc.cache_interval_multiplier = readOptionalSize(kv->value, "cache_interval_multiplier");
    else if (k == "cache_size")
      sc.cache_size               = readOptionalSize(kv->value, "cache_size");
    else if (k == "dynamic_split_fuse")
      sc.dynamic_split_fuse       = readOptionalBool(kv->value, "dynamic_split_fuse");
    else if (k == "enable_prefix_caching")
      sc.enable_prefix_caching    = readOptionalBool(kv->value, "enable_prefix_caching");
    else if (k == "max_num_batched_tokens")
      sc.max_num_batched_tokens    = readOptionalSize(kv->value, "max_num_batched_tokens");
    else if (k == "max_num_seqs")
      sc.max_num_seqs             = readOptionalSize(kv->value, "max_num_seqs");
    else if (k == "num_kv_blocks")
      sc.num_kv_blocks             = readOptionalSize(kv->value, "num_kv_blocks");
    else if (k == "num_linear_attention_blocks")
      sc.num_linear_attention_blocks = readOptionalSize(kv->value, "num_linear_attention_blocks");
    else if (k == "use_cache_eviction")
      sc.use_cache_eviction       = readOptionalBool(kv->value, "use_cache_eviction");
    else if (k == "use_sparse_attention")
      sc.use_sparse_attention     = readOptionalBool(kv->value, "use_sparse_attention");
    // Unknown keys are silently ignored (ovgenai decoupling from the wider schema).
  }
  return sc;
}

// --- Public: modelSchedulerConfigToJson ------------------------------------
//
// Emit only the set fields (an all-nullopt config -> the empty object).  This is
// exactly what `show-config` places under the top-level "scheduler_config" key.
// Fields are emitted in the same alphabetical order as the struct.
std::string modelSchedulerConfigToJson(const ModelSchedulerConfig& c) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  auto addInt = [&](const char* key, std::optional<std::size_t> val) {
    if (val.has_value()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetUint64(*val);
      out.AddMember(k, v, a);
    }
  };
  auto addBool = [&](const char* key, std::optional<bool> val) {
    if (val.has_value()) {
      rapidjson::Value k(key, a);
      rapidjson::Value v;
      v.SetBool(*val);
      out.AddMember(k, v, a);
    }
  };
  addInt("cache_interval_multiplier", c.cache_interval_multiplier);
  addInt("cache_size",                c.cache_size);
  addBool("dynamic_split_fuse",       c.dynamic_split_fuse);
  addBool("enable_prefix_caching",    c.enable_prefix_caching);
  addInt("max_num_batched_tokens",    c.max_num_batched_tokens);
  addInt("max_num_seqs",              c.max_num_seqs);
  addInt("num_kv_blocks",             c.num_kv_blocks);
  addInt("num_linear_attention_blocks", c.num_linear_attention_blocks);
  addBool("use_cache_eviction",       c.use_cache_eviction);
  addBool("use_sparse_attention",     c.use_sparse_attention);
  // `out` is already the (possibly-empty) scheduler object: when nothing was set
  // it stays "{}" (matching the free-form blobs that default to {}).
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

}  // namespace config
}  // namespace ovgenai
