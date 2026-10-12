// ovcli's model-load config: the "Model load parameters" boost section this
// class owns (loadOptionsDesc / usage), its parser (validated against the load
// enums + the free-JSON blobs), and its JSON round-trip.  The KV-cache /
// batching (scheduler) knobs it used to carry moved to ModelSchedulerConfig,
// whose section the parser now splices in alongside the request section.
// Model-load-only helpers live in this file's anonymous namespace; the
// argv / boost / number / JSON-object coercions all sections share come from
// config.hpp; the request + scheduler section descriptions are included so the
// parser splices them in.
#include "model_load_config.hpp"
#include "config.hpp"
#include "model_runtime_config.hpp"    // the runtime section, spliced into our parse
#include "model_scheduler_config.hpp"  // the scheduler section, spliced into our parse
#include "model_request_config.hpp"    // the request section, spliced into our parse

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // `po` alias is provided by config.hpp

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovcli {
namespace config {
namespace {

// --- The enums (the accepted member strings) --------
const std::vector<std::string> kEngines = {"optimum", "ovgenai", "openvino"};
const std::vector<std::string> kModelTypes = {
    "llm",         "vlm",         "whisper",
    "qwen3_asr",   "kokoro",      "qwen3_tts_custom_voice",
    "qwen3_tts_voice_design",     "qwen3_tts_voice_clone",
    "emb",         "rerank"};

// The one required / enum / derive pass BOTH entry points share: after either
// parser has populated a ModelLoadConfig (strings + optionals), finalize() fills
// the one it derives and validates the other required / enum fields.
//   * model  -- derived by *neither*; the required flag. The load_config JSON
//             reader sets it first if the LOAD message named it; the CLI leaves it
//             empty -> finalize() reports it missing.
//   * engine, model_type, device -- validated against the k-enums.
void finalize(ModelLoadConfig& cfg) {
  std::vector<std::string> missing;
  if (cfg.model.empty()) missing.push_back("--model");
  if (cfg.engine.empty()) missing.push_back("--engine");
  if (cfg.model_type.empty()) missing.push_back("--model-type");
  if (cfg.device.empty()) missing.push_back("--device");
  if (!missing.empty())
    throw ConfigError("missing required load option(s): " + joinList(missing, ", "));

  if (!inSet(cfg.engine, kEngines))
    throw ConfigError("--engine '" + cfg.engine + "' must be one of: " +
                        joinList(kEngines, ", "));
  if (!inSet(cfg.model_type, kModelTypes))
    throw ConfigError("--model-type '" + cfg.model_type + "' must be one of: " +
                        joinList(kModelTypes, ", "));
}

// (The former free-form `runtime_config` blob reader lived here;  that blob is
//  now its own first-class ModelRuntimeConfig,  so it is gone --  and the one
//  knob that moved with it,  cache_dir,  lives there too:  ov::cache_dir is an
//  OpenVINO Runtime property,  so it is set on the runtime side,  not here.)

}  // close the anonymous namespace of model-load-only helpers

// --- The "Model load parameters" boost section this class owns ---------------
//
// The four required flags plus the optional model-load knobs (the KV-cache /
// batching scheduler knobs now have their own section: see
// ModelSchedulerConfig::schedulerOptionsDesc, which this parser splices in).
//  Each with a single-letter short where its first letter is not claimed by a
// longer-name prefix. Single source of truth for both usage() (its contribution
// to the shared --help) and parseModelLoadConfig (spliced alongside the request
// + scheduler sections the other classes own).
po::options_description ModelLoadConfig::loadOptionsDesc() {
  po::options_description s(
      "Model load parameters (applied at model load time; shared by worker / run / show-config)",
      kHelpLineLength, kHelpMinDescriptionLength);
  s.add_options()
    // --- required ---
    ("model,m", po::value<std::string>(),
        "Dir of the OpenVINO IR model; the model's identity (required)")
    ("engine,e", po::value<std::string>(),
        "Loader: optimum, ovgenai or openvino (required)")
    ("model-type,t", po::value<std::string>(),
        "Pipeline: llm, vlm, whisper, qwen3_asr, kokoro, qwen3_tts_*, emb or rerank (required)")
    ("device,d", po::value<std::string>(),
        "OpenVINO device, e.g. CPU, GPU, NPU (required)")
    // --- optional model-load knobs (CACHE_DIR is now ModelRuntimeConfig) ---
    ("draft-model", po::value<std::string>(),
        "Draft model dir for speculative decoding (no short: prefix-conflicts --device)")
    ("draft-device", po::value<std::string>(),
        "Device for the draft model (default CPU; empty clears it; no short: --device conflict)")
    // --- free-form JSON blob (authored;  carried verbatim) ---
    ("config-blocks,b", po::value<std::string>(),
        "Per-request default blocks (sampler_config, ...) as a JSON object");
  return s;
}

// This class's slice of the --help listing (its "Model load parameters"
// section only); the combined three-section view is config.cpp's
// combinedConfigUsage.
std::string ModelLoadConfig::usage() {
  std::ostringstream os;
  os << ModelLoadConfig::loadOptionsDesc();
  return os.str();
}

// --- Public: parse and validate the load-model args -------------------------
//
// Uses boost::program_options (this class's load section spliced alongside the
// request + scheduler sections the other classes own, so the generation /
// scheduler flags a caller also passes are "known") for the option-name /
// shorthand handling, then applies the domain validation (enums, numerics, JSON
// objects) via the helpers below, producing one ModelLoadConfig that both
// `ovcli worker` and `ovcli run` share.
ModelLoadConfig parseModelLoadConfig(const std::vector<std::string>& args) {
  const PreProcessed pp = preProcess(args);
  if (pp.positionalCount > 1) {
    throw ConfigError("only one positional (the model path) may be given; found "
                       + std::to_string(pp.positionalCount)
                       + ". ovcli loads exactly one model -- set it with --model");
  }
  std::vector<std::string> cleanArgs = pp.args;
  cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                  cleanArgs.end());

  // Parse over the full argv: all three sections (our load section plus the
  // request + scheduler sections the other classes own), so a caller's load /
  // request / scheduler flags are all "known" (not "unknown option") while we
  // read only the load fields.
  po::options_description desc;
  desc.add(ModelLoadConfig::loadOptionsDesc());
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelRequestConfig::requestOptionsDesc());
  desc.add(ModelSchedulerConfig::schedulerOptionsDesc());
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

  // Read each field by its canonical long name. vm.count(name) is 1 iff the
  // option was supplied via either its long or short spelling; boost has
  // already normalised both into one entry under the registered name, which
  // for the comma syntax --longname,short is the LONG name.  The request +
  // scheduler fields (present in the same vm for the other parsers) are simply
  // not accessed here.
  ModelLoadConfig cfg;  // strings and optionals at their struct defaults
  if (vm.count("model")) cfg.model = vm["model"].as<std::string>();
  if (vm.count("engine"))     cfg.engine     = vm["engine"].as<std::string>();
  if (vm.count("model-type")) cfg.model_type = vm["model-type"].as<std::string>();
  if (vm.count("device"))     cfg.device     = vm["device"].as<std::string>();
  if (vm.count("draft-model")) {
    const std::string v = vm["draft-model"].as<std::string>();
    if (!v.empty()) cfg.draft_model = v;
  }
  if (vm.count("draft-device")) {
    const std::string v = vm["draft-device"].as<std::string>();
    if (v.empty()) cfg.draft_device = std::nullopt;  // "" clears the default
    else            cfg.draft_device = v;
  }
  // (The former free-form `--runtime-config` blob is now ModelRuntimeConfig's
  //  own typed section,  spliced into this parser above so the runtime flags
  //  are "known" here though read by the runtime parser,  not this one.)

  finalize(cfg);
  return cfg;
}

// --- Public: modelLoadConfigFromJson ---------------------------------------
//
// Parse a load_config JSON object. Tolerant like the parser (unknown keys are
// dropped, not a hard error) because the config no longer travels inside
// a fixed schema; finalize() then validates.  The "scheduler_config" key a
// caller may have nested inside is now a foreign key from this reader's point
// of view (the scheduler is its own config) and is simply skipped like any
// other unknown key.
ModelLoadConfig modelLoadConfigFromJson(const std::string& json) {
  rapidjson::Document doc;
  if (doc.Parse<0>(json.c_str()).HasParseError())
    throw ConfigError("load_config is not valid JSON at position " +
                        std::to_string(doc.GetErrorOffset()));
  if (!doc.IsObject()) throw ConfigError("load_config must be a JSON object");

  ModelLoadConfig cfg;  // optionals at their defaults
  for (auto kv = doc.MemberBegin(); kv != doc.MemberEnd(); ++kv) {
    const std::string k = kv->name.GetString();
    const rapidjson::Value& v = kv->value;
    if (k == "model") {
      if (v.IsString()) cfg.model = v.GetString();
      else if (!v.IsNull()) throw ConfigError("load_config.model: expected a string");
    } else if (k == "model_type") {
      if (v.IsString()) cfg.model_type = v.GetString();
      else if (!v.IsNull()) throw ConfigError("load_config.model_type: expected a string");
    } else if (k == "engine") {
      if (v.IsString()) cfg.engine = v.GetString();
      else if (!v.IsNull()) throw ConfigError("load_config.engine: expected a string");
    } else if (k == "device") {
      if (v.IsString()) cfg.device = v.GetString();
      else if (!v.IsNull()) throw ConfigError("load_config.device: expected a string");
    } else if (k == "draft_model") {
      if (v.IsString() && !std::string(v.GetString()).empty())
        cfg.draft_model = v.GetString();
      else if (!v.IsNull() && !v.IsString())
        throw ConfigError("load_config.draft_model: expected a string");
    } else if (k == "draft_device") {
      if (v.IsString()) {
        const std::string s(v.GetString());
        if (s.empty()) cfg.draft_device = std::nullopt;
        else            cfg.draft_device = s;
      } else if (!v.IsNull())
        throw ConfigError("load_config.draft_device: expected a string");
    }
    // A caller may nest a "runtime_config" object inside;  it is a foreign key
    // from this reader's point of view now (the runtime config is its own
    // ModelRuntimeConfig) -- like the not-exposed context_window /
    // worker_max_respawns / worker_line_limit,  and the now-separate
    // scheduler_config,  it is simply ignored.
  }

  finalize(cfg);
  return cfg;
}

std::string modelLoadConfigToJson(const ModelLoadConfig& c) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  // A key is emitted only when its field actually carries a value: an empty
  // string is treated as "not set" (the wire reader reads an absent / empty
  // string the same way as a JSON null), so an empty key is dropped from the
  // output instead of showing up as "".
  auto addStr = [&](const char* key, const std::string& s) {
    if (s.empty()) return;
    rapidjson::Value k(key, a);
    rapidjson::Value v(s.c_str(), s.size(), a);
    out.AddMember(k, v, a);
  };
  addStr("model", c.model);
  addStr("model_type", c.model_type);
  addStr("engine", c.engine);
  addStr("device", c.device);
  // (The former "runtime_config" key was a free-form blob here;  it is now the
  //  ModelRuntimeConfig's own "runtime_config" section in show-config.)
  // Optional model-load knobs: omit the key entirely when it is not set
  // (nullopt, or an empty string) -- no "null" placeholders in the output.
  // draft_device's "CPU" struct default is a real value, so a model with no
  // draft still shows it, just as max_tokens keeps its 16384 default.
  if (c.draft_model && !c.draft_model->empty()) addStr("draft_model", *c.draft_model);
  if (c.draft_device && !c.draft_device->empty()) addStr("draft_device", *c.draft_device);

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

// A single-line human summary (for logging what resolved).
std::string describeConfig(const ModelLoadConfig& c) {
  std::ostringstream os;
  os << "model '" << "' [" << c.model_type << ", " << c.engine << " @ "
     << c.device << "] " << c.model;
  if (c.draft_model)
    os << "; draft '" << *c.draft_model << "' on "
       << (c.draft_device ? *c.draft_device : "CPU");
  return os.str();
}

}  // namespace config
}  // namespace ovcli
