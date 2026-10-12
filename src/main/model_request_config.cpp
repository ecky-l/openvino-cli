// ovcli's per-request generation config: the "Model request parameters" boost
// section this class owns (requestOptionsDesc / usage), its parser, and its JSON
// round-trip (snake_case keys, so the wire form is identical on both sides).
// Request-only helpers live in this file's
// anonymous namespace; the argv / boost / number / JSON coercions all sections
// share come from config.hpp (ConfigError is one of them); the load + scheduler
// section descriptions are included so the parser splices them in alongside our
// own.
#include "model_request_config.hpp"
#include "config.hpp"
#include "model_load_config.hpp"       // the load section, spliced into our parse
#include "model_runtime_config.hpp"    // the runtime section, spliced into our parse
#include "model_scheduler_config.hpp"  // the scheduler section, spliced into our parse

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

// --- The tool-call output format enum (the accepted member strings) ---
const std::vector<std::string> kToolParsers = {
    "qwen35", "hermes", "gemma4", "museglimmer"};

// Parse a double from the CLI; a malformed / trailing-garbage value is a ConfigError.
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

// --- JSON field readers (used by modelRequestConfigFromJson) ----------------
// Absent or null → leave the field at its struct default (non-optional) or at
// nullopt (optional). A present non-number / negative int / wrong type is an error.

// Read one optional int from a JSON object (absent or null → nullopt; negative
// integer or non-integer type → ConfigError).
void readOptInt(rapidjson::Document& doc, const char* key,
                std::optional<std::size_t>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsInt() && !it->value.IsInt64())
    throw ConfigError(std::string(path) + ": expected an integer");
  if (it->value.GetInt64() < 0)
    throw ConfigError(std::string(path) + ": must be non-negative");
  out = static_cast<std::size_t>(it->value.GetUint64());
}

void readOptDouble(rapidjson::Document& doc, const char* key,
                   std::optional<double>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsNumber())
    throw ConfigError(std::string(path) + ": expected a number");
  out = it->value.GetDouble();
}

void readOptStr(rapidjson::Document& doc, const char* key,
                std::optional<std::string>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsString()) throw ConfigError(std::string(path) + ": expected a string");
  out = it->value.GetString();
}

void readDefaultDouble(rapidjson::Document& doc, const char* key,
                       double& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;  // keep struct default
  if (!it->value.IsNumber()) throw ConfigError(std::string(path) + ": expected a number");
  out = it->value.GetDouble();
}

void readDefaultSize(rapidjson::Document& doc, const char* key,
                     std::size_t& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;  // keep struct default
  if (!it->value.IsInt() && !it->value.IsInt64())
    throw ConfigError(std::string(path) + ": expected an integer");
  if (it->value.GetInt64() < 0)
    throw ConfigError(std::string(path) + ": must be non-negative");
  out = static_cast<std::size_t>(it->value.GetUint64());
}

}  // close the anonymous namespace of request-only helpers

// --- The "Model request parameters" boost section this class owns ------------
//
// The generation-config flags (the ones the model-load section does not carry),
// each with a single-letter short where its first letter is not already claimed
// by a longer-name prefix. Single source of truth for both usage() (its
// contribution to the shared --help) and parseModelRequestConfig (spliced
// alongside the load section the other class owns).
po::options_description ModelRequestConfig::requestOptionsDesc() {
  po::options_description s(
      "Model request parameters (worker: per-request defaults; run: the one request)",
      kHelpLineLength, kHelpMinDescriptionLength);
  s.add_options()
    ("num-assistant-tokens,n", po::value<std::string>(),
        "Draft tokens per speculative step — per-request default; protocol can override")
    ("assistant-confidence-threshold,a", po::value<std::string>(),
        "Confidence threshold for accepting draft tokens — per-request default; protocol can override")
    ("tool-call-parser", po::value<std::string>(),
        "Tool-call output format: qwen35, hermes, gemma4 or museglimmer "
        "(no short: prefix-conflicts --model-type)")
    ("max-tokens", po::value<std::string>(),
        "Max new tokens to generate (default 16384)")
    ("temperature", po::value<std::string>(),
        "Sampling temperature > 0; 0 or negative → greedy (default 1.0)")
    ("top-k", po::value<std::string>(),
        "Top-k sampling; 0 disables (default 50)")
    ("top-p", po::value<std::string>(),
        "Top-p (nucleus) sampling threshold (default 1.0)")
    ("repetition-penalty", po::value<std::string>(),
        "Repetition penalty; 1.0 = no penalty (default)")
    ("seed", po::value<std::string>(),
        "Random seed for reproducible sampling")
    ("frequency-penalty", po::value<std::string>(),
        "Frequency penalty (non-optional default 0; 0 = off / nullopt)")
    ("presence-penalty", po::value<std::string>(),
        "Presence penalty (non-optional default 0; 0 = off / nullopt)")
    ("chat-template-kwargs", po::value<std::string>(),
        "Chat template kwargs as a JSON object; e.g. "
        "{\"add_generation_prompt\":true}");
  return s;
}

// This class's slice of the --help listing (its "Model request parameters"
// section only); the combined two-section view is config.cpp's
// combinedConfigUsage.
std::string ModelRequestConfig::usage() {
  std::ostringstream os;
  os << ModelRequestConfig::requestOptionsDesc();
  return os.str();
}

// --- Public: parseModelRequestConfig (for ovcli run) -------------------------
//
// Reads only the generation-config fields. Parsed over the full argv (this
// class's section plus the load section the other class owns) so boost doesn't
// reject the load flags that this function simply doesn't read.
ModelRequestConfig parseModelRequestConfig(const std::vector<std::string>& args) {
  const PreProcessed pp = preProcess(args);
  if (pp.positionalCount > 1) {
    throw ConfigError("only one positional (the model path) may be given; found "
                       + std::to_string(pp.positionalCount)
                       + ". ovcli loads exactly one model -- set it with --model");
  }
  std::vector<std::string> cleanArgs = pp.args;
  cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                  cleanArgs.end());

  // Parse over the full argv: all three sections (our request section plus the
  // load + scheduler sections the other classes own), so boost doesn't reject
  // their flags a caller also passes (not "unknown option"); we read only the
  // request fields.
  po::options_description desc;
  desc.add(ModelRequestConfig::requestOptionsDesc());
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelLoadConfig::loadOptionsDesc());
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

  // Only read the generation-config fields; load-section fields (present in the
  // same vm but for the other parser) are simply not accessed.
  ModelRequestConfig cfg;  // non-optional fields at struct defaults
  if (vm.count("max-tokens"))
    cfg.max_tokens = static_cast<std::size_t>(
        parseSize(vm["max-tokens"].as<std::string>(), "--max-tokens"));
  if (vm.count("temperature"))
    cfg.temperature = parseDouble(vm["temperature"].as<std::string>(), "--temperature");
  if (vm.count("top-k"))
    cfg.top_k = static_cast<std::size_t>(
        parseSize(vm["top-k"].as<std::string>(), "--top-k"));
  if (vm.count("top-p"))
    cfg.top_p = parseDouble(vm["top-p"].as<std::string>(), "--top-p");
  if (vm.count("repetition-penalty"))
    cfg.repetition_penalty = parseDouble(vm["repetition-penalty"].as<std::string>(),
                                        "--repetition-penalty");
  if (vm.count("seed"))
    cfg.seed = static_cast<std::size_t>(
        parseSize(vm["seed"].as<std::string>(), "--seed"));
  if (vm.count("frequency-penalty"))
    cfg.frequency_penalty = parseDouble(vm["frequency-penalty"].as<std::string>(),
                                        "--frequency-penalty");
  if (vm.count("presence-penalty"))
    cfg.presence_penalty = parseDouble(vm["presence-penalty"].as<std::string>(),
                                       "--presence-penalty");
  if (vm.count("num-assistant-tokens"))
    cfg.num_assistant_tokens = static_cast<std::size_t>(
        parseSize(vm["num-assistant-tokens"].as<std::string>(),
                   "--num-assistant-tokens"));
  if (vm.count("assistant-confidence-threshold"))
    cfg.assistant_confidence_threshold =
        parseDouble(vm["assistant-confidence-threshold"].as<std::string>(),
                     "--assistant-confidence-threshold");
  if (vm.count("tool-call-parser")) {
    const std::string v = vm["tool-call-parser"].as<std::string>();
    if (!v.empty()) cfg.tool_call_parser = v;
  }
  if (vm.count("chat-template-kwargs"))
    cfg.chat_template_kwargs = std::string(requireJsonObject(
        vm["chat-template-kwargs"].as<std::string>(),
        "--chat-template-kwargs"));

  if (cfg.tool_call_parser.has_value() && !cfg.tool_call_parser->empty() &&
      !inSet(*cfg.tool_call_parser, kToolParsers))
    throw ConfigError("--tool-call-parser '" + *cfg.tool_call_parser +
                      "' must be one of: " + joinList(kToolParsers, ", "));
  return cfg;
}

// --- Public: modelRequestConfigFromJson -------------------------------------
ModelRequestConfig modelRequestConfigFromJson(const std::string& json) {
  rapidjson::Document doc;
  if (doc.Parse<0>(json.c_str()).HasParseError())
    throw ConfigError("runtime config is not valid JSON at position " +
                        std::to_string(doc.GetErrorOffset()));
  if (!doc.IsObject()) throw ConfigError("runtime config must be a JSON object");

  ModelRequestConfig cfg;  // non-optional at struct defaults
  // Non-optional (have struct defaults; skip if absent or null)
  readDefaultSize(doc,   "max_tokens",         cfg.max_tokens,          "runtime_config.max_tokens");
  readDefaultDouble(doc, "temperature",        cfg.temperature,          "runtime_config.temperature");
  readDefaultSize(doc,   "top_k",              cfg.top_k,               "runtime_config.top_k");
  readDefaultDouble(doc, "top_p",              cfg.top_p,               "runtime_config.top_p");
  readDefaultDouble(doc, "repetition_penalty", cfg.repetition_penalty,   "runtime_config.repetition_penalty");
  // Optional
  readOptInt(doc,    "seed",                          cfg.seed,                         "runtime_config.seed");
  readOptDouble(doc, "frequency_penalty",             cfg.frequency_penalty,            "runtime_config.frequency_penalty");
  readOptDouble(doc, "presence_penalty",              cfg.presence_penalty,             "runtime_config.presence_penalty");
  readOptInt(doc,    "num_assistant_tokens",          cfg.num_assistant_tokens,         "runtime_config.num_assistant_tokens");
  readOptDouble(doc, "assistant_confidence_threshold",cfg.assistant_confidence_threshold,"runtime_config.assistant_confidence_threshold");
  readOptStr(doc,    "tool_call_parser",              cfg.tool_call_parser,             "runtime_config.tool_call_parser");
  // chat_template_kwargs: a JSON object; compact it if present.
  {
    auto it = doc.FindMember("chat_template_kwargs");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
      if (!it->value.IsObject())
        throw ConfigError("runtime_config.chat_template_kwargs: expected an object");
      cfg.chat_template_kwargs = std::string(compactJsonOf(it->value));
    }
  }
  return cfg;
}

std::string modelRequestConfigToJson(const ModelRequestConfig& c) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  auto addInt    = [&](const char* key, std::size_t v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetUint64(v);
    out.AddMember(k, val, a);
  };
  auto addDouble = [&](const char* key, double v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetDouble(v);
    out.AddMember(k, val, a);
  };
  auto addStr    = [&](const char* key, const std::string& v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val(v.c_str(), v.size(), a);
    out.AddMember(k, val, a);
  };
  // Non-optional fields carry a real default (max_tokens / temperature /
  // top_k / top_p / repetition_penalty), so they are always emitted -- a
  // key that has a default value is *set*, and the output keeps that default
  // even when the request did not ask for it (matches pydantic's non-Option
  // fields; e.g. ModelRequestConfig::max_tokens = 16384).
  addInt("max_tokens",         c.max_tokens);
  addDouble("temperature",     c.temperature);
  addInt("top_k",             c.top_k);
  addDouble("top_p",           c.top_p);
  addDouble("repetition_penalty", c.repetition_penalty);
  // Optional generation knobs: a not-set (nullopt) key is omitted entirely
  // (no "null" placeholder) -- a key appears only when the request sets it.
  // An explicitly supplied 0 / false is a *set* value and is still emitted.
  if (c.seed.has_value()) addInt("seed", *c.seed);
  if (c.frequency_penalty.has_value()) addDouble("frequency_penalty", *c.frequency_penalty);
  if (c.presence_penalty.has_value()) addDouble("presence_penalty", *c.presence_penalty);
  if (c.num_assistant_tokens.has_value()) addInt("num_assistant_tokens", *c.num_assistant_tokens);
  if (c.assistant_confidence_threshold.has_value())
    addDouble("assistant_confidence_threshold", *c.assistant_confidence_threshold);
  // String optionals: omit when unset (nullopt, or an empty string).
  if (c.tool_call_parser && !c.tool_call_parser->empty())
    addStr("tool_call_parser", *c.tool_call_parser);
  if (c.chat_template_kwargs && !c.chat_template_kwargs->empty()) {
    rapidjson::Document tmp;
    rapidjson::Value node;
    if (!tmp.Parse<0>(c.chat_template_kwargs->c_str()).HasParseError())
      node.CopyFrom(tmp, a);
    else node.SetObject();
    out.AddMember(rapidjson::Value("chat_template_kwargs", a), std::move(node), a);
  }

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

}  // namespace config
}  // namespace ovcli
