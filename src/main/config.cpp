// Shared config *plumbing* (namespace ovcli::config) that none of the three
// concrete configs (model-load, scheduler, per-request) owns on its own: the
// ConfigError type, the argv / number / JSON-object coercions all parsers share,
// showConfigJson (the composite that names all three configs' JSON --
// show-config's --) and combinedConfigUsage (renders the three option sections,
// each as its owning class renders it, for the shared --help). It holds NO
// parameters: the boost options_descriptions live in the three classes that name
// their flags.
#include "config.hpp"

// The three concrete config headers: showConfigJson composes their serializers
// and combinedConfigUsage renders their option sections, so the full
// ModelLoadConfig / ModelSchedulerConfig / ModelRequestConfig definitions must
// all be visible here (config.hpp only forward-declares them to avoid an include
// cycle).
#include "model_load_config.hpp"
#include "model_runtime_config.hpp"
#include "model_scheduler_config.hpp"
#include "model_request_config.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace po = boost::program_options;

namespace ovcli {
namespace config {
namespace {

// --- The internals of preProcess: token classification into options / values /
//     positionals (so boost sees a positional-free list for the single model).
//     File-local: only preProcess (a shared helper below) needs these. ------

// (Unused enum kept for the classify() machinery's readability.)
enum class TokKind { TERM, OPTION, VALUE, POSITIONAL };

bool optionTakesNextSlot(const std::string& optTok) {
  if (optTok == "-" || optTok == "--help" || optTok == "-h") return false;
  if (optTok.find('=') != std::string::npos) return false;
  return true;
}

bool tokenIsModelPathOption(const std::string& tok) {
  if (tok == "--model" || tok == "-m") return true;
  if (tok.rfind("--model=", 0) == 0) return true;
  if (tok.rfind("-m=", 0) == 0) return true;
  return false;
}

struct Classified {
  std::vector<std::string> optToks;
  std::vector<std::string> valueToks;
  std::vector<std::string> positionalToks;
  bool explicitModelPath = false;
  std::size_t positionalCount = 0;
};

Classified classify(const std::vector<std::string>& args) {
  Classified c;
  bool afterDashDash = false;
  bool prevTakesNext = false;
  for (const auto& tok : args) {
    if (!afterDashDash && tok == "--") {
      c.optToks.push_back(tok);
      afterDashDash = true;
      prevTakesNext = false;
      continue;
    }
    if (afterDashDash) {
      c.positionalToks.push_back(tok);
      c.positionalCount++;
      continue;
    }
    if (!tok.empty() && tok[0] == '-') {
      if (tok == "--") {
        c.optToks.push_back(tok);
        continue;
      }
      c.optToks.push_back(tok);
      if (tokenIsModelPathOption(tok)) c.explicitModelPath = true;
      prevTakesNext = optionTakesNextSlot(tok);
      continue;
    }
    if (prevTakesNext) {
      c.valueToks.push_back(tok);
      prevTakesNext = false;
    } else {
      c.positionalToks.push_back(tok);
      c.positionalCount++;
    }
  }
  return c;
}

bool hasConflict(const Classified& c) {
  return c.explicitModelPath && c.positionalCount > 0;
}

// Raw base-10 integer parse (accepts an optional sign); used only by parseSize
// below. Throws ConfigError on a malformed / trailing-garbage value.
long long parseInt(const std::string& s, const char* flag) {
  try {
    size_t pos = 0;
    long long v = std::stoll(s, &pos, 10);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    return v;
  } catch (...) {
    throw ConfigError(std::string("invalid integer for ") + flag + ": '" + s + "'");
  }
}

}  // close the anonymous namespace of config-internal helpers

// --- Parse helpers shared by both the model-load and the per-request parsers --
//     (declared in config.hpp; defined here so each class's parser links to one
//     implementation. config.cpp owns none of the parameters: the option
//     sections are defined in the two classes that name their flags.)

bool inSet(const std::string& v, const std::vector<std::string>& s) {
  for (const auto& x : s)
    if (x == v) return true;
  return false;
}

std::string joinList(const std::vector<std::string>& v, const std::string& sep) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += sep;
    out += v[i];
  }
  return out;
}

// Re-serialize any parsed/constructed value compactly (used to canonicalize
// the free-form JSON blobs that travel between argv and JSON).
std::string compactJsonOf(rapidjson::Value& v) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  v.Accept(w);
  return std::string(sb.GetString());
}

// A JSON string (a flag value or a member) that must be a JSON *object*,
// re-emitted compactly; an absent/garbage/non-object value is an error.
std::string requireJsonObject(const std::string& s, const char* flag) {
  rapidjson::Document d;
  if (d.Parse<0>(s.c_str()).HasParseError())
    throw ConfigError(std::string(flag) + ": not valid JSON at position " +
                        std::to_string(d.GetErrorOffset()));
  if (!d.IsObject()) throw ConfigError(std::string(flag) + ": expected a JSON object");
  return compactJsonOf(d);
}

// Like parseInt but also rejects negative values; for use with size_t fields.
long long parseSize(const std::string& s, const char* flag) {
  long long v = parseInt(s, flag);
  if (v < 0)
    throw ConfigError(std::string("negative value not allowed for ") + flag + ": '" + s + "'");
  return v;
}

// --- A tiny positional-friendly preprocessor for boost ----------------------
// ovcli's UX has a single leading positional stand in for --model, but
// boost 1.90's positional_options_description rejects "the option cannot be
// specified more than once" when both are given. So before handing args to
// boost we:
//   * count the leading positionals (anything before `--` that doesn't start
//     with `-`; anything at/after `--` is also a positional);
//   * if there is more than one, leave it to the caller (each parser reports
//     the "only one positional" ConfigError);
//   * drop the `--` terminator so boost doesn't choke on it;
//   * if an explicit --model / -m is ALSO present, drop the positional
//     (explicit wins); otherwise keep it so boost's positional_options_description
//     maps it onto `model`.
PreProcessed preProcess(const std::vector<std::string>& args) {
  PreProcessed pp;
  const Classified c = classify(args);
  pp.positionalCount = c.positionalCount;
  pp.explicitModelPath = c.explicitModelPath;

  if (!hasConflict(c)) {
    pp.args = args;
    pp.args.erase(std::remove(pp.args.begin(), pp.args.end(), std::string("--")),
                   pp.args.end());
    return pp;
  }
  size_t optIdx = 0, valIdx = 0, posIdx = 0;
  bool afterDashDash = false;
  for (const auto& tok : args) {
    if (!afterDashDash && tok == "--") { afterDashDash = true; continue; }
    if (afterDashDash) { posIdx++; continue; }
    if (!tok.empty() && tok[0] == '-') {
      pp.args.push_back(tok);
      optIdx++;
      continue;
    }
    if (valIdx < c.valueToks.size() && c.valueToks[valIdx] == tok) {
      pp.args.push_back(tok);
      valIdx++;
    } else {
      posIdx++;
    }
  }
  return pp;
}

// Turn a boost parser error into a ConfigError in ovcli's own vocabulary
// (so tests / callers see "unknown option" / "requires a value" consistently).
ConfigError rephraseBoost(const po::error& e) {
  const std::string msg = e.what();
  auto liftName = [&]() {
    auto q1 = msg.find('\'');
    if (q1 == std::string::npos) return std::string{};
    auto q2 = msg.find('\'', q1 + 1);
    if (q2 == std::string::npos || q2 <= q1) return std::string{};
    return msg.substr(q1 + 1, q2 - q1 - 1);
  };
  if (msg.find("unrecognised option") != std::string::npos ||
      msg.find("unknown option")       != std::string::npos) {
    const std::string name = liftName();
    return ConfigError("unknown option" + (name.empty() ? std::string{} : " " + name));
  }
  if (msg.find("required argument for option") != std::string::npos) {
    const std::string name = liftName();
    return ConfigError((name.empty() ? std::string("--x") : name) + " requires a value");
  }
  return ConfigError("option parsing error: " + msg);
}

// --- showConfigJson (composite of load + scheduler + request configs) --------
//
// Parse the three sub-serializers' output as rapidjson Documents and copy them
// into a parent object under the "load_config" / "scheduler_config" /
// "request_config" keys (the three top-level sections the user asked for). The
// output is a single compact JSON string.
std::string showConfigJson(const ModelLoadConfig& loadConfig,
                           const ModelRuntimeConfig& runtimeConfig,
                           const ModelSchedulerConfig& schedulerConfig,
                           const ModelRequestConfig& requestConfig) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  auto addSection = [&](const char* key, const std::string& json) {
    rapidjson::Document tmp;
    rapidjson::Value node;
    if (!json.empty() && !tmp.Parse<0>(json.c_str()).HasParseError())
      node.CopyFrom(tmp, a);
    else
      node.SetObject();
    out.AddMember(rapidjson::Value(key, a), node, a);
  };
  addSection("load_config",      modelLoadConfigToJson(loadConfig));
  addSection("runtime_config",   modelRuntimeConfigToJson(runtimeConfig));
  addSection("scheduler_config",  modelSchedulerConfigToJson(schedulerConfig));
  addSection("request_config",   modelRequestConfigToJson(requestConfig));

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

// --- combinedConfigUsage: the shared --help view (a composition, not an
//     ownership) -------------------------------------------------------------
//
// Render the three option sections side by side, each exactly as the class that
// OWNS it renders it, so `ovcli <cmd> --help` shows all three parameter groups
// under their own headers. This mirrors the former makeLoadOptions: the --help /
// -h switch is registered on the untitled root section (added after the three
// named sub-descriptions) so a stray --help in a direct test still parses clean;
// streaming the combined description then prints the help tip, the "Model load
// parameters" section, then the "Scheduler Options" section, then the "Model
// request parameters" section.
std::string combinedConfigUsage() {
  po::options_description desc;   // root, untitled (carries the --help switch)
  desc.add(ModelLoadConfig::loadOptionsDesc());
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelSchedulerConfig::schedulerOptionsDesc());
  desc.add(ModelRequestConfig::requestOptionsDesc());
  desc.add_options()("help", po::bool_switch(), "Show this help message");
  std::ostringstream os;
  os << desc;
  return os.str();
}

}  // namespace config
}  // namespace ovcli
