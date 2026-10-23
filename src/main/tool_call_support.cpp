// Method bodies for the model-GENERIC tool-call machinery declared in tool_call_support.hpp
// -- the ContentStreamer that carries a selected IncrementalParser into the engine, plus the
// registry / detect / select.  The model-SPECIFIC parsers each live in their own .cpp under
// toolcallparsers/ (this phase: qwen35_tool_call_parser.cpp).  See the header for the design
// rationale; this TU only carries the BODIES so the header stays a clean interface.
#include "tool_call_support.hpp"

namespace ovgenai {
namespace toolcall {

// --- ContentStreamer (carries the parser; the engine runs the loop + copies the result) --
ContentStreamer::ContentStreamer(const ov::genai::Tokenizer& tok,
                                 std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                                 std::string& acc)
  : ov::genai::TextParserStreamer(tok, std::move(parsers)), m_acc(acc) {}

ov::genai::StreamingStatus ContentStreamer::write(ov::genai::JsonContainer& message) {
  m_acc += message["content"].as_string().value_or(std::string());   // -> result
  return ov::genai::StreamingStatus::RUNNING;
}

// --- the registry: every advertised name -> an entry (wired iff `make` is non-null) --------
// The `qwen35` entry is wired (builds the model-specific Qwen35ToolCallParser; its kSentfunc
// / kSentry sentinels -- the model-intrinsic example_function_name / example_parameter -- are
// owned by the per-model header).  The others stay advertised-but-unwired today (resolve to
// null -> graceful: no parser, no turn-stop, finish_reason stays stop).
const std::vector<ToolCallFormatSpec>& knownToolCallFormats() {
  static const std::vector<ToolCallFormatSpec> kSpecs = {
    { "qwen35", { kSentfunc, kSentry },
      []() { return std::static_pointer_cast<ov::genai::IncrementalParser>(
                     std::make_shared<Qwen35ToolCallParser>()); } },
    { "hermes",      { }, nullptr },
    { "gemma4",      { }, nullptr },
    { "museglimmer", { }, nullptr },
  };
  return kSpecs;
}

std::vector<std::string> advertisedToolCallParsers() {
  std::vector<std::string> v;
  for (const auto& s : knownToolCallFormats())
    v.push_back(s.name);
  return v;
}

std::shared_ptr<ov::genai::IncrementalParser> makeToolCallParserByName(const std::string& name) {
  for (const auto& s : knownToolCallFormats())
    if (s.name == name && s.make)
      return s.make();
  return nullptr;
}

// --- detect the model's format from its OWN loaded template (render a probe, match sentinels)
std::optional<std::string> detectToolCallFormat(const ov::genai::Tokenizer& tok) {
  // Probe tool arg: one dummy function tool, BUILT via the engine's own JsonContainer
  // API rather than hand-typed JSON, so the outer array and every nested object are
  // balanced by construction (a hand-typed literal here had dropped the trailing object
  // close + array close, making from_json_string throw -> the catch returned nullopt and
  // auto-detect silently never fired). The literal tokens below ("_probe"/"x"/"probe")
  // deliberately do NOT collide with the model-intrinsic example_function_name /
  // example_parameter sentinels the parser keys on -- those come from the template.
  ov::genai::JsonContainer xProp = ov::genai::JsonContainer::object();
  xProp["type"] = std::string("string");
  xProp["description"] = std::string("probe");
  ov::genai::JsonContainer props = ov::genai::JsonContainer::object();
  props["x"] = xProp;
  ov::genai::JsonContainer params = ov::genai::JsonContainer::object();
  params["type"] = std::string("object");
  params["properties"] = props;
  ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
  fn["name"] = std::string("_probe");
  fn["description"] = std::string("probe");
  fn["parameters"] = params;
  ov::genai::JsonContainer tool = ov::genai::JsonContainer::object();
  tool["type"] = std::string("function");
  tool["function"] = fn;
  ov::genai::JsonContainer tools = ov::genai::JsonContainer::array();
  tools.push_back(tool);
  ov::genai::ChatHistory history(ov::genai::JsonContainer::from_json_string(
      "[{\"role\":\"user\",\"content\":\"hi\"}]"));
  std::string rendered;
  try {
    rendered = tok.apply_chat_template(history, true, std::string(),
                                       std::optional<ov::genai::JsonContainer>(tools),
                                       std::nullopt);
  } catch (...) {
    return std::nullopt;
  }
  for (const auto& s : knownToolCallFormats()) {
    if (s.sentinels.empty())
      continue;
    bool all = true;
    for (const auto& needle : s.sentinels)
      if (rendered.find(needle) == std::string::npos) { all = false; break; }
    if (all)
      return s.name;
  }
  return std::nullopt;
}

// matchFormat -- the MODEL-FREE core of the auto-detect gate (no model, no render), split
// out from detectToolCallFormat (which renders the loaded template and then calls this) so
// the gate is unit-testable without a model: the entry whose `sentinels` are ALL present in
// `rendered` wins; nullopt if none match (graceful: no parser, no stop).
std::optional<std::string> matchFormat(const std::string& rendered) {
  for (const auto& s : knownToolCallFormats()) {
    if (s.sentinels.empty())
      continue;
    bool has = true;
    for (const auto& needle : s.sentinels)
      if (rendered.find(needle) == std::string::npos) { has = false; break; }
    if (has)
      return s.name;
  }
  return std::nullopt;
}

// selectToolCallParserName -- selection precedence: an explicit --tool-call-parser name
// (override) wins; else auto-detect from the model's OWN loaded template; else no parser
// (graceful: no stop, finish_reason stays stop, no tool_calls -- today's behaviour).
std::optional<std::string> selectToolCallParserName(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out) {
  if (override_name.has_value() && !override_name->empty())
    return *override_name;
  auto d = detectToolCallFormat(tok);
  if (detected_out)
    *detected_out = d;
  return d;
}

}  // namespace toolcall
}  // namespace ovgenai



