// Method bodies for the model-GENERIC tool-call machinery declared in tool_call_support.hpp
// -- the ContentStreamer that carries a selected IncrementalParser into the engine, plus the
// registry / detect / select.  The model-SPECIFIC parsers each live in their own .cpp under
// toolcallparsers/ (this phase: qwen35_tool_call_parser.cpp).  See the header for the design
// rationale; this TU only carries the BODIES so the header stays a clean interface.
#include "tool_call_support.hpp"

#include <openvino/genai/llm_pipeline.hpp>       // DecodedResults (full type; the bridge target)
#include <openvino/genai/text_streamer.hpp>      // TextParserStreamer + get_parsed_message (the source)
#include <openvino/genai/parsers.hpp>            // ReasoningIncrementalParser (the prepended reasoning splitter)

#include <cstdio>                                // fputs / fputc / fflush -- the per-chunk JSON emit

namespace ovgenai {
namespace toolcall {

// --- ContentStreamer (carries the parser; the OpenArc Qwen35ToolCallStreamer analog; the engine
// runs the loop + copies the result) -- The base's ctor hardcodes EMPTY detokenization params,
// so its decode() would fall back to the tokenizer's rt_info default (skip_special_tokens = true
// for these fixtures) and STRIP a leading ‹/thinking› -- for the 0.8B that close-of-thinking IS
// the whole lead-in before the ‹tool_call›, so the surfaced content would collapse to empty.
// Setting skip_special_tokens = false here keeps that 0.8B no-think lead-in verbatim (a plain
// TextStreamer decodes a specials-preserving token stripped) instead of collapsing it to empty.
// The carried parser routes the tool block out into the tool_calls array; a gated
// ReasoningIncrementalParser on the list (composeParserList prepends it only when thinking is on)
// would also split reasoning into a reasoning_content key, and the base fills the remaining
// content either way.
ContentStreamer::ContentStreamer(const ov::genai::Tokenizer& tok,
                                 std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                                 std::string& acc)
  : ov::genai::TextParserStreamer(tok, std::move(parsers)), m_acc(acc) {
  m_additional_detokenization_params = ov::AnyMap{ ov::genai::skip_special_tokens(false) };
}

ov::genai::StreamingStatus ContentStreamer::write(ov::genai::JsonContainer& message) {
  // Stream the result: on EVERY decoded chunk, emit the string repr of the delta message the parsers
  // just filled (content, the reasoner's reasoning_content, and a one-shot tool_calls on the chunk
  // where a call completes) as one compact JSON line -- replacing the old "accumulate then emit at the
  // end".  to_json_string(-1) is the compact dumper (the worker stays one physical line).
  std::string line = message.to_json_string(-1);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
  // Also keep a plain (reasoning-stripped) content accumulator: unused by the terminal emitResult
  // (the content now streams per chunk), but still read by the per-model leak/obs test checks.
  m_acc += message["content"].as_string().value_or(std::string());
  return ov::genai::StreamingStatus::RUNNING;
}

// The bridge the *VLM* pipeline omitted (see the header rationale): dynamic_pointer_cast is the gate --
// it hits only for a parser-carrying streamer (our ContentStreamer, a TextParserStreamer) and misses
// for a plain TextStreamer (the no-parser engine path, where we must touch nothing).  When it hits we
// copy the streamer's folded get_parsed_message() into res.parsed[0]; that folded message carries BOTH
// the content and the one-shot tool_calls array the parser wrote into its private internal message.
void bridgeParsedMessage(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                         ov::genai::DecodedResults& res) {
  auto tps = std::dynamic_pointer_cast<ov::genai::TextParserStreamer>(streamer);
  if (!tps) return;
  if (res.parsed.empty()) res.parsed.resize(1);
  res.parsed[0] = tps->get_parsed_message();
}

// --- the registry: every advertised name -> an entry (wired iff `make` is non-null) --------
// The `qwen35` entry is wired (builds the model-specific Qwen35ToolCallParser; its kSentfunc
// / kSentry sentinels -- the model-intrinsic example_function_name / example_parameter -- are
// owned by the per-model header).  The others stay advertised-but-unwired today (resolve to
// null -> graceful: no parser, no turn-stop, finish_reason stays stop).
const std::vector<ToolCallFormatSpec>& knownToolCallFormats() {
  static const std::vector<ToolCallFormatSpec> kSpecs = {
    { "qwen35", { kSentfunc, kSentry },
      // Just this format's tool parser.  A reasoning splitter is NOT in here: a separate, gated
      // ReasoningIncrementalParser is composed beside it by composeParserList and prepended only
      // when the rendered template shows thinking is on.
      []() { return std::vector<std::shared_ptr<ov::genai::IncrementalParser>>{
        std::static_pointer_cast<ov::genai::IncrementalParser>(
            std::make_shared<Qwen35ToolCallParser>()),
      }; } },
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

std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name) {
  for (const auto& s : knownToolCallFormats())
    if (s.name == name && s.make)
      return s.make();
  return {};
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

// One DYNAMIC reasoner, always at slot 0 (the user's one-parser both-sides design). Reasoning is a
// runtime TOGGLE on it (setActive), not a compose-time decision: a model may render a reasoning
// template yet not actually reason, so render-based auto-detect is not trustworthy.
std::shared_ptr<DynamicReasoningIncrementalParser> makeDynamicReasoningParser() {
  return std::make_shared<DynamicReasoningIncrementalParser>(
      /*expect_open_tag=*/false, /*keep_original_content=*/false,
      kThinkingOpen, kThinkingClose);
}
std::shared_ptr<ov::genai::IncrementalParser> makeReasoningParser() {
  return std::static_pointer_cast<ov::genai::IncrementalParser>(makeDynamicReasoningParser());
}
std::vector<std::shared_ptr<ov::genai::IncrementalParser>>
    composeParserList(const std::optional<std::string>& name, bool reasoner_active) {
  auto dyn = makeDynamicReasoningParser();
  if (reasoner_active)
    dyn->setActive(true);
  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> powers;
  powers.push_back(std::static_pointer_cast<ov::genai::IncrementalParser>(dyn));
  if (name && !name->empty()) {
    const auto tool = makeToolCallParserByName(*name);
    powers.insert(powers.end(), tool.begin(), tool.end());
  }
  return powers;
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



