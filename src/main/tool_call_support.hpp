// Tool-call support for the `vlm` subcommand -- the MODEL-GENERIC selection + carrying
// machinery.  It is a header + cpp pair (the project's model_*_config.{hpp,cpp} idiom):
// THIS header DECLARES the registry / detect / select functions and the ContentStreamer that
// carries a selected IncrementalParser into the engine; the specific carrier is the OpenArc
// Qwen35ToolCallStreamer analog (ContentStreamer, see the .cpp) -- it keeps the engine's parser
// loop and content accumulation; there is NO reasoning splitter (reasoning stays on the normal
// content output, only the tool block is parsed out into the tool_calls array); their method
// BODIES live in
// tool_call_support.cpp.  The MODEL-SPECIFIC parsers each live in their own
// toolcallparsers/<model>_tool_call_parser.{hpp,cpp}: this phase is Qwen35ToolCallParser
// (with its tag/sentinel constants + its tr/argValue/qwenBlockToCall free helpers); a later
// phase adds gemma4 / hermes / museglimmer beside it.
//
// Future-proof design (the generic part): a model's OWN chat template is the source of truth
// for HOW it states a call, so the CLI derives BOTH "parse or not" and "which parser" by
// rendering the loaded template and matching a model-intrinsic sentinel (the functions below).
// The engine runs the parser inside a TextParserStreamer subclass and, in the (tools|kwargs)
// pre-apply path, copies the streamer's get_parsed_message() (which, on a completed call,
// holds .tool_calls + a TOOL_CALL_STOP status) into res.parsed[0].  So we leave cfg.parsers
// empty and run the parser IN the streamer; a non-empty tool_calls then makes the CLI report
// the uniform finish_reason "tool_calls" (a small model can under-rand its outer close tag;
// the parser finalises on the inner function close, so the turn is still reported as a call).
//
// Bracket-safety guardrail (shared with the per-model headers): the tag BYTES are built from
// the \u003C / \u003E escape constants in the per-model headers; this file only NAMES those
// (via the included per-model header) plus the model-intrinsic example_function_name /
// example_parameter sentinels -- no literal angle-bracket pair sits here.
#pragma once

#include <openvino/genai/parsers.hpp>          // IncrementalParser / StreamingStatus
#include <openvino/genai/text_streamer.hpp>     // TextParserStreamer (pure-virtual write(JsonContainer&))
#include <openvino/genai/tokenizer.hpp>         // Tokenizer + apply_chat_template / get_original_chat_template
#include <openvino/genai/chat_history.hpp>      // ChatHistory
#include <openvino/genai/generation_config.hpp>  // (for downstream; not used here directly)
#include <openvino/genai/json_container.hpp>     // JsonContainer

#include "toolcallparsers/qwen35_tool_call_parser.hpp"  // Qwen35ToolCallParser + its tag/sentinel constants + helpers

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declaration (the full type arrives via llm_pipeline.hpp, included in the .cpp) -- the engine-side
// result whose `parsed` member the bridge fills.  Forward-declared here to keep the header light.
namespace ov { namespace genai { class DecodedResults; } }

namespace ovgenai {
namespace toolcall {

// The reasoning tags == the ReasoningIncrementalParser's defaults == the close qwen emits; defined
// here (not the qwen35 header) so the reasoning gate is model-generic.  Built from the per-model
// header's generic angle-byte constants (LT/GT) -- no literal angle byte here.
inline const std::string kThinkingOpen  = LT + "think"   + GT;   // the reasoning OPEN tag
inline const std::string kThinkingClose = LT + "/think" + GT;   // the reasoning CLOSE tag

// A TextParserStreamer subclass that carries an IncrementalParser (the OpenArc
// Qwen35ToolCallStreamer analog): the base keeps the parser loop and copies its
// get_parsed_message() into res.parsed[0], while this streamer appends the (specials-preserving)
// content to the caller's accumulated output.  A reasoning splitter may sit on the parser list
// (composeParserList prepends a gated ReasoningIncrementalParser only when thinking is on); in that
// case the same per-chunk write also streams reasoning_content, otherwise the base just fills the
// content key (the no-think case).  Only the tool block is parsed out (by a carried parser) into the
// tool_calls array.  The pure-virtual write(JsonContainer&) is the one that fires; the string-
// write base drives the parser loop and calls it.  (Method bodies in tool_call_support.cpp.)
class ContentStreamer : public ov::genai::TextParserStreamer {
public:
  ContentStreamer(const ov::genai::Tokenizer& tok,
                  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                  std::string& acc);
  ov::genai::StreamingStatus write(ov::genai::JsonContainer& message) override;
private:
  std::string& m_acc;                        // the output the caller drains (specials preserved)
};

// The engine-side copy the *VLM* pipeline never performs: only the *LLM* pipeline's run_generate_with_parsers
// copies a TextParserStreamer's folded get_parsed_message() into res.parsed[0].  The VLM pipeline that
// `vlm` drives leaves res.parsed EMPTY, so a tool-call turn would otherwise report the engine's own
// finish_reason "tool_calls" (set from the parser's TOOL_CALL_STOP) yet carry NO tool_calls array.  The
// CLI still holds the carrier, so we reproduce the copy: for a TextParserStreamer (our ContentStreamer) we
// copy its folded message -- content PLUS the one-shot tool_calls array the parser wrote into its private
// message -- into res.parsed[0]; a plain TextStreamer is not a TextParserStreamer, so the cast misses and
// res.parsed is left exactly as the engine produced it (graceful fallback: no tool_calls, finish_reason
// stays as the engine's own).
void bridgeParsedMessage(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                         ov::genai::DecodedResults& res);

// --- the registry + detection / selection (bodies in tool_call_support.cpp) ------------
// One source of truth: every advertised (validated) name is an entry; an entry is "wired"
// iff its `make` is non-null.  `sentinels` are the substrings whose presence in the model's
// RENDERED template selects this parser (empty = not auto-detected yet).
struct ToolCallFormatSpec {
  std::string name;
  std::vector<std::string> sentinels;
  // This format's tool parser(s) only -- a reasoning splitter is NOT in here; a separate, gated
  // ReasoningIncrementalParser is composed beside it by composeParserList (gated on the rendered
  // chat template).  For qwen35 this returns just the model-specific Qwen35ToolCallParser (it adds
  // the tool_calls key).  An unwired (not-yet-implemented) format carries an empty (null) function.
  std::function<std::vector<std::shared_ptr<ov::genai::IncrementalParser>>()> make;
};

// Every advertised name -> an entry (wired iff `make` is non-null); the registry owns the
// kSentfunc / kSentry sentinels via the included per-model header.
const std::vector<ToolCallFormatSpec>& knownToolCallFormats();
// The flat list of every advertised (validated) --tool-call-parser name (the one source the
// CLI-flag validation reads).
std::vector<std::string> advertisedToolCallParsers();
// Build the format's tool parser(s) for a (validated) name, or an empty list if advertised-but-unwired.
// For qwen35 this returns just the model-specific Qwen35ToolCallParser.  A reasoning splitter is NOT
// returned here -- a separate, gated ReasoningIncrementalParser is prepended by composeParserList only
// when the rendered chat template shows thinking is on.
std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name);
// Render the model's OWN loaded template with a probe tool arg and match that against the
// registry sentinels; nullopt if none match (graceful: no parser, no turn-stop).
std::optional<std::string> detectToolCallFormat(const ov::genai::Tokenizer& tok);
// The MODEL-FREE core of the auto-detect gate (no model, no render) so the gate is unit
// testable: the entry whose `sentinels` are ALL present in `rendered` wins; nullopt if none.
std::optional<std::string> matchFormat(const std::string& rendered);

// The reasoning splitter is a runtime TOGGLE, not a compose-time gate. One
// DynamicReasoningIncrementalParser is ALWAYS at slot 0 of the parser list (the tool parser(s)
// after it) -- one parser for both the reasoning_content split AND the tool_calls split.  The
// old render-based gates (reasoningEnabledFromRender / detectReasoningEnabled) are removed because
// a model may render a reasoning template yet not actually reason (the Qwen3.5-0.8B), so the
// rendered prompt is not a trustworthy on/off.  Reasoning is instead an EXPLICIT user control
// (the --enable-thinking flag, or a per-request chat_template_kwargs enable_thinking) applied by
// calling setActive() on the kept handle; a per-request value wins over the flag over the model
// default.  Default OFF: a non-thinking model's lead-in is routed to content (not a spurious
// reasoning_content); ON routes reasoning into reasoning_content (the engine's normal split).
class DynamicReasoningIncrementalParser : public ov::genai::ReasoningIncrementalParser {
 public:
  using ov::genai::ReasoningIncrementalParser::ReasoningIncrementalParser;
  void setActive(bool active) { m_active = active; }
  bool isActive() const { return m_active; }
  std::string parse(ov::genai::JsonContainer& dm, std::string& delta_text,
                    const std::optional<std::vector<int64_t>>& delta_tokens) override {
    if (!m_active)
      return delta_text;
    return ov::genai::ReasoningIncrementalParser::parse(dm, delta_text, delta_tokens);
  }
 private:
  bool m_active = false;
};
std::shared_ptr<DynamicReasoningIncrementalParser> makeDynamicReasoningParser();
std::shared_ptr<ov::genai::IncrementalParser> makeReasoningParser();
std::vector<std::shared_ptr<ov::genai::IncrementalParser>>
    composeParserList(const std::optional<std::string>& name, bool reasoner_active);
// Selection precedence: an explicit --tool-call-parser name override wins; else auto-detect
// from the model's OWN loaded template; else no parser (graceful: no stop, finish_reason
// stays stop, no tool_calls -- exactly the no-parser default today).
std::optional<std::string> selectToolCallParserName(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out = nullptr);

}  // namespace toolcall
}  // namespace ovgenai
