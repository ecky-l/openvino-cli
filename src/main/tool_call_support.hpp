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

// A TextParserStreamer subclass that carries an IncrementalParser (the OpenArc
// Qwen35ToolCallStreamer analog): the base keeps the parser loop and copies its
// get_parsed_message() into res.parsed[0], while this streamer appends the (specials-preserving)
// content to the caller's accumulated output.  There is NO reasoning splitter -- reasoning stays
// on the normal content output; only the tool block is parsed out (by the parser) into the
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
  // The ordered parser list for this format (see makeToolCallParserByName).  For qwen35 it is two,
  // in order: a ReasoningIncrementalParser (expect_open_tag=false) that splits reasoning off into a
  // reasoning_content key and rewrites content, then the model-specific tool parser that adds the
  // tool_calls key.  An unwired (not-yet-implemented) format carries an empty (null) function.
  std::function<std::vector<std::shared_ptr<ov::genai::IncrementalParser>>()> make;
};

// Every advertised name -> an entry (wired iff `make` is non-null); the registry owns the
// kSentfunc / kSentry sentinels via the included per-model header.
const std::vector<ToolCallFormatSpec>& knownToolCallFormats();
// The flat list of every advertised (validated) --tool-call-parser name (the one source the
// CLI-flag validation reads).
std::vector<std::string> advertisedToolCallParsers();
// Build the ordered parser list for a (validated) name, or an empty list if advertised-but-unwired.
// For qwen35 the list is two, in order: a ReasoningIncrementalParser (expect_open_tag=false -- Qwen's
// open-tag is pre-filled in the prompt) that strips reasoning into reasoning_content and rewrites the
// content key, then the model-specific Qwen35ToolCallParser that adds the tool_calls key.
std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name);
// Render the model's OWN loaded template with a probe tool arg and match that against the
// registry sentinels; nullopt if none match (graceful: no parser, no turn-stop).
std::optional<std::string> detectToolCallFormat(const ov::genai::Tokenizer& tok);
// The MODEL-FREE core of the auto-detect gate (no model, no render) so the gate is unit
// testable: the entry whose `sentinels` are ALL present in `rendered` wins; nullopt if none.
std::optional<std::string> matchFormat(const std::string& rendered);
// Selection precedence: an explicit --tool-call-parser name override wins; else auto-detect
// from the model's OWN loaded template; else no parser (graceful: no stop, finish_reason
// stays stop, no tool_calls -- exactly the no-parser default today).
std::optional<std::string> selectToolCallParserName(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out = nullptr);

}  // namespace toolcall
}  // namespace ovgenai

