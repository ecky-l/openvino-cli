// Tool-call support for the `vlm` subcommand -- the MODEL-GENERIC tool-call registry plus the
// render-based detect / selection that picks which named format's parser(s) run, and the VLM-pipeline
// bridge that copies a carrier's folded parse into res.parsed[0].  Header + cpp (the project's
// model_*_config.{hpp,cpp} idiom); the .cpp carries the bodies.
//
// It now #includes streaming_support.{hpp,cpp}, which owns the TWO pieces of the engine-side parser
// carrier this registry is composed around: ContentStreamer (the TextParserStreamer that carries an
// IncrementalParser list into the engine) and the always-slot-0 DynamicReasoningIncrementalParser (a
// ReasoningIncrementalParser with a runtime on/off split, plus the kThinkingOpen / kThinkingClose
// reasoning tags).  composeParserList pairs the slot-0 reasoner with a format's tool parser(s); the
// reasoner and its tags live with the streamer here-included, beside the registry it composes.
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
#include "streaming_support.hpp"        // the moved ContentStreamer + slot-0 reasoner (composeParserList's slot 0; source of the kThinking* tags)

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
struct ModelFamilySpec {
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
const std::vector<ModelFamilySpec>& knownModelFamilies();
// The flat list of every advertised (validated) --model-family name (the one source the
// CLI-flag validation reads).
std::vector<std::string> advertisedModelFamilies();
// Build the format's tool parser(s) for a (validated) name, or an empty list if advertised-but-unwired.
// For qwen35 this returns just the model-specific Qwen35ToolCallParser.  A reasoning splitter is NOT
// returned here -- a separate, gated ReasoningIncrementalParser is prepended by composeParserList only
// when the rendered chat template shows thinking is on.
std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name);
// Render the model's OWN loaded template with a probe tool arg and match that against the
// registry sentinels; nullopt if none match (graceful: no parser, no turn-stop).
std::optional<std::string> detectModelFamily(const ov::genai::Tokenizer& tok);
// The MODEL-FREE core of the auto-detect gate (no model, no render) so the gate is unit
// testable: the entry whose `sentinels` are ALL present in `rendered` wins; nullopt if none.
std::optional<std::string> matchFormat(const std::string& rendered);

std::vector<std::shared_ptr<ov::genai::IncrementalParser>>
    composeParserList(const std::optional<std::string>& name, bool reasoner_active);
// Selection precedence: an explicit --model-family name override wins; else auto-detect
// from the model's OWN loaded template; else no parser (graceful: no stop, finish_reason
// stays stop, no tool_calls -- exactly the no-parser default today).
std::optional<std::string> selectModelFamily(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out = nullptr);
}  // namespace toolcall
}  // namespace ovgenai

