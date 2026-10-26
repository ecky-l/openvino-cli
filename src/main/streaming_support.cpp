// Method bodies for the streaming-support machinery (declared in streaming_support.hpp) -- moved
// out of tool_call_support.cpp, which now keeps only the registry / detect / select / bridge /
// compose.  This TU carries the bodies for:
//   * ContentStreamer (the OpenArc Qwen35ToolCallStreamer analog) -- the per-chunk JSON emit;
//   * DynamicReasoningIncrementalParser (the slot-0 reasoner factory + the makeReasoningParser
//     alias).  See streaming_support.hpp for the design rationale.
#include "streaming_support.hpp"

#include <cstdio>    // fputs / fputc / fflush -- the per-chunk JSON emit the ContentStreamer does

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
}  // namespace toolcall
}  // namespace ovgenai

