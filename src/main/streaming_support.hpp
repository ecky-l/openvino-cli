// Streaming support for the `vlm` subcommand (moved out of tool_call_support.{hpp,cpp}) -- the
// content / reasoning parser-carrying machinery the engine-side streamer drives.  It keeps the
// project's model_*_config / support-header idiom but is its OWN file now, so tool_call_support
// holds only the tool-call registry / detect / select / bridge.  This header DECLARES, and the
// matching streaming_support.cpp DEFINES:
//   * ContentStreamer -- a TextParserStreamer subclass that carries an IncrementalParser into the
//     engine (the OpenArc Qwen35ToolCallStreamer analog): the base keeps the parser loop / content
//     accumulation; the carried parser routes the tool block out into the tool_calls array.
//   * DynamicReasoningIncrementalParser -- a ReasoningIncrementalParser with a runtime on/off
//     (setActive) split; plus the kThinkingOpen / kThinkingClose reasoning tags (built from the
//     qwen35 header's generic LT / GT angle-byte constants -- no literal angle byte here).
// They stay in the SAME ovgenai::toolcall namespace as the registry (tool_call_support.hpp, which
// #includes this header so composeParserList can run the slot-0 reasoner and the per-model parsers).
#pragma once

#include <openvino/genai/parsers.hpp>      // IncrementalParser / ReasoningIncrementalParser / StreamingStatus
#include <openvino/genai/text_streamer.hpp> // TextParserStreamer (ContentStreamer's base)
#include <openvino/genai/tokenizer.hpp>      // Tokenizer (ContentStreamer ctor arg)
#include <openvino/genai/json_container.hpp> // JsonContainer (the parse / write payloads)

#include "toolcallparsers/qwen35_tool_call_parser.hpp"  // LT/GT angle bytes + Qwen35ToolCallParser (the reasoner's tag source)

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
// write base drives the parser loop and calls it.  (Method bodies in streaming_support.cpp.)
class ContentStreamer : public ov::genai::TextParserStreamer {
public:
  ContentStreamer(const ov::genai::Tokenizer& tok,
                  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                  std::string& acc);
  ov::genai::StreamingStatus write(ov::genai::JsonContainer& message) override;
private:
  std::string& m_acc;                        // the output the caller drains (specials preserved)
};

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
}  // namespace toolcall
}  // namespace ovgenai

