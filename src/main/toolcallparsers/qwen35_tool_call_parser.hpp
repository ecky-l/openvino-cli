// Qwen3.5 tool-call parser (incremental).  Extracted from tool_call_support.hpp into its
// own translation unit so a future <model>_tool_call_parser can sit beside it.  Like the
// toolkit's ReasoningIncrementalParser it accumulates decoded text and only acts on a
// COMPLETE close tag, so a call split across streaming chunks is still detected; on a
// completed call it flips StreamingStatus::TOOL_CALL_STOP so the engine stops generation
// and the CLI reports a uniform finish_reason "tool_calls".
//
// Model-native tag set: every tag is a plain-ASCII angle (the open byte U+003C, the close
// byte U+003E), with an optional name / =value inside the open tags -- a
// ‹call-open› … ‹call-close› frame, an inner ‹function=NAME› … ‹/function› pair and, per
// argument, one ‹parameter=NAME› … ‹/parameter› triple.  The model's own chat template
// also bakes the model-intrinsic "example_function_name" / "example_parameter" markers
// (independent of the tools the user passes); those drive the auto-detect gate which lives
// in tool_call_support.hpp.
//
// The tag BYTES are built from the \u003C / \u003E escape constants below (so a literal
// angle-bracket pair never sits in this source -- the project guardrail: the environment
// would otherwise re-run raw tool tags).  The live model output carries a single NEWLINE
// between every one of its tags (the model emits one after each close tag); the helpers
// strip the per-tag whitespace (tr()) and locate the contiguous tag literals with
// std::string::find, so those inter-tag newlines are ignored.
#pragma once

#include <openvino/genai/parsers.hpp>        // IncrementalParser / StreamingStatus
#include <openvino/genai/json_container.hpp> // JsonContainer

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ovgenai {
namespace toolcall {

// --- tag literals (built from the \u003C / \u003E escapes; NO literal bracket pair) ----
inline const std::string LT  = "\u003C";        // the open  byte  (U+003C)
inline const std::string GT  = "\u003E";        // the close byte (U+003E)
inline const std::string kTCOpen    = LT + "tool_call"  + GT;   // ‹tool_call›
inline const std::string kTCClose   = LT + "/tool_call" + GT;   // ‹/tool_call›
inline const std::string kFuncOpen  = LT + "function=";          // ‹function=NAME
inline const std::string kFuncClose = LT + "/function" + GT;      // ‹/function›
inline const std::string kParamOpen = LT + "parameter=";         // ‹parameter=P
inline const std::string kParamClose = LT + "/parameter" + GT;    // ‹/parameter›
inline const std::string kSentfunc  = "example_function_name";    // model-intrinsic detect sentinel
inline const std::string kSentry    = "example_parameter";        // model-intrinsic detect sentinel

// Incremental parser for the Qwen3.5 ‹call-open› … ‹call-close› block.  Its parse()/reset()
// bodies (plus the tr / argValue / qwenBlockToCall helpers they use) live in the .cpp.
class Qwen35ToolCallParser : public ov::genai::IncrementalParser {
public:
  std::string parse(ov::genai::JsonContainer& delta_message,
                    std::string& delta_text,
                    const std::optional<std::vector<int64_t>>& delta_tokens) override;
  void reset() override;

private:
  std::string m_text;                                          // everything decoded since the last reset
  std::size_t m_consumed = 0;                                // chars of m_text already routed out of content
  std::vector<ov::genai::JsonContainer> m_calls;             // the completed {name, arguments} calls
};

}  // namespace toolcall
}  // namespace ovgenai

