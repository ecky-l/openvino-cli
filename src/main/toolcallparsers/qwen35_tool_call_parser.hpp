// Qwen3.5 tool-call parser -- a faithful C++ port of OpenArc QwenXMLToolParser
// (../OpenArc/src/engine/ov_genai/tool_parse/qwen35.py): a TOKEN-ID-driven state machine.
// The carrier is ContentStreamer (tool_call_support.cpp) = the OpenArc Qwen35ToolCallStreamer analog.
//
// NO reasoning splitter: per the project's direction, reasoning output goes to the normal
// output (content) exactly as before -- ONLY the tool block is parsed out into the tool_calls
// array.  The close-of-thinking lead-in merely flows through as content (the carrier decodes
// with skip_special_tokens=false).  The parser still keeps a full state machine and COLLECTS
// the tool-call fragments (the function name + the argument values, accumulated across whatever
// deltas) plus the other content, then EMITs the consolidated tool_calls array.
//
// Model shape: an outer call-open ... call-close frame, an inner function-open=NAME ...
// function-close pair, and per argument a parameter-open=P ... parameter-close triple.  The
// outer frame tags are single, special=False tokens (id 248058 / 248059 -- a one-token tag in
// this vocab); the inner tags are multi-token ordinary text.  So the BOUNDARY is decided by
// token ID (the delta_tokens the engine hands us -- the raw token ids of the decoded delta),
// and the inner tags are matched at the text level inside the block.  Text outside a complete
// block -> content; the block -> the tool_calls array.
//
// Bracket-safety guardrail: NO literal angle-bracket tag pair in any comment; in CODE build
// every tag byte from the LT/U003C-GT constants below (a single-backslash escape, so the file
// keeps the canonical escape and my output carries no raw angle byte); in prose use the words
// open-tag / close-tag / function-open-tag / parameter-open-tag (the env would otherwise re-run
// raw tool tags, and a raw angle byte scrambles the serving stack).
#pragma once

#include <openvino/genai/parsers.hpp>        // IncrementalParser / StreamingStatus
#include <openvino/genai/json_container.hpp> // JsonContainer

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ovgenai {
namespace toolcall {

// --- tag literals (built from the \u003C / \u003E escapes; NO literal bracket pair in source) --
inline const std::string LT  = "\u003C";        // the open  byte  (U+003C)
inline const std::string GT  = "\u003E";        // the close byte (U+003E)
inline const std::string kTCOpen    = LT + "tool_call"  + GT;   // call-open tag  (single token)
inline const std::string kTCClose   = LT + "/tool_call" + GT;   // call-close tag (single token)
inline const std::string kFuncOpen  = LT + "function=";          // function-open tag (multi-token)
inline const std::string kFuncClose = LT + "/function" + GT;      // function-close tag
inline const std::string kParamOpen = LT + "parameter=";          // parameter-open tag (multi-token)
inline const std::string kParamClose = LT + "/parameter" + GT;    // parameter-close tag
inline const std::string kSentfunc   = "example_function_name";   // model-intrinsic detect sentinel
inline const std::string kSentry    = "example_parameter";        // model-intrinsic detect sentinel

// Boundary tag token IDs (validated identical across the Qwen3.5 exports; probe-confirmed 1 token
// each): a single special=False token encodes the open-tag (call-open) and another the close-tag
// (call-close).  They always arrive in delta_tokens even under skip_special_tokens.
inline const std::int64_t kToolOpenId  = 248058;   // the open-tag  (call-open)   token id
inline const std::int64_t kToolCloseId = 248059;   // the close-tag (call-close) token id

// Boundary tag (text, id) pairs -- the parser's boundary table (prose: open-tag / close-tag).
struct QBoundary { const std::string* tag; std::int64_t id; };
inline const std::vector<QBoundary> kQBoundaries = {
  { &kTCOpen,  kToolOpenId  },
  { &kTCClose, kToolCloseId },
};

// One (decoded-text, raw-token-ids) event, as fed to parse() (the engine) or synthesized for text.
using TokChunk = std::pair<std::string, std::optional<std::vector<std::int64_t>>>;

// --- the parser = faithful QwenXMLToolParser (no reasoning split; just the tool-call state) -------
class Qwen35ToolCallParser : public ov::genai::IncrementalParser {
public:
  // parse() (the IncrementalParser contract): take a decoded delta -- the text PLUS its raw token
  // ids (delta_tokens).  INVESTIGATE the token ids to detect a boundary (the open-tag id 248058 /
  // the close-tag id 248059), advance the state machine, return the content (everything except the
  // tool block; reasoning + inter-call text flow through).  On a COMPLETED call, consolidate the
  // fragments and write the full tool_calls ARRAY into delta_message (once per turn -- the caller
  // then stops, so a later delta never re-writes it, which keeps concatenate from throwing), and
  // flip TOOL_CALL_STOP so the engine stops and the CLI reports the uniform finish_reason "tool_calls".
  std::string parse(ov::genai::JsonContainer& delta_message,
                    std::string& delta_text,
                    const std::optional<std::vector<std::int64_t>>& delta_tokens) override;
  void reset() override;

  // At end of a turn: force-close an unterminated call.  The engine consumes the stop token and
  // gives us no further parse(), so there is no end()/finalize hook to hang this on -- the small
  // 0.8B under-runs one token short of the outer close-tag, and we detect the finished call in
  // parse() instead (see finishTruncated()).  Exposed so the model-free / offline path can call it.
  void finalize() { finishTruncated(); }

  // Every completed call so far, in completion order -- the consolidated tool_calls list.
  const std::vector<ov::genai::JsonContainer>& completed_calls() const { return m_calls; }

private:
  // The state machine (OUTSIDE / HEAD / FUNC_NAME / BODY / PARAM_NAME / PARAM / TAIL).  `atEnd`
  // means the buffer ends at a boundary / stream end, so no tag may be held back waiting for more.  ('final' is a C++ keyword, so named atEnd.)
  enum class State { OUTSIDE, HEAD, FUNC_NAME, BODY, PARAM_NAME, PARAM, TAIL };

  void machine(std::vector<std::string>& out, bool atEnd);
  void applyBoundary(std::int64_t tag_id);                 // a boundary just fired (prose: open/close)
  void startCall();                                       // open a new call (index == calls already done)
  void finishCall();                                      // close the current call into m_calls (no delta write)
  void consumeValue(const std::string& text, bool atEnd);  // a parameter-value delta
  void finishTruncated();                                 // the in-parse truncated-finalize
  void emitToolCalls(ov::genai::JsonContainer& msg);      // write the consolidated tool_calls ARRAY

  // The machine working buffer (mirror of Python self._pending).
  std::string  m_pending;
  State        m_state = State::OUTSIDE;
  bool         m_skip_rest = false;
  std::optional<ov::genai::JsonContainer> m_cur;   // the call under construction (carries id/name)
  std::string  m_id;                                // id chosen at startCall (cosmetic)
  std::size_t  m_cur_index = 0;                    // index of the current call (== calls completed)
  std::string  m_name;                             // current function name (collected fragment)
  std::string  m_arguments_buf;                    // running arguments-JSON (collected fragments)
  std::string  m_param_name;
  std::string  m_param_raw;
  std::size_t  m_param_index = 0;
  bool         m_value_started = false;
  bool         m_pending_newline = false;
  bool         m_stopped = false;
  std::vector<ov::genai::JsonContainer> m_calls;   // completed calls (== the tool_calls list)
  std::vector<std::string> m_errors;
};

// --- model-free / offline text helper = the OpenArc _TextTagSynthesizer analog (no reasoning
// split): split `text` into (chunk, ids) events so every COMPLETE boundary tag arrives in its own
// chunk carrying its token id, and no chunk ends on a partial boundary tag (that would wait for a
// never-coming token).  Lets the unit tests drive the real token-ID state machine with nothing but
// text; the live engine path feeds real ids directly.  A trailing held-back tail is released at
// stream end by flush().
class Qwen35BoundarySynthesizer {
public:
  std::vector<TokChunk> feed(const std::string& text);
  std::optional<std::vector<TokChunk>> flush();   // release a held tail + the buffer
  void reset();
private:
  std::string m_buf;
};

}  // namespace toolcall
}  // namespace ovgenai
