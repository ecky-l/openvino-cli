// Tool-call parsing for the `vlm` subcommand -- the CLI-side of (b).
//
// DERIVED FROM THE MODEL (future-proof): a model's own chat template is the source of
// truth for HOW it states a tool call, so we (1) DETECT the model's format by rendering
// its loaded template through the public tokenizer API and matching a model-intrinsic
// sentinel, then (2) run the matching IncrementalParser.  For Qwen3.5 that format is a
// nested tag block; every tag is a plain-ASCII angle (open byte U+003C, close byte U+003E):
//   outer  ‹tool_call› ... ‹/tool_call›
//   inner  ‹function=NAME› ... ‹/function›
//   args   ‹parameter=P›value‹/parameter›
// The template also bakes a model-intrinsic example that uses the literal markers
// `example_function_name` / `example_parameter`; those (rendered into the prompt the user
// does NOT control) are the sentinel we detect on (independent of the tools the user passes).
//
// The raw tag bytes are plain ASCII, but to keep a literal bracket pair out of this source
// (so a tooling scanner does not re-run it), they are built from the \u003C / \u003E escapes;
// the inert ‹…› marker is used only in prose.
//
// Consult-only the OpenVINO.GenAI toolkit: this only USES its public classes
// (IncrementalParser / TextParserStreamer / JsonContainer / Tokenizer); it is not modified.
#pragma once

#include <openvino/genai/parsers.hpp>          // IncrementalParser / StreamingStatus
#include <openvino/genai/text_streamer.hpp>     // TextParserStreamer (pure-virtual write(JsonContainer&))
#include <openvino/genai/tokenizer.hpp>         // Tokenizer + apply_chat_template / get_original_chat_template
#include <openvino/genai/chat_history.hpp>      // ChatHistory
#include <openvino/genai/generation_config.hpp>  // (for downstream; not used here directly)
#include <openvino/genai/json_container.hpp>     // JsonContainer

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ovgenai {
namespace toolcall {

// --- tag literals (built from escapes; NO literal bracket pair in source) ---------
inline const std::string LT = "\u003C";   // '<'  (U+003C)
inline const std::string GT = "\u003E";   // '>'  (U+003E)
inline const std::string kTCOpen   = LT + "tool_call"    + GT;  // ‹tool_call ›
inline const std::string kTCClose  = LT + "/tool_call"   + GT;  // ‹/tool_call ›
inline const std::string kFuncOpen  = LT + "function=";         // ‹function=NAME
inline const std::string kFuncClose = LT + "/function"    + GT; // ‹/function ›
inline const std::string kParamOpen = LT + "parameter=";        // ‹parameter=P
inline const std::string kParamClose = LT + "/parameter"  + GT; // ‹/parameter ›
inline const std::string kSentfunc = "example_function_name";   // model-intrinsic detect sentinel
inline const std::string kSentry = "example_parameter";         // model-intrinsic detect sentinel

// Trim the Qwen whitespace the model emits around a tag's value.
inline std::string tr(const std::string& s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// A `<value>` is usually a bare word/number; keep the smart case (parse as JSON -> typed),
// fall back to the raw string.  Never throws.
inline ov::genai::JsonContainer argValue(const std::string& raw) {
  const std::string v = tr(raw);
  if (v.empty())
    return ov::genai::JsonContainer(std::string());
  try {
    return ov::genai::JsonContainer::from_json_string(v);
  } catch (...) {
    return ov::genai::JsonContainer(v);
  }
}

// Parse one `‹function=NAME› … ‹/function›` block's inner body into the
// `{"name":…, "arguments":{ … }}` entry (one `‹parameter=P›val‹/parameter›` per arg).
// The body passed here is the text BETWEEN the outer `‹tool_call›`/`‹/tool_call›`.
inline ov::genai::JsonContainer qwenBlockToCall(const std::string& body) {
  const std::size_t fopen = body.find(kFuncOpen);
  if (fopen == std::string::npos)
    return ov::genai::JsonContainer();   // no function -> not a parseable call
  const std::size_t fe = body.find(GT, fopen + kFuncOpen.size());
  if (fe == std::string::npos)
    return ov::genai::JsonContainer();
  const std::string name = tr(body.substr(fopen + kFuncOpen.size(), fe - (fopen + kFuncOpen.size())));
  const std::size_t fc = body.find(kFuncClose, fe + 1);
  if (fc == std::string::npos || (fc < fe + 1))
    return ov::genai::JsonContainer();
  const std::string fbody = body.substr(fe + 1, fc - (fe + 1));

  const std::string arg = "arguments";
  ov::genai::JsonContainer args = ov::genai::JsonContainer::object();
  std::size_t pos = 0;
  while (true) {
    const std::size_t popen = fbody.find(kParamOpen, pos);
    if (popen == std::string::npos)
      break;
    const std::size_t pe = fbody.find(GT, popen + kParamOpen.size());
    if (pe == std::string::npos)
      break;
    const std::string pname = tr(fbody.substr(popen + kParamOpen.size(), pe - (popen + kParamOpen.size())));
    const std::size_t pc = fbody.find(kParamClose, pe + 1);   // 1 = past '>' that ended name
    if (pc == std::string::npos)
      break;
    const std::string val = fbody.substr(pe + 1, pc - (pe + 1));
    args[pname] = argValue(val);
    pos = pc + kParamClose.size();
  }
  return ov::genai::JsonContainer({{"name", name}, {arg, args}});
}

// Incremental parser for the Qwen3.5 `‹tool_call› … ‹/tool_call›` block.  Like the
// toolkit's ReasoningIncrementalParser it is incremental-and-split-across-chunks-safe
// (it accumulates text, and only acts on a COMPLETE close tag); unlike it it drives the
// OUTPUT parser path and on a completed call it flips StreamingStatus::TOOL_CALL_STOP so
// the engine stops and the CLI reports finish_reason = tool_calls.
class QwenXmlToolCallIncrementalParser : public ov::genai::IncrementalParser {
public:
  std::string parse(ov::genai::JsonContainer& delta_message,
                    std::string& delta_text,
                    const std::optional<std::vector<int64_t>>& /*delta_tokens*/) override {
    const std::size_t calls_before = m_calls.size();
    m_text += delta_text;

    // For each complete ‹/tool_call › at/after the consumed cursor, pair it back to the
    // nearest preceding ‹tool_call ›, parse the body between, record the call, advance.
    while (true) {
      const std::size_t close = m_text.find(kTCClose, m_consumed);
      if (close == std::string::npos)
        break;
      const std::size_t open = m_text.rfind(kTCOpen, close);   // nearest open at/before close
      if (open == std::string::npos || open < m_consumed) {
        m_consumed = close + kTCClose.size();                   // unpaired close: skip
        continue;
      }
      const std::size_t b0 = open + kTCOpen.size();
      if (b0 > close) { m_consumed = close + kTCClose.size(); continue; }
      m_calls.push_back(qwenBlockToCall(m_text.substr(b0, close - b0)));
      m_consumed = close + kTCClose.size();
    }

    // FINALIZE A TRUNCATED BLOCK -- the case this round fixes. A small model under-rands one
    // token short of the OUTER ‹/tool_call› (its |im_end| lands first): it has OPENED
    // ‹tool_call› and fully typed the INNER ‹function=NAME› ... ‹/function› but never the outer
    // close. The IncrementalParser has no end()/finalize hook (the engine consumes |im_end| as a
    // stop token and hands us no more parse() calls), so we detect "the call is done" HERE: an
    // open block whose inner function is complete, with no 2nd function in flight. Deferring on
    // a later function avoids truncating a multi-call turn -- still the documented "stop at the
    // first completed call". The "call completed" block below then forwards tool_calls + flips
    // TOOL_CALL_STOP; consuming to end-of-buffer swallows the dangling outer-open start so it is
    // not re-exposed in the result as stray angle tags.
    if (m_calls.size() == calls_before) {
      const std::size_t pendingOpen = m_text.find(kTCOpen, m_consumed);
      if (pendingOpen != std::string::npos) {
        const std::size_t funcOpen = m_text.find(kFuncOpen, pendingOpen + kTCOpen.size());
        if (funcOpen != std::string::npos) {
          const std::size_t fe = m_text.find(GT, funcOpen + kFuncOpen.size());
          if (fe != std::string::npos) {
            // require the inner function to be actually CLOSED, not merely opened
            const std::size_t fc = m_text.find(kFuncClose, fe + 1);
            if (fc != std::string::npos) {
              const std::size_t innerCloseEnd = fc + kFuncClose.size();
              // defer if a FURTHER function follows this one (a multi-call turn): never truncate it
              const std::size_t nextFunc = m_text.find(kFuncOpen, innerCloseEnd);
              if (nextFunc == std::string::npos) {
                const std::string body = m_text.substr(
                    pendingOpen + kTCOpen.size(),
                    innerCloseEnd - (pendingOpen + kTCOpen.size()));
                const auto call = qwenBlockToCall(body);
                if (!call.is_null()) {
                  m_calls.push_back(call);
                  m_consumed = m_text.size();
                }
              }
            }
          }
        }
      }
    }

    // Emit only the finalized, non-tool text: up to the next not-yet-closed ‹tool_call ›
    // (otherwise the in-progress block is held until its close arrives and is parsed out).
    std::size_t emit_to;
    const std::size_t pending = m_text.find(kTCOpen, m_consumed);
    emit_to = (pending == std::string::npos) ? m_text.size() : pending;
    std::string out = m_text.substr(m_consumed, emit_to - m_consumed);
    m_consumed = emit_to;

    if (m_calls.size() != calls_before) {
      // A call completed this step: forward the accumulated structured calls so the
      // engine copies them into the result's `.parsed` (res.parsed[0]["tool_calls"]),
      // then stop generation (finish reason becomes tool_calls via the engine mapping).
      ov::genai::JsonContainer arr = ov::genai::JsonContainer::array();
      for (const auto& c : m_calls)
        arr.push_back(c);
      delta_message["tool_calls"] = arr;
      set_status(ov::genai::StreamingStatus::TOOL_CALL_STOP);
    }
    return out;   // the engine sets delta_message["content"] = this
  }

  void reset() override {
    m_text.clear();
    m_consumed = 0;
    m_calls.clear();
    set_status(ov::genai::StreamingStatus::RUNNING);
  }

private:
  std::string m_text;                      // everything decoded since the last reset
  std::size_t m_consumed = 0;              // chars of m_text already routed out of "content"
  std::vector<ov::genai::JsonContainer> m_calls;   // completed {name, arguments} calls
};

// A TextParserStreamer subclass that carries an IncrementalParser (so the engine runs
// the parser loop AND copies its get_parsed_message() into res.parsed[0]) while the
// streamer appends the (specials-preserving) content to the caller's accumulated output.
// The pure-virtual write(JsonContainer&) is the one that fires; the string-write base
// drives the parser loop and calls it.
class ContentStreamer : public ov::genai::TextParserStreamer {
public:
  ContentStreamer(const ov::genai::Tokenizer& tok,
                  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                  std::string& acc)
    : ov::genai::TextParserStreamer(tok, std::move(parsers)), m_acc(acc) {}

  ov::genai::StreamingStatus write(ov::genai::JsonContainer& message) override {
    m_acc += message["content"].as_string().value_or(std::string());   // -> result
    return ov::genai::StreamingStatus::RUNNING;
  }

private:
  std::string& m_acc;
};

// --- the registry + detection / selection ----------------------------------------
// One source of truth: every advertised (validated) name is an entry; an entry is
// "wired" iff its `make` is non-null.  `sentinels` are the substrings whose presence in
// the model's RENDERED template selects this parser (empty = not auto-detected yet).
struct ToolCallFormatSpec {
  std::string name;
  std::vector<std::string> sentinels;
  std::function<std::shared_ptr<ov::genai::IncrementalParser>()> make;
};

inline const std::vector<ToolCallFormatSpec>& knownToolCallFormats() {
  static const std::vector<ToolCallFormatSpec> kSpecs = {
    { "qwen35", { kSentfunc, kSentry },
      []() { return std::static_pointer_cast<ov::genai::IncrementalParser>(
                     std::make_shared<QwenXmlToolCallIncrementalParser>()); } },
    { "hermes",      { }, nullptr },
    { "gemma4",      { }, nullptr },
    { "museglimmer", { }, nullptr },
  };
  return kSpecs;
}

inline std::vector<std::string> advertisedToolCallParsers() {
  std::vector<std::string> v;
  for (const auto& s : knownToolCallFormats())
    v.push_back(s.name);
  return v;
}

// Build a parser for a (validated) name, or nullptr if the name is advertised-but-unwired.
inline std::shared_ptr<ov::genai::IncrementalParser> makeToolCallParserByName(const std::string& name) {
  for (const auto& s : knownToolCallFormats())
    if (s.name == name && s.make)
      return s.make();
  return nullptr;
}

// DETECT the format from the model's own loaded template: render it (via the public
// API) with a probe tool arg -- the format instruction only materialises when tools are
// present -- then match the rendered text against the registry sentinels.  Returns the
// name of the matching wired format, or nullopt if none match (graceful).
inline std::optional<std::string> detectToolCallFormat(const ov::genai::Tokenizer& tok) {
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

// matchFormat -- the MODEL-FREE core of the auto-detect gate, split out adjacent to
// detectToolCallFormat (which renders the model's own loaded template and then calls it)
// so the gate is unit-testable WITHOUT a model: the registry entry whose `sentinels` are
// ALL present in `rendered` wins; nullopt if none match (graceful: no parser, no stop).
inline std::optional<std::string> matchFormat(const std::string& rendered) {
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

// Selection precedence: an explicit --tool-call-parser <name> (override) wins; else
// auto-detect from the model's loaded template; else no parser (graceful: no stop,
// finish_reason stays stop, no tool_calls -- exactly today's behaviour).
inline std::optional<std::string> selectToolCallParserName(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out = nullptr) {
  if (override_name.has_value() && !override_name->empty())
    return *override_name;
  auto d = detectToolCallFormat(tok);
  if (detected_out)
    *detected_out = d;
  return d;
}

}  // namespace toolcall
}  // namespace ovgenai
