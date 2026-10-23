// Qwen3.5 tool-call parser (incremental) -- free-helper and method bodies for the
// Qwen35ToolCallParser declared in the matching header.  The helpers are defined (in
// order) here so the parser's own parse() can call them; the header only declares them.
#include "qwen35_tool_call_parser.hpp"

namespace ovgenai {
namespace toolcall {

// Trim the Qwen whitespace the model emits around a tag's value.
std::string tr(const std::string& s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// A parameter value is usually a bare word/number; parse it as JSON to get a typed value,
// else fall back to the raw (trimmed) string.  Never throws.
ov::genai::JsonContainer argValue(const std::string& raw) {
  const std::string v = tr(raw);
  if (v.empty())
    return ov::genai::JsonContainer(std::string());
  try {
    return ov::genai::JsonContainer::from_json_string(v);
  } catch (...) {
    return ov::genai::JsonContainer(v);
  }
}

// Parse one ‹function=NAME› … ‹/function› block's inner body into a
// {"name":…, "arguments":{ … }} entry (one ‹parameter=P›value‹/parameter› per argument).
// The body passed in is the raw text BETWEEN the outer ‹tool_call› and ‹/tool_call›.
ov::genai::JsonContainer qwenBlockToCall(const std::string& body) {
  const std::size_t fopen = body.find(kFuncOpen);
  if (fopen == std::string::npos)
    return ov::genai::JsonContainer();   // no function -> not a parseable call
  const std::size_t fe = body.find(GT, fopen + kFuncOpen.size());
  if (fe == std::string::npos)
    return ov::genai::JsonContainer();
  const std::string name =
      tr(body.substr(fopen + kFuncOpen.size(), fe - (fopen + kFuncOpen.size())));
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
    const std::string pname =
        tr(fbody.substr(popen + kParamOpen.size(), pe - (popen + kParamOpen.size())));
    const std::size_t pc = fbody.find(kParamClose, pe + 1);   // 1 = past the '>' ending the name
    if (pc == std::string::npos)
      break;
    const std::string val = fbody.substr(pe + 1, pc - (pe + 1));
    args[pname] = argValue(val);
    pos = pc + kParamClose.size();
  }
  return ov::genai::JsonContainer({{"name", name}, {arg, args}});
}

std::string Qwen35ToolCallParser::parse(ov::genai::JsonContainer& delta_message,
                            std::string& delta_text,
                            const std::optional<std::vector<int64_t>>& /*delta_tokens*/) {
  const std::size_t calls_before = m_calls.size();
  m_text += delta_text;

  // Each complete ‹call-close› at/after the cursor is paired to the nearest preceding
  // ‹call-open›; the body is parsed into a recorded call and the cursor advances.
  while (true) {
    const std::size_t close = m_text.find(kTCClose, m_consumed);
    if (close == std::string::npos)
      break;
    const std::size_t open = m_text.rfind(kTCOpen, close);   // nearest open at/before the close
    if (open == std::string::npos || open < m_consumed) {
      m_consumed = close + kTCClose.size();                   // unpaired close: skip it
      continue;
    }
    const std::size_t b0 = open + kTCOpen.size();
    if (b0 > close) { m_consumed = close + kTCClose.size(); continue; }
    m_calls.push_back(qwenBlockToCall(m_text.substr(b0, close - b0)));
    m_consumed = close + kTCClose.size();
  }

  // FINALIZE A TRUNCATED BLOCK (the small-model case).  A small model can under-rand one
  // token short of the OUTER ‹call-close› (its |im_end| lands first): it has opened
  // ‹call-open› and fully typed the INNER ‹function=NAME› … ‹/function› but never the outer
  // close.  There is no end()/finalize hook (the engine consumes |im_end| as the stop token
  // and hands us no further parse() calls), so we detect a finished call HERE: an open block
  // whose inner function is CLOSED, with no 2nd function in flight.  Then forward tool_calls
  // + flip TOOL_CALL_STOP; consuming to end-of-buffer swallows the dangling outer-open start
  // so it is not re-exposed in the result as stray angle tags.
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
            // defer if a FURTHER function follows (a multi-call turn): never truncate it
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

  // Emit only the finalized, non-tool text: up to the next not-yet-closed ‹call-open›
  // (an in-progress block is held until its close arrives and is parsed out).
  std::size_t emit_to;
  const std::size_t pending = m_text.find(kTCOpen, m_consumed);
  emit_to = (pending == std::string::npos) ? m_text.size() : pending;
  std::string out = m_text.substr(m_consumed, emit_to - m_consumed);
  m_consumed = emit_to;

  if (m_calls.size() != calls_before) {
    // A call completed this step: forward the accumulated structured calls so the engine
    // copies them into res.parsed[0]["tool_calls"], then stop (the CLI maps TOOL_CALL_STOP
    // to the uniform finish_reason "tool_calls").
    ov::genai::JsonContainer arr = ov::genai::JsonContainer::array();
    for (const auto& c : m_calls)
      arr.push_back(c);
    delta_message["tool_calls"] = arr;
    set_status(ov::genai::StreamingStatus::TOOL_CALL_STOP);
  }
  return out;   // the engine then sets delta_message["content"] = this
}

void Qwen35ToolCallParser::reset() {
  m_text.clear();
  m_consumed = 0;
  m_calls.clear();
  set_status(ov::genai::StreamingStatus::RUNNING);
}

}  // namespace toolcall
}  // namespace ovgenai


