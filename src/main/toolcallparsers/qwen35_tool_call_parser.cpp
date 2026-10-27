// Qwen3.5 tool-call parser (the IncrementalParser body) -- a faithful C++ port of OpenArc's
// QwenXMLToolParser (qwen35.py): a TOKEN-ID-driven state machine.  See the header for the model
// (boundary by token id; inner tags by text; reasoning is NOT split, so it flows through as
// content) and the bracket-safety guardrail.  Only the member bodies + the file-local helpers
// live here so the header stays a clean interface.
#include "qwen35_tool_call_parser.hpp"

#include <openvino/genai/parsers.hpp>        // StreamingStatus (RUNNING / TOOL_CALL_STOP)

#include <algorithm>     // std::min_element
#include <cstdio>         // std::snprintf

namespace ovgenai {
namespace toolcall {

namespace {

std::string q35_tr(const std::string& s) {                  // trim the model's inter-tag whitespace
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Escape a raw string fragment for embedding inside a JSON string (so the concatenated argument
// fragments form valid JSON) -- mirrors QwenXMLToolParser._json_escape_fragment.
std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char ch : s) {
    const int c = static_cast<unsigned char>(ch);
    if (ch == '"')                         out += "\\\"";
    else if (ch == '\\')                   out += "\\\\";
    else if (ch == '\n')                   out += "\\n";
    else if (ch == '\t')                   out += "\\t";
    else if (ch == '\r')                   out += "\\r";
    else if (c < 0x20)  { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
    else                            out.push_back(ch);
  }
  return out;
}

// True if `text` is a strict prefix of one of the (multi-token) inner tags, so a delta ending
// mid-tag must not read as complete -- QwenXMLToolParser._is_partial.
bool isPartial(const std::string& text, const std::vector<std::string>& tags) {
  for (const auto& t : tags)
    if (t != text && t.rfind(text, 0) == 0) return true;
  return false;
}

// Length of the longest suffix of `text` that is a strict prefix of one of the tags (that many
// bytes must wait for the next delta) -- QwenXMLToolParser._holdback_suffix.
std::size_t holdbackSuffix(const std::string& text, const std::vector<std::string>& tags) {
  if (text.empty() || tags.empty()) return 0;
  std::size_t maxLen = 0;
  for (const auto& t : tags) if (t.size() > maxLen) maxLen = t.size();
  if (maxLen == 0) return 0;
  const std::size_t limit = std::min(text.size(), maxLen - 1);
  std::size_t maxHold = 0;
  for (std::size_t n = 1; n <= limit; ++n) {
    const std::string suffix = text.substr(text.size() - n);
    for (const auto& t : tags)
      if (t != suffix && t.rfind(suffix, 0) == 0) { maxHold = n; break; }
  }
  return maxHold;
}

}  // namespace

// ---- Qwen35ToolCallParser -------------------------------------------------------------

std::string Qwen35ToolCallParser::parse(
    ov::genai::JsonContainer& delta_message, std::string& delta_text,
    const std::optional<std::vector<std::int64_t>>& delta_tokens) {
  if (m_stopped) return std::string();
  const std::size_t completed_before = m_calls.size();
  m_pending += delta_text;

  // INVESTIGATE the token ids: a boundary tag fires only when its id (the open-tag 248058 or the
  // close-tag 248059) is in the delta.  The two are single special=False tokens, so their ids
  // survive even under skip_special_tokens; the engine hands us the raw id list.
  std::map<std::int64_t, int> remaining;
  if (delta_tokens)
    for (const std::int64_t t : *delta_tokens)
      for (const QBoundary& b : kQBoundaries)
        if (b.id == t) ++remaining[t];

  std::vector<std::string> out;   // content fragments for this delta (joined and returned)
  while (true) {
    // The earliest boundary event this delta: a boundary tag whose id is still available AND whose
    // tag text is present in the pending buffer.
    std::vector<std::pair<std::size_t, QBoundary>> events;
    for (const QBoundary& b : kQBoundaries)
      if (remaining[b.id] > 0) {
        const auto pos = m_pending.find(*b.tag);
        if (pos != std::string::npos) events.emplace_back(pos, b);
      }
    if (events.empty()) { machine(out, false); break; }
    auto& ev = *std::min_element(events.begin(), events.end(),
        [](const auto& x, const auto& y) { return x.first < y.first; });
    const std::string segment   = m_pending.substr(0, ev.first);           // text before the tag
    const std::string remainder = m_pending.substr(ev.first + ev.second.tag->size());
    // The machine must see exactly the text before the boundary; atEnd guarantees it drains it.
    m_pending = segment;
    if (!segment.empty()) machine(out, true);
    m_pending = remainder;
    --remaining[ev.second.id];
    applyBoundary(ev.second.id);
    if (m_stopped) { m_pending.clear(); break; }   // stop_after_tool_call (unused here today)
  }

  // The engine consumes the stop token and gives us no further parse(), so detect a finished-but-
  // unterminated call HERE (the small-model under-run): the inner function is closed but the
  // outer close-tag never arrived.  finishTruncated() appends to m_calls, so the grow-check below
  // emits it as a (single-element) tool_calls array.
  finishTruncated();

  // A call completed this step -> write the consolidated tool_calls ARRAY (only this once: the
  // caller then stops, so a later delta never re-writes the key, which keeps concatenate happy),
  // and flip the status so the engine halts and the CLI reports the uniform "tool_calls".
  if (m_calls.size() > completed_before) {
    emitToolCalls(delta_message);
    set_status(ov::genai::StreamingStatus::TOOL_CALL_STOP);
    m_stopped = true;
  }

  std::string content;
  for (auto& s : out) content += s;
  return content;
}

void Qwen35ToolCallParser::reset() {
  m_pending.clear();
  m_state = State::OUTSIDE;
  m_skip_rest = false;
  m_cur = std::nullopt;
  m_id.clear();
  m_cur_index = 0;
  m_name.clear();
  m_arguments_buf.clear();
  m_param_name.clear();
  m_param_raw.clear();
  m_param_index = 0;
  m_value_started = false;
  m_pending_newline = false;
  m_stopped = false;
  m_calls.clear();
  m_errors.clear();
  set_status(ov::genai::StreamingStatus::RUNNING);
}

// The state machine (QwenXMLToolParser._machine).  atEnd means the buffer ends at a boundary or at
// stream end, so no tag may be held back waiting for a token that will never come.
void Qwen35ToolCallParser::machine(std::vector<std::string>& out, bool atEnd) {
  while (true) {
    if (m_skip_rest) { m_pending.clear(); return; }

    if (m_state == State::OUTSIDE) {
      if (m_pending.empty()) return;
      out.push_back(m_pending);       // everything outside a block is content (reasoning lead-in...)
      m_pending.clear();
      return;
    }
    if (m_state == State::TAIL) {
      const std::string t = q35_tr(m_pending);
      if (!t.empty()) m_errors.push_back("unexpected text after the function close: " + t.substr(0, 20));
      m_pending.clear();
      return;
    }
    if (m_state == State::HEAD) {
      std::string s = q35_tr(m_pending);
      m_pending = s;
      if (s.rfind(kFuncOpen, 0) == 0) { m_pending = s.substr(kFuncOpen.size()); m_state = State::FUNC_NAME; continue; }
      if (s.empty()) return;
      if (!atEnd && isPartial(s, std::vector<std::string>{ kFuncOpen })) return;    // wait for more
      m_errors.push_back("expected a function-open tag after the call open: " + s.substr(0, 20));
      m_skip_rest = true; m_pending.clear();
      return;
    }
    if (m_state == State::FUNC_NAME) {
      const auto gt = m_pending.find(GT);
      if (gt == std::string::npos) {
        if (m_pending.find(LT) != std::string::npos || atEnd) {
          m_errors.push_back("malformed function name: " + m_pending.substr(0, 20));
          m_skip_rest = true; m_pending.clear();
        }
        return;
      }
      const std::string name = q35_tr(m_pending.substr(0, gt));
      m_pending = m_pending.substr(gt + 1);
      m_name = name;
      if (m_cur) {
        (*m_cur)["function"]["name"] = name;    // collected function-name fragment
        // The name just resolved -> forward {index, function:{name: NAME}} (req 2).
        ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
        f["index"] = static_cast<std::int64_t>(m_cur_index);
        ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
        ff["name"] = name;
        f["function"] = ff;
        frag(f);
      }
      m_state = State::BODY;
      continue;
    }
    if (m_state == State::BODY) {
      std::string s = q35_tr(m_pending);
      m_pending = s;
      if (s.rfind(kParamOpen, 0) == 0) { m_pending = s.substr(kParamOpen.size()); m_state = State::PARAM_NAME; continue; }
      if (s.rfind(kFuncClose, 0) == 0) { m_pending = s.substr(kFuncClose.size()); m_state = State::TAIL; continue; }
      if (s.empty()) return;
      if (!atEnd && isPartial(s, std::vector<std::string>{ kParamOpen, kFuncClose })) return;
      m_errors.push_back("unexpected text inside a function block: " + s.substr(0, 20));
      m_pending = s.substr(1);                          // drop one byte, keep scanning for a tag
      continue;
    }
    if (m_state == State::PARAM_NAME) {
      const auto gt = m_pending.find(GT);
      if (gt == std::string::npos) {
        if (m_pending.find(LT) != std::string::npos || atEnd) {
          m_errors.push_back("malformed parameter name: " + m_pending.substr(0, 20));
          m_skip_rest = true; m_pending.clear();
        }
        return;
      }
      m_param_name = q35_tr(m_pending.substr(0, gt));
      m_pending = m_pending.substr(gt + 1);
      m_param_raw.clear();
      m_value_started = false;
      m_pending_newline = false;
      m_state = State::PARAM;
      // The reference does two fragments here: `json.dumps(param_name)` (a QUOTED key) then a
      // lone '"' that opens the value string -- quoting the key is mandatory (a bare name would
      // give `{city: ...`, invalid JSON).  Then `: ` + the opening quote of the value.
      const std::string sep = (m_param_index == 0) ? "{" : ", ";
      m_param_index += 1;
      // The opening-argument fragment `{<key>: "}` (the quoted key + ": " + the value's open quote).
      // The SAME string is appended to the running buffer AND forwarded as two fragments -- mirrors
      // QwenXMLToolParser, which _frag's f"{sep}{json.dumps(name)}: " then a lone '"' -- so the streamed
      // tool_calls (req 2) and the consolidated arguments stay in lock-step.
      const std::string kwarg = sep + "\"" + jsonEscape(m_param_name) + "\": \"";
      m_arguments_buf += kwarg;
      {
        ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
        f["index"] = static_cast<std::int64_t>(m_cur_index);
        ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
        ff["arguments"] = kwarg;
        f["function"] = ff;
        frag(f);
      }
      {
        ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
        f["index"] = static_cast<std::int64_t>(m_cur_index);
        ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
        ff["arguments"] = std::string("\"");
        f["function"] = ff;
        frag(f);
      }
      continue;
    }
    if (m_state == State::PARAM) {
      const auto idx = m_pending.find(kParamClose);
      if (idx == std::string::npos) {
        if (atEnd) { consumeValue(m_pending, true); m_pending.clear(); m_state = State::BODY; continue; }
        const std::size_t hold = holdbackSuffix(m_pending, std::vector<std::string>{ kParamClose });
        const std::string safe = m_pending.substr(0, m_pending.size() - hold);
        m_pending = m_pending.substr(m_pending.size() - hold);
        if (!safe.empty()) consumeValue(safe, false);
        return;
      }
      const std::string value = m_pending.substr(0, idx);
      m_pending = m_pending.substr(idx + kParamClose.size());
      consumeValue(value, true);
      m_state = State::BODY;
      continue;
    }
    return;
  }
}

void Qwen35ToolCallParser::applyBoundary(std::int64_t tag_id) {
  if (tag_id == kToolOpenId) {
    if (m_cur) {
      m_errors.push_back("new tool call opened before the previous one closed");
      finishCall();
    }
    startCall();
  } else {  // kToolCloseId
    if (m_state == State::TAIL) { finishCall(); return; }
    if (!m_cur) { m_errors.push_back("tool call close outside a tool block"); return; }
    m_errors.push_back("tool call closed before the function close");
    finishCall();
  }
}

void Qwen35ToolCallParser::startCall() {
  m_cur_index = m_calls.size();
  m_id = "call_" + std::to_string(m_cur_index);
  ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
  fn["name"] = std::string("");
  fn["arguments"] = std::string("");
  ov::genai::JsonContainer call = ov::genai::JsonContainer::object();
  call["id"] = m_id;
  call["type"] = std::string("function");
  call["function"] = fn;
  m_cur = call;                                 // the in-progress call (carries id + the open flag)
  // A call just OPENED -> forward the initial empty fragment {index, id, type, function:{name:"",
  // arguments:""}} so the carrier streams the (name/arguments-empty) tool_calls the instant a tool
  // is detected (req 2).  This is the live path; when m_on_fragment is null it is a no-op.
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = static_cast<std::int64_t>(m_cur_index);
    f["id"] = m_id;
    f["type"] = std::string("function");
    ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
    ff["name"] = std::string("");
    ff["arguments"] = std::string("");
    f["function"] = ff;
    frag(f);
  }
  m_name.clear();
  m_param_index = 0;
  m_state = State::HEAD;
  m_skip_rest = false;
  // Arguments are built purely by the PARAM_NAME (the {/seps) + consumeValue (the value) +
  // finishCall (the closing }/{}) fragments, exactly like the reference _start_call -- so
  // touching the running buffer here is unnecessary (and a reference-faithful reset() is the
  // only clear site).  Multi-call turns are out of scope for now, so no per-call reset of the
  // buffer is needed (the reference builds one finished call per turn).
}

void Qwen35ToolCallParser::finishCall() {
  if (!m_cur) { m_state = State::OUTSIDE; m_skip_rest = false; return; }
  // The closing } (or {} for a zero-argument function) -- the final arguments fragment.
  const std::string closeTok = (m_param_index > 0) ? "}" : "{}";
  m_arguments_buf += closeTok;
  // Forward {index, function:{arguments: closeTok}} (req 2) before the call is folded into m_calls.
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = static_cast<std::int64_t>(m_cur_index);
    ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
    ff["arguments"] = closeTok;
    f["function"] = ff;
    frag(f);
  }
  ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
  fn["name"] = m_name;
  fn["arguments"] = m_arguments_buf;
  ov::genai::JsonContainer finished = ov::genai::JsonContainer::object();
  finished["id"] = m_id;
  finished["type"] = std::string("function");
  finished["function"] = fn;
  m_cur = std::nullopt;
  m_state = State::OUTSIDE;
  m_skip_rest = false;
  m_calls.push_back(finished);
  m_param_index = 0;
  m_name.clear();
  m_arguments_buf.clear();
}

// A parameter-value delta -> a JSON-escaped fragment appended to the running arguments string.
// Handles the open-tag wrapper newline (dropped once) and a held value-leading newline.
void Qwen35ToolCallParser::consumeValue(const std::string& text, bool atEnd) {
  std::string v = text;
  if (!m_value_started && !v.empty() && v.front() == '\n')
    v = v.substr(1);
  if (!v.empty()) m_value_started = true;
  if (m_pending_newline) { v = "\n" + v; m_pending_newline = false; }
  if (!v.empty() && v.back() == '\n') {
    if (atEnd) v = v.substr(0, v.size() - 1);
    else { m_pending_newline = true; v = v.substr(0, v.size() - 1); }
  }
  if (!v.empty()) {
    m_param_raw += v;
    m_arguments_buf += jsonEscape(v);           // a parameter-value fragment
    // The raw value just decoded -> forward {index, function:{arguments: escaped}} (req 2).
    {
      ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
      f["index"] = static_cast<std::int64_t>(m_cur_index);
      ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
      ff["arguments"] = jsonEscape(v);
      f["function"] = ff;
      frag(f);
    }
  }
  if (atEnd) {
    m_arguments_buf += "\"";                    // close the current value's JSON string
    // Close the value's JSON string -> forward {index, function:{arguments: "\""}} (req 2).
    {
      ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
      f["index"] = static_cast<std::int64_t>(m_cur_index);
      ov::genai::JsonContainer ff = ov::genai::JsonContainer::object();
      ff["arguments"] = std::string("\"");
      f["function"] = ff;
      frag(f);
    }
  }
}

void Qwen35ToolCallParser::finishTruncated() {
  if (!m_cur) return;                           // nothing in flight
  if (m_state != State::TAIL) return;          // the inner function is not closed -> incomplete
  // If a fresh inner function began after this one in what we still hold (a 2nd call in a
  // multi-call turn), it is NOT a truncation -- a later close-tag will finish it.
  if (!m_pending.empty() && m_pending.find(kFuncOpen) != std::string::npos) return;
  m_errors.push_back("generation ended with an unterminated tool call (missing the close-tag)");
  finishCall();
}

void Qwen35ToolCallParser::emitToolCalls(ov::genai::JsonContainer& msg) {
  ov::genai::JsonContainer arr = ov::genai::JsonContainer::array();
  for (const auto& c : m_calls) arr.push_back(c);
  msg["tool_calls"] = arr;                    // the consolidated tool_calls array
}

void Qwen35ToolCallParser::frag(const ov::genai::JsonContainer& fragment) {
  if (m_on_fragment)
    m_on_fragment(fragment);   // forward to the wired sink (the carrier); no-op when driven directly
}

// ---- Qwen35BoundarySynthesizer (model-free text path) = the OpenArc _TextTagSynthesizer ----
// Split text into (chunk, ids) events so every COMPLETE boundary tag arrives in its own chunk
// carrying its token id and no chunk ends on a partial boundary tag.  Lets the unit tests drive
// the real token-id state machine with nothing but text; the live engine path feeds real ids.
std::vector<TokChunk> Qwen35BoundarySynthesizer::feed(const std::string& text) {
  m_buf += text;
  std::vector<TokChunk> out;
  while (!m_buf.empty()) {
    std::vector<std::pair<std::size_t, const QBoundary*>> events;
    for (const QBoundary& b : kQBoundaries) {
      const auto pos = m_buf.find(*b.tag);
      if (pos != std::string::npos) events.emplace_back(pos, &b);
    }
    if (!events.empty()) {
      auto& e = *std::min_element(events.begin(), events.end(),
          [](const auto& x, const auto& y) { return x.first < y.first; });
      const std::size_t pos = e.first;
      if (pos > 0) { out.emplace_back(m_buf.substr(0, pos), std::nullopt); m_buf = m_buf.substr(pos); }
      out.emplace_back(*e.second->tag, std::vector<std::int64_t>{ e.second->id });
      m_buf = m_buf.substr(e.second->tag->size());
      continue;
    }
    const std::vector<std::string> btags = { *kQBoundaries[0].tag, *kQBoundaries[1].tag };
    const std::size_t keep = m_buf.size() - holdbackSuffix(m_buf, btags);
    if (keep > 0) { out.emplace_back(m_buf.substr(0, keep), std::nullopt); m_buf = m_buf.substr(keep); }
    break;
  }
  return out;
}

std::optional<std::vector<TokChunk>> Qwen35BoundarySynthesizer::flush() {
  if (m_buf.empty()) return std::nullopt;
  const std::string chunk = m_buf;
  m_buf.clear();
  return std::vector<TokChunk>{ { chunk, std::nullopt } };
}

void Qwen35BoundarySynthesizer::reset() { m_buf.clear(); }

}  // namespace toolcall
}  // namespace ovgenai
