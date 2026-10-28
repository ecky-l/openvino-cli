// Method bodies for the consolidated streaming-support entry point (declared in streaming_support.hpp) --
// the registry (moved in from the now-removed tool_call_support.{hpp,cpp}) AND the ContentStreamer (which now also owns the
// per-FAMILY assembly that the vlm subcommand drives through its static instance() factory).  The
// model-specific parsers each live in their own .cpp under toolcallparsers/ (this phase:
// qwen35_tool_call_parser.cpp).  See the header for the design rationale.
#include "streaming_support.hpp"
#include <cstdio>                       // fputs / fputc / fflush -- the per-chunk JSON emit the ContentStreamer does
#include <map>                          // mergeToolCallFragments: per-call-index accumulator for the emitted tool_calls
#include <openvino/genai/llm_pipeline.hpp>       // DecodedResults (full type; the bridge target)

namespace ovgenai {
namespace toolcall {

// The bridge the *VLM* pipeline omitted (see the header rationale): dynamic_pointer_cast is the gate --
// it hits only for a parser-carrying streamer (our ContentStreamer, a TextParserStreamer) and misses
// for a plain TextStreamer (the no-parser engine path, where we must touch nothing).  When it hits we
// copy the streamer's folded get_parsed_message() into res.parsed[0]; that folded message carries BOTH
// the content and the one-shot tool_calls array the parser wrote into its private internal message.
void bridgeParsedMessage(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                         ov::genai::DecodedResults& res) {
  auto tps = std::dynamic_pointer_cast<ov::genai::TextParserStreamer>(streamer);
  if (!tps) return;
  if (res.parsed.empty()) res.parsed.resize(1);
  // A ContentStreamer carries the tool parser and assembles the consolidated (ALL calls -> one
  // index-free array) so a multi-call turn still lands fully in res.parsed; a plain TextParserStreamer
  // (no ContentStreamer / no tool parser) just gets the folded message.
  if (auto cs = std::dynamic_pointer_cast<ContentStreamer>(streamer))
    res.parsed[0] = cs->consolidatedMessage();
  else
    res.parsed[0] = tps->get_parsed_message();
}

// Assemble the full passed-through result (above): the base's accumulated content + reasoning_content,
// plus a one-time consolidated tool_calls array from the carried parser's completed_calls() (every
// call, all of them).  No-op add for a content-only turn / a non-tool family (m_tool null or no
// completed calls) -> the message is returned unchanged.
ov::genai::JsonContainer ContentStreamer::consolidatedMessage() const {
  ov::genai::JsonContainer msg = get_parsed_message();
  if (m_tool) {
    const std::vector<ov::genai::JsonContainer>& calls = m_tool->completed_calls();
    if (!calls.empty()) {
      ov::genai::JsonContainer arr = ov::genai::JsonContainer::array();
      for (const auto& c : calls)
        arr.push_back(c);                 // index-free OpenAI {id, type:function:{name,arguments}} -- every call
      msg["tool_calls"] = arr;
    }
  }
  return msg;
}

// --- the registry: every advertised name -> an entry (wired iff `make` is non-null) --------
// The `qwen35` entry is wired (builds the model-specific Qwen35ToolCallParser; its kSentfunc
// / kSentry sentinels -- the model-intrinsic example_function_name / example_parameter -- are
// owned by the per-model header).  The others stay advertised-but-unwired today (resolve to
// null -> graceful: no parser, no turn-stop, finish_reason stays stop).
const std::vector<ModelFamilySpec>& knownModelFamilies() {
  static const std::vector<ModelFamilySpec> kSpecs = {
    { "qwen35", { kSentfunc, kSentry },
      // Just this format's tool parser.  A reasoning splitter is NOT in here: a separate, gated
      // ReasoningIncrementalParser is composed beside it by instance() and prepended only
      // when the rendered template shows thinking is on.
      []() { return std::vector<std::shared_ptr<ov::genai::IncrementalParser>>{
        std::static_pointer_cast<ov::genai::IncrementalParser>(
            std::make_shared<Qwen35ToolCallParser>()),
      }; } },
    { "hermes",      { }, nullptr },
    { "gemma4",      { }, nullptr },
    { "museglimmer", { }, nullptr },
  };
  return kSpecs;
}

std::vector<std::string> advertisedModelFamilies() {
  std::vector<std::string> v;
  for (const auto& s : knownModelFamilies())
    v.push_back(s.name);
  return v;
}

std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name) {
  for (const auto& s : knownModelFamilies())
    if (s.name == name && s.make)
      return s.make();
  return {};
}

// --- detect the model's format from its OWN loaded template (render a probe, match sentinels)
std::optional<std::string> detectModelFamily(const ov::genai::Tokenizer& tok) {
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
  for (const auto& s : knownModelFamilies()) {
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

// matchFormat -- the MODEL-FREE core of the auto-detect gate (no model, no render), split
// out from detectModelFamily (which renders the loaded template and then calls this) so
// the gate is unit-testable without a model: the entry whose `sentinels` are ALL present in
// `rendered` wins; nullopt if none match (graceful: no parser, no stop).
std::optional<std::string> matchFormat(const std::string& rendered) {
  for (const auto& s : knownModelFamilies()) {
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

// selectModelFamily -- selection precedence: an explicit --model-family name
// (override) wins; else auto-detect from the model's OWN loaded template; else no parser
// (graceful: no stop, finish_reason stays stop, no tool_calls -- today's behaviour).
std::optional<std::string> selectModelFamily(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out) {
  if (override_name.has_value() && !override_name->empty())
    return *override_name;
  auto d = detectModelFamily(tok);
  if (detected_out)
    *detected_out = d;
  return d;
}

// --- ContentStreamer: the engine-side carrier the registry composes around (the OpenArc
// --- Qwen35ToolCallStreamer analog).  The 3-arg ctor (direct / test use) delegates to the 4-arg
// --- entry-point ctor; the 4-arg one (instance() calls it) retains the slot-0 reasoner handle so a
// --- later activateEnableThinking can flip it.  Both finalise skip_special_tokens = false so the base's
// --- decode() (which hardcodes EMPTY detokenization params, falling back to the tokenizer's rt_info
// --- default -- true, stripping a leading close-of-thinking; for the 0.8B that close IS the whole
// --- lead-in before the tool-call block) keeps that no-think lead-in verbatim instead of collapsing it.
ContentStreamer::ContentStreamer(const ov::genai::Tokenizer& tok,
                                   std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                                   std::string& acc)
  : ContentStreamer(tok, std::move(parsers), acc, nullptr) {}
ContentStreamer::ContentStreamer(const ov::genai::Tokenizer& tok,
                                   std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                                   std::string& acc,
                                   const std::shared_ptr<DynamicReasoningIncrementalParser>& dyn,
                                   int stream)
  : TextParserStreamer(tok, std::move(parsers)), m_acc(acc), m_dyn(dyn) {
  m_additional_detokenization_params = ov::AnyMap{ ov::genai::skip_special_tokens(false) };
  m_tool_calls = ov::genai::JsonContainer::array();   // the per-step tool_calls buffer (drained each write)
  m_buf_tools  = ov::genai::JsonContainer::array();   // this window's accumulated tool_calls fragments (drained on the next emit)
  // Set in the body (NOT the init list) so the init-list order stays declaration-ordered and the int
  // member would otherwise trip -Wreorder (m_tool_calls sits before m_stream in declaration order).
  m_stream = stream;
}
// Append one per-step tool fragment the carried parser forwarded (via on_fragment) to this delta's
// tool_calls buffer; the next write() drains it onto stdout (req 2).
void ContentStreamer::collectFragment(const ov::genai::JsonContainer& fragment) {
  m_tool_calls.push_back(fragment);
}

// Stream the result, honouring the --stream N aggregation.  Build the emitted line FROM SCRATCH so the
// output carries only keys with a non-empty value (and nothing for an all-empty delta):
//   * content / reasoning_content are omitted when absent OR an empty string (req 1) -- so a
//     tool-only or blank delta comes out with no "content":"", instead of the old always-on one;
//   * tool_calls is the per-STEP fragment list the carried parser forwarded (req 2).  It is read from
//     the m_tool_calls buffer, NOT from `message`: the consolidate-at-completion tool_calls that
//     emitToolCalls writes into `message` would throw in the next delta's concatenate() (array vs
//     array).  That consolidated array stays in the folded message for the get_parsed_message ->
//     res.parsed bridge; the streamed (gradual) one is what we accumulate into the window here.
// The ONLY change from the old per-token code is when a built line is EMITTED: a non-empty delta now
// ACCUMULATES into the window buffers (content / reasoning / tool_calls) instead of writing to stdout
// at once, and a physical line is produced from those buffers by emitAggregate() -- when the window's
// per-delta counter reaches N (a non-empty delta counts, all-empty ones do not, so --stream 1 is
// BYTE-FOR-BYTE the old per-token stream), OR at the end (end() -> emitAggregate() emits the last
// partial window, or, for --stream <= 0, the WHOLE stream as one total line).  to_json_string(-1) is
// the compact dumper (one physical line per emit).
ov::genai::StreamingStatus ContentStreamer::write(ov::genai::JsonContainer& message) {
  std::optional<std::string> content;
  // Guard each read with contains(): operator[](key).as_string() is a Read-mode access that THROWS
  // if the path is absent.  The base always sets content (so that key is present), but a no-think /
  // non-thinker turn has NO reasoning_content key at all -- an unguarded read would throw on the very
  // first tool-only delta (it would, on the real vlm path, abort the whole generation).
  if (message.contains("content"))
    content = message["content"].as_string();
  std::optional<std::string> reasoning;
  if (message.contains("reasoning_content"))
    reasoning = message["reasoning_content"].as_string();
  const bool has_content   = content.has_value()   && !content->empty();
  const bool has_reasoning = reasoning.has_value() && !reasoning->empty();
  const bool has_tools     = !m_tool_calls.empty();
  const bool delta_has_output = has_content || has_reasoning || has_tools;

  // Also keep a plain (reasoning-stripped) content accumulator: unused by the terminal emitResult
  // (the content streams per chunk), but still read by the per-model leak/obs test checks.  Feeding
  // only when `content` is present matches the old unconditional post-emit append exactly (an absent or
  // empty content appends nothing either way -> byte-identical m_acc).
  if (content.has_value())
    m_acc += *content;

  // Fold this delta's per-step tool_calls fragments into the current window's buffer, then reset
  // m_tool_calls for the next delta -- the same drain the old per-delta code did, just into the
  // accumulating buffer.
  if (has_tools)
    for (size_t i = 0; i < m_tool_calls.size(); ++i)
      m_buf_tools.push_back(m_tool_calls[i]);
  m_tool_calls = ov::genai::JsonContainer::array();

  if (m_stream <= 0) {
    // No streaming output at all: just accumulate everything; the totality is emitted once, at end().
    if (has_content)   m_buf_content   += *content;
    if (has_reasoning) m_buf_reasoning += *reasoning;
    return ov::genai::StreamingStatus::RUNNING;
  }

  // N>=1: accumulate the window; a line is emitted when the per-window counter reaches N.  A delta that
  // carries nothing to show neither counts toward N nor emits -- so --stream 1 reproduces today's
  // per-token stream byte-for-byte (each non-empty delta -> exactly the line the old code produced).
  if (delta_has_output) {
    if (has_content)   m_buf_content   += *content;
    if (has_reasoning) m_buf_reasoning += *reasoning;
    ++m_stream_count;
    if (m_stream_count >= m_stream)
      emitAggregate();
  }
  return ov::genai::StreamingStatus::RUNNING;
}

// --- merge a window's raw per-step tool_call fragments into the MEANINGFUL per-call array a line carries
//
// The per-step fragments the carried parser forwards (via on_fragment -> collectFragment -> m_buf_tools)
// are each the OpenAI shape { index, [id, type] function:{ [name] , [arguments] } } : a {call-open}
// fragment is { index, id, type, function:{ name:"" , arguments:"" } }, a {function-name} fragment is
// { index, function:{ name } }, and the argument / {}} fragments are { index, function:{ arguments } }.
// Today's emit dumped ALL of them as separate array elements (one per token), which is NOT what a
// --stream N line should be.  This rebuilds them so ONE emitted line carries ONE entry per call index
// seen in the window (in first-seen order), with the call's name / arguments ACCUMULATED WITHIN that one
// entry -- exactly the OpenAI streaming contract applied per window.  Consequences, all from ONE rule:
//   * a single-fragment window (N==1) reproduces that fragment VERBATIM, so --stream 1 stays byte-identical
//     to before (the {call-open}'s empty name/arguments keys are preserved because the fragment still carries
//     them);
//   * a 3-fragment window (--stream 3) yields one entry per call, e.g. { index, id, type,
//     function:{ name:<the name> , arguments:<the 3 concat'd argument fragments> } } -- the user's example;
//   * the WHOLE stream is one window for N<=0, so this yields the fully-consolidated array, exactly what
//     the parser's completed_calls() (the res.parsed[0] bridge) produces -- "the whole aggregate".
// Merge rules per call index (the qwen35 parser always carries function.{name,arguments} on every fragment,
// so a key's PRESENCE — not a non-empty value — drives whether it is emitted, which keeps the verbatim N==1
// shape): name = the last NON-EMPTY function.name seen (the {call-open}'s "" is skipped, but its PRESENCE
// still emits an empty name key when no real name arrived); arguments = the CONCATENATION of every
// function.arguments seen (the same m_arguments_buf the parser builds); id / type (top-level) are carried
// from whichever fragment set them (the {call-open}'s id / type).
static ov::genai::JsonContainer mergeToolCallFragments(const ov::genai::JsonContainer& fragments) {
  if (!fragments.is_array() || fragments.empty())
    return ov::genai::JsonContainer::array();   // nothing to merge (a content/reasoning-only delta or an all-empty window)
  // Per-call accumulator (one per index); the {call-open} sets id/type/{name:arguments} and later fragments
  // extend the name (last non-empty) / arguments (concatenation).  `order` keeps first-seen index order.
  struct Acc {
    bool        saw_name = false;   std::string name;       // function.name : key emitted iff seen; last non-empty wins
    bool        saw_args = false;   std::string arguments;  // function.arguments : key emitted iff seen; concatenation
    bool        saw_id   = false;   std::string id;         // top-level id   : carried from the {call-open}
    bool        saw_type = false;   std::string type;       // top-level type : carried from the {call-open}
  };
  std::vector<std::int64_t> order;
  std::map<std::int64_t, Acc> acc;
  for (size_t i = 0; i < fragments.size(); ++i) {
    const ov::genai::JsonContainer& f = fragments[i];   // one raw per-step fragment = an OpenAI-shaped object
    if (!f.is_object())
      continue;                                         // a non-object fragment carries no structured call data
    // The index a fragment lands under (every qwen35 fragment carries it; a missing key just defaults to 0).
    std::int64_t idx = 0;
    if (f.contains("index"))
      if (auto v = f["index"].as_int()) idx = *v;
    const bool isNew = !acc.count(idx);
    Acc& a = acc[idx];                                  // default-constructed on first sight of a new index
    if (isNew)
      order.push_back(idx);
    if (f.contains("id"))
      a.saw_id = true,   a.id   = f["id"].as_string().value_or(std::string());
    if (f.contains("type"))
      a.saw_type = true, a.type = f["type"].as_string().value_or(std::string());
    if (f.contains("function")) {
      const ov::genai::JsonContainer& fn = f["function"];
      if (fn.is_object()) {
        if (fn.contains("name")) {                      // last non-empty name wins; bare {call-open}'s "" is skipped
          a.saw_name = true;
          if (const std::string nm = fn["name"].as_string().value_or(std::string()); !nm.empty())
            a.name = nm;
        }
        if (fn.contains("arguments")) {                 // concatenate (the reference contract the parser itself uses)
          a.saw_args = true;
          a.arguments += fn["arguments"].as_string().value_or(std::string());
        }
      }
    }
  }
  // Materialise one entry per index (first-seen order), emulating the original key order
  // index -> id -> type -> function:{ name , arguments } (matching how the parser builds a call).
  ov::genai::JsonContainer out = ov::genai::JsonContainer::array();
  for (std::int64_t idx : order) {
    const Acc& a = acc[idx];
    ov::genai::JsonContainer entry = ov::genai::JsonContainer::object();
    entry["index"] = idx;                              // every entry carries its index (the {call-open}'s)
    if (a.saw_id)   entry["id"]   = a.id;
    if (a.saw_type) entry["type"] = a.type;
    // the function sub-object is emitted iff any fragment carried a function.* key -- present even for a
    // bare {call-open} (name / arguments keys, possibly empty) so the per-delta stream's shape is unchanged.
    if (a.saw_name || a.saw_args) {
      ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
      if (a.saw_name) fn["name"]      = a.name;
      if (a.saw_args) fn["arguments"] = a.arguments;
      entry["function"] = fn;
    }
    out.push_back(entry);
  }
  return out;
}

// Build one emitted line from the current window's accumulated buffers, then reset the buffers + the
// per-window counter.  The keys are set in the ORIGINAL content -> reasoning_content -> tool_calls
// order and an empty value is OMITTED (never an "":"" / "[]"), so for --stream 1 the line is
// byte-identical to the old per-token emit and a larger N simply spans more deltas per line.  Fires
// from write() (the N>=1 window boundary) AND from end() (the final remainder, or -- for N<=0 -- the
// whole stream).  The buffers are copied into locals first, then the members reset, so the same routine
// serves as the terminal flush at end() without leaking the just-emitted content into a later window.
void ContentStreamer::emitAggregate() {
  std::string content = m_buf_content;
  std::string reasoning = m_buf_reasoning;
  ov::genai::JsonContainer tools = m_buf_tools;
  const bool has_content   = !content.empty();
  const bool has_reasoning = !reasoning.empty();
  const bool has_tools     = !tools.empty();

  m_buf_content.clear();
  m_buf_reasoning.clear();
  m_buf_tools = ov::genai::JsonContainer::array();
  m_stream_count = 0;
  if (!has_content && !has_reasoning && !has_tools)
    return;                     // an all-empty window -> emit no line (matches the old guard)

  ov::genai::JsonContainer out = ov::genai::JsonContainer::object();
  if (has_content)   out["content"]           = content;
  if (has_reasoning) out["reasoning_content"] = reasoning;
  if (has_tools) {
    // MERGE the window's raw per-step fragments (m_buf_tools) into one entry per call index -- name /
    // arguments accumulated WITHIN each entry -- instead of dumping them as separate array elements, so the
    // line carries the MEANINGFUL per-call JSON (a 3-token window -> one entry per call, not 3 fragments;
    // the whole stream for N<=0 -> the fully-consolidated array).
    const ov::genai::JsonContainer merged = mergeToolCallFragments(tools);
    if (!merged.empty())
      out["tool_calls"] = merged;
  }

  std::string line = out.to_json_string(-1);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// The terminal end-of-stream hook.  Let the base flush its trailing decoded text FIRST -- TextStreamer's
// own end() decodes the last cached tokens and fires one more write(JsonContainer) (its subword
// callback), which routes through our write() and accumulates the trailing delta into the window.  THEN
// emit whatever the window still holds: for N>=1 the final partial chunk (any deltas short of N), for
// N<=0 the WHOLE stream (it emitted nothing mid-stream).  This is the "end of the output stream" that
// --stream N<=0 aggregates to.  Called exactly once per request, via the engine's generate loop
// (ThreadedStreamerWrapper::end -- idempotent, joins the worker thread before the base end()).
void ContentStreamer::end() {
  ov::genai::TextStreamer::end();
  emitAggregate();
}

void ContentStreamer::activateEnableThinking(bool activate) {
  if (m_dyn)
    m_dyn->setActive(activate);   // ContentStreamer is a friend of the reasoner -> reaches its private setter
}

// THE entry point: build + return the proper streamer for a modelFamily (the old makeDynamicReasoningParser +
// composeParserList + the qwen35 reasoning tags, all folded in here).  For "qwen35" the always-slot-0
// reasoner is the DynamicReasoningIncrementalParser(false, false, kThinkOpen, kThinkClose) and then the
// family's tool parser(s) (the model-specific Qwen35ToolCallParser); the reasoner on/off flips per
// enableThinking via the delegate below.  Any other / empty family the registry does not wire gets an
// empty-parser content streamer (graceful: content, no tool_calls).
std::shared_ptr<ContentStreamer> ContentStreamer::instance(
    const ov::genai::Tokenizer& tok, const std::string& modelFamily,
    bool enableThinking, std::string& acc, int stream) {

  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers;
  std::shared_ptr<DynamicReasoningIncrementalParser> reasoner;
  std::shared_ptr<Qwen35ToolCallParser> qwenTool;   // function-scoped (like parsers/reasoner) so the
                                                    // wiring below -- after the family branch closes --
                                                    // stays in scope; null for any non-qwen35 family
  if (modelFamily == "qwen35") {
	// The reasoning tags for THIS family only -- LOCAL here (not namespace-wide constants), built from the
	// per-model header's LT / GT angle-byte constants (bracket-safety: no literal angle byte in this source,
	// just the names).  A new family would define its OWN tags in this same switch.
	static const std::string kThinkOpen  = LT + "think"   + GT;
	static const std::string kThinkClose = LT + "/think" + GT;
    reasoner = std::make_shared<DynamicReasoningIncrementalParser>(
        /*expect_open_tag=*/false, /*keep_original_content=*/false, kThinkOpen, kThinkClose);
    parsers.push_back(std::static_pointer_cast<ov::genai::IncrementalParser>(reasoner));
    const auto tools = makeToolCallParserByName(modelFamily);   // the model-specific Qwen35ToolCallParser
    // Grab a concrete handle to THIS just-built tool parser so, once the streamer exists below, we
    // can forward each per-step tool fragment (req 2) into its stdout display (weak_ptr -> cycle-free).
    for (auto& p : tools) {
      if (auto q = std::dynamic_pointer_cast<Qwen35ToolCallParser>(p)) { qwenTool = q; break; }
    }
    parsers.insert(parsers.end(), tools.begin(), tools.end());
  }
  auto streamer = std::make_shared<ContentStreamer>(tok, std::move(parsers), acc, reasoner, stream);
  streamer->activateEnableThinking(enableThinking);
  // Wire the qwen35 tool parser's per-step fragment sink -> the streamer's display buffer (weak_ptr
  // so the streamer<->parser references form no ownership cycle).
  if (qwenTool) {
    streamer->m_tool = qwenTool;          // record it so consolidatedMessage() can build the all-calls array
    std::weak_ptr<ContentStreamer> weakStreamer = streamer;
    qwenTool->setOnFragment([weakStreamer](const ov::genai::JsonContainer& fragment) {
      if (auto s = weakStreamer.lock())
        s->collectFragment(fragment);
    });
  }
  return streamer;
}

}  // namespace toolcall
}  // namespace ovgenai
