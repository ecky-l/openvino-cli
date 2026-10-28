// Model-free unit tests for the Qwen3.5 tool-call parser (the QwenXMLToolParser C++ port).
//
// They drive the parser the way the engine does -- a fresh delta_message per chunk, the parser's
// returned content folded into an accumulated message via concatenate -- but supply REAL token
// ids by SPLITTING the text with Qwen35BoundarySynthesizer (no real tokenizer / model needed:
// the synthesizer emits each complete boundary tag in its own chunk carrying its token id).  The
// parser then INVESTIGATES delta_tokens for the boundary ids (248058/248059), runs its state
// machine, collects the call fragments + content, and EMITS the tool_calls array -- so a non-model
// box still proves "the tool_calls array is present and correct."  Live model + real-token
// coverage lives in qwen35_tool_call_parser_test.cpp.
//
// Naming note: a using-DIRECTIVE (not an alias) is required because `ovgenai` is re-opened flat by
// other headers; containing it here to this TU is harmless.
//
// The calls are the OpenAI non-streaming shape {id, type:function, function:{name, arguments}}
// with arguments a JSON string (the faithful QwenXMLToolParser output); helpers below read the
// name under "function" and the city by re-parsing that arguments string.
#include <gtest/gtest.h>

#include "streaming_support.hpp"                // ContentStreamer + the qwen35 header (parser + synth + consts)
#include <openvino/genai/json_container.hpp>
#include <openvino/genai/llm_pipeline.hpp>        // DecodedResults (the engine-side result the bridge fills)
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace ovgenai::toolcall;

namespace {

// Build a complete Qwen3.5 call block for `fn` with the given {param, value} args, in the exact
// shape the model's own template renders (an outer call-open ... call-close frame; an inner
// function-open=NAME ... function-close; per arg a parameter-open=P ... parameter-close).
std::string blockForCall(const std::string& fn,
                         const std::vector<std::pair<std::string, std::string>>& args) {
  std::string b = kTCOpen + kFuncOpen + fn + GT;
  for (const auto& a : args)
    b += kParamOpen + a.first + GT + a.second + kParamClose;
  b += kFuncClose + kTCClose;
  return b;
}

// Drive the parser over the synthesized (chunk, ids) events of `text`, mirroring the engine
// (fresh delta_message per chunk; the parser's returned content folded via concatenate; a
// streamer's get_parsed_message would be the accumulated message).  Returns the accumulated message.
ov::genai::JsonContainer runSynth(Qwen35ToolCallParser& p, Qwen35BoundarySynthesizer& s,
                                  const std::string& text) {
  ov::genai::JsonContainer acc = ov::genai::JsonContainer::object();
  auto feed = [&](std::vector<TokChunk> chunks) {
    for (const auto& ch : chunks) {
      ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
      std::string t = ch.first;
      const std::string content = p.parse(dm, t, ch.second);
      dm["content"] = content;
      acc.concatenate(dm);
    }
  };
  feed(s.feed(text));
  if (auto tail = s.flush()) feed(*tail);   // release a held (never-completing) tail
  return acc;
}

// Read name / city out of the OpenAI-shaped call (arguments is a JSON string we re-parse).
std::string callName(const ov::genai::JsonContainer& c) {
  return c["function"]["name"].as_string().value_or(std::string());
}
std::optional<std::string> argCity(const ov::genai::JsonContainer& c) {
  const std::string raw = c["function"]["arguments"].as_string().value_or(std::string());
  if (raw.empty()) return std::nullopt;
  try {
    const ov::genai::JsonContainer a = ov::genai::JsonContainer::from_json_string(raw);
    return a["city"].as_string();
  } catch (...) { return std::nullopt; }
}
int statusOf(ov::genai::StreamingStatus s) { return static_cast<int>(s); }

}  // namespace

// A single complete call -> the parser investigates the boundary ids, runs the machine, collects
// the fragments + content, and surfaces exactly ONE call in the tool_calls array (name + city).
TEST(ToolCallSupport, QwenBlockSingleCall) {
  Qwen35ToolCallParser p;
  Qwen35BoundarySynthesizer s;
  std::string in = blockForCall("get_weather", {{"city", "London"}, {"unit", "celsius"}});

  const ov::genai::JsonContainer acc = runSynth(p, s, in);

  ASSERT_TRUE(acc.contains("tool_calls")) << "no tool_calls array was produced";     // the ask
  ASSERT_TRUE(acc["tool_calls"].is_array()) << "tool_calls is not an array";
  ASSERT_EQ(acc["tool_calls"].size(), std::size_t{1});
  EXPECT_EQ(callName(acc["tool_calls"][0]), std::string("get_weather"));
  ASSERT_TRUE(argCity(acc["tool_calls"][0]).has_value());
  EXPECT_EQ(*argCity(acc["tool_calls"][0]), std::string("London"));
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
}

// Plain (non-tag) text passes through untouched as content and never sets a tool_calls key.
TEST(ToolCallSupport, NonToolTextPassesThrough) {
  Qwen35ToolCallParser p;
  Qwen35BoundarySynthesizer s;
  std::string in = "Sure, I will look that up.";

  const ov::genai::JsonContainer acc = runSynth(p, s, in);

  EXPECT_FALSE(acc.contains("tool_calls"));                                // no array
  EXPECT_EQ(acc["content"].as_string().value_or(std::string()), in);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));
}

// The reported small-model bug, proven without a model: the model opens the frame and fully types
// the inner function but the outer close-tag is one byte short (the |im_end| lands first).  The
// inner function IS complete, so this must still surface as ONE call in the array.
TEST(ToolCallSupport, ToolCallTruncatedOuterCloseStillFires) {
  Qwen35ToolCallParser p;
  Qwen35BoundarySynthesizer s;
  const std::string full = blockForCall("get_weather", {{"city", "London"}});
  ASSERT_FALSE(full.empty());
  ASSERT_EQ(full[full.size() - 1], GT[0]);        // last byte is the close-tag's closing byte
  std::string in = full.substr(0, full.size() - 1);

  const ov::genai::JsonContainer acc = runSynth(p, s, in);

  ASSERT_TRUE(acc.contains("tool_calls")) << "the truncated call was dropped, not surfaced";  // the ask
  ASSERT_EQ(acc["tool_calls"].size(), std::size_t{1});
  EXPECT_EQ(callName(acc["tool_calls"][0]), std::string("get_weather"));
  ASSERT_TRUE(argCity(acc["tool_calls"][0]).has_value());
  EXPECT_EQ(*argCity(acc["tool_calls"][0]), std::string("London"));
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
}

// The close lands in a LATER chunk than the open (the block straddles a chunk boundary): no
// premature trigger before the close, and the call is reported only once the close arrives.
TEST(ToolCallSupport, ToolCallSplitAcrossChunks) {
  Qwen35ToolCallParser p;
  Qwen35BoundarySynthesizer s;
  std::string full = blockForCall("get_weather", {{"city", "London"}});
  auto chunks = s.feed(full);
  ASSERT_GE(chunks.size(), std::size_t{1});

  // Feed the OPEN chunk first -> a started-but-unclosed block must stay RUNNING, no array yet.
  {
    ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
    std::string t = chunks[0].first;
    const std::string content = p.parse(dm, t, chunks[0].second);
    dm["content"] = content;
    EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING))
      << "premature trigger before the close chunk";
    EXPECT_FALSE(dm.contains("tool_calls"));
  }
  // Then feed the rest -> the close arrives -> the call is reported; the array is present.
  ov::genai::JsonContainer acc = ov::genai::JsonContainer::object();
  for (std::size_t i = 1; i < chunks.size(); ++i) {
    ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
    std::string t = chunks[i].first;
    const std::string content = p.parse(dm, t, chunks[i].second);
    dm["content"] = content;
    acc.concatenate(dm);
  }
  if (auto tail = s.flush())
    for (const auto& ch : *tail) {
      ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
      std::string t = ch.first;
      const std::string content = p.parse(dm, t, ch.second);
      dm["content"] = content;
      acc.concatenate(dm);
    }

  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  ASSERT_TRUE(acc.contains("tool_calls")) << "no tool_calls array after the close";    // the ask
  EXPECT_EQ(acc["tool_calls"].size(), std::size_t{1});
  EXPECT_EQ(callName(acc["tool_calls"][0]), std::string("get_weather"));
  ASSERT_TRUE(argCity(acc["tool_calls"][0]).has_value());
  EXPECT_EQ(*argCity(acc["tool_calls"][0]), std::string("London"));
}

// reset() clears the per-turn state so a fresh turn reports exactly ONE new call (not a stale
// carry-over from turn 1).
TEST(ToolCallSupport, ResetClearsPerTurnState) {
  Qwen35ToolCallParser p;
  Qwen35BoundarySynthesizer s;
  std::string d1 = blockForCall("get_weather", {{"city", "Paris"}});
  const ov::genai::JsonContainer acc1 = runSynth(p, s, d1);
  ASSERT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  ASSERT_TRUE(acc1.contains("tool_calls"));
  ASSERT_EQ(acc1["tool_calls"].size(), std::size_t{1});
  EXPECT_EQ(argCity(acc1["tool_calls"][0]).value_or(std::string()), std::string("Paris"));

  p.reset();
  s.reset();
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));

  std::string d2 = blockForCall("get_weather", {{"city", "Tokyo"}});
  const ov::genai::JsonContainer acc2 = runSynth(p, s, d2);
  ASSERT_TRUE(acc2.contains("tool_calls"));
  ASSERT_EQ(acc2["tool_calls"].size(), std::size_t{1});       // exactly one, not a stale carry-over
  EXPECT_EQ(argCity(acc2["tool_calls"][0]).value_or(std::string()), std::string("Tokyo"));
}

// D3: one registry = one source of truth.  qwen35 is the only wired format; the rest are
// advertised-but-unwired (resolve to null -> graceful: no parser, no stop, finish_reason stop).
TEST(ToolCallSupport, RegistryAdvertisesAllAndWiresQwenOnly) {
  EXPECT_EQ(advertisedModelFamilies(),
            (std::vector<std::string>{"qwen35", "hermes", "gemma4", "museglimmer"}));
  // qwen35 is wired -> its registry make returns JUST the Qwen35ToolCallParser (the tool parser);
  // a reasoning splitter is NOT in the registry anymore -- it is composed separately
  // (composeParserList, gated on the rendered template) and prepended only when thinking is on, so
  // the registry stays a model-agnostic tool-format list.  The earlier two-entry (reasoner + tool)
  // shape is gone.
  auto qwen = makeToolCallParserByName("qwen35");
  ASSERT_FALSE(qwen.empty()) << "qwen35 must be wired -> a non-empty parser list";
  ASSERT_EQ(qwen.size(), std::size_t{1});
  EXPECT_NE(dynamic_cast<Qwen35ToolCallParser*>(qwen[0].get()), nullptr)
    << "the list must be just the Qwen35ToolCallParser (the reasoner is composed separately)";
  EXPECT_TRUE(makeToolCallParserByName("hermes").empty());   // advertised, not wired
  EXPECT_TRUE(makeToolCallParserByName("nope").empty());
}

// The reasoner at RUNTIME (always in the list; toggled by setActive) -- model-free. INACTIVE: a
// reasoning-tagged chunk passes through verbatim (raw tags stay in content; no reasoning_content key).
// ACTIVE: the engine splits the span into reasoning_content and returns the part after the close.
TEST(ToolCallSupport, ReasonerToggleActiveInactive) {
  const std::string kOpen  = LT + "think" + GT;
  const std::string kClose = LT + "/think" + GT;
  auto dyn = std::make_shared<DynamicReasoningIncrementalParser>(false, false, kOpen, kClose);
  ov::genai::Tokenizer tok;
  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> list{
    std::static_pointer_cast<ov::genai::IncrementalParser>(dyn)};
  std::string acc;
  auto carrier = std::make_shared<ContentStreamer>(tok, std::move(list), acc, dyn);

  ASSERT_NE(dyn, nullptr);
  ASSERT_FALSE(dyn->isActive());
  carrier->activateEnableThinking(false);
  {
    ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
    std::string t = kOpen + "plain reasoning text" + kClose + " final";
    ASSERT_FALSE(t.empty());
    const std::string out = dyn->parse(dm, t, std::vector<int64_t>{});
    EXPECT_FALSE(dm.contains("reasoning_content"))
      << "inactive reasoning must NOT write a reasoning_content key";
    EXPECT_EQ(out, t);
    EXPECT_FALSE(out.find(kOpen) == std::string::npos);
  }
  carrier->activateEnableThinking(true);
  ASSERT_TRUE(dyn->isActive());
  {
    ov::genai::JsonContainer dm = ov::genai::JsonContainer::object();
    std::string t = kOpen + "reason here" + kClose + " the answer";
    const std::string out = dyn->parse(dm, t, std::vector<int64_t>{});
    ASSERT_TRUE(dm.contains("reasoning_content")) << "active reasoning must split it out";
    std::string rc = dm["reasoning_content"].as_string().value_or(std::string());
    EXPECT_NE(rc.find("reason here"), std::string::npos)
      << "active: reasoning must split the inner text into reasoning_content (whitespace-tolerant; the upstream base parser may add a benign leading space from a literal open tag)";
    EXPECT_EQ(out, " the answer");
  }
}
// D1/D5 -- the model-free core of the auto-detect gate: the model-intrinsic sentinels
// (example_function_name / example_parameter) that Qwen's template renders drive detection; a
// template with neither matches nothing (graceful); the sentinels are AND-ed so naming only ONE
// cannot false-trigger.
TEST(ToolCallSupport, MatchFormatIsModelFree) {
  const std::string hit =
    kTCOpen + kFuncOpen + kSentfunc + GT
    + kParamOpen + kSentry + GT + "value" + kParamClose + kFuncClose + kTCClose;
  auto got = matchFormat(hit);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, std::string("qwen35"));
  EXPECT_FALSE(matchFormat("a plain chat template with role and content and no tool call block at all").has_value());
  EXPECT_FALSE(matchFormat("it references " + kSentfunc + " only").has_value());
}

// The selection precedence: an explicit --model-family <name> override wins without
// auto-detect, so a default (empty) Tokenizer is a safe stand-in for the model-free half.
TEST(ToolCallSupport, SelectOverrideWinsWithoutModel) {
  ov::genai::Tokenizer tok;
  auto got = selectModelFamily(std::optional<std::string>("gemma4"), tok, nullptr);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, std::string("gemma4"));
}

// BRIDGE REGRESSION -- the model-free half of the real-world bug "finish_reason tool_calls but no
// tool_calls array".  Only the *LLM* pipeline's run_generate_with_parsers copies a streamer's folded
// get_parsed_message() into res.parsed[0]; the *VLM* pipeline that `vlm` drives has no such copy, so
// res.parsed was left EMPTY and emitResult reported the engine's own finish_reason "tool_calls" (set
// from the parser's TOOL_CALL_STOP) with NO tool_calls array.  The CLI now bridges it.  No model is
// needed to prove the GRACEFUL half: a plain TextStreamer is NOT a TextParserStreamer, so the bridge's
// dynamic-cast must MISS and leave res.parsed exactly (empty) as the engine produced it -- the no-parser
// path must never gain a spurious tool_calls.  (The positive half -- a parser-carrier actually landing
// tool_calls in res.parsed -- is the fixture-gated live test in qwen35_tool_call_parser_test.cpp.)
TEST(ToolCallSupport, BridgeIsNoOpForPlainStreamer) {
  ov::genai::DecodedResults res;                 // parsed empty -- as the VLM pipeline leaves it
  ASSERT_TRUE(res.parsed.empty()) << "precondition: a fresh result's parsed is empty";

  ov::genai::Tokenizer tok;
  auto plain = std::make_shared<ov::genai::TextStreamer>(
      tok, [](std::string) -> ov::genai::CallbackTypeVariant { return ov::genai::StreamingStatus::RUNNING; });
  std::shared_ptr<ov::genai::StreamerBase> base = plain;   // the engine holds the carrier as a StreamerBase

  bridgeParsedMessage(base, res);

  EXPECT_TRUE(res.parsed.empty())               // cast missed -> res.parsed untouched (no spurious tool_calls)
    << "bridge fired for a plain TextStreamer; res.parsed was not left as the engine produced it";
}

// REGRESSION for parser-bug.txt: the value's OPENING '"' was streamed twice.  The defect lived only
// in the per-step STREAMED fragment channel -- PARAM_NAME forwarded `{"city": "` (a fragment that
// already ends on the value's open quote) AND, as a SECOND fragment, a redundant lone `"`, so
// reassembling the streamed arguments produced `{"city": ""London"}` (a doubled open quote, invalid
// JSON).  The CONSOLIDATED result the other model-free tests read was already correct, so nothing
// caught it.  This drives the real state machine through its on_fragment sink (the very stream channel)
// and reassembles the streamed arguments exactly as a consumer would, so the doubled open quote is
// proven gone.  (bracket-safety: the tag bytes are the k*/LT/GT consts; the argument text below,
// e.g. the literal `{"city": "London"}`, carries no angle-bracket tag pair -- so it is safe here.)
TEST(ToolCallSupport, StreamedArgumentsReassembleNoDoubledOpenQuote) {
  Qwen35ToolCallParser p;                       // driven directly (the model-free path); sink wired below
  Qwen35BoundarySynthesizer s;

  // Capture every fragment the parser forwards through its on_fragment sink (a null sink would no-op).
  std::vector<ov::genai::JsonContainer> frags;
  p.setOnFragment([&frags](const ov::genai::JsonContainer& f) { frags.push_back(f); });

  // A complete single call (the blockForCall shape mirrors the model's own template).
  runSynth(p, s, blockForCall("get_weather", {{"city", "London"}}));                  // drive the parser; the sink fills frags + completed_calls

  // Reassemble each call's arguments from the streamed fragments, in stream order -- the OpenAI
  // streaming contract: a {index, function:{arguments:FRAG}} appends FRAG to that call's arguments.
  std::map<int64_t, std::string> reassembled;
  for (const auto& f : frags) {
    if (!f.contains("function")) continue;                    // id-only / name-only fragments carry no arguments
    const ov::genai::JsonContainer& fn = f["function"];
    if (!fn.contains("arguments")) continue;
    std::int64_t idx = 0;
    if (f.contains("index")) { if (auto iv = f["index"].as_int()) idx = *iv; }
    reassembled[idx] += fn["arguments"].as_string().value_or(std::string());
  }

  ASSERT_TRUE(reassembled.count(0)) << "no streamed arguments were collected for call 0";
  const std::string joined = reassembled[0];

  // (1) the bug's signature: a doubled open quote.  A correct `{"city": "London"}` carries no two
  //     adjacent '"'; the old double-emit left `{"city": ""London"}` (which does) -> this fails on it.
  EXPECT_EQ(joined.find("\"\""), std::string::npos)
    << "streamed arguments re-assemble with a doubled open quote: " << joined;

  // (2) the reassembled arguments re-parse to valid JSON and carry city=London -- argCity() re-parses
  //     and returns nullopt on a parse throw, so it fails cleanly on the doubled-quote string.
  ov::genai::JsonContainer call = ov::genai::JsonContainer::object();
  ov::genai::JsonContainer fn   = ov::genai::JsonContainer::object();
  fn["arguments"] = joined;
  call["function"] = fn;
  ASSERT_TRUE(argCity(call).has_value())
    << "re-assembled streamed arguments are not valid JSON: " << joined;
  EXPECT_EQ(argCity(call).value_or(std::string()), std::string("London"));

  // (3) the streamed channel now AGREES with the consolidated one (they used to diverge: the consolidated
  //     was correct, the stream doubled the open quote).  The consolidated is read from
  //     p.completed_calls() -- the same source the real bridge reads -- NOT from the drained buffer:
  //     wiring a non-null sink deliberately stops emitToolCalls from writing delta_message (a 2nd call
  //     re-setting an array key would throw the base concatenate), so a sinked run's buffer has no
  //     tool_calls while the consolidated stays fully populated.
  ASSERT_FALSE(p.completed_calls().empty())
    << "the parser produced no completed call to compare the streamed channel against";
  const std::string consolidatedCity = argCity(p.completed_calls()[0]).value_or(std::string());
  EXPECT_EQ(consolidatedCity, argCity(call).value_or(std::string()))
    << "the streamed reassembly and the consolidated call disagree";
  EXPECT_EQ(consolidatedCity, std::string("London"));
}

// mergeToolCallFragments is now the single GENERALISED consumer of a window's raw per-step tool_calls
// fragments: it keys on the OpenAI Chat Completions `type` ( "function" and "custom" today; the file-local
// kToolCallTypes table is the extension point for the later Responses-API members).  These tests drive it
// MODEL-FREE -- they feed a synthetic fragment sequence (what a --stream N window would hold) and assert the
// merged per-call shape -- so the generalization is provable without a live model emitting that type.
//
// A "function" call (the user's get_weather example, --stream 3) must fold to ONE entry per call, the name
// (last non-empty) and the argument (concatenated) WITHIN that one entry, carrying the id + type -- and the
// key regression guard: a value/name-ONLY window (no top-level `type`, as when a window falls BETWEEN two
// calls) must still emit { index, function:{...} } but must NOT gain a top-level `type` (this is why the
// generalization resolves the sub-object from the fragment's `function:` key, not from a top-level `type`
// the fragment never carried).
TEST(ToolCallSupport, MergeKeepsFunctionStreamedShape) {
  ov::genai::JsonContainer frags = ov::genai::JsonContainer::array();
  // {call-open}: index + id + type + function:{ name:"", arguments:"" }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    f["id"]    = std::string("call_0");
    f["type"]  = std::string("function");
    ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
    fn["name"]      = std::string("");
    fn["arguments"] = std::string("");
    f["function"] = fn;
    frags.push_back(f);
  }
  // {name} delta: index + function:{ name }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
    fn["name"] = std::string("get_weather");
    f["function"] = fn;
    frags.push_back(f);
  }
  // {argument} delta: index + function:{ arguments }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
    fn["arguments"] = std::string("{\"city\":\"London\"}");
    f["function"] = fn;
    frags.push_back(f);
  }
  const ov::genai::JsonContainer merged = mergeToolCallFragments(frags);
  ASSERT_TRUE(merged.is_array());
  ASSERT_EQ(merged.size(), std::size_t{1}) << merged.to_json_string();
  const ov::genai::JsonContainer e = merged[0];
  EXPECT_EQ(e["index"].as_int().value_or(std::int64_t{-1}), std::int64_t{0});
  EXPECT_TRUE(e.contains("id"));
  EXPECT_EQ(e["id"].as_string().value_or(std::string()), std::string("call_0"));
  EXPECT_TRUE(e.contains("type"));
  EXPECT_EQ(e["type"].as_string().value_or(std::string()), std::string("function"));
  ASSERT_TRUE(e.contains("function"));
  const ov::genai::JsonContainer fn = e["function"];
  ASSERT_TRUE(fn.contains("name"));               // the {call-open}'s presence keeps the key (even empty)
  EXPECT_EQ(fn["name"].as_string().value_or(std::string()), std::string("get_weather"));   // last non-empty wins
  ASSERT_TRUE(fn.contains("arguments"));
  EXPECT_EQ(fn["arguments"].as_string().value_or(std::string()), std::string("{\"city\":\"London\"}"));
}

// The generalization landed: a "custom" tool call (the OTHER OpenAI Chat Completions union member).  Its
// sub-object is `custom` (not `function`) and its value key is `input` (the FREE-FORM string the model
// generated -- NOT a JSON payload like `function.arguments`).  The same machine, keyed on `type`, rebuilds
// these four deltas into ONE entry:
//   final  : {"index":0,"id":"call_c","type":"custom","custom":{"name":"get_weather","input":"weather in London"}}
//   deltas : {index:0, id:call_c, type:custom, custom:{name:"",input:""}}   (the {call-open})
//            {index:0, custom:{name:"get_weather"}}
//            {index:0, custom:{input:"weather in"}}   {index:0, custom:{input:" London"}}
TEST(ToolCallSupport, MergeHandlesCustomToolType) {
  ov::genai::JsonContainer frags = ov::genai::JsonContainer::array();
  // {call-open}: index + id + type:"custom" + custom:{ name:"", input:"" }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    f["id"]    = std::string("call_c");
    f["type"]  = std::string("custom");
    ov::genai::JsonContainer cd = ov::genai::JsonContainer::object();
    cd["name"]  = std::string("");
    cd["input"] = std::string("");
    f["custom"] = cd;
    frags.push_back(f);
  }
  // {name} delta: index + custom:{ name }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    ov::genai::JsonContainer cd = ov::genai::JsonContainer::object();
    cd["name"] = std::string("get_weather");
    f["custom"] = cd;
    frags.push_back(f);
  }
  // {input} x2: the free-form input streamed in fragments, concatenated
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    ov::genai::JsonContainer cd = ov::genai::JsonContainer::object();
    cd["input"] = std::string("weather in");
    f["custom"] = cd;
    frags.push_back(f);
  }
  {
    ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
    f["index"] = std::int64_t{0};
    ov::genai::JsonContainer cd = ov::genai::JsonContainer::object();
    cd["input"] = std::string(" London");
    f["custom"] = cd;
    frags.push_back(f);
  }
  const ov::genai::JsonContainer merged = mergeToolCallFragments(frags);
  ASSERT_TRUE(merged.is_array());
  ASSERT_EQ(merged.size(), std::size_t{1}) << merged.to_json_string();
  const ov::genai::JsonContainer e = merged[0];
  EXPECT_EQ(e["index"].as_int().value_or(std::int64_t{-1}), std::int64_t{0});
  EXPECT_TRUE(e.contains("id"));
  EXPECT_EQ(e["id"].as_string().value_or(std::string()), std::string("call_c"));
  EXPECT_TRUE(e.contains("type"));
  EXPECT_EQ(e["type"].as_string().value_or(std::string()), std::string("custom"));
  ASSERT_TRUE(e.contains("custom"));
  EXPECT_FALSE(e.contains("function")) << "a custom call must not also carry a function sub-object";
  const ov::genai::JsonContainer cd = e["custom"];
  ASSERT_TRUE(cd.contains("name"));
  EXPECT_EQ(cd["name"].as_string().value_or(std::string()), std::string("get_weather"));
  ASSERT_TRUE(cd.contains("input"));
  EXPECT_EQ(cd["input"].as_string().value_or(std::string()), std::string("weather in London"));
}

// Regression guard, symmetric with the function case: a `custom` value/name-only window (NO top-level
// `type`/`id`, as when a window falls between calls) must still emit { index, custom:{...} } but must NOT
// gain a top-level `type` -- the sub-object is resolved from the fragment's `custom:` key.
TEST(ToolCallSupport, MergeCustomValueOnlyWindowHasNoType) {
  ov::genai::JsonContainer frags = ov::genai::JsonContainer::array();
  ov::genai::JsonContainer f = ov::genai::JsonContainer::object();
  f["index"] = std::int64_t{0};
  ov::genai::JsonContainer cd = ov::genai::JsonContainer::object();
  cd["input"] = std::string("lon");
  f["custom"] = cd;
  frags.push_back(f);

  const ov::genai::JsonContainer merged = mergeToolCallFragments(frags);
  ASSERT_EQ(merged.size(), std::size_t{1});
  const ov::genai::JsonContainer e = merged[0];
  EXPECT_EQ(e["index"].as_int().value_or(std::int64_t{-1}), std::int64_t{0});
  EXPECT_FALSE(e.contains("type")) << "a custom value-only window must not gain a top-level type";
  EXPECT_FALSE(e.contains("id"))   << "a custom value-only window must not gain a top-level id";
  ASSERT_TRUE(e.contains("custom"));
}
