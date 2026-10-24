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

#include "tool_call_support.hpp"                // ContentStreamer + the qwen35 header (parser + synth + consts)
#include <openvino/genai/json_container.hpp>
#include <openvino/genai/llm_pipeline.hpp>        // DecodedResults (the engine-side result the bridge fills)
#include <cstddef>
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
  EXPECT_EQ(advertisedToolCallParsers(),
            (std::vector<std::string>{"qwen35", "hermes", "gemma4", "museglimmer"}));
  EXPECT_NE(makeToolCallParserByName("qwen35"), nullptr);   // wired -> a real parser
  EXPECT_EQ(makeToolCallParserByName("hermes"), nullptr);   // advertised, not wired
  EXPECT_EQ(makeToolCallParserByName("nope"), nullptr);
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

// The selection precedence: an explicit --tool-call-parser <name> override wins without
// auto-detect, so a default (empty) Tokenizer is a safe stand-in for the model-free half.
TEST(ToolCallSupport, SelectOverrideWinsWithoutModel) {
  ov::genai::Tokenizer tok;
  auto got = selectToolCallParserName(std::optional<std::string>("gemma4"), tok, nullptr);
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
