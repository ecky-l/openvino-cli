// Tool-call parser / registry unit tests for (b) -- model-agnostic, future-proof tool
// CALL support in the `vlm` subcommand.  NOTHING here loads a model: the tests drive the
// CLI-side machinery directly -- the Qwen3.5 IncrementalParser (the nested  ‹tool_call› …
// ‹/tool_call›  tag block the model's own template emits, validated by rendering the model's
// loaded template through the public tokenizer API), the ContentStreamer wrapper, the
// name->parser registry (one source of truth), and the MODEL-FREE detection sentinel match.
// The model-GATED half of the gate (render the model's template then detect) is covered
// separately once a fixture with a tool-aware template lands in the tree.
//
// Tag literals are built from the header's escaped constants (LT / GT / kFuncOpen / ...),
// so no literal provider bracket pair appears in this source (it would otherwise re-run a
// tooling scanner); where a tag is shown in prose it is written with the inert ‹…› marker.
//
// NAMING: the members are pulled in with a using-DIRECTIVE (not a namespace ALIAS):
// `ovgenai` is re-opened FLAT by the project's other headers (commands/config/...), so a
// `using tc = ovgenai::toolcall;` alias is rejected by this g++ ("denotes no type"), whereas a
// `using namespace ovgenai::toolcall;` bringing the names in for THIS translation unit works
// (verified).  It is contained to this file (does not pollute the project's global namespace).
//
// The parser's 2nd arg is a non-const `std::string&` (the base class contract: "modified in
// place"), so every delta_text passed to parse() must be a NON-CONST lvalue.
#include <gtest/gtest.h>

#include "tool_call_support.hpp"

#include <openvino/genai/json_container.hpp>
#include <openvino/genai/streamer_base.hpp>
#include <openvino/genai/tokenizer.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace ovgenai::toolcall;

namespace {
// Build a complete Qwen3.5 call block for `fn` with the given {{param, value}} args, in the
// exact shape the model's own template renders -- the outer  ‹tool_call› … ‹/tool_call›
// wrapping an inner  ‹function=fn› … ‹/function›, each arg a  ‹parameter=p›v‹/parameter›.
std::string blockForCall(const std::string& fn,
                         const std::vector<std::pair<std::string, std::string>>& args) {
  std::string b = kTCOpen + kFuncOpen + fn + GT;
  for (const auto& a : args)
    b += kParamOpen + a.first + GT + a.second + kParamClose;
  b += kFuncClose + kTCClose;
  return b;
}

// Compare a StreamingStatus without relying on a gtest printer for the enum class.
int statusOf(ov::genai::StreamingStatus s) { return static_cast<int>(s); }
}  // namespace

// A single complete call -> the parser finalises it, flips TOOL_CALL_STOP (so the engine
// stops generation and the CLI reports finish_reason = tool_calls), and records
// {name, arguments} on the shared delta_message for the engine to copy into .parsed.
TEST(ToolCallSupport, QwenBlockSingleCall) {
  QwenXmlToolCallIncrementalParser p;
  ov::genai::JsonContainer msg = ov::genai::JsonContainer::object();
  std::string in = blockForCall("get_weather", {{"city", "London"}, {"unit", "celsius"}});

  const std::string filtered = p.parse(msg, in, std::nullopt);

  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  EXPECT_EQ(filtered, "");                        // the whole delta was the tool block
  const auto& calls = msg["tool_calls"];
  ASSERT_TRUE(calls.is_array());
  ASSERT_EQ(calls.size(), std::size_t{1});
  EXPECT_EQ(calls[0]["name"].get_string(), "get_weather");
  EXPECT_EQ(calls[0]["arguments"]["city"].get_string(), "London");
  EXPECT_EQ(calls[0]["arguments"]["unit"].get_string(), "celsius");
}

// Plain (non-tag) text passes through untouched and never sets a tool_calls key.
TEST(ToolCallSupport, NonToolTextPassesThrough) {
  QwenXmlToolCallIncrementalParser p;
  ov::genai::JsonContainer msg = ov::genai::JsonContainer::object();
  std::string in = "Sure, I will look that up.";

  const std::string filtered = p.parse(msg, in, std::nullopt);

  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));
  EXPECT_EQ(filtered, "Sure, I will look that up.");
  EXPECT_EQ(msg.size(), std::size_t{0});          // nothing was written into the delta_message
}

// The close tag straddles a chunk boundary: completion must still be detected on the later
// chunk (the parser accumulates all decoded text and re-scans from its consume-cursor).
TEST(ToolCallSupport, ToolCallSplitAcrossChunks) {
  QwenXmlToolCallIncrementalParser p;
  ov::genai::JsonContainer msg = ov::genai::JsonContainer::object();
  const std::string full = blockForCall("get_weather", {{"city", "London"}});

  ASSERT_FALSE(full.empty());
  ASSERT_EQ(full[full.size() - 1], GT[0]);        // last byte is the › that ends ‹/tool_call ›
  std::string a = full.substr(0, full.size() - 1);   // drop the › -> close is incomplete
  std::string b = full.substr(full.size() - 1);     // the missing › byte

  const std::string fa = p.parse(msg, a, std::nullopt);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));
  EXPECT_EQ(fa, "");                              // no completed close yet -> nothing, no false trigger

  const std::string fb = p.parse(msg, b, std::nullopt);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  EXPECT_EQ(fb, "");
  const auto& calls = msg["tool_calls"];
  ASSERT_TRUE(calls.is_array());
  ASSERT_EQ(calls.size(), std::size_t{1});
  EXPECT_EQ(calls[0]["name"].get_string(), "get_weather");
  EXPECT_EQ(calls[0]["arguments"]["city"].get_string(), "London");
}

// Leading prose (before any tool block) is emitted as content in the earlier pass, then a
// later pass that completes the block adds no content -- so the recorded content and the
// tool_calls carry disjoint information (the block itself is stripped out of the result).
TEST(ToolCallSupport, ProsePrefixIsEmittedBeforeTheBlock) {
  QwenXmlToolCallIncrementalParser p;
  ov::genai::JsonContainer msg = ov::genai::JsonContainer::object();
  std::string prefix = "The weather in London is: ";

  const std::string f1 = p.parse(msg, prefix, std::nullopt);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));
  EXPECT_EQ(f1, prefix);
  EXPECT_EQ(msg.size(), std::size_t{0});          // no tool block yet -> nothing recorded

  std::string in = blockForCall("get_weather", {{"city", "London"}});
  const std::string f2 = p.parse(msg, in, std::nullopt);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  EXPECT_EQ(f2, "");                             // the block itself contributes no content
  const auto& calls = msg["tool_calls"];
  ASSERT_TRUE(calls.is_array());
  ASSERT_EQ(calls.size(), std::size_t{1});
  EXPECT_EQ(calls[0]["name"].get_string(), "get_weather");
  EXPECT_EQ(calls[0]["arguments"]["city"].get_string(), "London");
}

// reset() clears the per-turn state (decoded-text buffer, consume-cursor, recorded calls) so
// a fresh turn completes cleanly: a second block after reset yields exactly ONE new call
// (not a stale carry-over from turn 1) and a RUNNING status.
TEST(ToolCallSupport, ResetClearsPerTurnState) {
  QwenXmlToolCallIncrementalParser p;
  ov::genai::JsonContainer msg1 = ov::genai::JsonContainer::object();
  std::string d1 = blockForCall("get_weather", {{"city", "Paris"}});
  p.parse(msg1, d1, std::nullopt);
  ASSERT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));

  p.reset();
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::RUNNING));

  ov::genai::JsonContainer msg2 = ov::genai::JsonContainer::object();
  std::string d2 = blockForCall("get_weather", {{"city", "Tokyo"}});
  const std::string f2 = p.parse(msg2, d2, std::nullopt);
  EXPECT_EQ(statusOf(p.get_status()), statusOf(ov::genai::StreamingStatus::TOOL_CALL_STOP));
  const auto& calls = msg2["tool_calls"];
  ASSERT_TRUE(calls.is_array());
  ASSERT_EQ(calls.size(), std::size_t{1});        // exactly one, not a stale carry-over
  EXPECT_EQ(calls[0]["arguments"]["city"].get_string(), "Tokyo");
}

// D3: one registry = one source of truth.  qwen35 is the only wired format; the rest are
// advertised-but-unwired (resolve to null -> graceful: no parser, no stop, today's behaviour).
TEST(ToolCallSupport, RegistryAdvertisesAllAndWiresQwenOnly) {
  EXPECT_EQ(advertisedToolCallParsers(),
            (std::vector<std::string>{"qwen35", "hermes", "gemma4", "museglimmer"}));
  EXPECT_NE(makeToolCallParserByName("qwen35"), nullptr);   // wired -> a real parser
  EXPECT_EQ(makeToolCallParserByName("hermes"), nullptr);   // advertised, not wired
  EXPECT_EQ(makeToolCallParserByName("nope"), nullptr);
}

// D1/D5 -- the model-free core of the auto-detect gate: the model-intrinsic sentinels
// example_function_name / example_parameter (which Qwen's template renders independent of the
// user's tools) drive detection; a template with neither matches nothing (graceful), and
// the sentinels are AND-ed so a template that happens to name only ONE cannot false-trigger.
TEST(ToolCallSupport, MatchFormatIsModelFree) {
  const std::string hit =
    LT + "tool_call" + GT + kFuncOpen + kSentfunc + GT
    + kParamOpen + kSentry + GT + "value" + kParamClose + kFuncClose + kTCClose;
  auto got = matchFormat(hit);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, std::string("qwen35"));

  EXPECT_FALSE(matchFormat(
    "a plain chat template with role and content and no tool call block at all").has_value());
  EXPECT_FALSE(matchFormat("it references " + kSentfunc + " only").has_value());
}

// The selection precedence: an explicit --tool-call-parser <name> override wins without
// auto-detect, so a default (empty) Tokenizer is a safe stand-in for this model-free half of
// the gate (detectToolCallFormat is only reached on the no-override path).
TEST(ToolCallSupport, SelectOverrideWinsWithoutModel) {
  ov::genai::Tokenizer tok;                        // Tokenizer() = default
  auto got = selectToolCallParserName(std::optional<std::string>("gemma4"), tok, nullptr);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, std::string("gemma4"));
}
