// Characterization coverage for the Qwen3.5 tool-call parser (Point 2 of the iteration),
// updated for the ported shape: the parser now emits the OpenAI-style 'tool_calls' ARRAY
// -- one entry per call, each { "id", "type":"function", "function":{ "name", "arguments" } },
// where "arguments" is the assembled argument JSON *string* (QwenXMLToolParser._finish_call).
//
// It drives the parser the way the ENGINE actually drives it, so a failure here is a real
// parser defect, not a test artefact: for each captured live model output in
// tool-call-qwen35.txt it (1) takes the model's `result` string LITERALLY (the verbatim
// decoded output, incl. the \n the model put between its tags and any leading
// close-of-thinking / prose), (2) TOKENIZES it with the real Qwen3.5 tokenizer, and (3) has a
// gmock stand-in for the VLMPipeline EMIT those token ids one at a time into a real
// ContentStreamer -- so the real TextParserStreamer chunk-flushes on the model's newlines and
// calls Qwen35ToolCallParser::parse() with BOTH the decoded delta_text and the delta_tokens,
// exactly as vlm.cpp's run does.  It then asserts (a) get_parsed_message()["tool_calls"] is ONE
// get_weather(call for city=London) call in the {function.name, arguments<json string>} shape,
// (b) the engine's non-empty-tool_calls rule reports a uniform finish_reason "tool_calls", and
// (c) the surfaced content carries no raw Qwen35 tag literals and no leaked close-of-thinking
// -- the parser routes the whole call block out (and, for a thinking turn, a separate gated reasoner
// moves reasoning into reasoning_content).  Run it with ctest -V to read one [obs <model>] line
// per model of what the engine actually surfaced (so the run records the real outcome).
//
// Builds/runs only when both the Qwen3.5 tokenizer fixture and the captured-outputs file are
// baked in at configure time; on an under-provisioned box the test skips, it never fails.
//
// Bracket-safety: no literal angle-bracket tag pair sits in this source (the project guardrail
// -- the env would otherwise re-run raw tool tags); the tag bytes are the k* / LT / GT constants
// from the per-model header, and a close-of-thinking is built as LT + "/thinking" + GT.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <openvino/core/descriptor/tensor.hpp>        // ov::Tensor + ov::AnyMap (any.hpp); encode output + image args
#include <openvino/genai/visual_language/pipeline.hpp>  // VLMPipelineBase / VLMDecodedResults / VideoMetadata
#include <openvino/genai/tokenizer.hpp>              // Tokenizer + TokenizedInputs + AnyMap

#include "streaming_support.hpp"                    // ContentStreamer + makeToolCallParserByName + the k*+LT+GT consts

// ---------------------------------------------------------------------------
// Reading the captured outputs (file-scope helpers; [[maybe_unused]] so an under-
// provisioned box that bakes neither macro still compiles the (skipping) suite).
// ---------------------------------------------------------------------------
struct Section { std::string label; std::string result; };   // one captured run: model + its (literal) result

// Index just past the '}' that closes the first top-level '{...}' object in `s`, starting the
// brace scan at the caller's `firstOpen`.  String/escape-aware so braces inside JSON strings
// (quotes, backslashes) don't miscount depth.
[[maybe_unused]]
static std::size_t findJsonObjectEnd(const std::string & s, std::size_t firstOpen) {
  int depth = 0;
  bool inStr = false, esc = false;
  for (std::size_t i = firstOpen; i < s.size(); ++i) {
    const char c = s[i];
    if (inStr) {                                  // inside a JSON string: only } that is not the
      if (esc) { esc = false; }                   //   unescaped close-quote ends it.
      else if (c == '\\') { esc = true; }
      else if (c == '"') { inStr = false; }
      continue;
    }
    if (c == '"') { inStr = true; }
    else if (c == '{') { ++depth; }
    else if (c == '}') { --depth; if (depth == 0) { return i + 1; } }
  }
  return std::string::npos;
}

// Split the file into its bullets (skipping the leading header before the first one) and pull each
// section's `result` string out of that section's JSON object -- taken LITERALLY, verbatim.
// Returns {} if the file is unreadable (the caller then skips).
[[maybe_unused]]
static std::vector<Section> readSections(const std::string & path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { return {}; }
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string file = ss.str();

  std::vector<Section> out;
  std::string curLabel, curBody;
  bool inSec = false;
  auto flush = [&]() {
    if (!inSec) { return; }
    const auto open = curBody.find('{');
    if (open != std::string::npos) {
      const auto end = findJsonObjectEnd(curBody, open);
      if (end != std::string::npos) {
        try {
          const ov::genai::JsonContainer obj =
            ov::genai::JsonContainer::from_json_string(curBody.substr(open, end - open));
          if (obj.is_object() && obj.contains("result")) {
            if (std::optional<std::string> r = obj["result"].as_string()) {
              out.push_back(Section{curLabel, *r});
            }
          }
        } catch (...) { /* a malformed section: drop it, keep scanning the rest */ }
      }
    }
    inSec = false;
    curLabel.clear();
    curBody.clear();
  };

  std::size_t lineStart = 0;
  while (lineStart < file.size()) {
    const auto nl = file.find('\n', lineStart);
    const std::string line =
      (nl == std::string::npos) ? file.substr(lineStart)
                                : file.substr(lineStart, nl - lineStart);
    const auto s = line.find_first_not_of(" \t");
    const bool bullet =
      (s != std::string::npos && s + 1 < line.size() &&
       line[s] == '*' && line[s + 1] == ' ');          // a bullet section start
    if (bullet) {
      flush();
      inSec = true;
      const std::string after = line.substr(s + 1);
      const auto b = after.find_first_not_of(" \t");
      curLabel =
        (b == std::string::npos) ? std::string()
        : after.substr(b, after.find_last_not_of(" \t") - b + 1);
    }
    if (inSec) { curBody += line + "\n"; }
    lineStart = (nl == std::string::npos) ? file.size() : nl + 1;
  }
  flush();
  return out;
}

// ---------------------------------------------------------------------------
// A gmock stand-in for the heavy, model-loading VLMPipeline.  Only the one method the
// engine actually invokes on our path is a real mock (its action EMITS the pre-tokenized
// result tokens into the streamer, exactly like the real per-token emit); EVERY other
// virtual of VLMPipelineBase is a trivial plain override -- that keeps the EXPECT_CALL on the
// single mocked generate unambiguous and makes the class concrete/instantiable.  The two
// single-image overloads are virtual-with-a-body in the base (they forward to the
// vector-images one), so they are inherited, not overridden.
class MockVLMPipeline : public ov::genai::VLMPipelineBase {
public:
  // The driven path: the engine calls this with the streamer; the test's action feeds tokens.
  MOCK_METHOD(ov::genai::VLMDecodedResults, generate,
              (const std::string& prompt,
               const std::vector<ov::Tensor>& images,
               const ov::genai::GenerationConfig& cfg,
               const ov::genai::StreamerVariant& streamer), (override));

  // Every other pure virtual of VLMPipelineBase the test never calls -- trivial bodies, taken
  // verbatim from the installed pipeline.hpp so no override misses the (abstract) base.
  ov::genai::VLMDecodedResults generate(
      const std::string&, const std::vector<ov::Tensor>&, const std::vector<ov::Tensor>&,
      const ov::genai::GenerationConfig&, const ov::genai::StreamerVariant&) override { return {}; }
  ov::genai::VLMDecodedResults generate(const std::string&, const ov::AnyMap&) override { return {}; }
  ov::genai::VLMDecodedResults generate(
      const ov::genai::ChatHistory&, const std::vector<ov::Tensor>&,
      const ov::genai::GenerationConfig&, const ov::genai::StreamerVariant&) override { return {}; }
  ov::genai::VLMDecodedResults generate(
      const ov::genai::ChatHistory&, const std::vector<ov::Tensor>&, const std::vector<ov::Tensor>&,
      const ov::genai::GenerationConfig&, const ov::genai::StreamerVariant&) override { return {}; }
  ov::genai::VLMDecodedResults generate(const ov::genai::ChatHistory&, const ov::AnyMap&) override { return {}; }
  ov::genai::VLMDecodedResults generate(
      const std::string&, const std::vector<ov::Tensor>&, const std::vector<ov::Tensor>&,
      const std::vector<ov::Tensor>&, const std::vector<ov::genai::VideoMetadata>&,
      const ov::genai::GenerationConfig&, const ov::genai::StreamerVariant&) override { return {}; }
  ov::genai::VLMDecodedResults generate(
      const ov::genai::ChatHistory&, const std::vector<ov::Tensor>&, const std::vector<ov::Tensor>&,
      const std::vector<ov::Tensor>&, const std::vector<ov::genai::VideoMetadata>&,
      const ov::genai::GenerationConfig&, const ov::genai::StreamerVariant&) override { return {}; }
  ov::genai::Tokenizer get_tokenizer() const override { return ov::genai::Tokenizer(); }
  void set_chat_template(const std::string&) override {}
  ov::genai::GenerationConfig get_generation_config() const override { return {}; }
  void set_generation_config(const ov::genai::GenerationConfig&) override {}
  bool supports_hidden_states_collection() const override { return false; }
  bool is_audio_output_enabled() const override { return false; }
};

#if defined(OVGENAI_VLM_FIXTURE) && defined(OVGENAI_TC_QWEN35_FILE)

TEST(qwen35_tool_call_parser, drivesParserFromTockenizedCapturedOutputs) {
  using namespace ovgenai::toolcall;   // ContentStreamer / makeToolCallParserByName / the k*+LT+GT consts

  // Fixture must still exist at runtime (needs the OpenVINO.GenAI runtime on LD_LIBRARY_PATH); a
  // box that cannot provide it skips rather than fails.
  if (::access(OVGENAI_VLM_FIXTURE, F_OK) != 0) {
    GTEST_SKIP() << "Qwen3.5 tokenizer fixture vanished at runtime; re-run ctest under the project env";
    return;
  }
  // The REAL Qwen3.5 tokenizer (the fixture ships its real tokenizer files) -- it both encodes the
  // literal result and, inside the streamer, decodes the re-emitted tokens.
  ov::genai::Tokenizer tok;
  try { tok = ov::genai::Tokenizer(std::filesystem::path(OVGENAI_VLM_FIXTURE)); }
  catch (...) { GTEST_SKIP() << "could not build the Qwen3.5 tokenizer (needs the runtime)"; return; }

  const std::vector<Section> sections = readSections(OVGENAI_TC_QWEN35_FILE);
  if (sections.empty()) { GTEST_SKIP() << "no captures in the tool-call-qwen35 file"; }

  // A redactor so a printed repr over the surfaced text can never itself contain a raw tag pair
  // (the project guardrail): every tag literal is swapped for the inert U+2039 / U+203A marker and
  // newlines flattened, so the [obs] line is safe to print verbatim.
  auto redact = [&](std::string s) {

    const std::string ph = "\u2039\u203A";
    // Scrub EVERY tag-shaped span so a printed [obs] line can leak no raw tag (the guardrail):
    // one pass over the angle-bracket spans (open-byte to the next close-byte -- the think /
    // /thinking / tool_call / function / parameter tags), and one over pipe-delimited sentinels
    // (the eos sentinel that has no angle byte at all); each run -> the inert placeholder pair `ph`.
    auto scrub = [&](std::string& t, const std::string& open, const std::string& close) {
      std::size_t from = 0;
      while (true) {
        std::size_t o = t.find(open, from);
        if (o == std::string::npos) break;
        std::size_t c = t.find(close, o + open.size());
        if (c == std::string::npos) { t.replace(o, open.size(), ph); break; }
        t.replace(o, (c + close.size()) - o, ph);
        from = o + ph.size();
      }
    };
    scrub(s, LT, GT);                 // every angle-bracket tag span
    scrub(s, "|", "|");              // every pipe-delimited sentinel span
    std::size_t q;
    while ((q = s.find('\n')) != std::string::npos)
      s.replace(q, 1, "\\n");
    return s;
  };

  for (const Section& sec : sections) {             // one captured live run per Qwen model
    // This gtest's unary EXPECT/ASSERT macros take no POSITIONAL message argument (the green
    // tests just omit it; see tool_call_parser_toolkit.cpp's bare EXPECT_EQ(a, b)).  Each
    // assertion below therefore carries its own diagnostic via the postfix << "message" form, and
    // this per-model trace line additionally names WHICH section fired (ctest reveals a failed
    // test's stdout via --output-on-failure).
    std::cout << "[model " << sec.label << "] ";
    EXPECT_FALSE(sec.label.empty());
    // (1) result verbatim -> (2) tokenize it into the token ids the live model emitted.
    const ov::AnyMap tokParams{{"add_special_tokens", false}};
    const ov::genai::TokenizedInputs enc = tok.encode(sec.result, tokParams);
    const auto* ids = enc.input_ids.data<int64_t>();
    std::vector<int64_t> emitted(ids, ids + enc.input_ids.get_size());
    ASSERT_FALSE(emitted.empty());
    // (3) the REAL parser run the way the engine runs it: a ContentStreamer carrying it and
    //     accumulating surfaced content into `acc`; the engine-facing carrier is its TextParserStreamer base.
    // The reasoning gate, per section: a close-of-thinking present in the captured OUTPUT means a
    // thinking turn (the model opened the think tag AND then closed it) -> wire the reasoner; no
    // close (the 0.8B goes straight to the tool-call block and never closes) means thinking is off
    // -> omit the reasoner so the base streamer fills content.  Compose the list for this section
    // (reasoner + tool when thinking; just the tool, or [] when neither -- qwen35 is always tool-wired).
    const std::string kThinkingClose = LT + "/think" + GT;
    const bool reasoning = sec.result.find(kThinkingClose) != std::string::npos;
    std::string acc;
    auto streamer = ::ovgenai::toolcall::ContentStreamer::instance(tok, "qwen35", reasoning, acc);
    auto base = std::static_pointer_cast<ov::genai::StreamerBase>(streamer);
    ov::genai::StreamerVariant sv = base;
    // (4) the gmock model EMITS the tokens one-at-a-time then flushes -- the engine's exact path, so
    //     parse() runs with decoded text AND the delta_tokens.  Args 1&2 are concrete types so this
    //     4-arg overload is picked (not the ChatHistory-first / single-image overloads).
    ::testing::NiceMock<MockVLMPipeline> pipe;
    EXPECT_CALL(pipe, generate(
        ::testing::An<const std::string&>(),
        ::testing::An<const std::vector<ov::Tensor>&>(),
        ::testing::_, ::testing::_))
      .Times(1)
      .WillOnce([&](const std::string&,
                    const std::vector<ov::Tensor>&,
                    const ov::genai::GenerationConfig&,
                    const ov::genai::StreamerVariant& s) -> ov::genai::VLMDecodedResults {
        auto& ss = std::get<1>(s);
        for (const int64_t t : emitted) {
          if (ss->write(t) != ov::genai::StreamingStatus::RUNNING) { break; }
        }
        ss->end();
        return ov::genai::VLMDecodedResults{};
      });
    std::vector<ov::Tensor> noImages;
    (void)pipe.generate(std::string(), noImages, ov::genai::GenerationConfig{}, sv);
    // (5) what the engine copies into res.parsed[0] -- the streamer's folded message.
    const ov::genai::JsonContainer msg = streamer->consolidatedMessage();
  // (the live, wired-sink parser no longer folds tool_calls into the per-delta message -- a 2nd call
  // set into it would throw the engine's base concatenate -- so the consolidated, all-calls result is
  // read via consolidatedMessage(): content + reasoning_content + the full tool_calls array, exactly
  // what vlm.cpp's bridge puts into res.parsed.)

    // (a) a tool_calls ARRAY was produced; it must be exactly one entry in the OpenAI call shape
    //     { "id", "type":"function", "function":{ "name", "arguments" } } -- name read from
    //     function.name, city read by re-parsing the arguments JSON string (the ported shape).
    //     Reasoning text is NOT split out: it stays in content (the user's direction); only the
    //     tool block is collected into tool_calls.
    EXPECT_TRUE(msg.contains("tool_calls")) << (sec.label + ": produced NO tool_calls at all").c_str();
    if (msg.contains("tool_calls")) {
      const ov::genai::JsonContainer& tc = msg["tool_calls"];
      EXPECT_TRUE(tc.is_array()) << (sec.label + ": tool_calls is not an array").c_str();
      EXPECT_EQ(tc.size(), static_cast<std::size_t>(1)) <<
                (sec.label + ": got " + std::to_string(tc.size()) + " tool_calls, expected exactly 1").c_str();
      if (tc.is_array() && tc.size() >= 1) {
        // The ported parser opens each entry with { id, type, function:{name, arguments} }.
        EXPECT_EQ(tc[0]["function"]["name"].as_string().value_or(std::string()), std::string("get_weather")) <<
                  (sec.label + ": wrong / absent tool name").c_str();
        // arguments is assembled as a JSON *string*; re-parse it to read the individual args.
        const std::string argsJson = tc[0]["function"]["arguments"].as_string().value_or(std::string());
        ov::genai::JsonContainer args;
        try { args = ov::genai::JsonContainer::from_json_string(argsJson); }
        catch (...) { args = ov::genai::JsonContainer(); }
        EXPECT_EQ(args["city"].as_string().value_or(std::string()), std::string("London")) <<
                  (sec.label + ": city was not London").c_str();
      }
    }
    // (b) the engine's finish_reason rule (vlm.cpp): a non-empty tool_calls is reported tool_calls.
    const bool hasToolCalls =
      msg.contains("tool_calls") && msg["tool_calls"].is_array() && msg["tool_calls"].size() > 0;
    EXPECT_TRUE(hasToolCalls) << (sec.label + ": engine would report finish_reason stop, not tool_calls").c_str();
    // (c) the surfaced content must be CLEAN of raw Qwen35 tags (the parser routes the whole call
    //     block out) and of any leaked close-of-thinking from the leading bare/real + prose prefix.
    EXPECT_EQ(acc.find(kTCClose),    std::string::npos) << (sec.label + ": a tool_call close tag leaked").c_str();
    EXPECT_EQ(acc.find(kFuncClose),  std::string::npos) << (sec.label + ": a function close tag leaked").c_str();
    EXPECT_EQ(acc.find(kParamClose), std::string::npos) << (sec.label + ": a parameter close tag leaked").c_str();
    const std::string thinkClose = LT + "/thinking" + GT;      // no literal close-tag pair in this source
    EXPECT_EQ(acc.find(thinkClose), std::string::npos) <<
              (sec.label + ": a close-of-thinking leaked into the surfaced content").c_str();
    // (d) BRIDGE REGRESSION: the CLI must reproduce the get_parsed_message() -> res.parsed[0] copy
    // that only the *LLM* pipeline does (the *VLM* pipeline that `vlm` drives leaves res.parsed empty,
    // so without this the real output showed finish_reason "tool_calls" from the engine's own finish --
    // set from the parser's TOOL_CALL_STOP -- yet NO tool_calls array).  Drive the added bridge over
    // the SAME streamer the engine copied from, into a fresh result, and assert the array actually
    // lands in res.parsed -- the exact step whose absence caused the real-world breakage.
    {
      ov::genai::DecodedResults bridgedRes;
      ASSERT_TRUE(bridgedRes.parsed.empty()) << (sec.label + ": pre-bridge res.parsed should be empty").c_str();
      bridgeParsedMessage(base, bridgedRes);
      ASSERT_FALSE(bridgedRes.parsed.empty()) << (sec.label + ": the bridge produced an empty res.parsed").c_str();
      ASSERT_TRUE(bridgedRes.parsed[0].is_object());
      ASSERT_TRUE(bridgedRes.parsed[0].contains("tool_calls")) <<
                 (sec.label + ": the bridge dropped the tool_calls array out of res.parsed").c_str();
      ASSERT_TRUE(bridgedRes.parsed[0]["tool_calls"].is_array());
      EXPECT_EQ(bridgedRes.parsed[0]["tool_calls"].size(), std::size_t{1});
      EXPECT_EQ(bridgedRes.parsed[0]["tool_calls"][0]["function"]["name"].as_string().value_or(std::string()),
                std::string("get_weather"));
    }
    // (e) the reasoning gate's effect on THIS turn: a thinking turn (a close-of-thinking in its
    //     output) must carry a non-empty reasoning_content (the wired reasoner moved it there); a
    //     no-think turn (the 0.8B) must have NO reasoning_content key -- it omitted the reasoner and
    //     the base filled content only.  This is the 130/131 -> 131/131 flip and the point of it.
    std::string rc;
    if (msg.is_object() && msg.contains("reasoning_content"))
      rc = msg["reasoning_content"].as_string().value_or(std::string());
    const std::size_t rc_len = rc.size();
    if (reasoning)
      EXPECT_FALSE(rc.empty())          << (sec.label + ": a thinking turn must carry reasoning_content");
    else
      EXPECT_FALSE(msg.contains("reasoning_content")) << (sec.label + ": a no-think turn must NOT carry reasoning_content");
    // OBSERVE (Point 2): the real per-model outcome -- a ctest -V run records what the engine
    // actually surfaced (not only each check's pass/fail); a discrepancy is what we chase next.
    const std::string tcName =
      (msg.contains("tool_calls") && msg["tool_calls"].is_array() && msg["tool_calls"].size() >= 1)
        ? msg["tool_calls"][0]["function"]["name"].as_string().value_or(std::string())
        : std::string();
    std::string tcCity;
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array() && msg["tool_calls"].size() >= 1) {
      const std::string aj = msg["tool_calls"][0]["function"]["arguments"].as_string().value_or(std::string());
      try { tcCity = ov::genai::JsonContainer::from_json_string(aj)["city"].as_string().value_or(std::string()); }
      catch (...) { tcCity = std::string(); }
    }
    std::cout << std::endl << "  [obs " << sec.label << "] acc_bytes=" << acc.size()
              << " tool_calls=" << (msg.contains("tool_calls") ? msg["tool_calls"].size() : 0)
              << " name=" << tcName << " city=" << tcCity
              << " reasoning=" << (reasoning ? "TRUE" : "FALSE") << " reasoning_bytes=" << rc_len
              << " hasToolCalls=" << (hasToolCalls ? "true" : "false")
              << " leaked[thinkClose,tool_call,function,parameter]="
              << (acc.find(thinkClose)  != std::string::npos ? "Y" : "n")
              << (acc.find(kTCClose)    != std::string::npos ? "Y" : "n")
              << (acc.find(kFuncClose)  != std::string::npos ? "Y" : "n")
              << (acc.find(kParamClose) != std::string::npos ? "Y" : "n")
              << " | surfaced=" << redact(acc).substr(0, 160) << std::endl;
  }
}

#else

TEST(qwen35_tool_call_parser, qwen35ToolCallOutputsNotBakedIn) {
  GTEST_SKIP() << "needs the Qwen3.5 tokenizer fixture AND the captured outputs baked in at configure "
                  "time; neither is present, so there is nothing to drive the parser (skipped)";
}

#endif
