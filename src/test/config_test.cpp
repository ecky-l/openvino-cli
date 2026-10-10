// Unit tests for the shared load-model config (src/main/config.cpp): the arg
// parser that both `ovcli worker` and `ovcli run` consume, and the tolerant
// JSON round-trip that is its from-LOAD-message analog.
//
// The parser is boost::program_options (static), with a single-letter short
// alias per option where the first letter is unclaimed.  An option whose
// first-letter prefix clashes with a shorter one (e.g. --device "d" vs
// --draft-device "d...") is offered long-form only; see config.hpp for the
// full alias map and the rationale.
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../main/config.hpp"
#include "../main/model_load_config.hpp"
#include "../main/model_scheduler_config.hpp"
#include "../main/model_request_config.hpp"

// The config symbols (structs + their parse / JSON helpers + ConfigError) now
// live in namespace ovcli::config; pull them in unqualified for the tests.
using namespace ovcli::config;

namespace {

// A parse/validation is expected to throw ConfigError mentioning `needle`.
void expectConfigErrorContains(std::function<void()> fn, const std::string& needle) {
  bool got = false;
  try {
    fn();
  } catch (const ConfigError& e) {
    got = std::string(e.what()).find(needle) != std::string::npos;
  }
  EXPECT_TRUE(got) << "expected a ConfigError mentioning '" << needle << "'";
}

// A minimal, valid argument set; tests append / mutate from here.
std::vector<std::string> validMin() {
  return {"--model", "/opt/qwen3-8b", "--engine", "ovgenai",
          "--model-type", "vlm", "--device", "CPU"};
}
std::vector<std::string> validMinAppend(std::vector<std::string> extra) {
  std::vector<std::string> v = validMin();
  v.insert(v.end(), extra.begin(), extra.end());
  return v;
}

}  // namespace

TEST(parseLoadModel, minimalLongFlagsDerivesName) {
  auto c = parseModelLoadConfig(validMin());
  EXPECT_EQ(c.model, "/opt/qwen3-8b");
  EXPECT_EQ(c.engine, "ovgenai");
  EXPECT_EQ(c.model_type, "vlm");
  EXPECT_EQ(c.device, "CPU");
  // optionals at their declared defaults
  EXPECT_FALSE(c.cache_dir.has_value());
  EXPECT_FALSE(c.draft_model.has_value());
  EXPECT_EQ(c.draft_device, std::optional<std::string>("CPU"));
  EXPECT_EQ(c.runtime_config_json, "{}");
  // The scheduler is now its own config; a minimal invocation leaves it all-nullopt.
  auto sc = parseModelSchedulerConfig(validMin());
  EXPECT_FALSE(sc.hasAnySet());
}

TEST(parseLoadModel, shortAliasesMatchLong) {
  // Single-letter shorts: -m, -e, -t, -d (see config.hpp for the full map).
  auto c = parseModelLoadConfig({"-m", "/opt/x", "-e", "openvino", "-t", "llm", "-d", "GPU"});
  EXPECT_EQ(c.model, "/opt/x");
  EXPECT_EQ(c.engine, "openvino");
  EXPECT_EQ(c.model_type, "llm");
  EXPECT_EQ(c.device, "GPU");
}

TEST(parseLoadModel, attachedForms) {
  // --long=value is the canonical attached form; boost strips the '=' for
  // double-dash options. Single-character shorts take a run-on form
  // '-xvalue' instead: for a single-dash short, boost 1.90 keeps a leading
  // '=' in the value, so "-x=value" would decode with a literal '=' prefix
  // and then fail the enum / number validation the way a genuine typo would.
  auto a = parseModelLoadConfig({"--model=/opt/qwen", "--engine=ovgenai",
                                  "--model-type=vlm", "--device=GPU"});
  EXPECT_EQ(a.model, "/opt/qwen");
  EXPECT_EQ(a.engine, "ovgenai");

  auto b = parseModelLoadConfig({"-m/opt/qwen", "-eovgenai", "-tvlm", "-dCPU"});
  EXPECT_EQ(b.model, "/opt/qwen");
  EXPECT_EQ(b.engine, "ovgenai");
}

TEST(parseLoadModel, deviceShortIsDistinctFromDraftLongs) {
  // "d" is a prefix of the long names --draft-device / --draft-model, but
  // boost matches the single-character short by exact name: -d must still land
  // on --device (not a draft option, which are long-only because of the clash).
  auto c = parseModelLoadConfig({"-m", "/opt/qwen", "-e", "ovgenai", "-t", "llm",
                                  "-d", "GPU",
                                  "--draft-device", "CPU",
                                  "--draft-model", "/opt/draft"});
  EXPECT_EQ(c.device, "GPU");
  EXPECT_EQ(c.draft_device, std::optional<std::string>("CPU"));
  EXPECT_EQ(c.draft_model, std::optional<std::string>("/opt/draft"));
}

TEST(parseLoadModel, draftDeviceEmptyClearsDefault) {
  auto c = parseModelLoadConfig(validMinAppend({"--draft-device", ""}));
  EXPECT_FALSE(c.draft_device.has_value());  // "" clears the "CPU" default
}

TEST(parseLoadModel, positionalIsModelPathFriend) {
  // A lone positional stands in for --model; --model wins if both
  // are given (the preprocessor drops the positional so boost sees a clean
  // list and doesn't report "cannot be specified more than once").
  auto c = parseModelLoadConfig({"gnmt", "-e", "ovgenai", "-t", "llm", "-d", "CPU"});
  EXPECT_EQ(c.model, "gnmt");

  auto both = parseModelLoadConfig(
      {"--model", "/explicit/path", "positional",
       "-e", "ovgenai", "-t", "llm", "-d", "CPU"});
  EXPECT_EQ(both.model, "/explicit/path");  // explicit flag wins over the positional
}

TEST(parseLoadModel, tooManyPositionalsError) {
  expectConfigErrorContains(
      [] { parseModelLoadConfig({"a", "b", "-e", "ovgenai", "-t", "llm", "-d", "CPU"}); },
      "positional");
}

TEST(parseLoadModel, missingRequiredReportedTogether) {
  // Only engine is given; --model/--model-type/--device are all missing.
  expectConfigErrorContains(
      [] { parseModelLoadConfig({"--engine", "ovgenai"}); }, "model");
  expectConfigErrorContains(
      [] { parseModelLoadConfig({"--engine", "ovgenai"}); }, "model-type");
  expectConfigErrorContains(
      [] { parseModelLoadConfig({"--engine", "ovgenai"}); }, "device");
}

TEST(parseLoadModel, badEnumsReject) {
  // Give the bad value as the SOLE occurrence of the option; otherwise boost
  // reports "cannot be specified more than once" for the duplicate before our
  // finalize() can flag the bad enum value.
  expectConfigErrorContains([] {
    parseModelLoadConfig({"--model", "/opt/x", "--engine", "ovgenai",
                           "--model-type", "not-a-type", "--device", "CPU"});
  }, "must be one of");
  expectConfigErrorContains([] {
    parseModelLoadConfig({"--model", "/opt/x", "--engine", "not-an-engine",
                           "--model-type", "llm", "--device", "CPU"});
  }, "must be one of");
  expectConfigErrorContains([] {
    parseModelRequestConfig(validMinAppend({"--tool-call-parser", "x"}));
  }, "must be one of");
}

TEST(parseLoadModel, unknownLongAndShortOptionReject) {
  // An unrecognised long, an unrecognised single-letter short, or (the options
  // ovcli dropped: context-window / worker-max-respawns
  // / worker-line-limit are not worker properties) -- each must produce a
  // ConfigError whose text mentions "unknown option".
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--nope", "1"})); },
                            "unknown option");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--foo", "1"})); },
                            "unknown option");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"-z", "1"})); },
                            "unknown option");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--model-name", "y"})); },
                            "unknown option");  // derived, not a flag
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--context-window", "auto"})); },
                            "unknown option");  // dropped, not a flag
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--worker-max-respawns", "5"})); },
                            "unknown option");  // dropped, supervisor-side
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--worker-line-limit", "256K"})); },
                            "unknown option");  // dropped, worker-only transport
}

TEST(parseLoadModel, helperOptionMissingValue) {
  expectConfigErrorContains([] { parseModelLoadConfig({"--model"}); }, "requires a value");
  expectConfigErrorContains([] { parseModelLoadConfig({"-m"}); }, "requires a value");
}

TEST(parseLoadModel, numericCoercionAndFloor) {
  // Sole-occurrence forms: boost 1.90 rejects a *duplicate* option with
  // "cannot be specified more than once"; to trigger our numeric validation
  // the bad value must be the only one for each option.
  auto rt = parseModelRequestConfig({
      "--model", "/opt/x", "--engine", "ovgenai", "--model-type", "llm",
      "--device", "CPU",
      "--num-assistant-tokens", "4",
      "--assistant-confidence-threshold", "0.5"});
  ASSERT_TRUE(rt.num_assistant_tokens.has_value());
  EXPECT_EQ(rt.num_assistant_tokens.value(), std::size_t(4));
  ASSERT_TRUE(rt.assistant_confidence_threshold.has_value());
  EXPECT_NEAR(rt.assistant_confidence_threshold.value(), 0.5, 1e-12);

  expectConfigErrorContains([] {
    parseModelRequestConfig(validMinAppend({"--num-assistant-tokens", "x"}));
  }, "integer");
  expectConfigErrorContains([] {
    parseModelRequestConfig(validMinAppend({"--assistant-confidence-threshold", "hi"}));
  }, "number");
  // The single-letter short forms exercise the same type-parsing path.
  expectConfigErrorContains([] {
    parseModelRequestConfig(validMinAppend({"-n", "x"}));
  }, "integer");
}

TEST(parseLoadModel, jsonObjectFlagsRoundThrough) {
  auto c = parseModelLoadConfig(validMinAppend({
      "--runtime-config", "{\"INFERENCE_NUM_THREADS\":4}",
      "--max-num-seqs", "2",
      "--config-blocks",  "{\"sampler_config\":{\"temperature\":0.5}}",
  }));
  EXPECT_EQ(c.runtime_config_json, "{\"INFERENCE_NUM_THREADS\":4}");
  // --max-num-seqs is a scheduler flag now: read it via the scheduler config.
  auto sc = parseModelSchedulerConfig(validMinAppend({"--max-num-seqs", "2"}));
  ASSERT_TRUE(sc.max_num_seqs.has_value());
  EXPECT_EQ(*sc.max_num_seqs, std::size_t(2));

  // A non-object JSON value where an object is required is rejected.
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--runtime-config", "[1,2]"})); },
                            "object");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--runtime-config", "nope"})); },
                            "valid JSON");
}

TEST(jsonRoundTrip, fullConfig) {
  std::vector<std::string> args = {
      "--model", "/opt/qwen3-8b-a3b", "--engine", "ovgenai", "--model-type", "llm",
      "--device", "GPU",        "--cache-dir", "/var/ovcache",
      "--draft-model", "/opt/draft", "--draft-device", "CPU",
      "--num-assistant-tokens", "4", "--assistant-confidence-threshold", "0.5",
      "--tool-call-parser", "qwen35",
      "--runtime-config", "{\"INFERENCE_NUM_THREADS\":4}",
      "--max-num-seqs", "2",
      "--config-blocks",    "{\"sampler_config\":{\"temperature\":0.5}}",
  };
  // Each sibling config parses the same arg list, reading only its own section.
  auto load  = parseModelLoadConfig(args);
  auto sched = parseModelSchedulerConfig(args);
  auto req   = parseModelRequestConfig(args);
  // Round-trip each through its own JSON and assert equality.
  EXPECT_EQ(load,  modelLoadConfigFromJson(modelLoadConfigToJson(load)));
  EXPECT_EQ(sched, modelSchedulerConfigFromJson(modelSchedulerConfigToJson(sched)));
  EXPECT_EQ(req,   modelRequestConfigFromJson(modelRequestConfigToJson(req)));
  // the emitted load JSON must not carry the three fields ovcli deliberately dropped
  std::string js = modelLoadConfigToJson(load);
  EXPECT_EQ(js.find("context_window"), std::string::npos);
  EXPECT_EQ(js.find("worker_max_respawns"), std::string::npos);
  EXPECT_EQ(js.find("worker_line_limit"), std::string::npos);
  // and the scheduler config actually captured the one set field
  ASSERT_TRUE(sched.max_num_seqs.has_value());
  EXPECT_EQ(*sched.max_num_seqs, std::size_t(2));
}

TEST(jsonRoundTrip, adoptsAndIgnoresForeignKeys) {
  // A config shaped like a supervisor's model_dump, complete with the keys
  // ovcli dropped (context_window / worker_max_respawns / worker_line_limit):
  // known keys are read, the rest simply
  // ignored (not fatal).
  auto c = modelLoadConfigFromJson(
      R"({"model":"/opt/qwen","model_type":"llm",
          "engine":"ovgenai","device":"CPU","context_window":"auto",
          "worker_max_respawns":3,"worker_line_limit":65536,"some_foreign_key":true})");
  EXPECT_EQ(c.model, "/opt/qwen");
  EXPECT_EQ(c.model_type, "llm");
  EXPECT_EQ(c.engine, "ovgenai");
  // re-serializing emits none of the dropped/foreign keys at all
  std::string js = modelLoadConfigToJson(c);
  EXPECT_EQ(js.find("context_window"), std::string::npos);
  EXPECT_EQ(js.find("worker_max_respawns"), std::string::npos);
  EXPECT_EQ(js.find("worker_line_limit"), std::string::npos);
  EXPECT_EQ(js.find("some_foreign_key"), std::string::npos);
}

TEST(jsonRoundTrip, derivesNameWhenUnsetAndStillRequiresRest) {
  auto c = modelLoadConfigFromJson(
      R"({"model":"/opt/qwen3-8b-a3b/","model_type":"vlm","engine":"openvino","device":"NPU"})");
  EXPECT_EQ(c.model, "/opt/qwen3-8b-a3b/");  // path itself is kept verbatim

  // A model-less config is still missing-required, even in the tolerant reader.
  expectConfigErrorContains(
      [] { modelLoadConfigFromJson(R"({"model_type":"llm","engine":"ovgenai","device":"CPU"})"); },
      "model");
}

TEST(jsonRoundTrip, modelPathWrongTypeRejects) {
  expectConfigErrorContains(
      [] { modelLoadConfigFromJson(R"({"model":3,"model_type":"llm","engine":"ovgenai","device":"CPU"})"); },
      "model");
}
