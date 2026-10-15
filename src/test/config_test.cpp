// Unit tests for the shared load-model config (src/main/config.cpp): the arg
// parser that both `ovgenai worker` and `ovgenai run` consume, and the tolerant
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
#include "../main/model_runtime_config.hpp"
#include "../main/model_scheduler_config.hpp"
#include "../main/model_request_config.hpp"

// The config symbols (structs + their parse / JSON helpers + ConfigError) now
// live in namespace ovgenai::config; pull them in unqualified for the tests.
using namespace ovgenai::config;

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
  return {"--model", "/opt/qwen3-8b", "--device", "CPU"};
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
  EXPECT_EQ(c.device, "CPU");
  // The load optionals at their defaults (cache_dir is now on the runtime
  // config,  so it is no longer a load optional):
  EXPECT_FALSE(c.draft_model.has_value());
  EXPECT_EQ(c.draft_device, std::optional<std::string>("CPU"));
  // validMin() carries --model (load requires it) but no --cache-dir, so the new
  // cache_dir *default* fires: this minimal runtime config is no longer all-unset --
  // it holds cache_dir = <model-dir>/model_cache  (that is the feature, by request;
  // before the default a truly-minimal runtime config was all-nullopt, and that is
  // still asserted where no --model is given, e.g. the explicit-empty and no-model
  // cases in cache_dir_default_test.cpp).
  auto rt = parseModelRuntimeConfig(validMin());
  ASSERT_TRUE(rt.cache_dir.has_value());
  EXPECT_EQ(*rt.cache_dir, std::string("/opt/qwen3-8b/model_cache"));
  EXPECT_TRUE(rt.hasAnySet());
  // The scheduler is now its own config; a minimal invocation leaves it all-nullopt.
  auto sc = parseModelSchedulerConfig(validMin());
  EXPECT_FALSE(sc.hasAnySet());
}

TEST(parseLoadModel, shortAliasesMatchLong) {
  // Single-letter shorts: -m, -e, -t, -d (see config.hpp for the full map).
  auto c = parseModelLoadConfig({"-m", "/opt/x", "-d", "GPU"});
  EXPECT_EQ(c.model, "/opt/x");
  EXPECT_EQ(c.device, "GPU");
}

TEST(parseLoadModel, attachedForms) {
  // --long=value is the canonical attached form; boost strips the '=' for
  // double-dash options. Single-character shorts take a run-on form
  // '-xvalue' instead: for a single-dash short, boost 1.90 keeps a leading
  // '=' in the value, so "-x=value" would decode with a literal '=' prefix
  // and then fail the enum / number validation the way a genuine typo would.
  auto a = parseModelLoadConfig({"--model=/opt/qwen", "--device=GPU"});
  EXPECT_EQ(a.model, "/opt/qwen");

  auto b = parseModelLoadConfig({"-m/opt/qwen", "-dCPU"});
  EXPECT_EQ(b.model, "/opt/qwen");
}

TEST(parseLoadModel, deviceShortIsDistinctFromDraftLongs) {
  // "d" is a prefix of the long names --draft-device / --draft-model, but
  // boost matches the single-character short by exact name: -d must still land
  // on --device (not a draft option, which are long-only because of the clash).
  auto c = parseModelLoadConfig({"-m", "/opt/qwen", "-d", "GPU",
                                  "--draft-device", "CPU",
                                  "--draft-model", "/opt/draft"});
  EXPECT_EQ(c.device, "GPU");
  EXPECT_EQ(c.draft_device, std::optional<std::string>("CPU"));
  EXPECT_EQ(c.draft_model, std::optional<std::string>("/opt/draft"));
}

TEST(parseLoadModel, badEnumsReject) {
  // Give the bad value as the SOLE occurrence of the option; otherwise boost
  // reports "cannot be specified more than once" for the duplicate before our
  // finalize() can flag the bad enum value.
  expectConfigErrorContains([] {
    parseModelRequestConfig(validMinAppend({"--tool-call-parser", "x"}));
  }, "must be one of");
}

TEST(parseLoadModel, draftDeviceEmptyClearsDefault) {
  auto c = parseModelLoadConfig(validMinAppend({"--draft-device", ""}));
  EXPECT_FALSE(c.draft_device.has_value());  // "" clears the "CPU" default
}

TEST(parseLoadModel, positionalIsModelPathFriend) {
  // A lone positional stands in for --model; --model wins if both
  // are given (the preprocessor drops the positional so boost sees a clean
  // list and doesn't report "cannot be specified more than once").
  auto c = parseModelLoadConfig({"gnmt", "-d", "CPU"});
  EXPECT_EQ(c.model, "gnmt");

  auto both = parseModelLoadConfig(
      {"--model", "/explicit/path", "positional",
       "-d", "CPU"});
  EXPECT_EQ(both.model, "/explicit/path");  // explicit flag wins over the positional
}

TEST(parseLoadModel, tooManyPositionalsError) {
  expectConfigErrorContains(
      [] { parseModelLoadConfig({"a", "b", "-e", "ovgenai", "-t", "llm", "-d", "CPU"}); },
      "positional");
}

TEST(parseLoadModel, unknownLongAndShortOptionReject) {
  // An unrecognised long, an unrecognised single-letter short, or (the options
  // ovgenai dropped: context-window / worker-max-respawns
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

// The three parameters removed in the subcommand redesign (d56480a: --engine and
// --model-type; 9537b7f: --config-blocks). None survive it, so each is now rejected
// as "unknown option" (not silently accepted and ignored) -- a load flag, a stale
// free-form block, and the device/mode selector that the per-pipeline subcommand
// name replaced respectively. Pinned so someone cannot quietly reintroduce one.
TEST(parseLoadModel, removedObsoleteFlagsAreRejected) {
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--engine", "x"})); },
                            "unknown option");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--model-type", "x"})); },
                            "unknown option");
  expectConfigErrorContains([] { parseModelLoadConfig(validMinAppend({"--config-blocks", "x"})); },
                            "unknown option");
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
      "--model", "/opt/x",
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

TEST(parseLoadModel, runtimeAndSchedulerFlagsRoundThrough) {
  // The runtime flags the LOAD parser now accepts (known, not "unknown option")
  // are read by the runtime parser;  --max-num-seqs by the scheduler parser.
  auto rt = parseModelRuntimeConfig(validMinAppend({
      "--kv-cache-precision", "fp16",
      "--performance-hint-num-requests", "4",
      "--enable-cpu-pinning", "true",
  }));
  ASSERT_TRUE(rt.kv_cache_precision.has_value());
  EXPECT_EQ(*rt.kv_cache_precision, std::string("fp16"));
  ASSERT_TRUE(rt.performance_hint_num_requests.has_value());
  EXPECT_EQ(*rt.performance_hint_num_requests, std::size_t(4));
  ASSERT_TRUE(rt.enable_cpu_pinning.has_value());
  EXPECT_EQ(*rt.enable_cpu_pinning, bool(true));

  auto sc = parseModelSchedulerConfig(validMinAppend({"--max-num-seqs", "2"}));
  ASSERT_TRUE(sc.max_num_seqs.has_value());
  EXPECT_EQ(*sc.max_num_seqs, std::size_t(2));

  // A hint value outside OpenVINO's accepted set is rejected (the SDPA-vs-paged
  // gate elsewhere keys on attention_backend;  a bad performance-hint is the
  // clearest "must be one of" the enum validator can throw).
  expectConfigErrorContains(
      [] { parseModelRuntimeConfig(validMinAppend({"--performance-hint", "nope"})); },
      "must be one of");
}

TEST(jsonRoundTrip, fullConfig) {
  std::vector<std::string> args = {
    "--model", "/opt/qwen3-8b-a3b", "--device", "GPU",
    "--cache-dir", "/var/ovcache",
      "--draft-model", "/opt/draft", "--draft-device", "CPU",
      "--num-assistant-tokens", "4", "--assistant-confidence-threshold", "0.5",
      "--tool-call-parser", "qwen35",
      "--kv-cache-precision", "bf16",
      "--max-num-seqs", "2",
  };
  // Each sibling config parses the same arg list, reading only its own section.
  auto load  = parseModelLoadConfig(args);
  auto rt    = parseModelRuntimeConfig(args);
  auto sched = parseModelSchedulerConfig(args);
  auto req   = parseModelRequestConfig(args);
  // Round-trip each through its own JSON and assert equality.
  EXPECT_EQ(load,  modelLoadConfigFromJson(modelLoadConfigToJson(load)));
  EXPECT_EQ(rt,    modelRuntimeConfigFromJson(modelRuntimeConfigToJson(rt)));
  EXPECT_EQ(sched, modelSchedulerConfigFromJson(modelSchedulerConfigToJson(sched)));
  EXPECT_EQ(req,   modelRequestConfigFromJson(modelRequestConfigToJson(req)));
  // and the runtime config actually captured the one set field
  ASSERT_TRUE(rt.kv_cache_precision.has_value());
  EXPECT_EQ(*rt.kv_cache_precision, std::string("bf16"));
  // the emitted load JSON must not carry the three fields ovgenai deliberately dropped
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
  // ovgenai dropped (context_window / worker_max_respawns / worker_line_limit):
  // known keys are read, the rest simply
  // ignored (not fatal).
  auto c = modelLoadConfigFromJson(
      R"({"model":"/opt/qwen","device":"CPU","context_window":"auto",
          "worker_max_respawns":3,"worker_line_limit":65536,"some_foreign_key":true})");
  EXPECT_EQ(c.model, "/opt/qwen");
  // re-serializing emits none of the dropped/foreign keys at all
  std::string js = modelLoadConfigToJson(c);
  EXPECT_EQ(js.find("context_window"), std::string::npos);
  EXPECT_EQ(js.find("worker_max_respawns"), std::string::npos);
  EXPECT_EQ(js.find("worker_line_limit"), std::string::npos);
  EXPECT_EQ(js.find("some_foreign_key"), std::string::npos);
}

TEST(jsonRoundTrip, derivesNameWhenUnsetAndStillRequiresRest) {
  auto c = modelLoadConfigFromJson(
      R"({"model":"/opt/qwen3-8b-a3b/","device":"NPU"})");
  EXPECT_EQ(c.model, "/opt/qwen3-8b-a3b/");  // path itself is kept verbatim

  // A model-less config is still missing-required, even in the tolerant reader.
  expectConfigErrorContains(
      [] { modelLoadConfigFromJson(R"({"device":"CPU"})"); },
      "model");
}

TEST(jsonRoundTrip, modelPathWrongTypeRejects) {
  expectConfigErrorContains(
      [] { modelLoadConfigFromJson(R"({"model":3,"device":"CPU"})"); },
      "model");
}
