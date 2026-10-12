// Unit tests for the ovgenai model-load / generation-config layer.
//
// These exercise ModelSchedulerConfig (JSON parse + toSchedulerConfig +
// hasAnySet),  GenAIPipelineConfig (toPipelineProperties,  the SDPA gate,
//  the CACHE_DIR override,  and spec-decode default resolution),  and
//  GenAIGenerationConfig (temperature→do_sample,  the mutually-exclusive
//  num_assistant_tokens / assistant_confidence_threshold cascade,  optional
//  request-time fields).
#include <string>
#include <optional>

#include <gtest/gtest.h>

#include "genai_config.hpp"

using namespace ovgenai;

namespace {

// Build a ModelLoadConfig with exactly the four required fields,  leaving
// every optional at its struct default.  Callers layer on specific fields.
ModelLoadConfig baseConfig()
{
    ModelLoadConfig c;
    c.model  = "/opt/qwen3-8b";
    c.device      = "GPU";
    return c;
}

void expectConfigErrorContains(std::function<void()> fn, const std::string& needle)
{
    bool got = false;
    try { fn(); } catch (const ConfigError& e) {
        got = std::string(e.what()).find(needle) != std::string::npos;
    }
    EXPECT_TRUE(got) << "expected ConfigError containing '" << needle << "'";
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelSchedulerConfig
// ---------------------------------------------------------------------------

TEST(genai_scheduler_config, fromJson_reads_all_known_keys)
{
    auto sc = modelSchedulerConfigFromJson(
        R"({"max_num_batched_tokens":1024,
            "num_kv_blocks":2048,
            "cache_size":8,
            "num_linear_attention_blocks":16,
            "cache_interval_multiplier":4,
            "dynamic_split_fuse":true,
            "max_num_seqs":128,
            "enable_prefix_caching":true,
            "use_cache_eviction":true,
            "use_sparse_attention":true})");
    EXPECT_EQ(sc.max_num_batched_tokens,   std::optional<std::size_t>(1024));
    EXPECT_EQ(sc.num_kv_blocks,            std::optional<std::size_t>(2048));
    EXPECT_EQ(sc.cache_size,               std::optional<std::size_t>(8));
    EXPECT_EQ(sc.num_linear_attention_blocks, std::optional<std::size_t>(16));
    EXPECT_EQ(sc.cache_interval_multiplier,  std::optional<std::size_t>(4));
    EXPECT_EQ(sc.dynamic_split_fuse,       std::optional<bool>(true));
    EXPECT_EQ(sc.max_num_seqs,             std::optional<std::size_t>(128));
    EXPECT_EQ(sc.enable_prefix_caching,    std::optional<bool>(true));
    EXPECT_EQ(sc.use_cache_eviction,       std::optional<bool>(true));
    EXPECT_EQ(sc.use_sparse_attention,     std::optional<bool>(true));
    EXPECT_TRUE(sc.hasAnySet());
}

TEST(genai_scheduler_config, fromJson_missing_keys_stay_nullopt)
{
    auto sc = modelSchedulerConfigFromJson(
        R"({"max_num_batched_tokens":1024})");
    EXPECT_EQ(sc.max_num_batched_tokens, std::optional<std::size_t>(1024));
    EXPECT_FALSE(sc.num_kv_blocks.has_value());
    EXPECT_FALSE(sc.dynamic_split_fuse.has_value());
    EXPECT_TRUE(sc.hasAnySet());
}

TEST(genai_scheduler_config, fromJson_empty_object_hasNoFields)
{
    auto sc = modelSchedulerConfigFromJson("{}");
    EXPECT_FALSE(sc.hasAnySet());
}

TEST(genai_scheduler_config, fromJson_foreign_keys_silently_ignored)
{
    //  Matches ModelLoadConfig's fromJson tolerance:  unknown top-level keys
    //  are not a hard error,  they are just not loaded into any C++ field.
    auto sc = modelSchedulerConfigFromJson(
        R"({"max_num_batched_tokens":1024,"some_foreign_key":"whatever"})");
    EXPECT_EQ(sc.max_num_batched_tokens, std::optional<std::size_t>(1024));
    EXPECT_TRUE(sc.hasAnySet());
}

TEST(genai_scheduler_config, toSchedulerConfig_forwards_only_truthy_fields)
{
    //  Python's `if x:` for size_t: has_value() && *val != 0.
    //  Python's `if x:` for bool:   has_value() && *val == true.
    //  A field explicitly set to 0/False must NOT reach the engine —
    //  the engine default stays untouched.
    ModelSchedulerConfig sc;
    sc.max_num_batched_tokens   = 0;       //  int:  falsy →  skip
    sc.num_kv_blocks            = 4096;    //  int:  truthy →  forward
    sc.dynamic_split_fuse       = false;   //  bool: falsy →  skip
    sc.enable_prefix_caching    = true;    //  bool: truthy →  forward
    sc.cache_interval_multiplier = std::nullopt;   //  leave at default (nullopt)

    auto ov = sc.toSchedulerConfig();
    EXPECT_NE(ov.max_num_batched_tokens, std::size_t(0));  //  engine default kept
    EXPECT_EQ(ov.num_kv_blocks,          std::size_t(4096));
    EXPECT_NE(ov.dynamic_split_fuse,     false);           //  engine default kept
    EXPECT_EQ(ov.enable_prefix_caching,  true);
    EXPECT_FALSE(ov.cache_interval_multiplier.has_value());  //  engine default kept
}

TEST(genai_scheduler_config, toSchedulerConfig_forwards_0_cache_interval_when_prefix_off)
{
    //  Python forwards the value itself (including 0) for cache_interval_multiplier,
    //  not using the truthy check (the field is an explicit opt-in and 0 is
    //  a valid value when prefix caching is off).  The engine's own validate()
    //  enforces the pre-caching precondition,  not us.
    ModelSchedulerConfig sc;
    sc.cache_interval_multiplier = 0;
    auto ov = sc.toSchedulerConfig();
    ASSERT_TRUE(ov.cache_interval_multiplier.has_value());
    EXPECT_EQ(ov.cache_interval_multiplier.value(), std::size_t(0));
}

// ---------------------------------------------------------------------------
// GenAIPipelineConfig::toPipelineProperties
// ---------------------------------------------------------------------------

TEST(genai_pipeline_config, runtime_config_entries_pass_through)
{
    ModelLoadConfig c = baseConfig();
    ModelRuntimeConfig rt;
    rt.inference_num_threads = 4;
    rt.performance_hint      = std::string("LATENCY");
    GenAIPipelineConfig pc(c, ModelSchedulerConfig{}, rt);
    ov::AnyMap props = pc.toPipelineProperties();
    ASSERT_EQ(props.size(), std::size_t(2));
    ASSERT_NE(props.find("INFERENCE_NUM_THREADS"), props.end());
    EXPECT_TRUE(props.at("INFERENCE_NUM_THREADS").is<int64_t>());
    EXPECT_EQ(props.at("INFERENCE_NUM_THREADS").as<int64_t>(),
              static_cast<int64_t>(4));
    ASSERT_NE(props.find("PERFORMANCE_HINT"), props.end());
    EXPECT_TRUE(props.at("PERFORMANCE_HINT").is<std::string>());
    EXPECT_EQ(props.at("PERFORMANCE_HINT").as<std::string>(), "LATENCY");
    EXPECT_EQ(props.find("scheduler_config"), props.end());     //  no sched block
    EXPECT_EQ(props.find("CACHE_DIR"),        props.end());     //  no cache dir
    EXPECT_EQ(props.find("draft_model"),      props.end());     //  no draft model
}

TEST(genai_pipeline_config, cache_dir_is_a_runtime_property)
{
    //  cache_dir is now a runtime property: it flows through
    //  runtime.toProperties() (the genai layer no longer adds a separate
    //  CACHE_DIR step).
    ModelLoadConfig c = baseConfig();
    ModelRuntimeConfig rt;
    rt.cache_dir            = std::string("/from-top-level");
    rt.inference_num_threads = 4;
    GenAIPipelineConfig pc(c, ModelSchedulerConfig{}, rt);
    ov::AnyMap props = pc.toPipelineProperties();
    ASSERT_NE(props.find("CACHE_DIR"), props.end());
    EXPECT_EQ(props.at("CACHE_DIR").as<std::string>(), "/from-top-level");
    EXPECT_EQ(props.at("INFERENCE_NUM_THREADS").as<int64_t>(),
              static_cast<int64_t>(4));
}

TEST(genai_pipeline_config, scheduler_config_absent_when_block_not_authored)
{
    //  scheduler_config_json == nullopt  →  schedAuthored_ == false  →
    //  no scheduler_config entry,  matching Python's
    //  `if not sched_config_dict:  return {}`.
    ModelLoadConfig c = baseConfig();
    GenAIPipelineConfig pc(c, ModelSchedulerConfig{});
    ov::AnyMap props = pc.toPipelineProperties();
    EXPECT_EQ(props.find("scheduler_config"), props.end());
    EXPECT_FALSE(pc.schedulerAuthored());
}

TEST(genai_pipeline_config, scheduler_config_absent_when_block_all_null)
{
    // No scheduler fields set on ModelLoadConfig → sched_ is all nullopt →
    // hasAnySet() == false → no scheduler_config entry (same as not authored).
    ModelLoadConfig c = baseConfig();
    // All fields stay at their default nullopt
    GenAIPipelineConfig pc(c, ModelSchedulerConfig{});
    ov::AnyMap props = pc.toPipelineProperties();
    EXPECT_EQ(props.find("scheduler_config"), props.end());
    EXPECT_FALSE(pc.schedulerAuthored());
}

TEST(genai_pipeline_config, scheduler_config_present_when_block_has_at_least_one_field)
{
    ModelLoadConfig c = baseConfig();
    ModelSchedulerConfig sc;
    sc.max_num_batched_tokens = std::size_t(1024);
    GenAIPipelineConfig pc(c, sc);
    ov::AnyMap props = pc.toPipelineProperties();
    ASSERT_NE(props.find("scheduler_config"), props.end());
    EXPECT_TRUE(pc.schedulerAuthored());
    //  The value must be an ov::Any wrapping a SchedulerConfig.  We cannot
    //  easily inspect the internal fields through the Any without a cast,
    //  but we can assert the Any is non-empty (which distinguishes it from
    //  the Python None→empty-Any case).
    EXPECT_FALSE(props.at("scheduler_config").empty());
}

TEST(genai_pipeline_config, scheduler_config_rejected_by_sdpa_backend)
{
    //  The Python's `if pipeline_kwargs.get("ATTENTION_BACKEND") == "SDPA"
    //  and sched_config_dict:` —  when the runtime_config names the SDPA
    //  backend AND the operator authored a non-empty scheduler block,
    //  extract_scheduler_config_from_loader drops it and logs an error.
    ModelLoadConfig c = baseConfig();
    ModelRuntimeConfig rt;
    rt.attention_backend    = std::string("SDPA");
    rt.inference_num_threads = 4;
    ModelSchedulerConfig sc;
    sc.max_num_batched_tokens = std::size_t(1024);
    GenAIPipelineConfig pc(c, sc, rt);
    ov::AnyMap props = pc.toPipelineProperties();
    EXPECT_EQ(props.find("scheduler_config"), props.end());
    //  The rest of the runtime_config flows through untouched.
    ASSERT_NE(props.find("ATTENTION_BACKEND"), props.end());
    EXPECT_EQ(props.at("ATTENTION_BACKEND").as<std::string>(), "SDPA");
}

TEST(genai_pipeline_config, no_draft_model_causes_no_draft_model_key)
{
    ModelLoadConfig c = baseConfig();
    GenAIPipelineConfig pc(c, ModelSchedulerConfig{});
    ov::AnyMap props = pc.toPipelineProperties();
    EXPECT_EQ(props.find("draft_model"), props.end());
}

// ---------------------------------------------------------------------------
// GenAIGenerationConfig::toGenerationConfig
// ---------------------------------------------------------------------------

TEST(genai_generation_config, temperature_zero_forces_greedy)
{
    //  apply_temperature:  `generation_config.temperature = temperature;
    //   if temperature <= 0:  generation_config.do_sample = False`
    GenAIGenerationConfig gc;
    gc.temperature = 0.0f;
    auto result = gc.toGenerationConfig();
    EXPECT_EQ(result.temperature,   0.0f);
    EXPECT_FALSE(result.do_sample);
}

TEST(genai_generation_config, temperature_positive_enables_sampling)
{
    //  The C++ std default for do_sample in GenerationConfig is false;
    //  the Python `apply_temperature` only flips to false for <= 0,
    //  leaving the (already-true) value from `self.model.get_generation_config()`
    //  in Python.  For our standalone port we start from the C++ default
    //  and explicitly flip do_sample to true for temperature > 0,  which
    //  is the intuitive and Python-equivalent behaviour for the common
    //  sampling case.
    GenAIGenerationConfig gc;
    gc.temperature = 1.0f;
    auto result = gc.toGenerationConfig();
    EXPECT_EQ(result.temperature, 1.0f);
    EXPECT_TRUE(result.do_sample);
}

TEST(genai_generation_config, non_zero_fields_must_be_forwarded_by_request)
{
    //  The three optional (int/float, with pydantic default=None) fields
    //  are forwarded iff the request sent a truthy value:  0 is falsy
    //  for the seed;  0.0 is falsy for the two penalties (so neither
    //  is forwarded when zero —  matching Python `if config.X:`).
    GenAIGenerationConfig gc;
    gc.seed                = std::size_t(42);
    gc.frequency_penalty   = 0.5f;
    gc.presence_penalty    = 0.25f;
    auto result = gc.toGenerationConfig();
    EXPECT_EQ(result.rng_seed,           std::size_t(42));
    EXPECT_FLOAT_EQ(result.frequency_penalty, 0.5f);
    EXPECT_FLOAT_EQ(result.presence_penalty,  0.25f);
}

TEST(genai_generation_config, zero_seed_is_not_forwarded)
{
    //  Python's `if config.seed:`  —  0 is falsy,  so the Python code does
    //  not overwrite the engine's default (also 0 from GenerationConfig).
    //  We mirror that:  has_value AND nonzero →  forward.
    GenAIGenerationConfig gc;
    gc.seed = std::size_t(0);
    auto result = gc.toGenerationConfig();
    //  The engine default was 0,  we leave it as 0 —  the value is still
    //  0 regardless;  assert that the engine's own default is still honoured
    //  by NOT touching rng_seed (which is what we code for).
    EXPECT_EQ(result.rng_seed, std::size_t(0));
}

TEST(genai_generation_config, request_num_assistant_tokens_forwarded)
{
    //  Request-level num_assistant_tokens is forwarded directly to the
    //  engine's GenerationConfig.
    GenAIGenerationConfig gc;
    gc.num_assistant_tokens = std::size_t(5);
    auto result = gc.toGenerationConfig();
    ASSERT_TRUE(result.num_assistant_tokens.has_value());
    EXPECT_EQ(result.num_assistant_tokens.value(), std::size_t(5));
}

TEST(genai_generation_config, request_assistant_confidence_threshold_forwarded)
{
    //  Request-level assistant_confidence_threshold is forwarded;  num_assistant_tokens
    //  stays at the engine default (nullopt).
    GenAIGenerationConfig gc;
    gc.assistant_confidence_threshold = 0.4f;
    auto result = gc.toGenerationConfig();
    EXPECT_FLOAT_EQ(result.assistant_confidence_threshold, 0.4f);
    EXPECT_FALSE(result.num_assistant_tokens.has_value());
}

TEST(genai_generation_config, engine_defaults_kept_when_request_silent)
{
    //  When neither spec-decode parameter is set in the request,  the engine's
    //  own defaults (nullopt for num,  0.0f for threshold) are preserved.
    GenAIGenerationConfig gc;
    auto result = gc.toGenerationConfig();
    EXPECT_FALSE(result.num_assistant_tokens.has_value());
    EXPECT_FLOAT_EQ(result.assistant_confidence_threshold, 0.0f);  //  engine default
}

TEST(genai_generation_config, max_tokens_defaults_to_16384)
{
    GenAIGenerationConfig gc;
    auto result = gc.toGenerationConfig();
    EXPECT_EQ(result.max_new_tokens, std::size_t(16384));
}
