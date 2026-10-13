// Unit tests for ovgenai's ModelSchedulerConfig (src/main/model_scheduler_config.{hpp,cpp}) --
// the KV-cache / batching knobs the pipeline is built with.  This is a dedicated,
// self-contained coverage of the class, complementing (and, where the engine-config
// layer's genai_config_test.cpp was thinner, deepening) the scheduler coverage there.
//
// The engine-config layer already covers modelSchedulerConfigFromJson over all 10
// keys + the "at-least-one-field" hasAnySet gate + a representative toSchedulerConfig
// forwarding case.  This file closes the surfaces that left gaps:
//   * parseModelSchedulerConfig (the CLI parser): EVERY one of the 10 flags, each by
//     its value-kind -- 6 size_t (the two num / block / cache-size / interval knobs),
//     4 bool -- plus the unsigned-reject-negative and bool true/false / reject-nonsense
//     rules (it, like the runtime parser, requires no model/device);
//   * modelSchedulerConfigToJson + fromJson round-tripping ALL 10 fields together
//     (round-tripping only ever proved a single field before) + the reader's
//     per-kind wrong-type / malformed / non-object -> ConfigError;
//   * toSchedulerConfig forwarding for the five fields the engine-config test did not
//     reach (cache_size / max_num_seqs / num_linear_attention_blocks / use_cache_eviction
//     / use_sparse_attention), plus the one rule-vs-exception the engine keeps:
//     a size_t forwards only when nonzero, a bool only when true, EXCEPT
//     cache_interval_multiplier forwards on has_value() alone (0 is valid when prefix
//     caching is off -- the engine's own validate() enforces the precondition);
//   * the boost "Scheduler Options" section this class owns (header + a sample flag
//     from each kind + long-form only, since every scheduler name's first letter --
//     c/m/n/u -- is already claimed as a single-letter prefix by the load / request
//     sections, so they keep long form);
//   * the class's hasAnySet() (unlike the runtime config there are no string fields,
//     so it is a plain "any field non-nullopt" gate).
//
// Field name == the JSON key == the (dashed) CLI flag, all in alphabetical order,
// mirroring the header so a regression is line-local.  ov::genai::SchedulerConfig
// is public: only cache_interval_multiplier is std::optional (nullopt = auto),
// the rest are plain size_t / bool at the engine's own defaults
// (max_num_batched_tokens/max_num_seqs = 256; the block / cache-size / linear
// fields = 0; dynamic_split_fuse = true; the other bools = false).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/genai/scheduler_config.hpp>  // the typed config toSchedulerConfig builds

#include "../main/config.hpp"
#include "../main/model_scheduler_config.hpp"

using namespace ovgenai::config;

namespace {

// A parse/validation is expected to throw ConfigError mentioning `needle`.
void expectSchedulerErrorContains(std::function<void()> fn, const std::string& needle) {
  bool got = false;
  try {
    fn();
  } catch (const ConfigError& e) {
    got = std::string(e.what()).find(needle) != std::string::npos;
  }
  EXPECT_TRUE(got) << "expected a ConfigError mentioning '" << needle << "', but none did";
}

// A ModelSchedulerConfig with all 10 fields set to representative, clean values,
// distinct and all unlike the engine's own defaults, so the toSchedulerConfig
// forward path and the JSON round-trip have unambiguous values to assert.
ModelSchedulerConfig allFieldsSet() {
  ModelSchedulerConfig c;
  c.cache_interval_multiplier   = std::size_t(4);   // optional; the one exception
  c.cache_size                  = std::size_t(8);
  c.dynamic_split_fuse          = true;
  c.enable_prefix_caching       = true;
  c.max_num_batched_tokens      = std::size_t(1024);
  c.max_num_seqs               = std::size_t(128);
  c.num_kv_blocks              = std::size_t(2048);
  c.num_linear_attention_blocks = std::size_t(16);
  c.use_cache_eviction         = true;
  c.use_sparse_attention       = true;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// parseModelSchedulerConfig: every one of the 10 flags, each by value-kind.
// ---------------------------------------------------------------------------
TEST(scheduler_parse, allTenFlagsFromCli) {
  auto c = parseModelSchedulerConfig({
      "--cache-interval-multiplier", "4",
      "--cache-size", "8",
      "--dynamic-split-fuse", "true",
      "--enable-prefix-caching", "true",
      "--max-num-batched-tokens", "1024",
      "--max-num-seqs", "128",
      "--num-kv-blocks", "2048",
      "--num-linear-attention-blocks", "16",
      "--use-cache-eviction", "true",
      "--use-sparse-attention", "true"});
  EXPECT_EQ(*c.cache_interval_multiplier,   std::size_t(4));
  EXPECT_EQ(*c.cache_size,                  std::size_t(8));
  EXPECT_EQ(*c.dynamic_split_fuse,          bool(true));
  EXPECT_EQ(*c.enable_prefix_caching,       bool(true));
  EXPECT_EQ(*c.max_num_batched_tokens,      std::size_t(1024));
  EXPECT_EQ(*c.max_num_seqs,               std::size_t(128));
  EXPECT_EQ(*c.num_kv_blocks,              std::size_t(2048));
  EXPECT_EQ(*c.num_linear_attention_blocks, std::size_t(16));
  EXPECT_EQ(*c.use_cache_eviction,         bool(true));
  EXPECT_EQ(*c.use_sparse_attention,       bool(true));
  EXPECT_TRUE(c.hasAnySet());
}

// The six size_t knobs go through parseSize, which rejects a negative value
// (the two *num*/block/cache/interval knobs mirror that; only the thread knobs
// on the *runtime* config accept a sign -- these never do).
TEST(scheduler_parse, sizeTFieldsRejectNegative) {
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--cache-interval-multiplier", "-1"}); },
                               "negative");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--cache-size", "-1"}); },
                               "negative");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--max-num-batched-tokens", "-1"}); },
                               "negative");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--max-num-seqs", "-1"}); },
                               "negative");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--num-kv-blocks", "-1"}); },
                               "negative");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--num-linear-attention-blocks", "-1"}); },
                               "negative");
}

// The four bool knobs: true / false accepted (incl. the 1 / 0 / on / off / yes /
// no spellings), a nonsense value rejected.  Unlike the runtime parser (whose
// readBool treats an empty value as "unset"), the scheduler's parseBool is used
// directly, so an EMPTY value is an "invalid boolean" -- covered here, a useful
// pin so it is not silently loosened later.
TEST(scheduler_parse, boolFieldsAcceptValuesRejectNonsenseAndEmpty) {
  auto c = parseModelSchedulerConfig({
      "--dynamic-split-fuse", "false",
      "--enable-prefix-caching", "1",
      "--use-cache-eviction", "off",
      "--use-sparse-attention", "yes"});
  EXPECT_EQ(*c.dynamic_split_fuse,    bool(false));
  EXPECT_EQ(*c.enable_prefix_caching, bool(true));
  EXPECT_EQ(*c.use_cache_eviction,    bool(false));
  EXPECT_EQ(*c.use_sparse_attention,  bool(true));

  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--dynamic-split-fuse", "maybe"}); },
                               "invalid boolean");
  expectSchedulerErrorContains([] { parseModelSchedulerConfig({"--use-sparse-attention", ""}); },
                               "invalid boolean");  // scheduler (unlike runtime) does not clear-on-empty
}

// ---------------------------------------------------------------------------
// hasAnySet(): no string fields, so it is the plain "any field non-nullopt"
// gate -- a field set to 0 / false still counts (contrast: the runtime config's
// string fields count only when non-empty).
// ---------------------------------------------------------------------------
TEST(scheduler_hasAnySet, defaultIsFalse) {
  EXPECT_FALSE(ModelSchedulerConfig{}.hasAnySet());  // all 10 nullopt by default
}

TEST(scheduler_hasAnySet, anyFieldMakingItTrueEvenIfZeroOrFalse) {
  ModelSchedulerConfig a; a.cache_size = std::size_t(0);     // 0 still counts as set
  EXPECT_TRUE(a.hasAnySet());
  ModelSchedulerConfig b; b.use_sparse_attention = false;    // false still counts as set
  EXPECT_TRUE(b.hasAnySet());
}

// ---------------------------------------------------------------------------
// toSchedulerConfig(): forward a field only when truthy (size_t nonzero, bool
// true) -- EXCEPT cache_interval_multiplier, forwarded on has_value() alone (0 is
// valid when prefix caching is off; the engine's own validate() enforces the
// other preconditions).  A falsy / nullopt field is NOT forwarded, leaving the
// engine's own default (256 / true / false / 0) in place.
// ---------------------------------------------------------------------------
TEST(scheduler_toSchedulerConfig, forwardsEveryTruthyField) {
  auto c = allFieldsSet();          // all 10 set to distinct, non-default values
  auto ov = c.toSchedulerConfig();
  ASSERT_TRUE(ov.cache_interval_multiplier.has_value());     // the optional one
  EXPECT_EQ(ov.cache_interval_multiplier.value(), std::size_t(4));
  EXPECT_EQ(ov.cache_size,                  std::size_t(8));
  EXPECT_EQ(ov.dynamic_split_fuse,          bool(true));
  EXPECT_EQ(ov.enable_prefix_caching,       bool(true));
  EXPECT_EQ(ov.max_num_batched_tokens,      std::size_t(1024));
  EXPECT_EQ(ov.max_num_seqs,               std::size_t(128));
  EXPECT_EQ(ov.num_kv_blocks,              std::size_t(2048));
  EXPECT_EQ(ov.num_linear_attention_blocks, std::size_t(16));
  EXPECT_EQ(ov.use_cache_eviction,         bool(true));
  EXPECT_EQ(ov.use_sparse_attention,       bool(true));
}

TEST(scheduler_toSchedulerConfig, zeroKeptAsDefaultExceptTheIntervalException) {
  // A size_t set to 0 is skipped (0 is its "leave the engine default" sentinel) --
  // provable for the two fields whose engine default is NOT 0: max_num_batched_tokens
  // (256) and (via the same rule) the other size_t fields keep their 0 default.
  ModelSchedulerConfig c;
  c.max_num_batched_tokens = std::size_t(0);   // rule: skip -> stays at the engine default 256
  c.cache_interval_multiplier = std::size_t(0);// the exception: 0 is forwarded verbatim
  c.use_cache_eviction       = false;          // rule: skip -> stays at the engine default false
  auto ov = c.toSchedulerConfig();

  ASSERT_NE(ov.max_num_batched_tokens, std::size_t(0));   // not 0 -> the default (256) was kept
  EXPECT_EQ(ov.use_cache_eviction,     bool(false));
  ASSERT_TRUE(ov.cache_interval_multiplier.has_value());  // the ONLY field forwarded at 0
  EXPECT_EQ(ov.cache_interval_multiplier.value(), std::size_t(0));
}

// ---------------------------------------------------------------------------
// The JSON round-trip (the flat "scheduler_config" object show-config emits and a
// LOAD message may carry): to-JSON emits only the set fields (all-nullopt -> "{}");
// from-JSON reads them back; the two agree for the full 10-field set -- the gap
// the previous tests left (they only ever round-tripped a single field).
// ---------------------------------------------------------------------------
TEST(scheduler_json, allTenFieldsRoundTrip) {
  ModelSchedulerConfig c = allFieldsSet();
  std::string js = modelSchedulerConfigToJson(c);
  ModelSchedulerConfig back = modelSchedulerConfigFromJson(js);
  EXPECT_EQ(c, back);
  // every key is actually emitted (one per field, an all-unset config would drop it)
  EXPECT_NE(js.find("\"cache_interval_multiplier\""), std::string::npos);
  EXPECT_NE(js.find("\"cache_size\""), std::string::npos);
  EXPECT_NE(js.find("\"dynamic_split_fuse\""), std::string::npos);
  EXPECT_NE(js.find("\"enable_prefix_caching\""), std::string::npos);
  EXPECT_NE(js.find("\"max_num_batched_tokens\""), std::string::npos);
  EXPECT_NE(js.find("\"max_num_seqs\""), std::string::npos);
  EXPECT_NE(js.find("\"num_kv_blocks\""), std::string::npos);
  EXPECT_NE(js.find("\"num_linear_attention_blocks\""), std::string::npos);
  EXPECT_NE(js.find("\"use_cache_eviction\""), std::string::npos);
  EXPECT_NE(js.find("\"use_sparse_attention\""), std::string::npos);
}

TEST(scheduler_json, allUnsetIsEmptyObject) {
  EXPECT_EQ(modelSchedulerConfigToJson(ModelSchedulerConfig{}), std::string("{}"));
  EXPECT_FALSE(modelSchedulerConfigFromJson("{}").hasAnySet());
}

TEST(scheduler_json, foreignKeysAreSilentlyIgnored) {
  // A key with no matching field is dropped (not a hard error) -- matches the
  // sibling configs' tolerance of a wider schema; a known key is still read.
  auto c = modelSchedulerConfigFromJson(R"({"max_num_seqs":128,"some_foreign_key":7})");
  ASSERT_TRUE(c.max_num_seqs.has_value());
  EXPECT_EQ(*c.max_num_seqs, std::size_t(128));
  EXPECT_FALSE(c.cache_size.has_value());  // the foreign key surfaced as no field
  EXPECT_TRUE(c.hasAnySet());
}

TEST(scheduler_json, wrongTypePerFieldIsConfigError) {
  // a size_t field given a JSON string
  expectSchedulerErrorContains([] { modelSchedulerConfigFromJson(R"({"max_num_seqs":"two"})"); },
                               "expected an integer");
  // a size_t field given a negative -> the non-negative rule
  expectSchedulerErrorContains([] { modelSchedulerConfigFromJson(R"({"max_num_seqs":-1})"); },
                               "must be non-negative");
  // a bool field given a string
  expectSchedulerErrorContains([] { modelSchedulerConfigFromJson(R"({"enable_prefix_caching":"true"})"); },
                               "expected a boolean");
}

TEST(scheduler_json, malformedJsonAndNonObjectRejected) {
  expectSchedulerErrorContains([] { modelSchedulerConfigFromJson("{not json"); }, "not valid JSON");
  expectSchedulerErrorContains([] { modelSchedulerConfigFromJson(R"([])"); }, "expected a JSON object");
}

// ---------------------------------------------------------------------------
// The boost "Scheduler Options" section this class owns -- a light check: the
// caption, one representative flag from each of the two value-kinds, and (since
// every scheduler name's first letter is claimed as a prefix by the load /
// request sections) long-form only -- asserted by the absence of boost's
// " [ --" short-alias marker, far more robust than a raw ", " (a comma-space
// appears throughout the flag descriptions).
// ---------------------------------------------------------------------------
TEST(scheduler_optionsDesc, hasHeaderAndASampleFlagFromEachKind) {
  std::string s;  { std::ostringstream os; os << ModelSchedulerConfig::schedulerOptionsDesc(); s = os.str(); }
  EXPECT_NE(s.find("Scheduler Options"), std::string::npos);            // the section caption
  EXPECT_NE(s.find("--max-num-seqs"),   std::string::npos);             // a size_t flag
  EXPECT_NE(s.find("--enable-prefix-caching"), std::string::npos);      // a bool flag
  EXPECT_EQ(s.find(" [ --"), std::string::npos);                       // no single-letter shorts
}
