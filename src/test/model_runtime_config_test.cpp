// Unit tests for ovgenai's ModelRuntimeConfig (src/main/model_runtime_config.{hpp,cpp}) --
// the first-class config added for the OpenVINO Runtime / compile options.  It is the
// single home for those options, exposed three ways out of one set of 28 std::optional
// fields, and this file covers all three surfaces plus the closed-set (enum) validation:
//   * parseModelRuntimeConfig (the CLI --flags, long-form only) -- every one of the 28
//     parameters, the value-kind each maps to, the "empty clears the field" rule, the
//     signed-vs-unsigned split for the thread knobs, and the case-insensitive enum set;
//   * runtimeOptionsDesc / usage (the boost section this class owns, spliced into every
//     subcommand's shared --help) -- lightly, via a flag-count / header check;
//   * hasAnySet() -- the "was this block authored" gate: string fields count only when
//     non-empty, bool/int/int32/int-double fields whenever has_value();
//   * toProperties() -- the ov::AnyMap a LLMPipeline / VLMPipeline / ... constructor
//     consumes: one entry under each field's UPPER_SNAKE property name, strings/bools/
//     ints/floats as ov::Any of the matching C++ type, an explicit false forwarded but
//     an empty string/ nullopt omitted so the engine default is left untouched;
//   * the JSON modelRuntimeConfigFromJson / modelRuntimeConfigToJson round-trip -- every
//     field, "foreign" keys silently ignored, a wrong JSON type per field is a ConfigError
//     (matching the tolerant-reader style of the sibling configs).
//
// The field name IS the lower-cased OpenVINO property name, so the CLI flag (dashed),
// the "runtime_config" JSON key, and the UPPER_SNAKE property key are all a deterministic
// transform of the one field -- the tests assert that invariant rather than re-listing it.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/any.hpp>  // ov::Any / ov::AnyMap (what toProperties builds)

#include "../main/config.hpp"
#include "../main/model_runtime_config.hpp"

using namespace ovgenai::config;

namespace {

// A parse/validation is expected to throw ConfigError mentioning `needle`.
void expectRuntimeErrorContains(std::function<void()> fn, const std::string& needle) {
  bool got = false;
  try {
    fn();
  } catch (const ConfigError& e) {
    got = std::string(e.what()).find(needle) != std::string::npos;
  }
  EXPECT_TRUE(got) << "expected a ConfigError mentioning '" << needle << "', but none did";
}

// A ModelRuntimeConfig with all 28 fields set to representative, clean values, one
// per value-kind.  Used as the model for the from-JSON -> to-JSON -> from-JSON round-trip
// (and, where asserted, the to-properties forwards).  The "true" and "false" bools are
// both present on purpose: a false is a real, explicit value that must still be
// forwarded (0 / false are not the "unset" sentinel for a bool, unlike an empty string).
ModelRuntimeConfig allFieldsSet() {
  ModelRuntimeConfig c;
  c.activations_scale_factor    = static_cast<double>(0.5);  // the one double field
  c.attention_backend           = std::string("PA");
  c.auto_batch_timeout          = static_cast<std::size_t>(33);
  c.cache_dir                  = std::string("/data/cache");
  c.cache_mode                = std::string("OPTIMIZE_SPEED");
  c.compilation_num_threads    = 8;
  c.enable_cpu_pinning         = true;
  c.enable_cpu_reservation     = false;  // explicit false -> still forwarded
  c.enable_hyper_threading     = true;
  c.enable_mmap                = false;   // explicit false -> still forwarded
  c.enable_weightless          = true;
  c.execution_mode_hint        = std::string("PERFORMANCE");
  c.force_tbb_terminate        = false;   // explicit false -> still forwarded
  c.inference_num_threads      = 4;
  c.inference_precision_hint   = std::string("bf16");
  c.key_cache_group_size        = static_cast<std::size_t>(128);
  c.key_cache_precision         = std::string("u4");
  c.kv_cache_precision          = std::string("f16");
  c.log_level                  = std::string("LOG_INFO");
  c.model_priority             = std::string("HIGH");
  c.num_streams                = static_cast<std::size_t>(4);
  c.performance_hint           = std::string("THROUGHPUT");
  c.performance_hint_num_requests = static_cast<std::size_t>(7);
  c.scheduling_core_type        = std::string("PCORE_ONLY");
  c.value_cache_group_size       = static_cast<std::size_t>(64);
  c.value_cache_precision        = std::string("u8");
  c.weights_path                = std::string("/data/weights.bin");
  c.workload_type               = std::string("EFFICIENT");
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// parseModelRuntimeConfig: every one of the 28 parameters, and which C++ field /
// value-kind each flag lands in.  The enum fields are given mixed-case input so
// this also proves the case-insensitive -> canonical-form matching.
// ---------------------------------------------------------------------------
TEST(runtime_parse, allFlagsSetEveryField) {
  auto c = parseModelRuntimeConfig({
      "--activations-scale-factor", "0.5",
      "--attention-backend", "pa",
      "--auto-batch-timeout", "33",
      "--cache-dir", "/data/cache",
      "--cache-mode", "optimize_speed",
      "--compilation-num-threads", "8",
      "//  (the 6 bool knobs, a mix of true / false -- false is a real value)",
      "--enable-cpu-pinning", "true",
      "--enable-cpu-reservation", "false",
      "--enable-hyper-threading", "true",
      "--enable-mmap", "false",
      "--enable-weightless", "true",
      "--force-tbb-terminate", "false",
      "--execution-mode-hint", "performance",
      "--inference-num-threads", "4",
      "--inference-precision-hint", "bf16",
      "--key-cache-group-size", "128",
      "--key-cache-precision", "u4",
      "--kv-cache-precision", "f16",
      "--log-level", "log_info",
      "--model-priority", "high",
      "--num-streams", "4",
      "--performance-hint", "throughput",
      "--performance-hint-num-requests", "7",
      "--scheduling-core-type", "pcore_only",
      "--value-cache-group-size", "64",
      "--value-cache-precision", "u8",
      "--weights-path", "/data/weights.bin",
      // workload-type is a *free-form* string (not a closed enum), so the
      // operator's spelling is stored verbatim -- feed the canonical spelling.
      "--workload-type", "EFFICIENT"});

  // one line per field, so a regression in any one maps to a line that obviously should not change
  EXPECT_EQ(*c.activations_scale_factor, 0.5);
  EXPECT_EQ(*c.attention_backend, std::string("PA"));  // "pa" canonicalised -> "PA"
  EXPECT_EQ(*c.auto_batch_timeout, std::size_t(33));
  EXPECT_EQ(*c.cache_dir, std::string("/data/cache"));
  EXPECT_EQ(*c.cache_mode, std::string("OPTIMIZE_SPEED"));  // "optimize_speed" -> "OPTIMIZE_SPEED"
  EXPECT_EQ(*c.compilation_num_threads, std::int32_t(8));
  EXPECT_EQ(*c.enable_cpu_pinning, bool(true));
  EXPECT_EQ(*c.enable_cpu_reservation, bool(false));  // false stored (not cleared)
  EXPECT_EQ(*c.enable_hyper_threading, bool(true));
  EXPECT_EQ(*c.enable_mmap, bool(false));
  EXPECT_EQ(*c.enable_weightless, bool(true));
  EXPECT_EQ(*c.force_tbb_terminate, bool(false));
  EXPECT_EQ(*c.execution_mode_hint, std::string("PERFORMANCE"));
  EXPECT_EQ(*c.inference_num_threads, std::int32_t(4));
  EXPECT_EQ(*c.inference_precision_hint, std::string("bf16"));
  EXPECT_EQ(*c.key_cache_group_size, std::size_t(128));
  EXPECT_EQ(*c.key_cache_precision, std::string("u4"));
  EXPECT_EQ(*c.kv_cache_precision, std::string("f16"));
  EXPECT_EQ(*c.log_level, std::string("LOG_INFO"));
  EXPECT_EQ(*c.model_priority, std::string("HIGH"));
  EXPECT_EQ(*c.num_streams, std::size_t(4));
  EXPECT_EQ(*c.performance_hint, std::string("THROUGHPUT"));
  EXPECT_EQ(*c.performance_hint_num_requests, std::size_t(7));
  EXPECT_EQ(*c.scheduling_core_type, std::string("PCORE_ONLY"));
  EXPECT_EQ(*c.value_cache_group_size, std::size_t(64));
  EXPECT_EQ(*c.value_cache_precision, std::string("u8"));
  EXPECT_EQ(*c.weights_path, std::string("/data/weights.bin"));
  EXPECT_EQ(*c.workload_type, std::string("EFFICIENT"));
}

// An empty value clears a string field to "not set" (nullopt), so the engine's own
// default is left untouched -- the runtime config's uniform "unset" rule across value kinds.
TEST(runtime_parse, emptyStringClearsAStringField) {
  auto c = parseModelRuntimeConfig({"--cache-dir", ""});
  EXPECT_FALSE(c.cache_dir.has_value());
  EXPECT_TRUE(c.hasAnySet() == false);  // nothing else set -> the block is unauthored

  // a non-empty string is a real, set value
  auto c2 = parseModelRuntimeConfig({"--cache-dir", "/var/cache"});
  ASSERT_TRUE(c2.cache_dir.has_value());
  EXPECT_EQ(*c2.cache_dir, std::string("/var/cache"));
  EXPECT_TRUE(c2.hasAnySet());
}

// The five *unsigned* (size_t) knobs reject a negative value; the two *signed*
// thread knobs accept it (0 = "all cores", negative = all-but-N).
TEST(runtime_parse, unsignedFieldsRejectNegative) {
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--num-streams", "-1"}); },
                             "negative");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--auto-batch-timeout", "-1"}); },
                             "negative");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--key-cache-group-size", "-1"}); },
                             "negative");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--value-cache-group-size", "-1"}); },
                             "negative");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--performance-hint-num-requests", "-1"}); },
                             "negative");
}

TEST(runtime_parse, signedThreadFieldsAcceptNegativeAndZero) {
  // 0 and a negative are both valid for the two signed thread knobs.
  auto c = parseModelRuntimeConfig({"--inference-num-threads", "0",
                                    "--compilation-num-threads", "-4"});
  ASSERT_TRUE(c.inference_num_threads.has_value());
  EXPECT_EQ(*c.inference_num_threads, std::int32_t(0));
  ASSERT_TRUE(c.compilation_num_threads.has_value());
  EXPECT_EQ(*c.compilation_num_threads, std::int32_t(-4));
}

// A value outside a hint / enum field's closed set is a ConfigError naming the
// accepted members (boost does not validate the open string; our reader does).
TEST(runtime_parse, badEnumValueIsRejectedMustBeOneOf) {
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--attention-backend", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--cache-mode", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--execution-mode-hint", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--log-level", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--model-priority", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--performance-hint", "nope"}); },
                             "must be one of");
  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--scheduling-core-type", "nope"}); },
                             "must be one of");
}

// A closed-set value is matched case-insensitively, but the *canonical* (UPPER_SNAKE)
// OpenVINO spelling is what is stored, so the engine parses exactly what it expects.
TEST(runtime_parse, enumValuesCanonicaliseCaseInsensitively) {
  auto c = parseModelRuntimeConfig({
      "--attention-backend", "Pa",
      "--cache-mode", "Optimize_Speed",
      "--execution-mode-hint", "accuraCy",
      "--log-level", "Log_Info",
      "--model-priority", "High",
      "--performance-hint", "Cumulative_Throughput",
      "--scheduling-core-type", "ecore_Only"});
  EXPECT_EQ(*c.attention_backend, std::string("PA"));
  EXPECT_EQ(*c.cache_mode, std::string("OPTIMIZE_SPEED"));
  EXPECT_EQ(*c.execution_mode_hint, std::string("ACCURACY"));
  EXPECT_EQ(*c.log_level, std::string("LOG_INFO"));
  EXPECT_EQ(*c.model_priority, std::string("HIGH"));
  EXPECT_EQ(*c.performance_hint, std::string("CUMULATIVE_THROUGHPUT"));
  EXPECT_EQ(*c.scheduling_core_type, std::string("ECORE_ONLY"));
}

// The opposite behaviour for the *free-form* string fields: the operator's exact
// spelling is stored verbatim (an element type, a path, a free string), whereas
// the seven closed-set hint/enum fields canonicalise to OPENVINO's accepted form.
// An element-precision (kv_cache_precision and co.) is not a closed set the CLI
// polices -- the engine decides which element types a device actually supports --
// so that family is free-form, matching workload_type / the open string fields.
TEST(runtime_parse, freeFormStringsAreVerbatimWhileEnumsCanonicalise) {
  auto c = parseModelRuntimeConfig({
      "--performance-hint", "latency",             // closed set -> canonical "LATENCY"
      "--inference-precision-hint", "MixedCase",   // free form  -> verbatim "MixedCase"
  });
  ASSERT_TRUE(c.performance_hint.has_value());
  EXPECT_EQ(*c.performance_hint, std::string("LATENCY"));            // canonicalised
  ASSERT_TRUE(c.inference_precision_hint.has_value());
  EXPECT_EQ(*c.inference_precision_hint, std::string("MixedCase"));   // verbatim
}

// The six bool knobs: true/false accepted, a nonsense value rejected; empty clears.
TEST(runtime_parse, boolFieldsAcceptValuesRejectNonsense) {
  auto c = parseModelRuntimeConfig({"--enable-cpu-pinning", "true",
                                    "--enable-mmap", "false"});
  EXPECT_EQ(*c.enable_cpu_pinning, bool(true));
  EXPECT_EQ(*c.enable_mmap, bool(false));
  // the remaining four bools simply stay unset (nullopt) when not passed
  EXPECT_FALSE(c.enable_cpu_reservation.has_value());
  EXPECT_FALSE(c.enable_hyper_threading.has_value());
  EXPECT_FALSE(c.enable_weightless.has_value());
  EXPECT_FALSE(c.force_tbb_terminate.has_value());

  expectRuntimeErrorContains([] { parseModelRuntimeConfig({"--enable-cpu-pinning", "maybe"}); },
                             "invalid boolean");
}

// ---------------------------------------------------------------------------
// hasAnySet(): the "was this block authored" gate.
//   * string fields count only when non-empty (so an empty value is "not set");
//   * bool / int / int32 / double count whenever has_value() (incl. an explicit
//     false / 0 -- those are real values, not "unset").
// ---------------------------------------------------------------------------
TEST(runtime_hasAnySet, allUnsetIsFalse) {
  EXPECT_FALSE(ModelRuntimeConfig{}.hasAnySet());  // every field nullopt by default
}

TEST(runtime_hasAnySet, anyOneFieldSetMakesItTrue) {
  ModelRuntimeConfig s; s.cache_dir = std::string("/x");
  EXPECT_TRUE(s.hasAnySet());
  ModelRuntimeConfig bfalse; bfalse.enable_cpu_pinning = false;  // explicit false still counts
  EXPECT_TRUE(bfalse.hasAnySet());
  ModelRuntimeConfig i; i.num_streams = std::size_t(1);
  EXPECT_TRUE(i.hasAnySet());
  ModelRuntimeConfig i32; i32.inference_num_threads = 1;
  EXPECT_TRUE(i32.hasAnySet());
  ModelRuntimeConfig d; d.activations_scale_factor = 0.0;  // 0.0 is still a set value
  EXPECT_TRUE(d.hasAnySet());
  ModelRuntimeConfig boolset; boolset.enable_mmap = true;
  EXPECT_TRUE(boolset.hasAnySet());
}

TEST(runtime_hasAnySet, emptyStringDoesNotCount) {
  ModelRuntimeConfig s; s.cache_dir = std::string("");  // has_value() true, but empty
  EXPECT_FALSE(s.hasAnySet());
}

// ---------------------------------------------------------------------------
// toProperties(): the ov::AnyMap a pipeline constructor consumes -- one entry per
// set field, under its UPPER_SNAKE OpenVINO property name, as ov::Any of the
// matching C++ type.  Unset / empty-string fields are omitted (engine default kept);
// an explicit false is NOT special-cased -- it is forwarded as a normal value.
// ---------------------------------------------------------------------------
TEST(runtime_toProperties, forwardsEachValueKindUnderItsPropertyKey) {
  ModelRuntimeConfig c;
  c.activations_scale_factor = 0.5;              // float
  c.cache_dir                = std::string("/d/c");        // string
  c.enable_cpu_pinning       = true;             // bool
  c.enable_cpu_reservation   = false;            // bool == false -> still forwarded
  c.num_streams              = std::size_t(4);   // size_t -> int64
  c.inference_num_threads    = std::int32_t(2);  // int32 -> int64
  ov::AnyMap m = c.toProperties();

  ASSERT_NE(m.find("ACTIVATIONS_SCALE_FACTOR"), m.end());
  EXPECT_TRUE(m.at("ACTIVATIONS_SCALE_FACTOR").is<float>());
  EXPECT_FLOAT_EQ(m.at("ACTIVATIONS_SCALE_FACTOR").as<float>(), 0.5f);

  ASSERT_NE(m.find("CACHE_DIR"), m.end());
  EXPECT_TRUE(m.at("CACHE_DIR").is<std::string>());
  EXPECT_EQ(m.at("CACHE_DIR").as<std::string>(), std::string("/d/c"));

  ASSERT_NE(m.find("ENABLE_CPU_PINNING"), m.end());
  EXPECT_TRUE(m.at("ENABLE_CPU_PINNING").is<bool>());
  EXPECT_EQ(m.at("ENABLE_CPU_PINNING").as<bool>(), bool(true));

  // an explicit false is forwarded (a real value, not "unset")
  ASSERT_NE(m.find("ENABLE_CPU_RESERVATION"), m.end());
  EXPECT_TRUE(m.at("ENABLE_CPU_RESERVATION").is<bool>());
  EXPECT_EQ(m.at("ENABLE_CPU_RESERVATION").as<bool>(), bool(false));

  ASSERT_NE(m.find("NUM_STREAMS"), m.end());
  EXPECT_TRUE(m.at("NUM_STREAMS").is<int64_t>());
  EXPECT_EQ(m.at("NUM_STREAMS").as<int64_t>(), static_cast<int64_t>(4));

  ASSERT_NE(m.find("INFERENCE_NUM_THREADS"), m.end());
  EXPECT_TRUE(m.at("INFERENCE_NUM_THREADS").is<int64_t>());
  EXPECT_EQ(m.at("INFERENCE_NUM_THREADS").as<int64_t>(), static_cast<int64_t>(2));

  // exactly the six fields above, no more
  EXPECT_EQ(m.size(), std::size_t(6));
}

TEST(runtime_toProperties, allUnsetYieldsEmptyMap) {
  EXPECT_TRUE(ModelRuntimeConfig{}.toProperties().empty());
}

TEST(runtime_toProperties, omittedWhenEmptyAndForwardedWhenExplicit) {
  // An empty string is "not set" -> omitted from the map.  (find() and end() must
  // come from the SAME map: comparing an iterator of one temporary AnyMap against
  // end() of a second is not valid.)
  ModelRuntimeConfig c; c.cache_dir = std::string("");  // has_value(), but empty
  ov::AnyMap cm = c.toProperties();
  EXPECT_EQ(cm.find("CACHE_DIR"), cm.end());

  // But a non-empty value is forwarded.
  ModelRuntimeConfig d; d.cache_dir = std::string("/x");
  ov::AnyMap dm = d.toProperties();
  EXPECT_NE(dm.find("CACHE_DIR"), dm.end());
}

// ---------------------------------------------------------------------------
// The JSON round-trip (the "runtime_config" object form show-config emits and a
// LOAD message carries): to-JSON emits only the set fields; from-JSON reads them
// back; the two agree for the full 28-field set.
// ---------------------------------------------------------------------------
TEST(runtime_json, allFieldsRoundTrip) {
  ModelRuntimeConfig c = allFieldsSet();
  std::string js = modelRuntimeConfigToJson(c);
  ModelRuntimeConfig back = modelRuntimeConfigFromJson(js);
  EXPECT_EQ(c, back);
  // (EXPECT_EQ(c, back) already proved the values survive; here we only confirm the
  // keys are actually emitted -- one representative per value-kind -- so a key that
  // got silently dropped is caught (the struct equality would not notice a key the
  // writer skips but the reader never reads).
  EXPECT_NE(js.find("\"activations_scale_factor\""), std::string::npos);   // a double
  EXPECT_NE(js.find("\"attention_backend\""), std::string::npos);          // a string/enum
  EXPECT_NE(js.find("\"num_streams\""), std::string::npos);                // size_t  -> int64
  EXPECT_NE(js.find("\"inference_num_threads\""), std::string::npos);      // int32    -> int64
  EXPECT_NE(js.find("\"compilation_num_threads\""), std::string::npos);    // a negative int32
  EXPECT_NE(js.find("\"enable_cpu_reservation\""), std::string::npos);     // a false bool
  EXPECT_NE(js.find("\"workload_type\""), std::string::npos);              // a free-form string
}

TEST(runtime_json, allUnsetIsEmptyObject) {
  EXPECT_EQ(modelRuntimeConfigToJson(ModelRuntimeConfig{}), std::string("{}"));
  EXPECT_FALSE(modelRuntimeConfigFromJson("{}").hasAnySet());
}

TEST(runtime_json, keepsNegativeInt32AcrossTheRoundTrip) {
  ModelRuntimeConfig c; c.compilation_num_threads = -1;
  EXPECT_NE(modelRuntimeConfigToJson(c).find(R"("compilation_num_threads":-1)"),
            std::string::npos);
  ModelRuntimeConfig back = modelRuntimeConfigFromJson(modelRuntimeConfigToJson(c));
  ASSERT_TRUE(back.compilation_num_threads.has_value());
  EXPECT_EQ(*back.compilation_num_threads, std::int32_t(-1));
}

TEST(runtime_json, foreignKeysAreSilentlyIgnored) {
  // A key with no matching field is dropped (not a hard error), like the sibling
  // configs tolerate a supervisor's schema drift; known keys are still read.
  auto c = modelRuntimeConfigFromJson(
      R"({"some_foreign_key":1,"another_foreign":true,"num_streams":4})");
  ASSERT_TRUE(c.num_streams.has_value());
  EXPECT_EQ(*c.num_streams, std::size_t(4));
  // no foreign key surfaced as a real field
  EXPECT_FALSE(c.attention_backend.has_value());
  EXPECT_TRUE(c.hasAnySet());
}

TEST(runtime_json, wrongTypePerFieldIsConfigError) {
  // a string field given a JSON number
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"({"cache_dir":123})"); },
                             "expected a string");
  // a size_t field given a string
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"({"num_streams":"two"})"); },
                             "expected an integer");
  // a size_t field given a negative -> the non-negative rule
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"({"num_streams":-1})"); },
                             "must be non-negative");
  // a bool field given a string
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"({"enable_cpu_pinning":"true"})"); },
                             "expected a boolean");
  // a double field given a non-number
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"({"activations_scale_factor":"hi"})"); },
                             "expected a number");
}

TEST(runtime_json, malformedJsonAndNonObjectRejected) {
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson("{not json"); }, "not valid JSON");
  expectRuntimeErrorContains([] { modelRuntimeConfigFromJson(R"([])"); }, "expected a JSON object");
}

// ---------------------------------------------------------------------------
// The boost section this class owns (one long flag per property, alphabetically,
// long-form only) is spliced into every subcommand's shared --help -- a light
// check that the section header and a flag from each value-kind survive into it.
// (The deep parse coverage above is the real gate; this pins the help surface.)
// ---------------------------------------------------------------------------
TEST(runtime_optionsDesc, hasHeaderAndASampleFlagFromEachKind) {
  std::string s;  { std::ostringstream os; os << ModelRuntimeConfig::runtimeOptionsDesc(); s = os.str(); }
  EXPECT_NE(s.find("Runtime config options"), std::string::npos);  // the section caption
  // one representative flag from each of the five value-kinds
  EXPECT_NE(s.find("--activations-scale-factor"), std::string::npos);   // double
  EXPECT_NE(s.find("--attention-backend"), std::string::npos);          // hint / enum string
  EXPECT_NE(s.find("--enable-cpu-pinning"), std::string::npos);         // bool
  EXPECT_NE(s.find("--num-streams"), std::string::npos);                // size_t
  EXPECT_NE(s.find("--compilation-num-threads"), std::string::npos);    // signed int
  EXPECT_NE(s.find("--kv-cache-precision"), std::string::npos);         // free-form string
  // ...and every flag is long-form only.  boost marks a single-letter short alias
  // with a " [ --" prefix (e.g. "-m [ --model ] arg" in the load section); with
  // "~28 options a single letter is no sensible alias", so the runtime section has
  // none -- asserted by the absence of that marker, which is far more robust than
  // watching for a ", " (a comma-space appears throughout the flag descriptions).
  EXPECT_EQ(s.find(" [ --"), std::string::npos);
}
