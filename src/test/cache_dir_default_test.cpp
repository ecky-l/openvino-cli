// ovgenai's `--cache-dir` model-dir default: the one model-dependent runtime
// knob.  When `--cache-dir` is NOT given but `--model` is, the plugin
// compile/cache dir defaults to a subdirectory "model_cache" within the model's
// own dir:
//     --cache-dir (unset)  ->  resolveModelPath(--model) + "/model_cache"
// (the model is resolved first, so an `hf://...` --model co-locates its cache with
// the resolved snapshot dir; a plain path is used verbatim).  The result lands in
// the runtime config's `cache_dir` and thus in the CACHE_DIR property the pipeline
// constructor consumes at load time (GenAIPipelineConfig::toPipelineProperties
// seeds its AnyMap from runtime_.toProperties()).  An explicit --cache-dir --
// including an empty "" that means "disable caching" -- is left as the operator
// set it: the not-given state is detected at the CLI parse via vm.count, not via
// the field (empty and absent both land at nullopt after the reads, else
// indistinguishable), so an opt-out is not overwritten by the default.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/any.hpp>            // ov::Any / ov::AnyMap (what toProperties builds)

#include "../main/config.hpp"              // env_or / ConfigError
#include "../main/util.hpp"               // resolveModelPath (+ resolveModelPathIn / hfHubDir)
#include "../main/model_runtime_config.hpp" // parseModelRuntimeConfig + ModelRuntimeConfig

namespace {
// --- (re)declared here (this file is a separate translation unit from
//     model_path_resolve_test.cpp; anonymous-namespace helpers don't travel):
//     a scratch fake of the HuggingFace cache + a save/restore env-var scope ---

// The user's example's snapshot hash (refs/main of a tiny-random qwen3.5).
inline constexpr const char* kHash = "07ad2f323e908adc8d023136610ded7656a40ff0";

inline std::string refLine() { return std::string("main ") + kHash; }

// A per-process scratch root (so parallel binaries / reruns don't collide).
inline std::string scratchBase() {
  return (std::filesystem::temp_directory_path() /
          ("ovgenai-cache-" + std::to_string(::getpid()))).string();
}

// Build a fresh scratch hub dir and (when modelDir is non-empty) the cache tree a
// test expects inside it: `models--<...>/refs/main` + (if makeSnapshot)
// snapshots/<kHash>/config.json (a real, existing snapshot dir).
inline std::string makeCache(const std::string& label, const std::string& modelDir,
                             const std::string& ref, bool makeSnapshot) {
  using namespace std::filesystem;
  const path hub = path(scratchBase()) / ("hub-" + label);
  remove_all(hub);
  create_directories(hub);
  if (!modelDir.empty()) {
    const path repo = hub / modelDir;
    create_directories(repo / "refs");
    std::ofstream o(repo / "refs" / "main");
    o << ref << "\n";
    if (makeSnapshot) {
      create_directories(repo / "snapshots" / kHash);
      std::ofstream(repo / "snapshots" / kHash / "config.json") << "{}";
    }
  }
  return hub.string();
}

// Save/restore one env var: sets `val` (or unsets when empty); restores prior
// state on scope-exit so the HF_HUB_CACHE / HOME juggling does not leak into other
// tests in this process.
struct EnvScoped {
  std::string key, old;
  bool wasSet;
  EnvScoped(const std::string& k, const std::string& v)
      : key(k), old(), wasSet(std::getenv(k.c_str()) != nullptr) {
    if (wasSet) old = std::getenv(key.c_str());
    if (v.empty()) ::unsetenv(key.c_str()); else ::setenv(key.c_str(), v.c_str(), 1);
  }
  ~EnvScoped() {
    if (wasSet) ::setenv(key.c_str(), old.c_str(), 1); else ::unsetenv(key.c_str());
  }
};

}  // namespace

// --- Part A: the cache_dir default --------------------------------------------

// 1. --cache-dir absent, --model a plain path -> "<model>/model_cache"  (plain
//    paths resolve verbatim, so this is deterministic regardless of the env).
TEST(cacheDirDefault, notSpecifiedDefaultsToModelDir) {
  auto c = ovgenai::config::parseModelRuntimeConfig({"--model", "/opt/qwen3-8b", "--device", "CPU"});
  ASSERT_TRUE(c.cache_dir.has_value());
  EXPECT_EQ(*c.cache_dir, std::string("/opt/qwen3-8b/model_cache"));
  EXPECT_TRUE(c.hasAnySet());
  // it flows to the CACHE_DIR property + the JSON (the field name)
  EXPECT_TRUE(c.toProperties().find("CACHE_DIR") != c.toProperties().end());
  EXPECT_NE(ovgenai::config::modelRuntimeConfigToJson(c).find("\"cache_dir\":\"/opt/qwen3-8b/model_cache\""),
            std::string::npos);
}

// 2. the model is resolved FIRST, so an `hf://` --model co-locates its cache with
//    the resolved snapshot dir (never a literal "hf://.../model_cache").
TEST(cacheDirDefault, hfModelCoLocatesWithResolvedSnapshot) {
  const std::string hub =
      makeCache("cache-default", "models--tiny-random--qwen3.5", refLine(), true);
  EnvScoped cache("HF_HUB_CACHE", hub), home("HF_HOME", "");
  auto c = ovgenai::config::parseModelRuntimeConfig(
      {"--model", "hf://model/tiny-random/qwen3.5", "--device", "CPU"});
  ASSERT_TRUE(c.cache_dir.has_value());
  EXPECT_EQ(*c.cache_dir,
            (std::filesystem::path(hub) / "models--tiny-random--qwen3.5" / "snapshots" / kHash /
             "model_cache").string());
}

// 3. an explicitly-given --cache-dir wins over the default (no override).
TEST(cacheDirDefault, explicitCacheDirWins) {
  auto c = ovgenai::config::parseModelRuntimeConfig(
      {"--cache-dir", "/var/c", "--model", "/opt/qwen3-8b", "--device", "CPU"});
  ASSERT_TRUE(c.cache_dir.has_value());
  EXPECT_EQ(*c.cache_dir, std::string("/var/c"));
}

// 4. an explicit empty --cache-dir "" (the "disable caching" opt-out) + a --model does
//    NOT trigger the default: the two are told apart via vm.count at the CLI parse
//    (the field is likewise nullopt, indistinguishable).
TEST(cacheDirDefault, explicitEmptyDisablesNotDefaulted) {
  auto c = ovgenai::config::parseModelRuntimeConfig(
      {"--cache-dir", "", "--model", "/opt/qwen3-8b", "--device", "CPU"});
  EXPECT_FALSE(c.cache_dir.has_value());
  EXPECT_FALSE(c.hasAnySet());   // the empty value is "not set"
}

// 5. no --model and no --cache-dir: nothing to base a default on -> stays nullopt
//    (the guard `!model.empty()`), so the engine's own cache default is used.
TEST(cacheDirDefault, noModelLeavesCacheDirUnset) {
  auto c = ovgenai::config::parseModelRuntimeConfig({"--kv-cache-precision", "f16"});
  EXPECT_FALSE(c.cache_dir.has_value());
  ASSERT_TRUE(c.kv_cache_precision.has_value());   // sanity: the request/block still parsed
  EXPECT_EQ(*c.kv_cache_precision, std::string("f16"));
}

// --- Part B: the (resolved) runtime config reaches the pipeline's properties ----

// 6. toProperties() -- the exact ov::AnyMap GenAIPipelineConfig::toPipelineProperties
//    seeds its result from (and the VLMPipeline / LLMPipeline constructor then
//    receives as the `properties` arg it forwards to ov::Core::compile_model()) --
//    carries the resolved `CACHE_DIR` = "<model>/model_cache" after a Part-A default,
//    so the cache dir is in hand at construction, exactly when the model is loaded.
TEST(cacheDirDefault, resolvesIntoThePropertiesTheConstructorReceives) {
  auto c = ovgenai::config::parseModelRuntimeConfig({"--model", "/opt/qwen3-8b", "--device", "CPU"});
  ov::AnyMap m = c.toProperties();
  auto it = m.find("CACHE_DIR");
  ASSERT_NE(it, m.end());
  // ov::Any exposes no is_type<T>() predicate here; presence + the value assertion
  // below are the point, and CACHE_DIR is always stored as a std::string by
  // toProperties()' addStr, so as<std::string>() is type-correct (and would throw
  // and fail the test if it ever were not).
  EXPECT_EQ(it->second.as<std::string>(), std::string("/opt/qwen3-8b/model_cache"));
}
