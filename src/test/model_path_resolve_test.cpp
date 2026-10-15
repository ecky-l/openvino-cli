// Tests for the `--model hf://<name>` convenience (src/main/util.cpp): the fold
// of a HuggingFace-cache repo reference into the concrete `.../snapshots/<hash>`
// path.  These never touch the real cache -- a scratch directory stands in for it
// -- so they are hermetic and fast:
//   * resolveModelPath is a verbatim no-op for a (non-`hf://`) plain model path,
//     doing no env / filesystem work;
//   * resolution: `$HF_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`
//     for the cache root, `models--<repo-id>` (a leading repo-type segment and the
//     repo id's `/` written as `--`) for the repo dir, `refs/main` -> the snapshot
//     hash, `<hash>` -> the returned `snapshots/<hash>`;
//   * a reference that is missing / incomplete is REPORTED -- not downloaded --
//     with an error naming the `hf download <name>` to run (`<name>` = ref minus
//     "hf://");
//   * the same fold flows through parseModelLoadConfig (the load-config finalize),
//     so a `--model hf://...` resolves end to end.
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "../main/util.hpp"             // resolveModelPath / resolveModelPathIn / hfHubDir
#include "../main/config.hpp"           // ovgenai::config::ConfigError
#include "../main/model_load_config.hpp" // parseModelLoadConfig (the finalize wiring)

namespace fs = std::filesystem;
using ovgenai::resolveModelPath;
using ovgenai::resolveModelPathIn;
using ovgenai::hfHubDir;

namespace {

// The user's example's snapshot hash (refs/main of a tiny-random qwen3.5).  We
// can't rely on the real cache existing on this box, so the tests fabricate a
// minimal cache tree with this snapshot name and assert the resolved path points
// at it.
constexpr const char* kHash = "07ad2f323e908adc8d023136610ded7656a40ff0";

// The one ref line a git-like refs/main carries: "<rev> <hash>" (huggingface_hub
// writes "refs: main\n<hash>\n"; either way the LAST whitespace token is the hash
// the resolver reads).
std::string refLine() { return std::string("main ") + kHash; }

// A per-process scratch root (so parallel test binaries / reruns don't collide).
std::string scratchBase() {
  static const std::string base =
      (fs::temp_directory_path() / ("ovgenai-resolve-" + std::to_string(::getpid()))).string();
  return base;
}

// Build a fresh scratch hub dir and (when modelDirName is non-empty) the
// cache tree a test expects inside it: `models--<...>/refs/main` (writing the
// ref line) plus -- if makeSnapshot -- a `snapshots/<hash>/config.json` so the
// snapshot is a real, existing directory (a stand-in for a real OpenVINO IR).
// modelDirName is the dir name RELATIVE to the hub (including any "models--"
// prefix the test wants), matching what the resolver builds from a repo id.
std::string makeCache(const std::string& label, const std::string& modelDirName,
                      const std::string& ref, bool makeSnapshot) {
  const fs::path hub = fs::path(scratchBase()) / ("hub-" + label);
  fs::remove_all(hub);
  fs::create_directories(hub);
  if (!modelDirName.empty()) {
    const fs::path repo = hub / modelDirName;
    fs::create_directories(repo / "refs");
    std::ofstream o(repo / "refs" / "main");
    o << ref << "\n";                          // "<rev> <hash>\n" (or "\n" when ref is "")
    if (makeSnapshot) {
      fs::create_directories(repo / "snapshots" / kHash);
      std::ofstream(repo / "snapshots" / kHash / "config.json") << "{}";
    }
  }
  return hub.string();
}

// Save/restore a single env var: sets `key` to `val` (or unsets it when `val`
// is empty), remembering the prior state and restoring it on scope-exit -- so
// tests that play with the HF cache env vars don't leak into other tests in the
// same (shared) test process.
struct EnvScoped {
  std::string key, old;
  bool wasSet;
  EnvScoped(const std::string& k, const std::string& v)
      : key(k), old(), wasSet(std::getenv(k.c_str()) != nullptr) {
    if (wasSet) old = std::getenv(key.c_str());
    if (v.empty()) ::unsetenv(key.c_str());    // empty => "ensure unset"
    else           ::setenv(key.c_str(), v.c_str(), 1);
  }
  ~EnvScoped() {
    if (wasSet) ::setenv(key.c_str(), old.c_str(), 1);   // restore the prior value
    else        ::unsetenv(key.c_str());                // ... or that it was unset
  }
};

// A model not (fully) in the cache must be *reported*, not downloaded: the
// resolver throws config::ConfigError and its message must name the exact
// `hf download <name>` the operator runs (`<name>` = the reference minus
// "hf://", per the request -- so a non-type-prefixed ref and the example's
// type-prefixed ref are both named verbatim after the scheme).
void expectHfMissing(const std::function<void()>& fn, const std::string& wantName) {
  bool got = false;
  std::string msg;
  try {
    fn();
  } catch (const ovgenai::config::ConfigError& e) {
    msg = e.what();
    const std::string needle = "hf download " + wantName;
    got = msg.find(needle) != std::string::npos;
  }
  EXPECT_TRUE(got) << "expected a ConfigError telling the operator to run\n  "
                   << "'hf download " << wantName << "',\nbut got: " << msg;
}

}  // namespace

// --- A plain model path is returned verbatim; resolution does no work. --------

TEST(hfresolve, plainPathsAreVerbatimNoop) {
  EXPECT_EQ(resolveModelPath("/opt/qwen3-8b"), "/opt/qwen3-8b");
  EXPECT_EQ(resolveModelPath("./local --model dir/"), "./local --model dir/");
  EXPECT_EQ(resolveModelPath("relative/model"), "relative/model");
  // A non-`hf://` scheme is never even handed to the cache resolver.
  EXPECT_EQ(resolveModelPathIn("not an hf ref", "/dev/null/hub"), "not an hf ref");
  EXPECT_EQ(resolveModelPathIn("hf: not a real scheme", "/dev/null/hub"), "hf: not a real scheme");
  EXPECT_EQ(resolveModelPathIn("Hf://Upper", "/dev/null/hub"), "Hf://Upper");  // scheme is case-sensitive
}

// --- The user's example, end to end: the type segment is absorbed, '/' -> '--',
//     refs/main -> the hash, snapshots/<hash> is the returned path. -----------

TEST(hfresolve, exampleResolvesToSnapshot) {
  const std::string hub =
      makeCache("example", "models--tiny-random--qwen3.5", refLine(), /*makeSnapshot=*/true);

  const std::string got = resolveModelPathIn("hf://model/tiny-random/qwen3.5", hub);
  EXPECT_EQ(got, (fs::path(hub) / "models--tiny-random--qwen3.5" / "snapshots" / kHash).string());
}

TEST(hfresolve, slashesBecomeDoubleDashAndTypeSegmentAbsentKeepsFullId) {
  // No leading repo-type segment: the whole repo id is kept, just '/' -> '--'.
  const std::string hub =
      makeCache("slash", "models--org--Model--subdir", refLine(), /*makeSnapshot=*/true);
  EXPECT_EQ(resolveModelPathIn("hf://org/Model/subdir", hub),
            (fs::path(hub) / "models--org--Model--subdir" / "snapshots" / kHash).string());
}

TEST(hfresolve, leadingTypeSegmentIsAbsorbedByPrefix) {
  // A leading repo-type segment is absorbed by the constant 'models--' prefix
  // (as is the example's leading 'model/--'), so it never reappears after it.
  const std::string hub =
      makeCache("type", "models--owner--name", refLine(), /*makeSnapshot=*/true);
  EXPECT_EQ(resolveModelPathIn("hf://datasets/owner/name", hub),
            (fs::path(hub) / "models--owner--name" / "snapshots" / kHash).string());
}

// --- The cache root: the standard huggingface_hub env fall-through. -----------

TEST(hfresolve, hubDirEnvPriority) {
  {   // (a) $HF_HUB_CACHE wins outright
    EnvScoped a("HF_HUB_CACHE", "/custom/hub"), h("HF_HOME", ""), home("HOME", "/home/user");
    EXPECT_EQ(hfHubDir(), "/custom/hub");
  }
  {   // (b) else $HF_HOME/hub
    EnvScoped a("HF_HUB_CACHE", ""), h("HF_HOME", "/h"), home("HOME", "/home/user");
    EXPECT_EQ(hfHubDir(), "/h/hub");
  }
  {   // (c) else $HOME/.cache/huggingface/hub (the default)
    EnvScoped a("HF_HUB_CACHE", ""), h("HF_HOME", ""), home("HOME", "/home/user");
    EXPECT_EQ(hfHubDir(), "/home/user/.cache/huggingface/hub");
  }
}

// --- A missing / incomplete entry is reported, never downloaded. --------------

TEST(hfresolve, missingRepoIsReportedNotDownloaded) {
  const std::string hub = makeCache("missing-repo", "", "", /*makeSnapshot=*/false);
  expectHfMissing([&] { resolveModelPathIn("hf://model/tiny-random/qwen3.5", hub); },
                  "model/tiny-random/qwen3.5");
}

TEST(hfresolve, emptyRefsMainIsReported) {
  const std::string hub = makeCache("empty-refs", "models--tiny-random--qwen3.5", "", /*makeSnapshot=*/false);
  expectHfMissing([&] { resolveModelPathIn("hf://model/tiny-random/qwen3.5", hub); },
                  "model/tiny-random/qwen3.5");
}

TEST(hfresolve, refToAbsentSnapshotIsReported) {
  // refs/main names a snapshot whose directory is not there -- a partial / pruned cache.
  const std::string hub = makeCache("missing-snap", "models--tiny-random--qwen3.5",
                                     refLine(), /*makeSnapshot=*/false);
  expectHfMissing([&] { resolveModelPathIn("hf://model/tiny-random/qwen3.5", hub); },
                  "model/tiny-random/qwen3.5");
}

// --- The fold reaches the real load path via the load-config finalize() -------

TEST(hfresolve, loadConfigParseResolvesHfRef) {
  {   // Drive the cache root through the env, exactly as the real run would.
    const std::string hub =
        makeCache("parse-wire", "models--tiny-random--qwen3.5", refLine(), /*makeSnapshot=*/true);
    EnvScoped cache("HF_HUB_CACHE", hub), hfhome("HF_HOME", "");
    const ovgenai::config::ModelLoadConfig cfg = ovgenai::config::parseModelLoadConfig(
        {"--model", "hf://model/tiny-random/qwen3.5", "--device", "CPU"});
    EXPECT_EQ(cfg.model,
              (fs::path(hub) / "models--tiny-random--qwen3.5" / "snapshots" / kHash).string());
    EXPECT_EQ(cfg.device, "CPU");
  }
}

// Regression: a plain (non-`hf://`) model is parsed verbatim by the load config.
TEST(hfresolve, plainModelConfigUnaffected) {
  const ovgenai::config::ModelLoadConfig cfg = ovgenai::config::parseModelLoadConfig(
      {"--model", "/opt/qwen3-8b", "--device", "CPU"});
  EXPECT_EQ(cfg.model, "/opt/qwen3-8b");
  EXPECT_EQ(cfg.device, "CPU");
}
