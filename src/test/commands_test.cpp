// Unit tests for ovgenai's subcommand front-end (src/main/commands.{hpp,cpp}) --
// the "one subcommand per OpenVINO.GenAI C++ pipeline" surface this build phase
// introduced: 14 pipeline subcommands (llm / vlm / whisper / asr / tts / omni /
// embedding / text-embedding / rerank / the image- and video-generation
// pipelines) plus the `show-config` helper, each rendered by boost as its own
// --help and dispatched by a single table, every pipeline also taking a valueless
// --worker switch for the long-lived piped-worker form (omit it for a one-shot run).
//
// Covered here, at the in-process unit level (the companion black-box layer is
// worker_cli_test.cpp, which drives the built binary):
//   * the table's shape -- one entry per pipeline (+ the helper), each in the
//     right category, the pipeline name-set pinned as a snapshot so a dropped /
//     added pipeline is caught, no duplicate names, no blank blurb;
//   * the top-level `ovgenai --help` index -- both category headers and every
//     subcommand name render, plus the USAGE / EXAMPLES framing;
//   * dispatch() contract -- no args -> 2, help/-h/--help -> 0,
//     version/--version -> 0, an unknown token -> 2 ("unknown command" on stderr);
//   * each registered subcommand's behaviour -- a valid one-shot / worker run
//     resolves the model and exits 0 (reporting -- "worker mode" / "one-shot" --
//     to stderr, the one contract a supervisor would read), and a run missing the
//     load model or device exits 2.  A pipeline's own --help exits 0 and shows its
//     --worker switch; a helper's --help shows none.
//
// The stubs write to stdout/stderr via C stdio, so each dispatch is run under an
// in-process fd redirect (dup2 the two stdio streams onto pipes, then drain after
// the call) -- the same capture a subprocess test needs, but without forking, so
// it runs inside the test process itself (which links the OpenVINO runtime, as the
// other unit tests do).  Result codes come from the return value; the mode text is
// asserted on the drained stderr so a one-shot is provably *not* reported as a
// worker (the bound-bool pitfall buildOptionsDesc exists to avoid).
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "../main/commands.hpp"

using namespace ovgenai::commands;

namespace {

// The 14 OpenVINO.GenAI C++ pipelines, one subcommand each, as registered in
// commands.cpp -- pinned here as the snapshot the "one subcommand per pipeline"
// deliverable promises.  The snapshot test guards the *set* (order-independent),
// while the behavioural tests iterate whatever the table actually registers, so an
// implementation that drops or adds a pipeline trips both.
const char* const kPipelineNames[] = {
    "llm", "vlm", "whisper", "asr", "tts", "omni",     "embedding", "text-embedding",
    "rerank", "text2image", "image2image", "inpainting", "text2video", "image2video",
};
constexpr std::size_t kNumPipelines = sizeof(kPipelineNames) / sizeof(kPipelineNames[0]);

// Captured outcome of one dispatch(): the exit code plus the drained
// stdout / stderr of the (redirected) call.
struct DispatchResult {
  int rc = -1;              // dispatch()'s return; -1 if capture could not even start
  std::string out;
  std::string err;
};

// Read `fd` to EOF, appending to `*s`.
void readUntil(std::string* s, int fd) {
  char buf[8192];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof buf)) > 0)
    s->append(buf, static_cast<std::size_t>(n));
}

// Run subcommandTable().dispatch("ovgenai", args) with the process's stdout /
// stderr redirected onto pipes, returning the exit code plus the captured text.
// in-process dup2 (not a fork), so it runs inside the test binary even though that
// binary links the OpenVINO runtime.  The real fds are restored and the pipes
// drained imperatively before return, so both the code and the text are on `r`.
// A subcommand lets a parse failure escape only as its own caught ConfigError -> 2,
// so the dispatch call does not throw for any argv we feed.
DispatchResult runDispatch(std::vector<std::string> args) {
  DispatchResult r;
  int pout[2], perr[2];
  if (::pipe(pout) != 0 || ::pipe(perr) != 0)
    return r;  // rc stays -1: the caller treats that as "could not capture"
  int realOut = ::dup(STDOUT_FILENO);
  int realErr = ::dup(STDERR_FILENO);
  if (realOut < 0 || realErr < 0) {
    for (int fd : {pout[0], pout[1], perr[0], perr[1]}) ::close(fd);
    return r;
  }
  ::dup2(pout[1], STDOUT_FILENO);
  ::dup2(perr[1], STDERR_FILENO);

  int code = -1;
  try {
    code = subcommandTable().dispatch("ovgenai", args);
  } catch (...) {
    code = -1;  // defensive: none of our argvs make a subcommand throw
  }
  r.rc = code;

  // Flush the C-stdio buffers (now sitting on the pipes), hand the readers EOF,
  // restore the real fds, then drain -- all before return (not deferred) so both
  // the code and the text are on `r`.
  std::fflush(nullptr);
  ::close(pout[1]);
  ::close(perr[1]);
  if (realOut >= 0) { ::dup2(realOut, STDOUT_FILENO); ::close(realOut); }
  if (realErr >= 0) { ::dup2(realErr, STDERR_FILENO); ::close(realErr); }
  readUntil(&r.out, pout[0]); ::close(pout[0]);
  readUntil(&r.err, perr[0]); ::close(perr[0]);
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// The table's shape.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, registersOneEntryPerPipelinePlusTheHelper) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  // 14 pipelines + 1 helper = 15 entries.
  ASSERT_EQ(e.size(), 15u);
  std::size_t pipelineCount = 0, helperCount = 0;
  for (const Subcommand& s : e) {
    if (s.name == "show-config") { ++helperCount; EXPECT_EQ(s.category, std::string("helper")); }
    else { EXPECT_EQ(s.category, std::string("pipeline")); ++pipelineCount; }
  }
  EXPECT_EQ(pipelineCount, kNumPipelines);
  EXPECT_EQ(helperCount, 1u);
}

// Snapshot of "one subcommand per OpenVINO.GenAI C++ pipeline": the set of
// pipeline entries is exactly the 14 expected tokens.  Order-independent, so this
// guards the *set* -- a dropped or added pipeline fails it -- without being brittle
// to a reordering of the table.
TEST(ovgenai_commands, pipelineNameSetIsEveryGenAIPipeline) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  std::vector<std::string> actual;
  for (const Subcommand& s : e)
    if (s.category == "pipeline") actual.push_back(s.name);
  ASSERT_EQ(actual.size(), kNumPipelines);
  std::vector<std::string> expected(std::begin(kPipelineNames), std::end(kPipelineNames));
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

TEST(ovgenai_commands, noBlankBlurbAndNoDuplicateNames) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  std::vector<std::string> names;
  names.reserve(e.size());
  for (const Subcommand& s : e) {
    ASSERT_FALSE(s.name.empty());
    ASSERT_FALSE(s.description.empty());  // every entry is captioned
    names.push_back(s.name);
  }
  std::sort(names.begin(), names.end());
  bool dup = false;
  for (std::size_t i = 1; i < names.size(); ++i)
    if (names[i] == names[i - 1]) dup = true;
  EXPECT_FALSE(dup) << "two subcommands share a name";
}

TEST(ovgenai_commands, showConfigIsHelperCategoryAndPipelineCarriesItsClassName) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  const Subcommand* sc = nullptr;
  for (const Subcommand& s : e)
    if (s.name == "show-config") sc = &s;
  ASSERT_NE(sc, nullptr);
  EXPECT_EQ(sc->category, std::string("helper"));
  EXPECT_NE(sc->description.find("Print the resolved load config"), std::string::npos);
  // a pipeline entry carries the class name that backs it (llm -> LLMPipeline, ...)
  const Subcommand* llm = nullptr;
  for (const Subcommand& s : e)
    if (s.name == "llm") llm = &s;
  ASSERT_NE(llm, nullptr);
  EXPECT_EQ(llm->category, std::string("pipeline"));
  EXPECT_NE(llm->description.find("LLMPipeline"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The top-level `ovgenai --help` index renders under its two category headers.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, topLevelHelpRendersBothCategoriesAndEverySubcommand) {
  const std::string usage = subcommandTable().usage("ovgenai");
  EXPECT_NE(usage.find("USAGE:"), std::string::npos);
  EXPECT_NE(usage.find("Pipeline subcommands"), std::string::npos);
  EXPECT_NE(usage.find("Helper subcommands"), std::string::npos);
  EXPECT_NE(usage.find("EXAMPLES:"), std::string::npos);
  // every subcommand is listed (a dropped one would be missing from the index)
  for (const char* p : kPipelineNames)
    EXPECT_NE(usage.find(p), std::string::npos) << "missing pipeline subcommand in --help: " << p;
  EXPECT_NE(usage.find("show-config"), std::string::npos);
}

// ---------------------------------------------------------------------------
// dispatch()'s top-level contract.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, noArgsPrintsUsageAndExitsTwo) {
  DispatchResult r = runDispatch({});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.out.find("USAGE"), std::string::npos);
}

TEST(ovgenai_commands, helpFlagsPrintUsageExitZero) {
  for (const std::string h : {"help", "-h", "--help"}) {
    DispatchResult r = runDispatch({h});
    EXPECT_EQ(r.rc, 0) << "for " << h;
    EXPECT_NE(r.out.find("USAGE"), std::string::npos) << "for " << h;
  }
}

TEST(ovgenai_commands, versionPrintsVersionExitZero) {
  for (const std::string v : {"version", "--version"}) {
    DispatchResult r = runDispatch({v});
    EXPECT_EQ(r.rc, 0) << "for " << v;
    EXPECT_NE(r.out.find("ovgenai"), std::string::npos) << "for " << v;
  }
}

TEST(ovgenai_commands, unknownCommandExitsTwoAndNamesIt) {
  DispatchResult r = runDispatch({"frobnicate"});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.err.find("unknown command"), std::string::npos);
  // the offending token is named so the operator can see what they mistyped
  EXPECT_NE(r.err.find("frobnicate"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Each registered subcommand's behaviour.  Iterating the table (not the hardcoded
// set) exercises exactly what the implementation registers; the snapshot test
// above is the guard that the set is the expected one.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, everyPipelineRunsOneShotAndWorkerWithModelAndDevice) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  for (const Subcommand& s : e) {
    if (s.category != "pipeline") continue;
    // a valid one-shot run resolves the model to stderr and exits 0
    DispatchResult os = runDispatch({s.name, "--model", "/opt/m", "--device", "CPU"});
    EXPECT_EQ(os.rc, 0) << s.name << " one-shot";
    EXPECT_NE(os.err.find("(one-shot)"), std::string::npos) << s.name << " one-shot label";
    EXPECT_NE(os.err.find("/opt/m"), std::string::npos) << s.name << " logs the model it was given";
    // the same run with --worker exits 0 and (crucially) reports *worker* mode,
    // proving the bound-bool switch is not a phantom "always set" flag
    DispatchResult wk = runDispatch({s.name, "--worker", "--model", "/opt/m", "--device", "CPU"});
    EXPECT_EQ(wk.rc, 0) << s.name << " worker";
    EXPECT_NE(wk.err.find("(worker mode)"), std::string::npos) << s.name << " worker label";
    EXPECT_NE(wk.err.find("/opt/m"), std::string::npos) << s.name << " logs the model it was given";
    // --help for a pipeline exits 0 and shows the --worker switch
    DispatchResult help = runDispatch({s.name, "--help"});
    EXPECT_EQ(help.rc, 0) << s.name << " --help";
    // A pipeline (buildOptionsDesc(true)) offers the --worker switch.  Match its
    // distinctive description, not the bare "--worker" token: the shared help note
    // also mentions "--worker-max-respawns" (for every subcommand), which would trip
    // a naive search for "--worker".
    EXPECT_NE(help.out.find("Run as a long-lived worker subprocess"), std::string::npos)
        << s.name << " --help should offer the --worker switch";
  }
}

TEST(ovgenai_commands, pipelineMissingModelOrDeviceExitsTwo) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  for (const Subcommand& s : e) {
    if (s.category != "pipeline") continue;
    DispatchResult missModel = runDispatch({s.name, "--device", "CPU"});
    EXPECT_EQ(missModel.rc, 2) << s.name << " missing --model";
    EXPECT_NE(missModel.err.find("missing required"), std::string::npos) << s.name;
    DispatchResult missDevice = runDispatch({s.name, "--model", "/opt/m"});
    EXPECT_EQ(missDevice.rc, 2) << s.name << " missing --device";
    EXPECT_NE(missDevice.err.find("missing required"), std::string::npos) << s.name;
  }
}

// `show-config` is a helper (no --worker) whose resolve-and-print job is to print
// the JSON of what resolved and exit 0, or exit 2 on a config error.
TEST(ovgenai_commands, showConfigResolvesAndPrintsJsonExitZero) {
  DispatchResult r = runDispatch({"show-config", "--model", "./tiny-gpt2-ov", "--device", "CPU"});
  ASSERT_EQ(r.rc, 0);
  EXPECT_NE(r.out.find("\"model\":\"./tiny-gpt2-ov\""), std::string::npos);
}

TEST(ovgenai_commands, showConfigMissingModelExitsTwo) {
  DispatchResult r = runDispatch({"show-config", "--device", "CPU"});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.err.find("missing required"), std::string::npos);
}

// A subcommand's --help does not require the model (anyHelp short-circuits before
// parsing), and a helper's help shows no --worker line (only pipelines do).
TEST(ovgenai_commands, showConfigHelpExitsZeroWithoutModelAndWithoutWorker) {
  DispatchResult r = runDispatch({"show-config", "--help"});
  ASSERT_EQ(r.rc, 0);
  EXPECT_NE(r.out.find("Model load parameters"), std::string::npos);
  // A helper (buildOptionsDesc(false)) offers no --worker switch.  Again match the
  // distinctive description, not the token: the shared help note mentions
  // "--worker-max-respawns" for every subcommand, so a bare "--worker" search would
  // spuriously match a helper's help even while the switch itself is correctly absent.
  EXPECT_EQ(r.out.find("Run as a long-lived worker subprocess"), std::string::npos)
      << "a helper shows no --worker switch";
}
