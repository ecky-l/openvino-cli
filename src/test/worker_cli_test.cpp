// End-to-end smoke test: launches the built `ovgenai` binary as a subprocess and
// checks the front-end dispatch + the shared load-model args, from outside the
// process (the unit tests in config_test.cpp cover the parser logic directly).
//
// OVGENAI_WORKER_BIN is baked in at configure time (see CMakeLists); when it is
// not set the whole file is compiled out. When the binary can't even exec (e.g.
// the OpenVINO runtime .so are not on LD_LIBRARY_PATH -- run ctest under
// `eval $(./setupenv.sh)`), each test GTEST_SKIPs instead of failing a box that
// simply isn't provisioned for inference.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if defined(OVGENAI_WORKER_BIN)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::string readAll(int fd) {
  std::string s;
  char buf[8192];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof buf)) > 0) s.append(buf, static_cast<size_t>(n));
  return s;
}

// Run the ovgenai binary with `args` (everything after the program name). Returns
// the exit status, or -1 when the binary could not be exec'd / ran to a non-clean
// end -- a caller treats -1 as "skip". stdout/stderr are drained to *out/*err.
int runOvgenai(const std::vector<std::string>& args, std::string* out, std::string* err) {
  int outp[2], errp[2];
  if (::pipe(outp) != 0 || ::pipe(errp) != 0) return -1;
  pid_t pid = ::fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    // Drained output goes to the captured pipes; no stdin (the phase-tested
    // subcommands read no stdin, and /dev/null makes a future stdin reader EOF
    // instead of hanging inside an inherited terminal/pipe).
    int devnull = ::open("/dev/null", O_RDONLY);
    ::dup2(devnull, STDIN_FILENO);
    ::dup2(outp[1], STDOUT_FILENO);
    ::dup2(errp[1], STDERR_FILENO);
    ::close(devnull);
    ::close(outp[0]); ::close(outp[1]); ::close(errp[0]); ::close(errp[1]);
    std::vector<char*> av;
    av.push_back(const_cast<char*>(OVGENAI_WORKER_BIN));
    for (const auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    ::execv(OVGENAI_WORKER_BIN, av.data());
    ::_exit(127);  // exec failed (e.g. a missing shared library)
  }
  ::close(outp[1]);
  ::close(errp[1]);
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) { ::close(outp[0]); ::close(errp[0]); return -1; }
  // Safe to read after the child exits: its (tiny) output is buffered until we do.
  if (out) *out = readAll(outp[0]);
  ::close(outp[0]);
  if (err) *err = readAll(errp[0]);
  ::close(errp[0]);
  if (!WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}

// True when the binary at least links and execs cleanly (catches a box without
// the OpenVINO runtime on LD_LIBRARY_PATH, so the tests skip rather than fail).
bool ovgenaiRuns() {
  std::string out, err;
  return runOvgenai({"version"}, &out, &err) == 0;
}

}  // namespace

TEST(vcli_worker, topLevelUsageExitsTwo) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvgenai({}, &out, &err), 2);  // no subcommand -> usage, exit 2
  EXPECT_NE(out.find("USAGE"), std::string::npos);
}

TEST(vcli_worker, unknownCommandExitsTwo) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvgenai({"frobnicate"}, &out, &err), 2);
  EXPECT_NE(err.find("unknown command"), std::string::npos);
}

TEST(vcli_worker, subcommandHelpExitsZero) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvgenai({"show-config", "--help"}, &out, &err), 0);
  EXPECT_NE(out.find("Model load parameters"), std::string::npos);
  EXPECT_NE(out.find("--model"), std::string::npos);
  EXPECT_NE(out.find("--device"), std::string::npos);
}

TEST(vcli_worker, showConfigPrintsResolvedJson) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // rapidjson emits compact JSON (no space after ':').
  int rc = runOvgenai({
      "show-config", "--model", "./tiny-gpt2-ov", "--device", "CPU",
  },

  &out, &err);
  EXPECT_EQ(rc, 0);
  EXPECT_NE(out.find("\"model\":\"./tiny-gpt2-ov\""), std::string::npos);
}

TEST(vcli_worker, configErrorOnBadEnumAndMissingRequired) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // Bad model-type -> exit 2, named on stderr.
  EXPECT_EQ(runOvgenai({"show-config", "--model", "/x"}, &out, &err),
            2);
  // Missing --model -> exit 2, named on stderr.
  EXPECT_EQ(runOvgenai({"show-config", "--device", "CPU"}, &out, &err),
            2);
  EXPECT_NE(err.find("model"), std::string::npos);
}

TEST(vcli_worker, pipelineSubcommandsResolveAndWorkerExitsZero) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // Every pipeline subcommand consumes the identical shared load-model surface
  // and (this phase) exits 0 after resolving it, having reported the model to
  // stderr.  --worker must flip the reported mode -- and it is a *bound* bool,
  // not the notify-defaulted variables_map (which would read a "present" flag
  // even for an absent one) -- so a one-shot (worker omitted) must NOT report
  // itself as a worker; assert both directions to catch that pitfall.
  // --worker present -> the stub reports "worker mode":
  EXPECT_EQ(runOvgenai(
      {"llm", "--worker", "--model", "./tiny-gpt2-ov", "--device", "CPU"},
      &out, &err),
       0);
  EXPECT_NE(err.find("worker mode"), std::string::npos);
  EXPECT_NE(err.find("/tiny-gpt2-ov"), std::string::npos);
  // --worker omitted -> `vlm` is the first pipeline with a REAL body: it loads a
  // VLMPipeline and generates, so it needs a genuine (random-weight) VLM model --
  // NOT the LLM tiny-gpt2-ov used for the llm check above -- plus something to
  // generate, via --input (this runner feeds /dev/null on stdin, which now is
  // simply not read while --input is present).  A clean run is rc 0, reports
  // "one-shot", logs the model, and emits a result line; an un-exec'able box /
  // a missing fixture skips rather than fails.
#if defined(OVGENAI_VLM_FIXTURE)
  if (::access(OVGENAI_VLM_FIXTURE, F_OK) == 0) {
    std::string vout, verr;
    int rc = runOvgenai(
        {"vlm", "--model", OVGENAI_VLM_FIXTURE, "--device", "CPU",
         "--input", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
         "--max-tokens", "4"},
        &vout, &verr);
    if (rc == -1)
      GTEST_SKIP() << "ovgenai could not exec here (run ctest under `eval $(./setupenv.sh)`)";
    else if (rc != 0)
      FAIL() << "vlm real one-shot exited " << rc << " (stderr):\n" << verr;
    EXPECT_NE(verr.find("one-shot"), std::string::npos);
    EXPECT_NE(verr.find("tiny-random--qwen3.5-ov"), std::string::npos);
    EXPECT_NE(vout.find("\"finish_reason\""), std::string::npos);
  } else {
    GTEST_SKIP() << "baked VLM fixture path is gone at runtime";
  }
#else
  GTEST_SKIP() << "OVGENAI_VLM_FIXTURE was not baked at configure time";
#endif
}

// --input and --worker are two exclusive input surfaces, so giving both is a
// usage error the front-end must reject on stderr with exit 1, before any model
// load -- hence this test needs no VLM fixture, only that the binary execs
// (the ovgenaiRuns gate, shared with the other vcli_worker tests).  The conflict
// is detected from the bound --worker bool and the optional --input, so the
// --input value is irrelevant (any token) and no --model/--device is even
// required to reach this early return.
TEST(vcli_worker, inputAndWorkerConflictExitsOne) {
  if (!ovgenaiRuns()) GTEST_SKIP() << "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  int rc = runOvgenai({"vlm", "--input", "{\"messages\":[]}", "--worker"},
                      &out, &err);
  EXPECT_EQ(rc, 1) << "--input and --worker together must exit 1 (stderr):\n" << err;
  EXPECT_NE(err.find("mutually exclusive"), std::string::npos)
      << "expected the --input/--worker conflict text on stderr:\n" << err;
}

#else  // OVGENAI_WORKER_BIN not defined at configure time

TEST(vcli_worker, binary_path_not_baked_in) { GTEST_SKIP() << "OVGENAI_WORKER_BIN not set at configure time"; }

#endif
