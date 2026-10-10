// End-to-end smoke test: launches the built `ovcli` binary as a subprocess and
// checks the front-end dispatch + the shared load-model args, from outside the
// process (the unit tests in config_test.cpp cover the parser logic directly).
//
// OVCLI_WORKER_BIN is baked in at configure time (see CMakeLists); when it is
// not set the whole file is compiled out. When the binary can't even exec (e.g.
// the OpenVINO runtime .so are not on LD_LIBRARY_PATH -- run ctest under
// `eval $(./setupenv.sh)`), each test GTEST_SKIPs instead of failing a box that
// simply isn't provisioned for inference.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if defined(OVCLI_WORKER_BIN)
#include <fcntl.h>
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

// Run the ovcli binary with `args` (everything after the program name). Returns
// the exit status, or -1 when the binary could not be exec'd / ran to a non-clean
// end -- a caller treats -1 as "skip". stdout/stderr are drained to *out/*err.
int runOvcli(const std::vector<std::string>& args, std::string* out, std::string* err) {
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
    av.push_back(const_cast<char*>(OVCLI_WORKER_BIN));
    for (const auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    ::execv(OVCLI_WORKER_BIN, av.data());
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
bool ovcliRuns() {
  std::string out, err;
  return runOvcli({"version"}, &out, &err) == 0;
}

}  // namespace

TEST(vcli_worker, topLevelUsageExitsTwo) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvcli({}, &out, &err), 2);  // no subcommand -> usage, exit 2
  EXPECT_NE(out.find("USAGE"), std::string::npos);
}

TEST(vcli_worker, unknownCommandExitsTwo) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvcli({"frobnicate"}, &out, &err), 2);
  EXPECT_NE(err.find("unknown command"), std::string::npos);
}

TEST(vcli_worker, subcommandHelpExitsZero) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  EXPECT_EQ(runOvcli({"show-config", "--help"}, &out, &err), 0);
  EXPECT_NE(out.find("Model load parameters"), std::string::npos);
  EXPECT_NE(out.find("--model"), std::string::npos);
  EXPECT_NE(out.find("--device"), std::string::npos);
}

TEST(vcli_worker, showConfigPrintsResolvedJson) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // rapidjson emits compact JSON (no space after ':').
  int rc = runOvcli({
      "show-config", "--model", "./tiny-gpt2-ov",
      "--engine", "ovgenai", "--model-type", "llm", "--device", "CPU",
  },

  &out, &err);
  EXPECT_EQ(rc, 0);
  EXPECT_NE(out.find("\"model\":\"./tiny-gpt2-ov\""), std::string::npos);
  EXPECT_NE(out.find("\"engine\":\"ovgenai\""), std::string::npos);
  // the deliberately-dropped fields must NOT appear
  EXPECT_EQ(out.find("context_window"), std::string::npos);
  EXPECT_EQ(out.find("worker_max_respawns"), std::string::npos);
}

TEST(vcli_worker, configErrorOnBadEnumAndMissingRequired) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // Bad model-type -> exit 2, named on stderr.
  EXPECT_EQ(runOvcli({"show-config", "--model", "/x", "--engine", "ovgenai",
                       "--model-type", "notatype", "--device", "CPU"},
                      &out, &err),
            2);
  EXPECT_NE(err.find("model-type"), std::string::npos);
  // Missing --model -> exit 2, named on stderr.
  EXPECT_EQ(runOvcli({"show-config", "--engine", "ovgenai", "--model-type", "llm",
                       "--device", "CPU"},
                      &out, &err),
            2);
  EXPECT_NE(err.find("model"), std::string::npos);
}

TEST(vcli_worker, workerAndRunResolveAndExitZero) {
  if (!ovcliRuns()) GTEST_SKIP() << "ovcli not runnable here (needs `eval $(./setupenv.sh)`)";
  std::string out, err;
  // Both consume the identical shared load-model surface and (this phase) exit 0
  // after resolving it, having reported to stderr.
  EXPECT_EQ(runOvcli({"worker", "--model", "./tiny-gpt2-ov",
                       "--engine", "ovgenai", "--model-type", "vlm", "--device", "CPU"},
                      &out, &err),
            0);
  EXPECT_EQ(runOvcli({"run", "--model", "./tiny-gpt2-ov",
                       "--engine", "openvino", "--model-type", "emb", "--device", "CPU"},
                      &out, &err),
            0);
  // The derived name (basename of the path) is what a supervisor logs.
  EXPECT_NE(err.find("/tiny-gpt2-ov"), std::string::npos);
}

#else  // OVCLI_WORKER_BIN not defined at configure time

TEST(vcli_worker, binary_path_not_baked_in) { GTEST_SKIP() << "OVCLI_WORKER_BIN not set at configure time"; }

#endif
