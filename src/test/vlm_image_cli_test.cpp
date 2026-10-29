// End-to-end coverage of the `vlm` pipeline's image-source surface (req 4). Each test
// launches the built `ovgenai` binary as a subprocess -- now `ovgenai run --pipeline
// vlm ...` -- against the committed tiny-random--qwen3.5-ov fixture and feeds a
// one-user-message chat whose single part is an image, with the image `url` in
// every accepted form:
//
//   * "data:<mime>;base64,XXX"  -- the built-in mime-type / base64 form (kept);
//   * "file://<absolute>"        -- an explicit file:// scheme;
//   * "<absolute>"/"<relative>"  -- a plain path (resolved against the CWD);
//   * "~..."                     -- a leading ~ expanded to $HOME.
//
// plus a deliberately missing file (which must fail with the "could not open
// image file" ConfigError) and, as a bonus, the same chat delivered on piped
// stdin instead of via --input (the non-direct input surface).  We do not care
// about the generated text (the fixture has random weights, so the output is
// rubbish) -- only that every accepted image source is *taken* and the run
// completes (rc 0, a result line emitted), and the one missing file is rejected.
//
// The image is a real, check-in-able JPEG baked at configure time as
// OVGENAI_VLM_IMAGE (src/test/fixtures/dog-237-536x354.jpg); the model dir is
// OVGENAI_VLM_FIXTURE and the worker binary OVGENAI_WORKER_BIN (see CMakeLists).
// When any of those is not baked, the whole file compiles out.  When the binary
// can't exec (a box without the OpenVINO runtime on LD_LIBRARY_PATH -- run ctest
// under `eval $(./setupenv.sh)`) or the fixture/image is gone at runtime, each
// test GTEST_SKIPs instead of failing an under-provisioned box.

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "util.hpp"   // ovgenai::base64_encode / env_or (ovgenai_core)

#if defined(OVGENAI_WORKER_BIN) && defined(OVGENAI_VLM_FIXTURE) && defined(OVGENAI_VLM_IMAGE)

using ovgenai::base64_encode;

namespace {

std::string readAll(int fd) {
  std::string s;
  char buf[8192];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof buf)) > 0) s.append(buf, static_cast<size_t>(n));
  return s;
}

std::string readWholeFile(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f) return std::string();
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

std::string dirOf(const std::string& p) {
  auto pos = p.find_last_of('/');
  return (pos == std::string::npos) ? std::string(".") : p.substr(0, pos);
}
std::string baseOf(const std::string& p) {
  auto pos = p.find_last_of('/');
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

// A one-user-message chat whose content is a text part + a single image part,
// with `uri` substituted verbatim as the image url.  The inserted `uri` is one
// of the accepted forms and (for the base64 data form) an A-Za-z0-9+/= blob, so
// it needs no further JSON-escaping; a hand-built string keeps the test compact.
std::string chatWithImage(const std::string& uri) {
  return std::string("{\"messages\":[{\"role\":\"user\",\"content\":[")
       + "{\"type\":\"text\",\"text\":\"describe this image\"},"
       + "{\"type\":\"image_url\",\"url\":\"" + uri + "\"}]}]}";
}

// The same one-user-message text+image chat as chatWithImage, but pretty-printed
// across many lines and delivered on piped stdin: a document whose JSON whitespace
// spans several lines, which exercises the whole-stream read-to-EOF input path that
// the old per-line reader (readOneLine) could not handle.  `uri` is substituted
// verbatim, as in chatWithImage (needs no further JSON escaping for the file:// form).
std::string chatWithImageMultiline(const std::string& uri) {
  return std::string("{\n")
       + "  \"messages\": [\n"
       + "    {\n"
       + "      \"role\": \"user\",\n"
       + "      \"content\": [\n"
       + "        { \"type\": \"text\", \"text\": \"describe this image\" },\n"
       + "        { \"type\": \"image_url\", \"url\": \"" + uri + "\" }\n"
       + "      ]\n"
       + "    }\n"
       + "  ]\n"
       + "}\n";
}

// Launch the built ovgenai with `args`, capturing its stdout/stderr.  `stdIn`,
// `home`, `cwd` are optional per-run overrides (empty = leave the inherited
// value): a non-empty stdIn is piped to the child (and is what makes the
// non-direct, non-tty input path read one line); a non-empty `home` is pushed
// into the child's environment as $HOME (for the ~ case); a non-empty `cwd` is
// chdir'd to in the parent (the child inherits it) -- enough to make a
// CWD-relative image path resolve.  Returns the exit status, or -1 when the run
// could not exec / finish -- a caller treats -1 as "skip".
int runVlm(std::vector<std::string> args,
           const std::string& stdIn,
           const std::string& home,
           const std::string& cwd,
           std::string* out, std::string* err) {
  // Snapshot what we will override so a failed mid-test (which restores nothing)
  // cannot leave the parent process's HOME/CWD corrupted for later tests.
  const char* oh = ::getenv("HOME");
  const bool hadHome = (oh != nullptr);
  const std::string oldHome = hadHome ? std::string(oh) : std::string();
  const bool setHome = !home.empty();
  if (setHome) ::setenv("HOME", home.c_str(), 1);

  char cwdbuf[4096];
  const std::string oldCwd = ::getcwd(cwdbuf, sizeof cwdbuf) ? std::string(cwdbuf) : std::string();
  bool didChdir = false;
  if (!cwd.empty()) {
    if (::chdir(cwd.c_str()) != 0) {
      if (setHome) { if (hadHome) ::setenv("HOME", oldHome.c_str(), 1); else ::unsetenv("HOME"); }
      return -1;
    }
    didChdir = true;
  }

  int rc = -1;
  const bool piped = !stdIn.empty();
  int outp[2], errp[2], inp[2];
  if (::pipe(outp) != 0 || ::pipe(errp) != 0 || (piped && ::pipe(inp) != 0)) {
    // fall through to restore with rc == -1
  } else {
    pid_t pid = ::fork();
    if (pid < 0) {
      ::close(outp[0]); ::close(outp[1]); ::close(errp[0]); ::close(errp[1]);
      if (piped) { ::close(inp[0]); ::close(inp[1]); }
    } else if (pid == 0) {
      ::close(outp[0]); ::close(errp[0]);
      if (piped) ::close(inp[1]);
      ::dup2(outp[1], STDOUT_FILENO);
      ::dup2(errp[1], STDERR_FILENO);
      if (piped) {
        ::dup2(inp[0], STDIN_FILENO);            // child reads piped input
      } else {
        int dn = ::open("/dev/null", O_RDONLY);  // --input present: stdin unused
        ::dup2(dn, STDIN_FILENO);
        ::close(dn);
      }
      ::close(outp[1]); ::close(errp[1]);
      std::vector<char*> av;
      av.push_back(const_cast<char*>(OVGENAI_WORKER_BIN));
      for (auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
      av.push_back(nullptr);
      ::execv(OVGENAI_WORKER_BIN, av.data());
      ::_exit(127);  // exec failed (a missing shared library, etc.)
    } else {
      ::close(outp[1]); ::close(errp[1]);
      if (piped) {
        ::close(inp[0]);
        std::size_t off = 0;
        while (off < stdIn.size()) {
          ssize_t w = ::write(inp[1], stdIn.data() + off, stdIn.size() - off);
          if (w < 0) break;
          off += static_cast<std::size_t>(w);
        }
        ::close(inp[1]);   // EOF to the child once the whole line is delivered
      }
      int status = 0;
      if (::waitpid(pid, &status, 0) < 0) {
        ::close(outp[0]); ::close(errp[0]);
      } else {
        if (!WIFEXITED(status)) {
          ::close(outp[0]); ::close(errp[0]);
        } else {
          rc = WEXITSTATUS(status);
          if (out) *out = readAll(outp[0]);
          ::close(outp[0]);
          if (err) *err = readAll(errp[0]);
          ::close(errp[0]);
        }
      }
    }
  }

  if (setHome) { if (hadHome) ::setenv("HOME", oldHome.c_str(), 1); else ::unsetenv("HOME"); }
  if (didChdir && !oldCwd.empty()) ::chdir(oldCwd.c_str());
  return rc;
}

// Run `ovgenai` with the standard vlm arg set, the chat in `uri` (the only
// thing that varies between the positive variants) and the per-run env/cwd
// overrides; asserts a clean completion with a result line.  A run that the box
// cannot exec (rc == -1) skips rather than fails an un-provisioned box.
void runImageVariant(const std::string& uri,
                     const std::string& stdIn,
                     const std::string& home,
                     const std::string& cwd,
                     const char* what) {
  const std::string in = chatWithImage(uri);
  std::vector<std::string> args =
      {"run", "--pipeline", "vlm", "--model", OVGENAI_VLM_FIXTURE, "--device", "CPU"};
  if (stdIn.empty()) {
    args.push_back("--input");
    args.push_back(in);
  }
  args.push_back("--max-tokens");
  args.push_back("4");
  std::string out, err;
  int rc = runVlm(args, stdIn, home, cwd, &out, &err);
  if (rc == -1)
    GTEST_SKIP() << "ovgenai could not exec here (run ctest under `eval $(./setupenv.sh)`) [" << what << "]";
  EXPECT_EQ(rc, 0) << "image source " << what << " (stderr):\n" << err;
  EXPECT_NE(out.find("finish_reason"), std::string::npos)
      << "no result line from an accepted image source " << what << " (stdout):\n" << out
      << "\n(stderr):\n" << err;
}

// Reachable?  Gate every test: the binary must exec AND the baked fixture dir and
// image file must still exist at runtime -- otherwise skip.  Sets `*why`.
bool vlmReady(std::string* why) {
  if (::access(OVGENAI_VLM_FIXTURE, F_OK) != 0) { *why = "baked VLM fixture dir is gone at runtime"; return false; }
  if (::access(OVGENAI_VLM_IMAGE, F_OK) != 0)  { *why = "baked image (dog jpg) is gone at runtime"; return false; }
  std::string out, err;
  if (runVlm({"version"}, "", "", "", &out, &err) != 0) {
    *why = "ovgenai not runnable here (needs `eval $(./setupenv.sh)`)"; return false;
  }
  return true;
}

}  // namespace

// The built-in mime / base64 form: read the real JPEG, base64-encode it, and run
// with "data:image/jpeg;base64,<payload>".  The payload is computed at runtime
// (so the giant blob need not live in source) and taken by the data: branch of
// the resolver (kept, in addition to the file forms).
TEST(vlm_image, dataUrlIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  std::string bytes = readWholeFile(OVGENAI_VLM_IMAGE);
  ASSERT_FALSE(bytes.empty()) << "fixture image read back empty";
  const std::string uri = "data:image/jpeg;base64," + base64_encode(bytes);
  runImageVariant(uri, "", "", "", "data:image/jpeg;base64");
}

// An explicit file:// scheme; the resolver strips the 7-char "file://" prefix.
TEST(vlm_image, fileSchemeIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  runImageVariant(std::string("file://") + OVGENAI_VLM_IMAGE, "", "", "", "file:// scheme");
}

// A plain absolute path (no scheme); opened directly.
TEST(vlm_image, absolutePathIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  runImageVariant(OVGENAI_VLM_IMAGE, "", "", "", "absolute path");
}

// A path relative to the CWD: the runner chdir's the parent (child inherits it)
// to the image's directory and the url is just the bare file name.
TEST(vlm_image, relativePathIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  const std::string dir = dirOf(OVGENAI_VLM_IMAGE);
  runImageVariant(baseOf(OVGENAI_VLM_IMAGE), "", "", dir, "relative-to-CWD path");
}

// A leading ~ (== $HOME): the runner pushes the image's directory onto $HOME for
// the child and the url is "~/dog-...jpg", so expandTilde resolves it.
TEST(vlm_image, tildePathIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  const std::string uri = std::string("~/") + baseOf(OVGENAI_VLM_IMAGE);
  runImageVariant(uri, "", dirOf(OVGENAI_VLM_IMAGE), "", "tilde ($HOME) path");
}

// The non-direct surface: the same image chat (a file:// url) delivered on piped
// stdin -- no --input -- so the resolver is still exercised, now via the
// piped-one-line input path rather than the --input value.
TEST(vlm_image, pipedStdinIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  runImageVariant(std::string("file://") + OVGENAI_VLM_IMAGE,
                  chatWithImage(std::string("file://") + OVGENAI_VLM_IMAGE), "", "",
                  "piped stdin (no --input)");
}

// Regression guard for the read-to-EOF change (req (a)): the SAME image chat as
// pipedStdinIsAccepted, but pretty-printed across many lines on piped stdin (no
// --input).  The old per-line reader (readOneLine) stopped at the first '\n' so the
// chat would fail to parse; readAllInput reads the whole document to EOF, so it passes.
TEST(vlm_image, multiLinePipedStdinIsAccepted) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  const std::string uri = std::string("file://") + OVGENAI_VLM_IMAGE;
  runImageVariant(uri, chatWithImageMultiline(uri), "", "",
                  "multi-line piped stdin (no --input)");
}

// The negative case: a url that no resolver branch can open must be a clean
// ConfigError (exit 2) naming the file, not a crash and not a silent success.
TEST(vlm_image, missingFileIsRejected) {
  std::string why;
  if (!vlmReady(&why)) { GTEST_SKIP() << why; return; }
  std::vector<std::string> args = {
      "run", "--pipeline", "vlm", "--model", OVGENAI_VLM_FIXTURE, "--device", "CPU",
      "--input", chatWithImage("file:///no/such/img-xyz.png"),
      "--max-tokens", "4"};
  std::string out, err;
  int rc = runVlm(args, "", "", "", &out, &err);
  if (rc == -1)
    GTEST_SKIP() << "ovgenai could not exec here (run ctest under `eval $(./setupenv.sh)`)";
  EXPECT_NE(rc, 0) << "a missing image file was NOT rejected (stderr):\n" << err;
  EXPECT_EQ(rc, 2);
  EXPECT_NE(err.find("could not open image file"), std::string::npos)
      << "missing image file did not report the expected ConfigError (stderr):\n" << err;
}

#else  // one of OVGENAI_WORKER_BIN / OVGENAI_VLM_FIXTURE / OVGENAI_VLM_IMAGE is not baked

TEST(vlm_image, image_fixtures_not_baked_in) {
  GTEST_SKIP() << "ovgenai image-input coverage is baked only when the worker "
                  "binary and the tiny-random VLM fixture + dog JPEG image are "
                  "present at configure time";
}

#endif
