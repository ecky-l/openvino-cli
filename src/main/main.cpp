// ovcli -- CLI front-end. One binary, three entry points that ALL derive the
// model to load from the command line (the shared --model/--engine/
// --model-type/--device/... flags), rather than the `LOAD` message a parent
// used to push down the pipe in the previous worker design:
//
//   ovcli worker [model-dir] <load-model options>
//     Long-lived protocol process (the drop-in for the previous worker).
//     Today it parses + validates the single load config and reports it, since
//     the stdin/stdout line-JSON loop (PING/GENERATE/...) is ported later.
//   ovcli run [model-dir] <load-model options>
//     One-shot: reads a request (text or base64) on stdin, infers via OpenVINO
//     .GenAI, writes the result on stdout. Today it resolves the config; the
//     inference backend arrives next.
//   ovcli show-config [model-dir] <load-model options>
//     Prints the resolved load config as JSON and exits (loads nothing) -- a fast
//     way to see exactly what `worker`/`run` would build.
//
// Every load-model flag is shared verbatim across all three and documented in
// config.hpp/.cpp (which replace a supervisor's per-model `load_config`);
// `--context-window` and `--worker-max-respawns` are intentionally absent (see
// the note there). Each also accepts --help / -h for a full flag listing.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "config.hpp"                // shared parse helpers + ConfigError + showConfigJson
#include "model_load_config.hpp"     // load model + parser / JSON round-trip / describe / usage
#include "model_scheduler_config.hpp"// scheduler config + parser / JSON round-trip / usage
#include "model_request_config.hpp"  // request config + parser / JSON
#include "ovcli_config.h"
#include "util.hpp"  // (unused here; keeps the front-end a thin core consumer)

namespace {

std::string topUsage(const char* argv0) {
  return std::string(argv0) +
      " -- self-contained OpenArc inference worker / one-shot CLI\n"
      "USAGE:\n"
      "  " + std::string(argv0) + " worker [model-dir] <options>   long-lived inference worker\n"
      "  " + std::string(argv0) + " run    [model-dir] <options>   one-shot inference (stdin->stdout)\n"
      "  " + std::string(argv0) + " show-config [model-dir] <options>\n"
      "                                                  print the resolved load config as JSON\n"
      "  " + std::string(argv0) + " version                       print the version and exit\n"
      "\n"
      "All three take the SAME load-model options (they replace OpenArc's `LOAD`)\n"
      "and share one config, so a model is now loaded from the command line instead\n"
      "of a pipe message. `ovcli <cmd> --help` lists every shared flag.\n"
      "\nEXAMPLES:\n"
      "  echo '{\"v\":1,\"op\":\"PING\"}' | " + std::string(argv0) + " worker \\\n"
      "      --model ./model --engine ovgenai --model-type vlm --device CPU\n"
      "  echo 'hi' | " +
      std::string(argv0) + " run --model ./model --engine ovgenai \\\n"
      "      --model-type llm --device CPU\n";
}

std::string subHelp(const char* sub, const char* blurb) {
  std::string s;
  s += std::string("Usage: ") + "ovcli " + sub + " [model-dir] <--load-model options>\n";
  s += std::string("  ") + blurb + "\n\n";
  s += "The model is loaded from these options (shared by worker / run / show-config):\n";
  s += "  A positional <model-dir> is a friend for --model when it is the only\n"
       "  model. --context-window / --worker-max-respawns are not available here.\n";
  s += ovcli::config::combinedConfigUsage();
  return s;
}

// True if the subcommand's own args ask for help before we touch a value.
bool anyHelp(const std::vector<std::string>& a) {
  for (const auto& x : a)
    if (x == "--help" || x == "-h") return true;
  return false;
}

int runWorker(const std::string& argv0, const std::vector<std::string>& args) {
  std::vector<std::string> sa(args.begin() + 1, args.end());
  if (anyHelp(sa)) {
    std::printf("%s", subHelp("worker",
        "Long-lived inference worker: the drop-in for OpenArc's Python worker, "
        "spawned one per model. This build resolves the load config; the "
        "stdin/stdout line-JSON protocol arrives in a later phase. ")
        .c_str());
    return 0;
  }
  try {
    ovcli::config::ModelLoadConfig cfg = ovcli::config::parseModelLoadConfig(sa);
    // Not wiring the protocol loop yet: report the one model that will be served,
    // then stop (a real worker would now read the pipe). A supervisor can read
    // this to learn what its child will stand up; nothing is written to stdout
    // (that channel is reserved for the future protocol).
    std::fprintf(stderr, "ovcli: worker would serve %s\n", ovcli::config::describeConfig(cfg).c_str());
    std::fprintf(stderr,
                 "ovcli: the line-JSON protocol loop is not ported yet; "
                 "this build resolves the load config and exits.\n");
    return 0;
  } catch (const ovcli::config::ConfigError& e) {
    std::fprintf(stderr, "ovcli: worker: %s\n", e.what());
    return 2;
  }
}

int runRun(const std::string& argv0, const std::vector<std::string>& args) {
  std::vector<std::string> sa(args.begin() + 1, args.end());
  if (anyHelp(sa)) {
    std::printf("%s", subHelp("run",
        "One-shot: read a request (text or base64) on stdin, infer with "
        "OpenVINO/.GenAI, write the result on stdout. This build resolves the "
        "load config; the inference backend arrives in a later phase. ")
        .c_str());
    return 0;
  }
  try {
    ovcli::config::ModelLoadConfig cfg = ovcli::config::parseModelLoadConfig(sa);
    // Not inferring yet: report the resolved model to stderr (stdout is the
    // future result channel and must stay clean), then exit.
    std::fprintf(stderr, "ovcli: run: %s\n", ovcli::config::describeConfig(cfg).c_str());
    std::fprintf(stderr,
                 "ovcli: the inference backend is not ported yet; "
                 "this build resolves the load config and exits.\n");
    return 0;
  } catch (const ovcli::config::ConfigError& e) {
    std::fprintf(stderr, "ovcli: run: %s\n", e.what());
    return 2;
  }
}

int runShowConfig(const std::string& argv0, const std::vector<std::string>& args) {
  std::vector<std::string> sa(args.begin() + 1, args.end());
  if (anyHelp(sa)) {
    std::printf("%s", subHelp("show-config",
        "Print the resolved load config as JSON and exit (no model is loaded) -- "
        "a fast way to see what `worker` / `run` would build. ")
        .c_str());
    return 0;
  }
  try {
    ovcli::config::ModelLoadConfig      loadCfg  = ovcli::config::parseModelLoadConfig(sa);
    ovcli::config::ModelSchedulerConfig  schedCfg = ovcli::config::parseModelSchedulerConfig(sa);
    ovcli::config::ModelRequestConfig    reqCfg   = ovcli::config::parseModelRequestConfig(sa);
    std::printf("%s\n",
                ovcli::config::showConfigJson(loadCfg, schedCfg, reqCfg).c_str());
    return 0;
  } catch (const ovcli::config::ConfigError& e) {
    std::fprintf(stderr, "ovcli: show-config: %s\n", e.what());
    return 2;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) {
    std::printf("%s", topUsage(argv[0]).c_str());
    return 2;
  }
  const std::string& mode = args[0];
  if (mode == "help" || mode == "--help") {
    std::printf("%s", topUsage(argv[0]).c_str());
    return 0;
  }
  if (mode == "version" || mode == "--version") {
    std::printf("ovcli %s\n", OVCLI_VERSION);
    return 0;
  }
  if (mode == "worker") return runWorker(argv[0], args);
  if (mode == "run") return runRun(argv[0], args);
  if (mode == "show-config") return runShowConfig(argv[0], args);
  std::fprintf(stderr, "ovcli: unknown command '%s'\n", mode.c_str());
  std::printf("%s", topUsage(argv[0]).c_str());
  return 2;
}
