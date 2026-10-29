// ovgenai -- the CLI front-end.  The whole command surface lives in one place
// (commands.{hpp,cpp}): four subcommands -- `run` (run one pipeline one-shot) and
// `worker` (run one pipeline as the long-lived piped worker), each picking the
// OpenVINO.GenAI pipeline by a single required --pipeline / -p parameter, plus the
// `show-config` and `show-pipelines` helpers (the latter is the one place the full
// table of supported pipelines is now listed; the top-level --help and the run /
// worker --pipeline help just point there).  The old per-pipeline subcommands
// (one each for llm / vlm / whisper / ...) and their --worker switch are gone.
// main just hands argv to that table.
#include "commands.hpp"

#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  return ovgenai::commands::subcommandTable().dispatch(argv[0], args);
}
