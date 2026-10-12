// ovgenai -- the CLI front-end.  The whole command surface now lives in one
// place (commands.{hpp,cpp}):  a subcommand table of one entry per
// OpenVINO.GenAI C++ pipeline (llm / vlm / whisper / asr / tts / omni /
// embedding / text-embedding / rerank / the image- and video-generation
// pipelines; the full list is in commands.cpp), plus the `show-config` helper.
// main just hands argv to that table.  The previous `worker` / `run` entry
// points are gone, superseded by the per-pipeline subcommands each of which
// now carries a --worker switch for the long-lived piped-worker form.
#include "commands.hpp"

#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  return ovgenai::commands::subcommandTable().dispatch(argv[0], args);
}
