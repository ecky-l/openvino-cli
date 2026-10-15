// The `vlm` subcommand's real body (Design A: a thin pipe around vlmpipeline).
//
// The other pipeline subcommands still route through the shared runPipelineStub
// (see commands.cpp); `vlm` is the first to get a genuine body: load a
// vlmpipeline, take a request (a full chat on stdin for the default
// `--apply-chat-template true`, or a raw `--prompt`/`--images` for `false`), and
// emit one JSON result object to stdout (the decoded text with special/protocol
// tokens preserved, the engine's native finish_reason, and the perf metrics).
//
// Model-specific tool / reasoning *parsing* stays in the Python consumer (the
// OpenArc-style one): the CLI never splits
//
// `runVlm` has the Subcommand's run shape but takes only (prog, args) -- the two
// captured name/desc the table's lambda forwards to every stub are not needed
// here (the body knows it is "vlm"), so the table just calls it with (prog, args).
#ifndef OVGENAI_VLM_HPP
#define OVGENAI_VLM_HPP

#include <string>
#include <vector>

namespace ovgenai {

int runVlm(const std::string& /*prog*/, const std::vector<std::string>& /*args*/);

}  // namespace ovgenai

#endif  // OVGENAI_VLM_HPP
