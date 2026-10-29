// The `vlm` pipeline's real body (Design A: a thin pipe around VLMPipeline) -- now
// the body the run/worker front-end reaches via --pipeline vlm, not a subcommand.
//
// The other pipelines still route through the shared runPipelineStub (see
// commands.cpp); `vlm` is the first to get a genuine body: load a
// vlmpipeline, take a request -- a chat from --input (the run one-shot's direct
// surface), or (when --input is omitted) from stdin (a pipe -> one line; a terminal
// -> typed to a lone '.', the mail command's way) -- and let the engine apply the
// model template, and emit one JSON result object to stdout (the decoded text with
// special/protocol tokens preserved, the engine's native finish_reason, and the
// perf metrics).  For the `worker` subcommand the same executeRequest path runs
// GENERATE line-by-line on stdin instead; that form has no --input (its stdin IS the
// protocol), so the mode now comes from WHICH subcommand dispatched, not a --worker flag.
//
// Model-specific tool / reasoning *parsing* stays in the Python consumer (the
// OpenArc-style one): the CLI never splits.
//
// `runVlm` has the Subcommand's run shape but takes (prog, args, worker) -- `worker`
// is the mode the run / worker subcommand passed in (not a --worker flag, which is
// gone), and vlm no longer knows it is "vlm" by name (the front-end routed here by the
// --pipeline value), so it just runs one-shot (worker==false) or the long-lived
// loop (worker==true) for the loaded VLMPipeline.
#ifndef OVGENAI_VLM_HPP
#define OVGENAI_VLM_HPP

#include <string>
#include <vector>

namespace ovgenai {

int runVlm(const std::string& /*prog*/,
           const std::vector<std::string>& /*args*/,
           bool /*worker*/);

}  // namespace ovgenai

#endif  // OVGENAI_VLM_HPP
