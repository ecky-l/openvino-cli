// ovgenai's subcommand front-end -- the ONE place that owns the command surface.
//
// The command surface is now FOUR subcommands (it was once one per OpenVINO.GenAI
// pipeline, plus a `--worker` flag; both are gone):
//   * `run`    -- run ONE pipeline one-shot: `ovgenai run --pipeline <p> [model-dir]
//                 <options>`.  Selecting the pipeline by `--pipeline`/`-p` (a new,
//                 required, general parameter) -- not by a subcommand name -- is what
//                 makes this one subcommand cover every pipeline, the way the old
//                 per-pipeline subcommands did, while the pipeline bodies themselves
//                 are unchanged (`run --pipeline vlm ...` == the former `ovgenai vlm ...`).
//   * `worker` -- run ONE pipeline as the long-lived piped worker that the old
//                 `--worker` flag selected: `ovgenai worker --pipeline <p> ...`.
//   * `show-config` -- the helper, kept verbatim from the pre-redesign front-end.
//   * `show-pipelines` -- the helper that now lists every supported pipeline with a
//                 one-line explanation; the single place that full table lives (the
//                 top-level --help and the run / worker --pipeline help just point here).
//
// `--pipeline`/`-p` is a *general* parameter: it appears in the `--help` "general"
// section (the untitled root boost renders first, right where the one-shot `--input`
// sits) of BOTH `run` and `worker`.  It is REQUIRED (no default) and its description
// is now a single line pointing at the `show-pipelines` helper, which lists the whole
// set with a one-line explanation per pipeline (so it is no longer spelled out here).
// `--input`/`-i` is a *special*
// parameter of `run` only (the direct one-shot chat); `worker`'s stdin is instead
// its line-JSON protocol, so it offers no `--input` at all.
//
// The reason this is built on boost::program_options is that boost formats a
// description (the 2nd column) for free -- one auto-aligned, auto-wrapped line
// per entry -- which is what `ovgenai --help` must show, with the subcommands laid
// out by category.  The pipelines are no longer table *entries* (there are just the
// four subcommands): the canonical name/desc set lives in `canonicalPipelines()`
// (one source of truth for the `show-pipelines` listing, the `--pipeline` help pointer,
// the top-level index pointer, the body router, and the tests); a body is bound to a
// name in the constructor, as the old per-pipeline entry lambdas did.
#ifndef OVGENAI_COMMANDS_HPP
#define OVGENAI_COMMANDS_HPP

#include <functional>
#include <string>
#include <vector>

namespace ovgenai {
  // Forward-declared (fully defined in pipeline_support.hpp): the PARSED arguments a pipeline body
  // receives.  runModeEntry parses the command line ONCE (the general argument parsing lives in the
  // front-end, not in the bodies) and hands one of these in -- so a body neither re-parses the argv
  // nor needs it.  Forward-declared here (not #included) because this header only uses the type as a
  // `const&` parameter (runPipelineStub below); the full definition comes via pipeline_support.hpp.
  struct PipelineParsed;

namespace commands {

// One CLI subcommand entry.
//   name        -- the token the operator types: "run" / "worker" / "show-config"
//   category    -- "command" (run / worker) or "helper" (show-config)
//   description -- the blurb boost (the index, hand-laid) puts in the `--help` column
//   run         -- the behaviour; receives the program name (argv[0], for usage text) and the
//                  *full* argv tail (the subcommand token at index 0, then its options / a
//                  positional model).  The two real subcommands (run / worker) bind it to
//                  runModeEntry, which parses the command line ONCE and hands the selected body
//                  (a PipelineBinding, e.g. vlm's runVlmBody in pipeline_support.cpp) a
//                  PipelineParsed -- so the bodies no longer take / parse the argv at all.
struct Subcommand {
  std::string name;
  std::string category;
  std::string description;
  std::function<int(const std::string& /*prog*/, const std::vector<std::string>& /*args*/)> run;
};

// One OpenVINO.GenAI pipeline the CLI can drive: the operator-facing `--pipeline`
// token, the one-line blurb (the text the `show-pipelines` listing renders, one line
// per pipeline), and the body that runs it -- a run/worker entry
// that reads `--pipeline` and then forwards to the (unchanged) pipeline body, passing
// the subcommand-derived mode (worker=false for `run`, true for `worker`).
struct PipelineSpec {
  std::string name;         // the --pipeline token, e.g. "llm" / "vlm"
  std::string description;  // the short blurb (top-level index + --pipeline help)
};

// The single source of truth for the pipeline set (14 entries, insertion order):
// the canonical name/desc list every listing reads -- the `show-pipelines` body
// (now the one place the full table renders), the `--pipeline`/`-p` help pointer,
// the top-level `ovgenai --help` "Known pipelines" pointer, and the `pipelineNames()`
// snapshot test.
const std::vector<PipelineSpec>& canonicalPipelines();

// Just the `--pipeline` tokens of `canonicalPipelines()` (a stable snapshot for tests).
const std::vector<std::string>& pipelineNames();

// The `--pipeline`/`-p` option's description: a single line pointing the operator at the
// `show-pipelines` helper, which lists every pipeline with a one-line explanation (the
// full table no longer inlines in this --help).  Shared by the `run`/`worker` help --
// and reused by pipeline_support.cpp for its in-body `--pipeline` flag, so the body's
// parse stays consistent with the help.
std::string pipelineParamDesc();

// The infrastructure-only body shared by every not-yet-implemented pipeline (the rub the registry
// binds to the eleven pipelines with no real body yet): it NO LONGER parses the command line --
// runModeEntry parses it ONCE and hands the body a ready PipelineParsed -- so the stub just
// reflects, to stderr, the (already reported) route + the mode + the (already parsed) model / device
// it was given, as a supervisor would log it (stdout is reserved for the result / protocol channel),
// and exits 0.  It lives here -- not in pipeline_support.cpp -- so the stub and the real bodies
// share ONE help renderer; pipeline_support.cpp's registry calls it for the stubs.  Declared
// (external linkage), not in an anonymous namespace, precisely so the registry in another
// translation unit can reach it.
int runPipelineStub(const std::string& name, const PipelineParsed& parsed, bool worker);

// The front-end: the registry, the argv dispatch, and the `--help` index.
class SubcommandTable {
 public:
  SubcommandTable();

  // The full subcommand table, insertion order (run, worker, then the two helpers
  // show-config / show-pipelines).
  const std::vector<Subcommand>& entries() const;

  // `ovgenai`'s whole body:
  //   no args                     -> print usage() to stdout, exit 2
  //   help / -h / --help          -> print usage() to stdout, exit 0
  //   version / --version         -> print "ovgenai <version>", exit 0
  //   a known first token         -> that subcommand's run, returning its own code
  //   anything else               -> "unknown command" to stderr + usage, exit 2
  int dispatch(const std::string& /*prog*/, const std::vector<std::string>& /*args*/) const;

  // `ovgenai --help`'s index: an intro + a USAGE header, a "Commands" section
  // (run / worker), a "Known pipelines" line that just points at `show-pipelines`
  // (the full 14 no longer inline), a "Helper subcommands" section
  // (show-config + show-pipelines), and examples.
  std::string usage(const std::string& /*prog*/) const;

 private:
  std::vector<Subcommand> m_entries;
};

// The one table. A lazily-built shared instance: main.cpp just calls it (and a
// test could list what is registered). Building once keeps the entry order
// deterministic and, sidesteps the static-initialisation-order fiasco -- a body
// is bound to a pipeline name inside the constructor, not from a foreign global.
const SubcommandTable& subcommandTable();

}  // namespace commands
}  // namespace ovgenai

#endif  // OVGENAI_COMMANDS_HPP
