// ovgenai's subcommand front-end -- the ONE place that owns the command surface.
//
// The command surface is now THREE subcommands (it was once one per OpenVINO.GenAI
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
//
// `--pipeline`/`-p` is a *general* parameter: it appears in the `--help` "general"
// section (the untitled root boost renders first, right where the one-shot `--input`
// sits) of BOTH `run` and `worker`.  It is REQUIRED (no default) and its description
// lists every pipeline with a one-line blurb, so the operator sees the whole set
// where the top-level index used to spell it out.  `--input`/`-i` is a *special*
// parameter of `run` only (the direct one-shot chat); `worker`'s stdin is instead
// its line-JSON protocol, so it offers no `--input` at all.
//
// The reason this is built on boost::program_options is that boost formats a
// description (the 2nd column) for free -- one auto-aligned, auto-wrapped line
// per entry -- which is what `ovgenai --help` must show, with the pipelines laid
// out by category.  The pipelines are no longer table *entries* (there are just the
// three subcommands): the canonical name/desc set lives in `canonicalPipelines()`
// (one source of truth for the `--pipeline` help, the top-level index, the body
// router, and the tests); a body is bound to a name in the constructor, as the old
// per-pipeline entry lambdas did.
#ifndef OVGENAI_COMMANDS_HPP
#define OVGENAI_COMMANDS_HPP

#include <functional>
#include <string>
#include <vector>

namespace ovgenai {
namespace commands {

// One CLI subcommand entry.
//   name        -- the token the operator types: "run" / "worker" / "show-config"
//   category    -- "command" (run / worker) or "helper" (show-config)
//   description -- the blurb boost (the index, hand-laid) puts in the `--help` column
//   run         -- the behaviour; receives the program name (argv[0], for usage text)
//                  and the *full* argv tail (the subcommand token at index 0, its own
//                  options/positional after -- the same shape `runPipelineStub` /
//                  `runVlm` consume, which each drop index 0 for their own args).
struct Subcommand {
  std::string name;
  std::string category;
  std::string description;
  std::function<int(const std::string& /*prog*/, const std::vector<std::string>& /*args*/)> run;
};

// One OpenVINO.GenAI pipeline the CLI can drive: the operator-facing `--pipeline`
// token, the one-line blurb (the same text the top-level "Known pipelines" index and
// the `--pipeline` help line render), and the body that runs it -- a run/worker entry
// that reads `--pipeline` and then forwards to the (unchanged) pipeline body, passing
// the subcommand-derived mode (worker=false for `run`, true for `worker`).
struct PipelineSpec {
  std::string name;         // the --pipeline token, e.g. "llm" / "vlm"
  std::string description;  // the short blurb (top-level index + --pipeline help)
};

// The single source of truth for the pipeline set (14 entries, insertion order):
// the canonical name/desc list every surface reads -- the `--pipeline`/`-p` help
// (`pipelineListString`), the top-level `ovgenai --help` "Known pipelines" section,
// and the `pipelineNames()` snapshot test.
const std::vector<PipelineSpec>& canonicalPipelines();

// Just the `--pipeline` tokens of `canonicalPipelines()` (a stable snapshot for tests).
const std::vector<std::string>& pipelineNames();

// The `--pipeline`/`-p` option's description: a one-line-per-pipeline enumeration
// ("name = <blurb>") boost re-indents under the flags column.  Shared by the
// `run`/`worker` help -- and reused verbatim by vlm.cpp for its in-body `--pipeline`
// flag, so the body's parse stays consistent with the help.
std::string pipelineListString();

// The front-end: the registry, the argv dispatch, and the `--help` index.
class SubcommandTable {
 public:
  SubcommandTable();

  // The full subcommand table, insertion order (run, worker, then the helper).
  const std::vector<Subcommand>& entries() const;

  // `ovgenai`'s whole body:
  //   no args                     -> print usage() to stdout, exit 2
  //   help / -h / --help          -> print usage() to stdout, exit 0
  //   version / --version         -> print "ovgenai <version>", exit 0
  //   a known first token         -> that subcommand's run, returning its own code
  //   anything else               -> "unknown command" to stderr + usage, exit 2
  int dispatch(const std::string& /*prog*/, const std::vector<std::string>& /*args*/) const;

  // `ovgenai --help`'s index: an intro + a USAGE header, a "Commands" section
  // (run / worker), a "Known pipelines (select with --pipeline)" section (the full
  // 14, one line each), a "Helper subcommands" section (show-config), and examples.
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
