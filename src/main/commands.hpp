// ovgenai's subcommand front-end -- the ONE place that owns the command surface.
//
// The previous two front-ends (worker / run) are replaced by:
//   * one subcommand per OpenVINO.GenAI C++ pipeline -- vlm, llm, whisper, ...
//     (see commands.cpp for the full list, taken from the public .GenAI headers),
//   * the `show-config` helper, which is kept verbatim from the pre-redesign 3-entry
//     front-end.
//
// The reason this is built on boost::program_options is that boost formats a
// description (the 2nd column) for free -- one auto-aligned, auto-wrapped line
// per entry, under one caption per category -- which is exactly what
// `ovgenai --help` must show, i.e. the subcommands laid out by category: a
// "Pipeline subcommands" group (one entry per .GenAI pipeline) and a "Helper
// subcommands" group (only `show-config` for now).
//
// Aggregation on purpose: each subcommand is a data entry (name / category /
// description) plus a one-line run. A subcommand that grows a real body later can
// be split into its own {_subcommand.h,_subcommand.cpp} file(s) and register its
// entry here, without touching anything else (the user asked to aggregate for now
// and separate later).
#ifndef OVGENAI_COMMANDS_HPP
#define OVGENAI_COMMANDS_HPP

#include <functional>
#include <string>
#include <vector>

namespace ovgenai {
namespace commands {

// One subcommand entry.
//   name        -- the token the operator types, e.g. "llm" / "vlm" / "show-config"
//   category    -- "pipeline" (one per OpenVINO.GenAI C++ pipeline) or "helper"
//   description -- the blurb boost formats into the `--help` 2nd column
//   run         -- the behaviour; receives the program name (argv[0], for usage
//                  text) and the *full* argv tail (the subcommand token at index
//                  0, its own options/positional after) -- the same shape the
//                  pre-redesign runWorker / runRun used, so a subcommand reuses
//                  `args.begin() + 1` for its own arguments.
struct Subcommand {
  std::string name;
  std::string category;
  std::string description;
  std::function<int(const std::string& /*prog*/, const std::vector<std::string>& /*args*/)> run;
};

// The front-end: the registry, the argv dispatch, and the two-category --help.
class SubcommandTable {
 public:
  SubcommandTable();

  // The full table, insertion order (pipelines first, then helpers).
  const std::vector<Subcommand>& entries() const;

  // `ovgenai`'s whole body:
  //   no args                     -> print usage() to stdout, exit 2
  //   help / -h / --help          -> print usage() to stdout, exit 0
  //   version / --version         -> print "ovgenai <version>", exit 0
  //   a known first token         -> that subcommand's run, returning its own code
  //   anything else               -> "unknown command" to stderr + usage, exit 2
  int dispatch(const std::string& /*prog*/, const std::vector<std::string>& /*args*/) const;

  // `ovgenai <cmd> --help`'s two-category rendering: an intro + a USAGE header,
  // then the "Pipeline subcommands" / "Helper subcommands" sections (each a
  // boost options_description -- aligned name | wrapped description), then a
  // couple of examples.
  std::string usage(const std::string& /*prog*/) const;

 private:
  std::vector<Subcommand> m_entries;
};

// The one table. A lazily-built shared instance: main.cpp just calls it (and a
// test could list what is registered). Building once keeps the entry order
// deterministic and -- because the future per-pipeline files will register into
// this single table, not each into its own global -- sidesteps the
// static-initialisation-order fiasco.
const SubcommandTable& subcommandTable();

}  // namespace commands
}  // namespace ovgenai

#endif  // OVGENAI_COMMANDS_HPP
