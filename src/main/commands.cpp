// ovgenai's subcommand front-end -- see commands.hpp for the design (one
// subcommand per OpenVINO.GenAI C++ pipeline, plus the `show-config` helper).
// Each subcommand's own `--help` is boost-rendered; the top-level `ovgenai
// --help` index is hand-laid instead, so a subcommand shows as a bare word --
// boost would print a one-token option as "--name", which wrongly reads as a
// flag (see the renderCategory / renderEntry note near the top of this file).
//
// This file is the whole command surface. It leans on the existing, untouched
// config layer (config.hpp's parse helpers + the four config classes' boost
// sections + JSON), so a subcommand's `--help` and its parser share one
// options_description and cannot drift -- the same single-source principle
// config.cpp already applies.
#include "commands.hpp"

#include "config.hpp"                // ConfigError / preProcess / rephraseBoost / joinList
                                    // / kHelp* / showConfigJson / the shared parse helpers
#include "model_load_config.hpp"     // parseModelLoadConfig + loadOptionsDesc
#include "model_runtime_config.hpp"  // parseModelRuntimeConfig + runtimeOptionsDesc
#include "model_scheduler_config.hpp"// parseModelSchedulerConfig + schedulerOptionsDesc
#include "model_request_config.hpp"  // parseModelRequestConfig + requestOptionsDesc

#include "ovgenai_config.h"          // OVGENAI_VERSION

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // the `po` alias is provided by config.hpp

namespace ovgenai {
namespace commands {
namespace {

// True iff the subcommand's own args ask for help before we touch a value.
bool anyHelp(const std::vector<std::string>& a) {
  for (const auto& x : a)
    if (x == "--help" || x == "-h") return true;
  return false;
}

// The combined boost section set shared by every subcommand -- the load, then
// runtime, then scheduler, then request sections (each exactly as its owning
// class renders it) -- plus a valueless --worker switch when the subcommand
// offers the long-lived worker form (the pipelines do; `show-config`, a helper,
// does not), plus the --help switch.  Returning the description (not a string)
// lets a subcommand use the SAME object both to render its own --help and to
// parse, so the two can never disagree.  `buildOptionsDesc(false, _)` matches
// config.cpp's combinedConfigUsage() exactly (untitled root + the four sections
// + --help), which is why `show-config`'s output is byte-identical to before.
//
// The `--worker` switch is *bound* to a real bool the caller passes in (hence the
// 2nd parameter) instead of being read back out of the variables_map: `notify()`
// materialises a valueless bool_switch's default (false) into the map even for an
// *absent* option, so `vm.count("worker")` reports the flag as set when it was
// never passed -- a one-shot would then report that it was a worker (the classic
// pitfall).  Binding flips it true only when the flag is actually on the line,
// and `notify()` leaves a bound switch at that value, so the caller reading back
// the bound bool gets the reliable one-shot/worker distinction.
po::options_description buildOptionsDesc(bool withWorker, bool& worker) {
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  if (withWorker)
    s.add_options()("worker", po::bool_switch(&worker),
                     "Run as a long-lived worker subprocess (omit for a one-shot "
                     "run): stay up and read/write a line-JSON protocol on "
                     "stdin/stdout, one JSON object per line. [Not yet implemented "
                     "in this build -- the subcommand body is infrastructure only.]");
  else
    worker = false;
  s.add_options()("help", po::bool_switch(), "Show this help message");
  return s;
}

// --- The top-level `ovgenai --help` index (hand-rendered, not boost) ---------
// We deliberately do NOT render this index with a boost options_description:
// given a one-token entry, boost always prints it as "--llm" (it prepends "--"
// to a long option and exposes no switch to suppress that), which reads as a
// flag, not a subcommand.  boost still renders each subcommand's *own* --help
// (via subcommandHelp below, where the "--"-prefixed flags ARE correct); we just
// lay the *list of subcommands* out by hand as a two-column table -- the same
// column geometry boost uses (the kHelp* constants), but with the names bare.

// Word-wrap `desc` into lines no wider than `lineLen - colStart`, then emit it:
// the first wrapped line rides the same physical line as the already-written
// (padded) name; the remaining lines are indented to `colStart`.  Precomputing
// into a list of fully-constructed lines first keeps the trailing newlines
// trivial -- exactly one "\n" per blurb line, no special-casing of a (possibly
// empty) first line, so a blurb-less entry still prints its name + "\n".  A
// token longer than the column is left intact rather than split (these are
// short subcommand blurbs, not prose).
void renderEntry(std::ostream& os, const std::string& name, const std::string& desc,
                 size_t colStart, size_t lineLen) {
  const size_t blurbWidth = (lineLen > colStart) ? lineLen - colStart : 0;
  std::vector<std::string> lines;
  {
    std::string line;
    std::istringstream ss(desc);
    std::string tok;
    while (ss >> tok) {
      if (!line.empty() && line.size() + 1 + tok.size() > blurbWidth) {
        lines.push_back(line);
        line.clear();
      }
      line += line.empty() ? tok : std::string(" ") + tok;
    }
    if (!line.empty()) lines.push_back(line);
  }
  std::string paddedName(2, ' ');
  paddedName += name;
  if (paddedName.size() < colStart)
    paddedName.append(colStart - paddedName.size(), ' ');
  for (size_t i = 0; i < lines.size(); ++i) {
    os << (i == 0 ? paddedName : std::string(colStart, ' '));
    os << lines[i] << "\n";
  }
  if (lines.empty()) os << paddedName << "\n";  // a blurb-less entry is still one line
}

// One capioned category: the caption line, a run of entries, then a trailing
// blank line that separates it from whatever follows.  The 1st column is the
// category's longest name plus a 2-space gap; the blurb then wraps to the shared
// help line width, so an index line visually lines up with the boost-rendered
// sections the same subcommand later shows in its own --help.
void renderCategory(std::ostream& os, const std::string& caption,
                    const std::vector<Subcommand>& subs) {
  if (subs.empty()) return;                       // an empty category draws nothing
  os << caption << ":\n";
  size_t nameCol = 0;
  for (const auto& sc : subs)
    nameCol = std::max(nameCol, sc.name.size());
  const size_t colStart = 2 + nameCol + 2;        // 2-space indent, then the name column + a gap
  for (const auto& sc : subs)
    renderEntry(os, sc.name, sc.description, colStart, config::kHelpLineLength);
  os << "\n";
}

// `<prog> <sub> --help`: a Usage line + the subcommand's blurb, a short note,
// then the shared options rendered by boost (one source == the parser's desc).
std::string subcommandHelp(const std::string& prog, const std::string& sub,
                           const std::string& desc, bool withWorker) {
  std::ostringstream os;
  os << "Usage: " << prog << " " << sub << " [model-dir] <options>\n";
  os << "  " << desc << "\n\n";
  os << "A positional <model-dir> is a friend for --model when it is the only\n";
  os << "model. --context-window / --worker-max-respawns / --worker-line-limit are not\n";
  os << "available here. The model is loaded from these shared options:\n\n";
  bool worker = false;  // the --worker entry is pure presentation here (the
                        // caller only renders this help, never parses it)
  os << buildOptionsDesc(withWorker, worker);
  return os.str();
}

// The `show-config` helper, kept verbatim from the pre-redesign 3-entry front-end
// (its resolve-and-print behaviour): parse the (unchanged, all-boost) four
// configs and print their JSON, loading nothing. Only the per-subcommand --help
// renderer changed (subHelp -> subcommandHelp), so its stdout is byte-identical.
int runShowConfig(const std::string& prog, const std::vector<std::string>& args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the token
  if (anyHelp(sa)) {
    std::printf("%s",
                subcommandHelp(prog, "show-config",
                               "Print the resolved load config as JSON and exit "
                               "(no model is loaded)",
                               /*withWorker=*/false)
                    .c_str());
    return 0;
  }
  try {
    config::ModelLoadConfig      loadCfg  = config::parseModelLoadConfig(sa);
    config::ModelRuntimeConfig   rtCfg    = config::parseModelRuntimeConfig(sa);
    config::ModelSchedulerConfig schedCfg = config::parseModelSchedulerConfig(sa);
    config::ModelRequestConfig   reqCfg   = config::parseModelRequestConfig(sa);
    std::printf("%s\n",
                config::showConfigJson(loadCfg, rtCfg, schedCfg, reqCfg).c_str());
    return 0;
  } catch (const config::ConfigError& e) {
    std::fprintf(stderr, "ovgenai: show-config: %s\n", e.what());
    return 2;
  }
}

// Every pipeline subcommand shares this stub: parse the combined sections plus
// --worker (so a worker invocation is "known", not an "unknown option"), confirm
// the load model is specified (the one contract a worker must keep before going
// up), report -- to stderr, like the old worker/run -- the model a supervisor
// would log (stdout is reserved for the future result / protocol channel), and
// exit 0.  The real <Pipeline> body (load + the one-shot / worker loop) lands in
// a later phase; nothing here touches OpenVINO.
int runPipelineStub(const std::string& name, const std::string& desc,
                    const std::string& prog, const std::vector<std::string>& args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the subcommand token
  if (anyHelp(sa)) {
    std::printf("%s", subcommandHelp(prog, name, desc, /*withWorker=*/true).c_str());
    return 0;
  }
  // Bound to the --worker entry: true iff the worker form was actually requested
  // (see buildOptionsDesc -- a plain vm.count() is unreliable here).
  bool worker = false;
  try {
    const config::PreProcessed pp = config::preProcess(sa);
    if (pp.positionalCount > 1) {
      throw config::ConfigError(
          "only one positional (the model path) may be given; found " +
          std::to_string(pp.positionalCount) +
          ". ovgenai loads exactly one model -- set it with --model");
    }
    std::vector<std::string> cleanArgs = pp.args;
    cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                    cleanArgs.end());

    po::options_description desc = buildOptionsDesc(true, worker);
    po::positional_options_description posDesc;
    posDesc.add("model", 1);  // a lone positional binds to --model

    po::variables_map vm;
    try {
      po::store(po::command_line_parser(cleanArgs)
                     .options(desc)
                     .positional(posDesc)
                     .style(po::command_line_style::unix_style)
                     .run(),
                vm);
      po::notify(vm);
    } catch (const po::error& e) {
      throw config::rephraseBoost(e);
    }

    // Mirror the load config's required set (model / device):
    // a worker or one-shot should not even go up without them.
    std::vector<std::string> missing;
    if (!vm.count("model"))       missing.push_back("--model");
    if (!vm.count("device"))      missing.push_back("--device");
    if (!missing.empty())
      throw config::ConfigError("missing required load option(s): "
                                + config::joinList(missing, ", "));

    // A no-default value option gets no phantom entry from notify(), so a
    // not-passed flag reads back as vm.count()==0 -- safe to count directly.
    const std::string model =
        vm.count("model") ? vm["model"].as<std::string>() : std::string();
    std::fprintf(stderr,
                 "ovgenai: %s%s: not yet implemented (this build is "
                 "infrastructure only); would %s model '%s'\n",
                 name.c_str(),
                 worker ? " --worker (worker mode)" : " (one-shot)",
                 worker ? "run as a worker for" : "run one-shot for",
                 model.c_str());
    return 0;
  } catch (const config::ConfigError& e) {
    std::fprintf(stderr, "ovgenai: %s: %s\n", name.c_str(), e.what());
    return 2;
  }
}

}  // close the anonymous namespace of this file's front-end helpers

// Build the table: every OpenVINO.GenAI C++ pipeline (taken from the public
// .GenAI headers under openvino.genai/src/cpp/include/openvino/genai), one
// entry each, then the `show-config` helper.  A pipeline entry's run is a lambda
// that forwards to the shared stub carrying that pipeline's name + blurb.
SubcommandTable::SubcommandTable() {
  struct PL {
    const char* name;  // the token the operator types
    const char* desc;  // the blurb (ovgenai --help index + a subcommand's --help)
  };
  const PL kPipelines[] = {
      {"llm",            "LLMPipeline: causal text generation (text in -> text out)"},
      {"vlm",            "VLMPipeline: vision-language model (images + text in -> text out)"},
      {"whisper",        "WhisperPipeline: speech recognition (audio in -> transcribed text out)"},
      {"asr",            "ASRPipeline: automatic speech recognition (audio in -> text out)"},
      {"tts",            "Text2SpeechPipeline: text-to-speech (text in -> audio out)"},
      {"omni",           "OmniPipeline: audio-visual multi-modal model (audio + text)"},
      {"embedding",      "EmbeddingPipeline: inputs -> dense embedding vectors"},
      {"text-embedding", "TextEmbeddingPipeline: text -> embedding vectors"},
      {"rerank",         "TextRerankPipeline: rerank candidate texts against a query"},
      {"text2image",     "Text2ImagePipeline: text -> image generation (diffusion)"},
      {"image2image",    "Image2ImagePipeline: image + text -> image (img2img, diffusion)"},
      {"inpainting",     "InpaintingPipeline: image + mask + text -> edited image (diffusion)"},
      {"text2video",     "Text2VideoPipeline: text -> video generation (diffusion)"},
      {"image2video",    "Image2VideoPipeline: image + optional text -> video (diffusion)"},
  };
  for (const auto& p : kPipelines) {
    const std::string name = p.name;
    const std::string desc = p.desc;
    m_entries.push_back(Subcommand{
        name, "pipeline", desc,
        [name, desc](const std::string& prog, const std::vector<std::string>& a) {
          return runPipelineStub(name, desc, prog, a);
        }});
  }

  // The one operational helper today: `show-config` (kept from the pre-redesign
  // front-end; no --worker, since it loads nothing).
  m_entries.push_back(Subcommand{"show-config", "helper",
      "Print the resolved load config as JSON and exit (no model is loaded)",
      &runShowConfig});
}

const std::vector<Subcommand>& SubcommandTable::entries() const {
  return m_entries;
}

int SubcommandTable::dispatch(const std::string& prog,
                              const std::vector<std::string>& args) const {
  if (args.empty()) {
    std::printf("%s", usage(prog).c_str());
    return 2;
  }
  const std::string& mode = args[0];
  if (mode == "help" || mode == "-h" || mode == "--help") {
    std::printf("%s", usage(prog).c_str());
    return 0;
  }
  if (mode == "version" || mode == "--version") {
    std::printf("ovgenai %s\n", OVGENAI_VERSION);
    return 0;
  }
  for (const auto& sc : m_entries)
    if (sc.name == mode) return sc.run(prog, args);
  std::fprintf(stderr, "ovgenai: unknown command '%s'\n", mode.c_str());
  std::printf("%s", usage(prog).c_str());
  return 2;
}

std::string SubcommandTable::usage(const std::string& prog) const {
  std::ostringstream os;
  os << prog << "  -- OpenVINO/GenAI inference CLI\n\n";
  os << "USAGE:\n";
  os << "  " << prog << " <subcommand> [model-dir] <options>\n";
  os << "  " << prog << " version | --version        print the version and exit\n";
  os << "  " << prog << " -h | --help                show this help\n\n";
  os << "Each <subcommand> loads its model from the SAME shared flags; run\n";
  os << prog << " <subcommand> --help for a subcommand's full flag list, and add --worker\n";
  os << "for the long-lived piped worker form (omit it for a one-shot run).\n\n";

  // The two categories: boost owns each subcommand's own `--help` (above), but
  // would print a one-token subcommand as "--llm" here, which wrongly reads as a
  // flag -- so the index is hand-laid instead (see renderCategory).  boost is
  // simply not used for this listing.
  std::vector<Subcommand> pipelineSubs;
  std::vector<Subcommand> helperSubs;
  for (const auto& sc : m_entries)
    (sc.category == "pipeline" ? pipelineSubs : helperSubs).push_back(sc);
  renderCategory(os, "Pipeline subcommands (one per OpenVINO.GenAI C++ pipeline)", pipelineSubs);
  renderCategory(os, "Helper subcommands", helperSubs);

  os << "EXAMPLES:\n";
  os << "  " << prog << " llm --model ./model --device CPU\n";
  os << "  echo 'hi' | " << prog << " llm --worker --model ./model --device CPU\n";
  return os.str();
}

const SubcommandTable& subcommandTable() {
  static SubcommandTable t;
  return t;
}

}  // namespace commands
}  // namespace ovgenai
