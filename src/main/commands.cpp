// ovgenai's subcommand front-end -- the ONE place that owns the command surface.
// See commands.hpp for the design: three subcommands (run / worker / show-config)
// instead of a per-pipeline subcommand plus a --worker flag.  The pipeline is now a
// *general* required parameter, --pipeline/-p, shared by run and worker -- so the
// same one subcommand drives every OpenVINO.GenAI C++ pipeline (the old per-pipeline
// subcommands are gone): run picks the one-shot form, worker the long-lived one, and
// the pipeline name (the --pipeline value) selects which body executes.  `--input`
// is the special one-shot direct chat of the `run` subcommand; `worker` has no
// --input (its stdin IS the protocol).
//
// Each subcommand's own --help is boost-rendered (the general section -- its own
// root options, --pipeline / [--input] / --help -- boost draws first, the four
// named config sections after).  The *top-level* `ovgenai --help` index is
// hand-laid like it was, but now as  Commands (run / worker) / Known pipelines
// (the full --pipeline set, the one-line blurbs) / Helper subcommands (show-config).
#include "commands.hpp"

#include "config.hpp"                    // ConfigError / preProcess / rephraseBoost / joinList
                                        // / kHelp* / showConfigJson / the shared parse helpers
#include "model_load_config.hpp"         // parseModelLoadConfig + loadOptionsDesc
#include "model_runtime_config.hpp"      // parseModelRuntimeConfig + runtimeOptionsDesc
#include "model_scheduler_config.hpp"    // parseModelSchedulerConfig + schedulerOptionsDesc
#include "model_request_config.hpp"      // parseModelRequestConfig + requestOptionsDesc
#include "vlm.hpp"                        // runVlm -- the first (and, for now, only) real body

#include "ovgenai_config.h"              // OVGENAI_VERSION

#include <algorithm>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>    // the `po` alias is provided by config.hpp

namespace ovgenai {
namespace commands {

// --- The canonical pipeline set: the single source of truth every surface reads --
//   * the --pipeline / -p help (pipelineListString) -- a one-line-per-pipeline list;
//   * `pipelineNames` (a stable snapshot the tests assert);
//   * the SubcommandTable's top-level "Known pipelines" index;
//   * the run/worker router (a body is bound to each name in the constructor).
// Both are non-anonymous (declared in commands.hpp) so vlm.cpp reuses
// pipelineListString for its own in-body --pipeline flag, keeping the body's parse
// consistent with the help.  Entry order is the display order.
const std::vector<PipelineSpec>& canonicalPipelines() {
  static const std::vector<PipelineSpec> p = {
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
  return p;
}

const std::vector<std::string>& pipelineNames() {
  static const std::vector<std::string> n = [] {
    std::vector<std::string> out;
    for (const auto& e : canonicalPipelines()) out.push_back(e.name);
    return out;
  }();
  return n;
}

// The --pipeline description: an intro line, then one line per pipeline ("name = blurb").
// Each line is a separate boost description line, so boost re-indents the long
// wrapped ones under the flags column -- a scannable list, not a wall of text.
std::string pipelineListString() {
  // A lead-in line, then one "name = blurb" line per pipeline (a "\n" before each --
  // no trailing one, so boost re-indents every entry to the flags column yet emits no
  // blank line after the last).
  std::string s = "The OpenVINO.GenAI pipeline to run (required, no default). Choose one of:";
  for (const auto& e : canonicalPipelines())
    s += "\n" + e.name + " = " + e.description;
  return s;
}

namespace {  // file-local front-end helpers

// True iff the subcommand's own args ask for help before we touch a value.
bool anyHelp(const std::vector<std::string>& a) {
  for (const auto& x : a)
    if (x == "--help" || x == "-h") return true;
  return false;
}

// Word-wrap `desc` into lines no wider than `lineLen - colStart`, then emit: the
// first wrapped line rides the same physical line as the already-written (padded)
// name; the rest are indented to `colStart`.  Used only by the top-level index (a
// subcommand is a bare word, not flag); boost still renders each subcommand's own
// --help.  The token longer than the column is left intact rather than split.
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

// One capioned category of the top-level index: a (name | description) pair list,
// rendered with boost's column geometry (the shared kHelp* constants) so an index
// line visually lines up with the boost-rendered section the same subcommand later
// shows in its own --help.  One capter works for {name, description} pairs, which
// is both a subcommand entry (name + blurb) and a pipeline spec (name + blurb).
void renderCategory(std::ostream& os, const std::string& caption,
                    const std::vector<std::pair<std::string, std::string>>& kv) {
  if (kv.empty()) return;                        // an empty category draws nothing
  os << caption << ":\n";
  size_t nameCol = 0;
  for (const auto& kv2 : kv)
    nameCol = std::max(nameCol, kv2.first.size());
  const size_t colStart = 2 + nameCol + 2;       // 2-space indent, then the name column + a gap
  for (const auto& kv2 : kv)
    renderEntry(os, kv2.first, kv2.second, colStart, config::kHelpLineLength);
  os << "\n";
}

// The combined option set a `run` / `worker` parse (and, via subcommandHelp, its
// --help) shares: the four named config sections the class that owns them render,
// plus the GENERAL (untitled-root) section boost draws *first* -- the new
// `--pipeline`/`-p` (always; required) and, for the one-shot `run` form only,
// `--input`/`-i`.  A `worker` form passes isRunForm=false so its --help offers no
// --input (a worker's stdin IS its protocol) -- which is exactly why --input is a
// `run`-only special and a `worker --input ...` is rejected as "unknown option".
//
// The description is returned (not a string) so the SAME object renders the
// --help and parses, which cannot disagree; it is built identically to
// config.cpp's combinedConfigUsage() (untitled root + the four sections + --help)
// plus the general parameters, so a no-pipeline request still parses clean.
po::options_description generalOptionsDesc(bool isRunForm) {
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  // The new general parameter: which pipeline this run / worker drives.  Required
  // (no default -- validated by the front-end, which owns the "missing --pipeline"
  // error), value taken by -p; the help lists every pipeline as one line each
  // (`pipelineListString`), so the full set is visible right here.  Name / description
  // are C-strings: this boost build's easy_init only overloads operator() for
  // (const char* name, value, const char* description) (no std::string variant), so the
  // std::string from pipelineListString is held in a named local and passed by .c_str()
  // -- boost copies it inside add_option, so the pointer never dangles.
  const std::string pipelineDesc = pipelineListString();
  s.add_options()
    ("pipeline,p", po::value<std::string>(), pipelineDesc.c_str());
  if (isRunForm)
    s.add_options()
    // The special one-shot surface: only `run` offers it (a worker reads its
    // protocol from stdin).  When given it is used verbatim (no stdin read); when
    // omitted the chat is read from stdin (a pipe => one line; a terminal => a
    // lone '.').  Meaningful for chat-shaped pipelines (e.g. vlm); others ignore it.
    ("input,i", po::value<std::string>(),
        "The 'run' one-shot's direct input: the chat to run, as a gen_config JSON object "
        "({\"messages\":[...], optionally \"tools\" / \"chat_template_kwargs\"}). When given, "
        "no stdin is read. When omitted, the chat is read from stdin: a pipe yields one "
        "(possibly large) line, an interactive terminal is read a line at a time until a "
        "lone '.' (the mail command's way). Meaningful for pipelines that take a direct chat "
        "(e.g. vlm); a worker has no --input -- it reads its line-JSON protocol from stdin instead.");
  s.add_options()("help", po::bool_switch(), "Show this help message");
  return s;
}

// `<prog> <run|worker> --help`: a Usage line + a note, then the shared options
// (the general section + the four config sections) rendered by boost -- one source
// for both the print and the parse.  isRunForm gives `run` (offers --input); the
// worker's prose explains why it has none.
std::string subcommandHelp(const std::string& prog, const std::string& modeName,
                           bool isRunForm) {
  std::ostringstream os;
  if (isRunForm) {
    os << "Usage: " << prog << " run --pipeline <p> [model-dir] <options>\n";
    os << "  Choose the pipeline with --pipeline / -p (required); then run it ONCE: the chat\n";
    os << "  comes from --input when given, otherwise from stdin (a pipe -> one (possibly large)\n";
    os << "  line; a terminal -> line-by-line, a lone '.' ending it, as in mail).  The result is\n";
    os << "  one JSON object on stdout.\n\n";
  } else {
    os << "Usage: " << prog << " worker --pipeline <p> [model-dir] <options>\n";
    os << "  Choose the pipeline with --pipeline / -p (required); then, instead of one-shot, STAY\n";
    os << "  UP and read/write the line-JSON protocol on stdin/stdout (GENERATE / PING / CANCEL /\n";
    os << "  UNLOAD -> READY / RESULT / PONG / CANCEL / DONE / ERROR -- one JSON object per line).\n";
    os << "  A worker has NO --input: its stdin IS that protocol (a direct --input is the\n";
    os << "  'run' one-shot's surface, which is why the two are now two subcommands).\n\n";
  }
  os << generalOptionsDesc(isRunForm);          // boost renders the section
  return os.str();
}

// `show-config`'s --help: the four shared config sections + --help, and NOTHING else
// -- it prints the resolved config and loads no model, so it neither picks a pipeline
// (--pipeline) nor takes a one-shot chat (--input) nor is a worker.
std::string showConfigHelp(const std::string& prog) {
  std::ostringstream os;
  os << "Usage: " << prog << " show-config <options>\n";
  os << "  Parse the (unchanged) load config from the command line and print it as JSON, then\n";
  os << "  exit -- no model is loaded, so no --model / --device is required just to see what\n";
  os << "  would resolve.  A helper: it neither selects a pipeline (--pipeline) nor takes the\n";
  os << "  'run' one-shot input (--input), and is not a (--worker) worker.\n\n";
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  s.add_options()("help", po::bool_switch(), "Show this help message");
  os << s;
  return os.str();
}

// The `show-config` helper, kept from the pre-redesign front-end (its
// resolve-and-print behaviour): parse the (unchanged, all-boost) four configs and
// print their JSON, loading nothing.  Only its --help renderer changed (now
// showConfigHelp -- no --pipeline / --input / --worker, since it is a helper).
int runShowConfig(const std::string& prog, const std::vector<std::string> args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the token
  if (anyHelp(sa)) {
    std::printf("%s", showConfigHelp(prog).c_str());
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

// The infrastructure-only body shared by every pipeline that has not yet grown a
// real one: it is reached (for both `run` and `worker`) with the routed call whose
// argv[0] is the `run`/`worker` token and whose remaining tokens still carry the
// general --pipeline it was routed by -- so it drops token 0 and parses the general
// section + the four named sections with boost (tolerating both --pipeline and
// --input via generalOptionsDesc, the way the old stub tolerated --worker) and then
// confirms the load model is specified (the one contract a worker must keep before
// going up), mirrors it to stderr, like a supervisor would log it (stdout is
// reserved for the future result / protocol channel), and exits 0.  The real
// <Pipeline> body (load + one-shot / worker loop) lands per pipeline later; nothing
// here touches OpenVINO.
int runPipelineStub(const std::string& name, const std::string& /*desc*/,
                    const std::string& /*prog*/, const std::vector<std::string>& args,
                    bool worker) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the run/worker token
  if (anyHelp(sa)) {  // the front-end intercepts --help before delegating, so this is
    // only a defensive fallback if a body were ever driven straight; keep it coherent
    // (the general section + four sections) rather than the deleted --worker switch.
    bool unused = false; (void) unused;
    std::printf("%s", subcommandHelp("", name, /*isRunForm=*/!worker).c_str());
    return 0;
  }
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

    // The general section (untitled root: --pipeline + the run form's --input) plus
    // the four named sections.  The stub reads only model / device from the map;
    // generalOptionsDesc(true) keeps --input "known" so a routed run that also passed
    // --input is not rejected (a worker route never carries --input anyway).
    po::options_description desc = generalOptionsDesc(/*isRunForm=*/true);
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

    // Mirror the load config's required set (model / device): a run must not even
    // go up without them.  (--pipeline is required too, but that is enforced by the
    // run / worker front-end that routed here, so a body is never reached without it.)
    std::vector<std::string> missing;
    if (!vm.count("model")) missing.push_back("--model");
    if (!vm.count("device")) missing.push_back("--device");
    if (!missing.empty())
      throw config::ConfigError("missing required load option(s): "
                                + config::joinList(missing, ", "));

    // The model a supervisor would log (stdout is reserved for the future result /
    // protocol channel); the (worker mode) / (one-shot) label names the form so a
    // run is provably *not* reported as the other.
    const std::string model   = vm["model"].as<std::string>();
    const std::string device  = vm.count("device") ? vm["device"].as<std::string>() : std::string();
    std::fprintf(stderr,
                 "ovgenai: %s%s: not yet implemented (this build is "
                 "infrastructure only); would %s model '%s' on '%s'\n",
                 name.c_str(),
                 worker ? " (worker mode)" : " (one-shot)",
                 worker ? "run it as a worker for" : "run a one-shot for",
                 model.c_str(), device.c_str());
    return 0;
  } catch (const config::ConfigError& e) {
    std::fprintf(stderr, "ovgenai: %s: %s\n", name.c_str(), e.what());
    return 2;
  }
}

// --- Pipeline routing: a name -> the body that runs it -------------------------
struct PipelineBinding {
  std::string name;
  std::string desc;
  std::function<int(const std::string& /*prog*/, const std::vector<std::string>& /*args*/,
                    bool /*worker*/)> run;
};

// Build the router from the canonical set, binding each name to its body: `vlm` is
// the first pipeline with a real body (loads a VLMPipeline and generates), the rest
// share the infrastructure-only stub (the per-pipeline lambda the old front-end used,
// now keyed off the --pipeline value instead of a subcommand name).
const std::vector<PipelineBinding>& pipelineBindings() {
  static const std::vector<PipelineBinding> b = [] {
    std::vector<PipelineBinding> v;
    for (const auto& p : canonicalPipelines()) {
      const std::string nm = p.name;
      const std::string d  = p.description;
      v.push_back(PipelineBinding{nm, d,
        [nm, d](const std::string& prog, const std::vector<std::string>& a, bool worker) {
          if (nm == "vlm")
            return ovgenai::runVlm(prog, a, worker);   // vlm is the first real body
          return runPipelineStub(nm, d, prog, a, worker);
        }});
    }
    return v;
  }();
  return b;
}

int runModeEntry(const std::string& prog, const std::vector<std::string>& args, bool worker) {
  const std::string modeName = worker ? "worker" : "run";
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the run/worker token
  if (anyHelp(sa)) {
    std::printf("%s", subcommandHelp(prog, modeName, /*isRunForm=*/!worker).c_str());
    return 0;
  }

  // One full combined parse over the (untitled) general section + the four named
  // sections: it rejects an unknown/missing-value option here (so a bad flag never
  // reaches a body) and yields the single general parameter --pipeline, REQUIRED,
  // which selects the body (it has no default -- boost has no clean "required"
  // option, so the front-end enforces it the way the model's required set is done).
  po::options_description desc = generalOptionsDesc(/*isRunForm=*/!worker);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(sa)
                  .options(desc)
                  .style(po::command_line_style::unix_style)
                  .run(),
              vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::fprintf(stderr, "ovgenai: %s: %s\n", modeName.c_str(), config::rephraseBoost(e).what());
    return 2;
  }
  if (!vm.count("pipeline")) {
    std::fprintf(stderr, "ovgenai: %s: missing required option: --pipeline (or -p)\n",
                 modeName.c_str());
    return 2;
  }

  // Route: the --pipeline value picks the body; the body re-parses the (unchanged)
  // argv, so the bodies themselves are untouched -- only the token that *selected*
  // them changed (a --pipeline value, not a subcommand name) and the mode now comes
  // from WHICH subcommand dispatched (run -> one-shot, worker -> long-lived) instead
  // of a --worker flag.  The full args (their run/worker token at index 0) are
  // forwarded verbatim: the body drops index 0 for its own options.
  const std::string pipeline = vm["pipeline"].as<std::string>();
  for (const auto& bind : pipelineBindings())
    if (bind.name == pipeline)
      return bind.run(prog, args, worker);
  std::fprintf(stderr, "ovgenai: %s: unknown pipeline '%s'; choose one of: %s\n",
               modeName.c_str(), pipeline.c_str(),
               config::joinList(pipelineNames(), ", ").c_str());
  return 2;
}

}  // close the anonymous namespace of this file's front-end helpers

// Build the table: the three real subcommands -- run (one-shot) and worker
// (long-lived), each routing by the --pipeline value to an unchanged pipeline body,
// then the show-config helper.  A body is bound to a pipeline name in the per-entry
// lambda, as the old per-pipeline entries did -- that is the move from "one
// subcommand per pipeline + a --worker flag" to "run / worker subcommand + a
// --pipeline parameter".
SubcommandTable::SubcommandTable() {
  m_entries.push_back(Subcommand{
      "run", "command",
      "Run one OpenVINO.GenAI pipeline one-shot; the pipeline is selected by the "
      "required --pipeline / -p (a lone <model-dir> stands in for --model).",
      [](const std::string& prog, const std::vector<std::string>& a) {
        return runModeEntry(prog, a, /*worker=*/false);
      }});

  m_entries.push_back(Subcommand{
      "worker", "command",
      "Run one OpenVINO.GenAI pipeline as a long-lived worker (the form the old --worker "
      "flag selected); the pipeline is chosen by the required --pipeline / -p, and it reads "
      "its line-JSON protocol from stdin (so it has no --input).",
      [](const std::string& prog, const std::vector<std::string>& a) {
        return runModeEntry(prog, a, /*worker=*/true);
      }});

  // The one operational helper today: `show-config` (no --pipeline / --input /
  // --worker -- it prints the resolved config and loads nothing).
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
  os << "  " << prog << " run      --pipeline <p> [model-dir] <options>   run one pipeline one-shot\n";
  os << "  " << prog << " worker   --pipeline <p> [model-dir] <options>   run one pipeline as a long-lived worker\n";
  os << "  " << prog << " version | --version                            print the version and exit\n";
  os << "  " << prog << " -h | --help                                    show this help\n\n";

  // The two operator-facing surfaces: run / worker (one entry each), then the helper
  // (show-config).  Categories carry them; the full pipeline set is rendered below
  // under its OWN "Known pipelines" section, not mixed into the Commands list.
  std::vector<std::pair<std::string, std::string>> commandSubs, helperSubs;
  for (const auto& sc : m_entries)
    (sc.category == "helper" ? helperSubs : commandSubs).push_back({sc.name, sc.description});
  renderCategory(os, "Commands (the pipeline is chosen with --pipeline / -p)",
                 commandSubs);

  // The full pipeline set: the entries the --pipeline value may take, one line each
  // with their blurb (the same blurbs the --pipeline description lists).  Kept as its
  // own section so the operator sees every pipeline without reading --help first.
  std::vector<std::pair<std::string, std::string>> pipelineSubs;
  for (const auto& e : canonicalPipelines())
    pipelineSubs.push_back({e.name, e.description});
  renderCategory(os, "Known pipelines (select with --pipeline / -p)", pipelineSubs);
  renderCategory(os, "Helper subcommands", helperSubs);

  os << "EXAMPLES:\n";
  os << "  " << prog << " run --pipeline llm --model ./model --device CPU\n";
  os << "  echo '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' | " << prog
       << " run --pipeline vlm --model ./model --device CPU\n";
  os << "  echo '{\"op\":\"GENERATE\"}' | " << prog
       << " worker --pipeline llm --model ./model --device CPU\n";
  return os.str();
}

const SubcommandTable& subcommandTable() {
  static SubcommandTable t;
  return t;
}

}  // namespace commands
}  // namespace ovgenai
