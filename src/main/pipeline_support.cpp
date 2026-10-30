// The pipeline-binding layer (see pipeline_support.hpp): the seam between the subcommand
// front-end (commands.cpp's runModeEntry, which selects a pipeline by --pipeline) and each
// pipeline's body.  This TU holds the vlm body -- the one real body today -- moved IN from the
// former vlm.cpp:
//   * its COMMAND-LINE PARSING (was the vlm-general two-pass boost parse here in vlm.cpp);
//   * its pipeline CREATION (load a VLMPipeline from the resolved config);
//   * its per-request generate -- the vlm-specific "the produce + emit" -- now injected into the
//     session-agnostic SessionDispatcher (session_support.{hpp,cpp}) as the run/worker driver the
//     dispatcher runs against a selected session's ChatHistory.
// It also builds the registry (kPipelineBindings): vlm -> runVlmBody; the 13 not-yet-implemented
// pipelines share the infrastructure-only stub (kept in commands.cpp -- it is front-end infra,
// not a pipeline body -- so the front-end's real --help stays the single source of truth there and
// this TU only references commands::runPipelineStub for it).
//
// `commands::runPipelineStub` (declared in commands.hpp, defined in commands.cpp) is referenced by
// name below; `ovgenai::runVlmBody` is defined just below.
#include "pipeline_support.hpp"
#include "commands.hpp"             // canonicalPipelines + pipelineParamDesc + runPipelineStub (the 13 stubs)
#include "session_support.hpp"      // SessionDispatcher + GenerateRequestFn + the shared ovgenai::jsonEscape (reused below)
#include "config.hpp"               // ConfigError / preProcess / rephraseBoost / joinList / kHelp*
#include "util.hpp"                 // base64_decode / env_or (the image-tensor resolver)
#include "model_load_config.hpp"    // ModelLoadConfig + parseModelLoadConfig
#include "model_runtime_config.hpp" // ModelRuntimeConfig + parseModelRuntimeConfig
#include "model_request_config.hpp" // ModelRequestConfig + parseModelRequestConfig
#include "model_scheduler_config.hpp" // ModelSchedulerConfig + parseModelSchedulerConfig
#include "genai_config.hpp"         // makeGenerationConfig + GenAIPipelineConfig
#include "streaming_support.hpp"    // ContentStreamer / selectModelFamily / makeToolCallParserByName / bridgeParsedMessage

#include <boost/program_options.hpp> // po (the `po` alias is provided, at global scope, by config.hpp)

#include <rapidjson/document.h>

#include <openvino/core/descriptor/tensor.hpp> // ov::Tensor / ov::Shape / ov::element::u8
#include <openvino/genai/generation_config.hpp>        // GenerationConfig
#include <openvino/genai/generation_handle.hpp>        // GenerationFinishReason
#include <openvino/genai/streamer_base.hpp>            // StreamerBase / StreamerVariant
#include <openvino/genai/text_streamer.hpp>            // TextParserStreamer (ContentStreamer's base)
#include <openvino/genai/tokenizer.hpp>               // Tokenizer + apply_chat_template
#include <openvino/genai/json_container.hpp>          // JsonContainer
#include <openvino/genai/chat_history.hpp>            // ChatHistory
#include <openvino/genai/visual_language/pipeline.hpp>     // VLMPipeline + VLMPipelineBase + VLMDecodedResults
#include <openvino/genai/visual_language/perf_metrics.hpp> // VLMPerfMetrics
#include <openvino/genai/visual_language/video_metadata.hpp> // VideoMetadata (the empty {} the history generate takes)

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ovgenai {
namespace {

// --- small, self-contained helpers (the vlm-specific ones; the INPUT readers + worker
// control lines moved to session_support; this file keeps the vlm translate / emit / generate).

// --- (The argument parsing used to live here: each body re-parsed the command line via its own
// --help detector + a strip of the general routing flags.  That is now a SINGLE general parse
// in commands.cpp -- runModeEntry, via the free function parsePipelineArgs -- produced ONCE per
// invocation and handed to each body as a PipelineParsed (so the bodies neither re-parse nor
// need the raw argv).  The --help the bodies used to render is likewise the front-end's.  This
// TU therefore keeps only the vlm-specific generate machinery below -- the request translate ->
// normalised ChatHistory -> applied template -> generate -> emit -- which the SessionDispatcher
// runs against a selected session; it no longer parses anything.)
// --- image-part decode -> a placeholder tensor. The engine keeps NO image-format decoder, so a
// file / the built-in data: base64 form is read + (kept raw as) a [1,1,1,3] placeholder tensor,
// enough to keep the image-part / image-tensor count one-to-one (so the engine's vision-index
// invariant holds).  `url` is one of (req 4): "data:<mime>;base64,.." | "file://.." | "~..." |
// a plain path (the tilde / file:// forms fall through to read bytes from a path).
std::string dataUrlBase64(const std::string& url) {
  if (url.rfind("data:", 0) == 0) {
    auto comma = url.find(',');
    if (comma != std::string::npos) return url.substr(comma + 1);
  }
  return url;
}

ov::Tensor makeImageTensorFromBytes(const std::string& bytes) {
  ov::Tensor t(ov::element::u8, ov::Shape{1, 1, 1, 3});
  auto p = t.data<unsigned char>();
  for (size_t i = 0; i < 3; ++i) p[i] = (i < bytes.size()) ? static_cast<unsigned char>(bytes[i]) : 0x00;
  return t;
}

std::string expandTilde(const std::string& path) {
  if (path == "~" || path.rfind("~/", 0) == 0) {
    std::string home = env_or("HOME", std::string());
    if (home.empty())
      throw config::ConfigError("vlm: image '" + path + "' uses ~ but $HOME is not set");
    if (path == "~") return home;
    return home + path.substr(1);                                 // keep the leading '/'
  }
  return path;
}

ov::Tensor imageTensorFromUrlOrPath(const std::string& url) {
  std::string bytes;
  if (url.rfind("data:", 0) == 0) {
    // the built-in base64 form (mime type + base64 payload).
    bytes = base64_decode(dataUrlBase64(url));
  } else {
    // a file form: "file://" / "~" / "~/" / absolute / relative-to-CWD.
    std::string path = (url.rfind("file://", 0) == 0) ? url.substr(7) : url;
    path = expandTilde(path);
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) throw config::ConfigError("vlm: could not open image file '" + url + "'");
    std::stringstream ss; ss << f.rdbuf();
    bytes = ss.str();
  }
  static bool warned = false;                                     // once, regardless of source
  if (!warned) {
    warned = true;
    std::fprintf(stderr,
                 "ovgenai: vlm: image pixel-decoding is not yet wired "
                 "(openvino.genai exports no image-format decoder); "
                 "decoding to a [1,1,1,3] placeholder tensor for now\n");
  }
  return makeImageTensorFromBytes(bytes);
}

// --- the request -> normalised ChatHistory + media translator (the one real bit of new code) ---
struct ParsedRequest {
  std::string normMessagesJson;             // compact JSON array of normalised messages
  std::vector<ov::Tensor> images;
  std::vector<ov::Tensor> videos;
  std::vector<ov::Tensor> audios;
  bool hasTools = false;                    // a *non-empty* tools array is present
  bool hasKwargs = false;                   // a chat_template_kwargs object is present
  bool addGenerationPrompt = true;          // for the (pre-apply) string path
  std::string toolsJson;                    // compact, only when hasTools (for apply_chat_template)
  std::string kwargsJson;                   // compact, only when hasKwargs
};

// Serialise one normalised message as a JSON object fragment, rewriting a multipart (list) *user*
// content into the engine's universal media parts (and collecting the decoded tensors onto `pr`);
// a *non-user* list content is flattened to its joined text so the engine's per-role
// content.get_string() never throws.  Every other member of the original message (role,
// tool_calls, tool_call_id, ...) is carried through verbatim.
std::string serializeMessage(const rapidjson::Value& m, ParsedRequest& pr) {
  bool isUser = false;
  {
    auto it = m.FindMember("role");
    isUser = (it != m.MemberEnd() && it->value.IsString() &&
              std::string(it->value.GetString()) == "user");
  }
  std::string out = "{";
  size_t sep = 0;   // 0 = first member, 1 = already have one (needs a leading ", ")
  auto put = [&](const std::string& frag) {
    if (sep) out += ", ";
    out += frag;
    sep = 1;
  };
  // role
  {
    auto it = m.FindMember("role");
    if (it != m.MemberEnd() && it->value.IsString())
      put(std::string("\"role\":\"") + jsonEscape(it->value.GetString()) + "\"");
  }
  // content
  {
    auto it = m.FindMember("content");
    if (it != m.MemberEnd()) {
      if (it->value.IsString()) {
        put(std::string("\"content\":\"") + jsonEscape(it->value.GetString()) + "\"");
      } else if (it->value.IsArray()) {
        if (isUser) {
          std::string parts = "[";
          size_t ps = 0;
          for (auto pit = it->value.Begin(); pit != it->value.End(); ++pit) {
            const std::string ptype =
                (pit->IsObject() && pit->HasMember("type") && (*pit)["type"].IsString())
                    ? std::string((*pit)["type"].GetString()) : "";
            if (ptype == "text") {
              std::string txt = ((*pit).HasMember("text") && (*pit)["text"].IsString())
                                    ? std::string((*pit)["text"].GetString()) : "";
              if (ps) parts += ",";
              parts += "{\"type\":\"text\",\"text\":\"" + jsonEscape(txt) + "\"}";
              ps = 1;
            } else if (ptype == "image_url" || ptype == "image") {
              if (ps) parts += ",";
              parts += "{\"type\":\"image\"}";
              ps = 1;
              if (pit->IsObject() && pit->HasMember("url") && (*pit)["url"].IsString())
                pr.images.push_back(imageTensorFromUrlOrPath(std::string((*pit)["url"].GetString())));
              else
                pr.images.push_back(makeImageTensorFromBytes(std::string()));
            } else if (ptype == "video_url" || ptype == "video") {
              if (ps) parts += ",";
              parts += "{\"type\":\"video\"}";
              ps = 1;
              pr.videos.push_back(makeImageTensorFromBytes(std::string()));
            } else if (ptype == "audio_url" || ptype == "audio") {
              if (ps) parts += ",";
              parts += "{\"type\":\"audio\"}";
              ps = 1;
              pr.audios.push_back(makeImageTensorFromBytes(std::string()));
            } else {
              // unknown / already-universal part: keep it verbatim (re-serialised).
              if (ps) parts += ",";
              parts += config::compactJsonOf(*pit);
              ps = 1;
            }
          }
          parts += "]";
          put(std::string("\"content\":") + parts);
        } else {
          // non-user list content -> its joined text (the engine reads get_string()).
          std::string joined;
          for (auto pit = it->value.Begin(); pit != it->value.End(); ++pit)
            if (pit->IsObject() && pit->HasMember("text") && (*pit)["text"].IsString())
              joined += (*pit)["text"].GetString();
          put(std::string("\"content\":\"") + jsonEscape(joined) + "\"");
        }
      } else {
        // content is a number/bool/null/not-an-object: re-serialise verbatim.
        put(std::string("\"content\":") + config::compactJsonOf(it->value));
      }
    }
  }
  // carry every other member through verbatim (role + content handled above).
  for (auto it = m.MemberBegin(); it != m.MemberEnd(); ++it) {
    const std::string k = it->name.GetString();
    if (k == "role" || k == "content") continue;
    put(std::string("\"") + jsonEscape(k) + "\":" + config::compactJsonOf(it->value));
  }
  out += "}";
  return out;
}

// Translate a gen_config object (`g`, read-only) into a ParsedRequest: build the normalised
// messages array as a compact JSON string (for from_json_string), and detect the tools/kwargs
// sub-rule + the pre-apply add_generation_prompt.
ParsedRequest translateGenConfig(const rapidjson::Value& g) {
  ParsedRequest pr;
  auto tc = g.FindMember("tools");
  if (tc != g.MemberEnd() && !tc->value.IsNull() && tc->value.IsArray() && !tc->value.Empty()) {
    pr.hasTools = true;
    pr.toolsJson = config::compactJsonOf(tc->value);
  }
  auto kc = g.FindMember("chat_template_kwargs");
  if (kc != g.MemberEnd() && !kc->value.IsNull()) {
    pr.hasKwargs = true;
    pr.kwargsJson = config::compactJsonOf(kc->value);
    auto agp = kc->value.FindMember("add_generation_prompt");
    if (agp != kc->value.MemberEnd() && agp->value.IsBool())
      pr.addGenerationPrompt = agp->value.GetBool();
  }
  auto agpTop = g.FindMember("add_generation_prompt");
  if (!pr.hasKwargs && agpTop != g.MemberEnd() && agpTop->value.IsBool())
    pr.addGenerationPrompt = agpTop->value.GetBool();

  std::string out = "[";
  size_t ms = 0;
  auto mk = g.FindMember("messages");
  if (mk != g.MemberEnd() && !mk->value.IsNull() && mk->value.IsArray()) {
    for (auto it = mk->value.Begin(); it != mk->value.End(); ++it) {
      if (!it->IsObject()) continue;
      if (ms) out += ",";
      out += serializeMessage(*it, pr);
      ms = 1;
    }
  }
  out += "]";
  pr.normMessagesJson = out;
  return pr;
}

// (c) native finish_reason -> the wire field (null for NONE).
std::string finishReasonName(ov::genai::GenerationFinishReason r) {
  switch (r) {
    case ov::genai::GenerationFinishReason::STOP:      return "stop";
    case ov::genai::GenerationFinishReason::LENGTH:    return "length";
    case ov::genai::GenerationFinishReason::TOOL_CALL: return "tool_calls";
    case ov::genai::GenerationFinishReason::NONE:      return std::string();
  }
  return std::string();
}

// One number member of the metrics object, formatted shortest-round (g).
void addNumMember(std::string& out, const std::string& key, double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", x);
  out += "\"";
  out += key;
  out += "\":";
  out += buf;
}

// Emit one result object to stdout (the "consume" side the Python consumer reads):
//   { "session_id","v?type","type":RESULT?,"req_id"?,"result" (omitted when streamed),
//     "tool_calls"?,"finish_reason":null|stop|length|tool_calls, "metrics":{...},
//     "gen_config":{...echoed...} }  -- one compact line.
// The produced `result` text (after the pure-tool-call blank + the res.texts fallback) and the
// hasToolCalls decision are computed ONCE by the caller (vlmGenerate) and passed in -- so emitResult
// and the appended assistant reply cannot diverge.  The new "session_id" is the first field (the
// thing that selected/the created this conversation); for the worker it precedes the v/type/req_id
// header, for the one-shot the result body.
void emitResult(bool isWorker, std::int64_t reqId, const std::string& sessionId,
                const std::string& out, bool hasToolCalls,
                const ov::genai::VLMDecodedResults& res,
                const ovgenai::config::ModelRequestConfig& reqConfig,
                bool streamed) {
  std::string line = "{";
  auto first = true;
  auto put = [&](const std::string& frag) {
    if (!first) line += ",";
    line += frag;
    first = false;
  };
  put("\"session_id\":\"" + jsonEscape(sessionId) + "\"");   // the selection key, first
  if (isWorker) {
    put("\"v\":1");
    put("\"type\":\"RESULT\"");
    put("\"req_id\":" + std::to_string(reqId));
  }
  if (!streamed)
    put("\"result\":\"" + jsonEscape(out) + "\"");

  // (b) the structured tool calls, if any -- one compact array so the worker's single-line
  // JSONL stays one physical line (to_json_string(-1) is the compact dumper).
  if (!streamed && hasToolCalls)
    put(std::string("\"tool_calls\":") + res.parsed[0]["tool_calls"].to_json_string(-1));

  // The PARSER is the authority for a tool-call turn: a non-empty tool_calls wins over the
  // engine's finish, which may read "stop" when the model under-rands its </tool_call> outer tag
  // (we finalize on the inner </function>, so the turn is still reported as a tool call, not a stop).
  std::vector<ov::genai::GenerationFinishReason> fr = res.finish_reasons;
  ov::genai::GenerationFinishReason finish =
      fr.empty() ? ov::genai::GenerationFinishReason::NONE : fr[0];
  std::string frName = hasToolCalls ? std::string("tool_calls") : finishReasonName(finish);
  if (frName.empty())
    put(std::string("\"finish_reason\":null"));
  else
    put("\"finish_reason\":\"" + frName + "\"");

  // metrics -- a modest, consumer-useful subset of the VLM perf metrics.
  std::string m = "\"metrics\":{";
  ov::genai::VLMPerfMetrics pm = res.perf_metrics;   // non-const copy: accessors are non-const
  size_t mfirst = 1;   // 1: content already opened, so the first added needs no leading ","
  auto mPut = [&](const std::string& frag) {
    if (mfirst) { m += frag; mfirst = 0; }
    else { m += ","; m += frag; }
  };
  auto mInt = [&](const std::string& k, long long x) {
    mPut("\"" + k + "\":" + std::to_string(x));
  };
  auto mDbl = [&](const std::string& k, double x) {
    char b[64]; std::snprintf(b, sizeof b, "%.6g", x);
    mPut("\"" + k + "\":" + b);
  };
  mInt("num_input_tokens",   static_cast<long long>(pm.get_num_input_tokens()));
  mInt("num_generated_tokens", static_cast<long long>(pm.get_num_generated_tokens()));
  mDbl("load_time_ms",                static_cast<double>(pm.get_load_time()));
  mDbl("mean_ttft_ms",                pm.get_ttft().mean);
  mDbl("mean_tpot_ms",                pm.get_tpot().mean);
  mDbl("mean_throughput_tps",         pm.get_throughput().mean);
  mDbl("mean_vision_encoding_ms",     pm.get_vision_encoding_duration().mean);
  mDbl("mean_text_embedding_ms",      pm.get_text_embedding_duration().mean);
  mInt("total_image_slice_count",     static_cast<long long>(pm.get_total_image_slice_count()));
  m += "}";
  put(m);

  // echo the gen_config that was used (so the consumer can re-derive / log it).
  std::string gcj = config::modelRequestConfigToJson(reqConfig);
  put(std::string("\"gen_config\":") + gcj);
  line += "}";

  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// The model's effective thinking switch for THIS request -- the SINGLE on/off the render and
// the parser agree on.  Precedence: a PER-REQUEST input's chat_template_kwargs enable_thinking
// (narrowest, wins -- a per-request --input/stdin flips it over the global --enable-thinking flag),
// then the global --enable-thinking flag, then NONE.  It reads enable_thinking directly (not the
// rendered prompt) to sidestep the engine's own inject-default (the render alone always looks
// thinking-on for a Qwen model, yet it may never actually reason).
std::optional<bool> resolveEffectiveThinking(const ParsedRequest& pr,
                                            const config::ModelRequestConfig& reqConfig) {
  if (pr.hasKwargs && !pr.kwargsJson.empty()) {
    std::optional<bool> per;
    try {
      auto kc = ov::genai::JsonContainer::from_json_string(pr.kwargsJson);
      if (kc.contains("enable_thinking"))
        per = kc["enable_thinking"].as_bool();
    } catch (...) {}
    if (per.has_value())
      return per;
  }
  return reqConfig.enable_thinking;
}

// The core one-request driver -- the per-pipeline `generate()` the SessionDispatcher runs against a
// selected session's ChatHistory, taking the pipeline (captured in the callback) + the per-request
// sampling (the CLI base for a one-shot, the line's gen_config folded over it for a worker
// GENERATE) + the selected session + this request's input (a gen_config chat).
//
// The session is the persistence point (req 1): this request's new (normalised) messages are
// appended to the session's ChatHistory FIRST, so the engine renders the ACCUMULATED conversation
// (a worker's later turns on the same session carry earlier context); the produced assistant reply
// is appended back at the end so the NEXT turn renders it too.  On first contact the session is
// empty, so the append seeds it with exactly this request == the old stateless single-turn behaviour
// (which is also why an empty input still hits the "no message" guard below).  Everything else
// mirrors the old executeRequest verbatim, just reading/writing the (now accumulating) session
// `history` instead of a throwaway local, and threading the sessionId through to the emit.
int vlmGenerate(const std::shared_ptr<ov::genai::VLMPipelineBase>& pipe,
                ov::genai::Tokenizer tokenizer,
                const std::string& sessionId,
                ov::genai::ChatHistory& history,
                const config::ModelRequestConfig& reqConfig,
                const std::string& inputJson,
                bool isWorker, std::int64_t reqId) {
  // Build the base GenerationConfig from the per-request sampling (the apply_chat_template gate is
  // set per-branch below: the (tools|kwargs) sub-rule decides whether the engine or the CLI applies
  // the model template).
  ov::genai::GenerationConfig cfg = makeGenerationConfig(reqConfig);

  // The one input surface: parse the (possibly multi-line) chat as a gen_config and translate it to
  // a normalised messages array + media tensors + the tools/kwargs sub-rules.  An empty input leaves
  // the messages "[]" so the engine path below reports "no message".
  ParsedRequest pr;
  if (!inputJson.empty()) {
    rapidjson::Document g;
    if (g.Parse<0>(inputJson.c_str()).HasParseError())
      throw config::ConfigError("vlm: input is not valid JSON at position " +
                                 std::to_string(g.GetErrorOffset()));
    pr = translateGenConfig(g);
  }

  // The top-level --chat-template-kwargs flag (a global override for THIS run, parsed in
  // config::ModelRequestConfig) must also force the (tools|kwargs) sub-rule ON here: otherwise
  // the engine path -- which by construction renders with NO kwargs -- would be taken and the flag
  // would never reach apply_chat_template, so an explicit chat_template_kwargs["enable_thinking"]
  // bool could not override the reasoning gate.  Fold it into pr only when the input did not
  // already carry its own chat_template_kwargs (an inline per-request object is more specific and
  // wins); then its JSON becomes both the render's extra_context (so the template's else branch
  // fires) and the override read by resolveEffectiveThinking below.
  if (!pr.hasKwargs && reqConfig.chat_template_kwargs && !reqConfig.chat_template_kwargs->empty()) {
    pr.hasKwargs = true;
    pr.kwargsJson = *reqConfig.chat_template_kwargs;
    rapidjson::Document kd;
    if (!kd.Parse<0>(reqConfig.chat_template_kwargs->c_str()).HasParseError()) {
      auto agp = kd.FindMember("add_generation_prompt");
      if (agp != kd.MemberEnd() && agp->value.IsBool())
        pr.addGenerationPrompt = agp->value.GetBool();
    }
  }

  // (1) Persist the session: append THIS request's new (normalised) messages to the selected
  // session's ChatHistory -- the engine renders the whole accumulated conversation, so a worker's
  // later turns on the same session carry context.  On first contact the session is empty, so this
  // seeds it with exactly this request's messages == the stateless single-turn behaviour.  Build /
  // check pr first so an empty input still hits the "no message" guard the engine path throws.
  if (!pr.normMessagesJson.empty() && pr.normMessagesJson != "[]") {
    ov::genai::JsonContainer arr = ov::genai::JsonContainer::from_json_string(pr.normMessagesJson);
    auto& msgs = history.get_messages();
    for (std::size_t i = 0; i < arr.size(); ++i)
      msgs.push_back(arr[i]);
  }

  // The one on/off for the reasoning splitter, shared by BOTH branches below (so they can never
  // disagree): a per-request chat_template_kwargs enable_thinking wins, then the --enable-thinking
  // flag, then the model's own default (nullopt -> off).  It drives the render (pre-apply: projected
  // onto extra_context) AND the parser (the dynamic reasoner's setActive) -- both from this one value.
  const auto effThinking = resolveEffectiveThinking(pr, reqConfig);
  const bool thinking = effThinking.has_value() ? *effThinking : false;
  // The --stream N token-aggregation size (default 1 == today's one line per token; >1 = N tokens
  // per line; <=0 = no mid-stream output, one total line at the end).  Resolved once here and shared
  // by BOTH generate branches (so the pre-apply and the engine path can never disagree).
  const int stream = reqConfig.stream.value_or(1);
  bool useEngineTemplate = !(pr.hasTools || pr.hasKwargs);   // the (tools|kwargs) sub-rule

  // (b) Resolve the model family for this request: an explicit --model-family name wins; else
  // auto-detect from the model's OWN rendered template (the model-intrinsic sentinel); else none.
  // A selected-but-unwired name degrades (instance() splices no tool parser for it) to
  // reasoner-only / plain text; an auto-detected wired format is reported on stderr (never the wire
  // stdout, so the protocol stays one-JSON-per-line).
  std::optional<std::string> detectedName;
  const auto modelFamily = ::ovgenai::toolcall::selectModelFamily(
      reqConfig.model_family, tokenizer, &detectedName);
  if (modelFamily.has_value() && !modelFamily->empty()) {
    std::fprintf(stderr, "ovgenai: vlm: using model family '%s'\n", modelFamily->c_str());
    if (::ovgenai::toolcall::makeToolCallParserByName(*modelFamily).empty())
      std::fprintf(stderr,
          "ovgenai: vlm: --model-family '%s' is advertised but not yet wired in the CLI; "
          "degrading to plain text (no tool_calls)\n", modelFamily->c_str());
  } else if (detectedName.has_value()) {
    std::fprintf(stderr,
        "ovgenai: vlm: auto-detected model family '%s' from the model's chat template\n",
        detectedName->c_str());
  }

  // The produced content (specials preserved) the streamer drains into; computed ONCE below (after
  // the pure-tool-call blank + the res.texts fallback) and fed to BOTH emitResult AND the appended
  // assistant reply so the wire result and the persisted session turn can never diverge.
  std::string acc;
  auto finalize = [&](const ov::genai::VLMDecodedResults& res) -> int {
    // (b) A non-empty tool_calls (the parser's authority) => finish_reason "tool_calls", the
    // specials-preserving `out` kept (NOT the engine's specials-stripped res.texts[0], which would
    // re-expose the raw tool-block tags).  A pure tool-call turn has no prose to show: blank out any
    // whitespace-only lead-in so it reports result:"" rather than a bare newline.
    bool hasToolCalls =
        !res.parsed.empty() && res.parsed[0].is_object() &&
        res.parsed[0].contains("tool_calls") &&
        res.parsed[0]["tool_calls"].is_array() &&
        res.parsed[0]["tool_calls"].size() > 0;
    std::string out = acc;
    if (!hasToolCalls && out.empty() && !res.texts.empty()) out = res.texts[0];
    if (hasToolCalls) {
      bool blank = true;
      for (const char c : out)
        if (!std::isspace(static_cast<unsigned char>(c))) { blank = false; break; }
      if (blank)
        out.clear();
    }
    // The content + reasoning_content (or tool_calls) have already streamed to stdout per chunk
    // (this streamer emits them live), so the terminal line carries only the header + finish +
    // metrics + gen_config -- pass /*streamed=*/true; it no longer re-emits result / tool_calls.
    emitResult(isWorker, reqId, sessionId, out, hasToolCalls, res, reqConfig,
               /*streamed=*/true);
    // (1) Append the assistant reply back to the session so the NEXT turn on this session renders
    // it too (this is the per-session context a long-lived worker relies on).  A pure tool-call turn
    // has `out` blanked, so it appends an empty-content assistant message -- harmless; the tool_calls
    // it produced are on the wire (the per-turn RESULT already carried them when not streamed).
    ov::genai::JsonContainer reply = ov::genai::JsonContainer::object();
    reply["role"] = std::string("assistant");
    reply["content"] = out;
    history.push_back(reply);
    return 0;
  };

  if (!useEngineTemplate) {
    // --- (tools | kwargs) pre-apply (string path, apply_chat_template = false).  Mirror OpenArc:
    // apply the template here (the model's own template -- "" = keep the loaded one -- plus tools /
    // extra_context), then hand the resulting string to the string generate with the gate OFF so the
    // engine does not template a second time.  The engine's own normalize_media_tags then maps the
    // (model-native) tags the template emitted onto the images by order.  The HISTORY rendered is the
    // (now accumulated) session -- the pre-apply path renders the same accumulated conversation the
    // engine path would, just hand-templated.
    cfg.apply_chat_template = false;
    std::optional<ov::genai::JsonContainer> tools = pr.hasTools
        ? std::optional<ov::genai::JsonContainer>(ov::genai::JsonContainer::from_json_string(pr.toolsJson))
        : std::nullopt;
    // Project the effective thinking switch onto the render (apply_chat_template's extra_context) so
    // the engine's applied template matches the parser split: enable_thinking == thinking, the SAME
    // effective value -- render + split can never disagree.  This also removes the engine's own
    // render-time inject-default from the pre-apply path (so a model that does not actually reason
    // no longer gets the reasoning-on form).
    rapidjson::Document ek;
    if (pr.hasKwargs && !pr.kwargsJson.empty())
      ek.Parse(pr.kwargsJson.c_str());
    if (!ek.IsObject()) ek.SetObject();
    auto& ekAlloc = ek.GetAllocator();
    ek.RemoveMember("enable_thinking");
    rapidjson::Value ekKey("enable_thinking", ekAlloc);
    rapidjson::Value ekVal; ekVal.SetBool(thinking);
    ek.AddMember(ekKey, ekVal, ekAlloc);
    std::optional<ov::genai::JsonContainer> extra =
      std::optional<ov::genai::JsonContainer>(
        ov::genai::JsonContainer::from_json_string(config::compactJsonOf(ek)));
    std::string rendered = tokenizer.apply_chat_template(
        history, pr.addGenerationPrompt, std::string(), tools, extra);

    // The single DynamicReasoningIncrementalParser is ALWAYS first (instance() puts it at slot 0, the
    // tool parser after) -- the one parser for both sides.  The pipeline sets its on/off via setActive
    // using the SAME effective thinking the render above used (render + split can never disagree), then
    // the ContentStreamer runs over the list.
    auto streamer = ::ovgenai::toolcall::ContentStreamer::instance(
        tokenizer, modelFamily.value_or(std::string()), thinking, acc, stream);
    auto res = pipe->generate(rendered, pr.images, cfg, ov::genai::StreamerVariant(streamer));
    // (b) BRIDGE: the VLM pipeline this command drives (unlike the LLM pipeline's
    // run_generate_with_parsers) never copies a TextParserStreamer's get_parsed_message() into
    // res.parsed, so a tool-call turn would leave res.parsed EMPTY and emitResult reported the
    // engine's own finish_reason "tool_calls" yet NO tool_calls array.  The CLI still holds the
    // carrier (a ContentStreamer, a TextParserStreamer), so bridge it: a plain TextStreamer makes
    // the cast miss (no-op).
    ::ovgenai::toolcall::bridgeParsedMessage(streamer, res);
    return finalize(res);
  }

  // --- engine path (default, no tools/kwargs): the HISTORY form, apply_chat_template = true -- the
  // history path always applies the template in the engine (the gate only affects the string form) and
  // splices <ov_genai_image_N>/<ov_genai_video_N>/<ov_genai_audio_N>, which the translator already put
  // in the content parts, so we just pass the (accumulated) history + this request's media.
  cfg.apply_chat_template = true;
  if (history.empty())
    throw config::ConfigError("vlm: no message to generate -- give a chat via --input,"
                              " pipe one on stdin, or type one at a terminal (a lone"
                              " '.' ends the input, as in the mail command)");
  auto streamer = ::ovgenai::toolcall::ContentStreamer::instance(
      tokenizer, modelFamily.value_or(std::string()), thinking, acc, stream);
  auto res = pipe->generate(history, pr.images, pr.videos, pr.audios, {}, cfg,
                           ov::genai::StreamerVariant(streamer));
  ::ovgenai::toolcall::bridgeParsedMessage(streamer, res);
  return finalize(res);
}

}  // close the file-local (vlm-specific helpers) anonymous namespace


// --- The vlm body (the one real body today) -- but it no longer parses the command line -- --------
// It receives the PARSED arguments runModeEntry produced ONCE for the whole subcommand (a
// PipelineParsed: the four typed configs + the run form's --input + the raw model/device the body
// was given) -- so the body owns only "create the pipeline + generate" (the PipelineBinding's job);
// the general argument parsing is the front-end's (commands.cpp runModeEntry -> parsePipelineArgs),
// and the --help this vlm used to render in-body is likewise the front-end's (the front-end
// intercepts --help BEFORE delegating, so a body is never reached for it).  With that, the body:
//   1. reports the mode + the model it was given, as the stubs do (so a failure is still visible);
//   2. CREATE the pipeline (load a VLMPipeline from the resolved load/runtime/scheduler config) --
//      the PipelineBinding's job, so the body, not the front-end, owns the load;
//   3. build a GenerateRequestFn that drives it (capturing the created pipeline + tokenizer) and
//      hand a SessionDispatcher built around that pipeline (its injected run/worker driver);
//   4. call dispatcher.run() (the run one-shot) or .worker() (the long-lived form) on it.
// For a `run` one-shot the dispatcher makes exactly one generated session; for a `worker` it hooks
// stdin and selects / creates a session per incoming line's session_id.  There is no config to MISS
// here (runModeEntry already validated the load set before routing, and handed it in), so a run
// failure is a load / generate throw (exit 1) -- not a usage error.
int runVlmBody(const std::string& prog, const PipelineParsed& args, bool worker) {
  // Report the mode + the *requested* model / device up front (mirrors the other pipelines' stub),
  // so the mode is visible on stderr even if a later step (the load) then fails.
  std::fprintf(stderr, "ovgenai: vlm%s: loading model '%s' on %s\n",
               worker ? " (worker mode)" : " (one-shot)", args.model.c_str(), args.device.c_str());
  // For a resolved ref (an `hf://` -> a snapshot dir) that is not the path actually loaded, report
  // where it resolved so a long load is identifiable.  A plain model path is returned verbatim, so
  // the resolve line is skipped for it.
  if (args.model.rfind("hf://", 0) == 0 && args.model != args.load.model)
    std::fprintf(stderr, "ovgenai: vlm: resolved %s -> %s\n",
                 args.model.c_str(), args.load.model.c_str());

  try {
    // Create the pipeline (the PipelineBinding's job) -- and for `vlm` the *create* is itself the
    // interesting part: a VLMPipeline is a pimpl (neither copyable nor moveable) so hold it as a
    // shared_ptr to the PUBLIC base (the base carries the virtual generate() overloads +
    // get_tokenizer the vlm generate reaches polymorphically) that the generate callback captures +
    // keeps alive for the (possibly many-turn) session.
    GenAIPipelineConfig pcfg(args.load, args.scheduler, args.runtime);
    std::shared_ptr<ov::genai::VLMPipeline> owned = std::make_shared<ov::genai::VLMPipeline>(
        args.load.model, args.load.device, pcfg.toPipelineProperties());
    ov::genai::Tokenizer tokenizer = owned->get_tokenizer();
    std::shared_ptr<ov::genai::VLMPipelineBase> base = owned;   // to the public base (virtual dispatch)

    // The per-pipeline generate() the SessionDispatcher runs: capture the created pipeline + tokenizer
    // (the closure stays alive with the dispatcher); the per-request sampling (the base reqConfig,
    // folded per request line for a worker), the request's input, and the selected session's history
    // are passed in per call, and vlmGenerate translates + appends that request to the session,
    // renders the accumulated history, generates, emits the RESULT line (with the session id), and
    // appends the assistant reply back to it -- the part that is still genuinely vlm-specific.
    GenerateRequestFn generate = [base, tokenizer](
        const std::string& sid, ov::genai::ChatHistory& hist,
        const config::ModelRequestConfig& rc, const std::string& in,
        bool isWorker, std::int64_t reqId) -> int {
      return vlmGenerate(base, tokenizer, sid, hist, rc, in, isWorker, reqId);
    };

    // Create the SessionDispatcher around the initialized pipeline -- the injected generate closure
    // captures the loaded VLMPipeline (+ its tokenizer); the per-request base reqConfig is the parsed
    // request config; the run form's direct chat (its --input, nullopt for a worker) is the cliInput
    // -- and run it in the requested form.
    SessionDispatcher dispatcher(std::move(generate), args.request, args.input);
    if (worker)
      return dispatcher.worker();
    return dispatcher.run();
  } catch (const config::ConfigError& e) {
    // A usage error that surfaced mid-run -- notably a bad image part: vlmGenerate throws
    // "vlm: could not open image file '...'" at DECODE time, after the load config is already
    // valid and the load is well under way.  It is still a ConfigError (a usage problem), so it
    // is exit 2 -- and this handler must sit BEFORE the std::exception catch below (a ConfigError
    // is-a std::exception, so the more-specific one must come first, or it would fall through to
    // the run-failure exit 1, which is what an earlier draft of this body got wrong).
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    // A run failure (a model-load / generate throw that is NOT a ConfigError) -- not a usage
    // error; the label was already printed, so just report and exit non-zero.  The load-config
    // set itself never reaches here: runModeEntry validated it -- and reported any miss -- before
    // routing; only a LATER mid-run ConfigError (the catch above), or a genuine run failure, does.
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 1;
  }
}

}  // namespace ovgenai

// --- The pipeline registry --------------------------------------------------------------
// The run/worker binding table, as a file-local `static const` map keyed by the --pipeline name
// (runModeEntry no longer builds or scans it: it parses the one argument set exactly once, then calls
// dispatchPipeline, which looks it up by name -- a single find, not a linear scan over the set).
// Each entry says ONLY which body runs that pipeline -- NOT its description: the descriptions stay in
// the canonical name set (it feeds show-pipelines + the "choose one of" routing message), so this
// table carries no descriptions.  (A name in the table but missing from the canonical set -- or
// vice-versa -- would show up at run time: the everyPipeline test runs every canonical pipeline by
// name and expects exit 0, so the two sets stay in lock-step.)
//
// Each entry's body is set EXPLICITLY (one line each) -- NOT built in a loop with an `if`: `vlm`
// (the one pipeline with a real body today) is its own lambda that drives runVlmBody; the other
// thirteen not-yet-implemented pipelines share one lambda onto runPipelineStub (the infrastructure-only
// stub the front-end owns as the single --help source).  When a pipeline grows a real body, only its
// one entry changes -- no shared stub code, no per-name `if`.
namespace ovgenai {
namespace commands {

// The single shared body for the not-yet-implemented pipelines: runPipelineStub, fed the pipeline name
// it just ran (parsed.pipeline) so the stub can name itself; the rest of its report comes from the
// parsed model / device.  One lambda, referenced by every non-vlm entry below.
static const std::function<int(const std::string&, const PipelineParsed&, bool)>
  runPipelineStubBody =
    [](const std::string& prog, const PipelineParsed& parsed, bool worker) -> int {
      return runPipelineStub(parsed.pipeline, parsed, worker);
    };

// The table itself (the requested "assign as static const ... we don't need a function at all"):
// one explicit entry per canonical pipeline -- `vlm` its own lambda (runVlmBody); the other thirteen
// share runPipelineStub above.  An entry stores only its name + its body (a PipelineBinding is a
// { name, run } pair): the name is the lookup key, run is the body dispatchPipeline forwards to with
// the parsed arguments + the program name + the mode.
static const std::unordered_map<std::string, PipelineBinding> kPipelineBindings = {
  { "llm",            { "llm",            runPipelineStubBody } },
  { "vlm",            { "vlm",            [](const std::string& prog, const PipelineParsed& parsed, bool worker) -> int { return ::ovgenai::runVlmBody(prog, parsed, worker); } } },
  { "whisper",        { "whisper",        runPipelineStubBody } },
  { "asr",            { "asr",            runPipelineStubBody } },
  { "tts",            { "tts",            runPipelineStubBody } },
  { "omni",           { "omni",           runPipelineStubBody } },
  { "embedding",      { "embedding",      runPipelineStubBody } },
  { "text-embedding", { "text-embedding", runPipelineStubBody } },
  { "rerank",         { "rerank",         runPipelineStubBody } },
  { "text2image",     { "text2image",     runPipelineStubBody } },
  { "image2image",    { "image2image",    runPipelineStubBody } },
  { "inpainting",     { "inpainting",     runPipelineStubBody } },
  { "text2video",     { "text2video",     runPipelineStubBody } },
  { "image2video",    { "image2video",    runPipelineStubBody } },
};

// run / worker's single delegated entry point: look the selected --pipeline up by name (a single find
// over kPipelineBindings) and run that binding's body -- so runModeEntry neither looks up nor routes a
// pipeline itself; it parses the one argument set exactly once, then calls this.  If the name is not
// in the table (unknown --pipeline) it is a usage error: named + pointed at show-pipelines (the
// canonical set via pipelineNames(), the same text that listing renders) + exit 2.  A hit runs the
// bound body and its exit code (0 ok / 1 a run failure / 2 a usage failure) is returned.
int dispatchPipeline(const std::string& modeName, const std::string& prog,
                     const PipelineParsed& parsed, bool worker) {
  const auto it = kPipelineBindings.find(parsed.pipeline);
  if (it == kPipelineBindings.end()) {
    std::fprintf(stderr, "ovgenai: %s: unknown pipeline '%s'; choose one of: %s\n",
                 modeName.c_str(), parsed.pipeline.c_str(),
                 config::joinList(pipelineNames(), ", ").c_str());
    return 2;
  }
  return it->second.run(prog, parsed, worker);
}

}  // namespace commands
}  // namespace ovgenai
