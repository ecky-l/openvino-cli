// The `vlm` subcommand's real body -- Design A: a thin C++ pipe around
// ov::genai::VLMPipeline.  See vlm.hpp for the run entry point.
//
// The subcommand owns exactly the translation the design requires, and nothing
// of the per-model *parsing* (that stays in the Python consumer):
//   * the input surface -- one only: the chat comes from --input, otherwise from
//     stdin (a pipe => one gen_config line; a terminal => typed line-by-line until
//     a lone '.', the mail command's way);
//   * an image part may carry a {"type":"image_url","url":...} that is a file
//     ("file://.." / ~/ / absolute / relative to the CWD) or the built-in base64
//     data: form;
//   * the JSON -> ChatHistory + media-tensor translator (the one bit of real new
//     code in the CLI): per image/video/audio part we (a) rewrite the content
//     item to the engine's universal `{"type":"image"|"video"|"audio"}` and
//     (b) push its decoded tensor onto the matching media vector in part order;
//   * the (b) specials-preserving streamer -- the engine decodes its own
//     result.texts with skip_special_tokens=true (it strips
//     </> etc.), so the emitted text is the text the
//     streamer accumulated, kept unstripped;
//   * the (c) native finish_reason mapping (NONE/STOP/LENGTH/TOOL_CALL).
//
// Engine-prompt construction is done *by hand* as JSON strings (built with a
// small escaper) and handed to the engine's own parsers/JsonContainer
// (from_json_string / apply_chat_template), so the CLI never has to build
// ov::Any/AnyMap/JsonContainer structs -- that footgun-prone work stays the
// engine's.  The CLI reads the request with rapidjson, constructs the output as
// JSON the same robust way, and only feeds the engine strings.
#include "vlm.hpp"

#include "config.hpp"                 // ConfigError / preProcess / rephraseBoost / compactJsonOf
#include "util.hpp"                   // base64_decode
#include "model_load_config.hpp"      // parseModelLoadConfig + ModelLoadConfig
#include "model_runtime_config.hpp"   // parseModelRuntimeConfig + ModelRuntimeConfig
#include "model_request_config.hpp"   // parseModelRequestConfig / modelRequestConfigFromJson / modelRequestConfigToJson
#include "model_scheduler_config.hpp" // parseModelSchedulerConfig + ModelSchedulerConfig
#include "tool_call_support.hpp"      // (b) the CLI-side tool-call parser + registry (ContentStreamer / select* / detect*)
#include "genai_config.hpp"           // makeGenerationConfig + GenAIPipelineConfig + re-exported config types

#include <boost/program_options.hpp>

#include <rapidjson/document.h>

#include <openvino/core/descriptor/tensor.hpp>   // ov::Tensor / ov::Shape / ov::element::u8 (also pulls any.hpp -> ov::AnyMap)
#include <openvino/genai/generation_config.hpp>        // GenerationConfig + apply_chat_template
#include <openvino/genai/generation_handle.hpp>       // GenerationFinishReason
#include <openvino/genai/streamer_base.hpp>           // StreamerBase / StreamingStatus
#include <openvino/genai/text_streamer.hpp>           // TextStreamer / CallbackTypeVariant / skip_special_tokens
#include <openvino/genai/tokenizer.hpp>              // Tokenizer + apply_chat_template
#include <openvino/genai/json_container.hpp>         // JsonContainer::from_json_string
#include <openvino/genai/chat_history.hpp>           // ChatHistory
#include <openvino/genai/visual_language/pipeline.hpp>     // VLMPipeline + VLMDecodedResults
#include <openvino/genai/visual_language/perf_metrics.hpp> // VLMPerfMetrics

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// A `po` alias at global scope; my vlm code lives in ovgenai (not ovgenai::config,
// whose config.hpp already has its own `po`), so it needs its own one.
namespace po = boost::program_options;

namespace ovgenai {
namespace {

// --- small, self-contained helpers --------------------------------------------

// --help / -h present anywhere in the subcommand's args (before we load a model).
bool anyHelp(const std::vector<std::string>& a) {
  for (const auto& x : a)
    if (x == "--help" || x == "-h") return true;
  return false;
}

// --- JSON construction helpers (hand-built, so I never fight Any/AnyMap/
// JsonContainer's allocators).  produce a valid JSON fragment of `in` as a string
// literal body (without the surrounding quotes) with ", \ and all control chars
// escaped; non-ASCII bytes pass through as UTF-8 the usual way.
std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"':  out += "\\\""; continue;
      case '\\': out += "\\\\"; continue;
      case '\b': out += "\\b";  continue;
      case '\f': out += "\\f";  continue;
      case '\n': out += "\\n";  continue;
      case '\r': out += "\\r";  continue;
      case '\t': out += "\\t";  continue;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

// Append a literal JSON string member ("k":"v") for a name/value pair.
void addStrMember(std::string& out, const std::string& key, const std::string& val) {
  out += "\"";
  out += jsonEscape(key);
  out += "\":\"";
  out += jsonEscape(val);
  out += "\"";
}

// Remove the vlm-only flags (and, for the value-taking one, its value token)
// from a copy of args, so the shared section parsers -- which do not know
// --input / --worker -- will not reject them as "unknown option".  `--input=X` /
// `-i=X` are dropped whole; `--input X` / `-i X` drop the flag and its one value
// (unless that value is itself an option); `--worker` is a zero-arg flag, so it
// is dropped on its own.
std::vector<std::string> stripVlmArgs(const std::vector<std::string>& args) {
  static const std::set<std::string> kSingle = {"--input", "-i"};
  std::vector<std::string> out;
  bool afterDashDash = false;
  for (size_t i = 0; i < args.size(); ) {
    const std::string& tok = args[i];
    if (!afterDashDash && tok == "--") { out.push_back(tok); afterDashDash = true; ++i; continue; }
    if (afterDashDash) { out.push_back(tok); ++i; continue; }  // positionals pass through
    if (tok == "--worker") { ++i; continue; }   // zero-arg flag; only the first pass knows it
    if (tok.rfind("--input=", 0) == 0 || tok.rfind("-i=", 0) == 0) {
      ++i;                                                    // the =form is one token
      continue;
    }
    if (kSingle.count(tok)) {
      if (i + 1 < args.size()) {
        const std::string& nx = args[i + 1];
        if (!(nx.size() > 1 && nx[0] == '-')) ++i;            // consume its (space) value
      }
      ++i;
      continue;
    }
    out.push_back(tok);
    ++i;
  }
  return out;
}

// Read one (possibly very large) line from `in`, bounded by `limit` bytes; an
// over-limit line is a ConfigError.  This is the line-JSON reader a --worker form
// needs (the protocol limit is large -- an inlined-base64 conversation is tens of
// MiB on one line -- so this must not assume a 64-KiB getline).
std::string readOneLine(std::istream& in, std::size_t limit) {
  std::string s;
  int ch = in.get();
  while (ch != EOF) {
    if (ch == '\n') break;
    if (limit && s.size() + 1 > limit)
      throw config::ConfigError("vlm input line exceeds the " + std::to_string(limit) +
                                 "-byte limit");
    s.push_back(static_cast<char>(ch));
    ch = in.get();
  }
  return s;
}

// The whole-stream reader the one-shot's piped / redirected (a non-tty) case needs: a
// multi-line / pretty-printed chat is ONE document, so read it all the way to EOF (no
// per-line "\n" stop like readOneLine).  Same char-by-char byte-cap shape as readOneLine
// (a runaway pipe still throws), but the worker keeps one-op-per-line via readOneLine --
// this stays the one-shot only.  The cap is the shared protocolLineLimit().
std::string readAllInput(std::istream& in, std::size_t limit) {
  std::string s;
  int ch = in.get();
  while (ch != EOF) {
    if (limit && s.size() + 1 > limit)
      throw config::ConfigError("vlm input exceeds the " + std::to_string(limit) +
                                 "-byte limit");
    s.push_back(static_cast<char>(ch));
    ch = in.get();
  }
  return s;
}

std::size_t protocolLineLimit() {
#ifdef OVGENAI_PROTOCOL_LINE_LIMIT
  return static_cast<std::size_t>(OVGENAI_PROTOCOL_LINE_LIMIT);
#else
  return (256u * 1024u * 1024u);
#endif
}

// Read chat input a line at a time from a terminal, the classic mail(1) way: a
// line that is a lone '.' (and nothing else) ends the input and is not part of
// it; every earlier line is kept (each with its trailing newline, so a
// pretty-printed chat JSON is reassembled verbatim).  EOF before a lone '.' also
// ends the input (whatever was typed is used).  std::getline grows the string, so
// a long single line is fine -- the lone '.' is what bounds an interactive
// session, not a byte cap.
std::string readMailInput() {
  std::string acc;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == ".") break;
    acc += line;
    acc.push_back('\n');
  }
  return acc;
}

// Acquire the one-shot chat input -- the precedent (req 2 + 3):
//   A. --input given  -> return its value verbatim; read NO stdin at all.
//   B. --input absent AND stdin is NOT a tty (a pipe / redirect) -> read the whole
//      stdin to EOF, then parse it: a multi-line / pretty-printed chat is ONE document.
//      (The --worker keeps one line per op via readOneLine; it never reaches here.)
//   C. --input absent AND stdin IS a tty (nothing piped) -> read the chat from the
//      command line until a lone '.', the mail command's way.
// A is decided first, so a present --input never touches stdin (req 2); the isatty
// probe runs only when --input was omitted (req 3 is the "nothing piped" case).
// An empty result is fine: executeRequest reports "no message".
std::string acquireInput(const std::optional<std::string>& cliInput) {
  if (cliInput.has_value()) return *cliInput;                      // A: --input given
  if (isatty(STDIN_FILENO)) return readMailInput();                // C: a terminal
  return readAllInput(std::cin, protocolLineLimit());               // B: piped -> read to EOF, then parse
}

// The base64 payload of a data: URL -- strip a "data:image/..;base64," prefix if
// present, else treat the whole string (a bare base64 blob) as the payload.
std::string dataUrlBase64(const std::string& url) {
  if (url.rfind("data:", 0) == 0) {
    auto comma = url.find(',');
    if (comma != std::string::npos) return url.substr(comma + 1);
  }
  return url;
}

// Decode an image part / --images entry to an ov::Tensor [N,H,W,3] (uint8).
//
// The openvino.genai C++ API takes *already-decoded* pixel tensors but exports
// no image-format (PNG/JPEG) decoder, so this decodes the base64 *payload* and
// keeps only the raw bytes as a single [1,1,1,3] placeholder tensor -- enough to
// keep the image-part / image-tensor count one-to-one (so the engine's vision-
// index invariant holds) without a real format decode.  Wiring a real raster
// decoder (e.g. a small stb_image) is a later enhancement; a warning is logged the
// first time an image is seen, so this is transparent, not hidden.
ov::Tensor makeImageTensorFromBytes(const std::string& bytes) {
  ov::Tensor t(ov::element::u8, ov::Shape{1, 1, 1, 3});
  auto p = t.data<unsigned char>();
  for (size_t i = 0; i < 3; ++i) p[i] = (i < bytes.size()) ? static_cast<unsigned char>(bytes[i]) : 0x00;
  return t;
}

// Expand a leading '~' (== $HOME) or '~/...' to a full home path; anything else
// is used as-is (an absolute '/...' path, or one relative to the CWD).  util's
// env_or is the shared $HOME lookup; an unset/empty HOME with a '~' is a ConfigError.
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

// Resolve the *source* of an image part's url and decode it to an ov::Tensor
// [N,H,W,3] (uint8).  The url is one of, in priority order (req 4):
//   * "data:<mime>;base64,XXX"  -> the built-in mime-type / base64 form: base64-
//                                   decode the payload (kept, in addition).
//   * a "file://..." URL        -> strip the scheme, resolve the rest as a path.
//   * "~" / "~/..."             -> expand ~ to $HOME, then read the file.
//   * an absolute "/..." / a relative (to the CWD) path -> read the file.
// openvino.genai exports no image-format decoder, so the (read / decoded) bytes
// are kept as a [1,1,1,3] placeholder tensor -- enough to keep the image-part /
// image-tensor count one-to-one (so the engine's vision-index invariant holds)
// without a real raster decode; a real decoder is a later enhancement.  The first
// time any image is seen a transparent note is logged once (not hidden).
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

// --- the request -> normalised ChatHistory + media translator -----------------
// (the one real bit of new code in the CLI; see the module header)
struct ParsedRequest {
  std::string normMessagesJson;             // compact JSON array of normalised messages
  std::vector<ov::Tensor> images;
  std::vector<ov::Tensor> videos;
  std::vector<ov::Tensor> audios;
  bool hasTools = false;                     // a *non-empty* tools array is present
  bool hasKwargs = false;                    // a chat_template_kwargs object is present
  bool addGenerationPrompt = true;          // for the (pre-apply) string path
  std::string toolsJson;                    // compact, only when hasTools (for apply_chat_template)
  std::string kwargsJson;                   // compact, only when hasKwargs
};

// Serialise one normalised message as a JSON object fragment, rewriting a
// multipart (list) *user* content into the engine's universal media parts
// (and collecting the decoded tensors onto `pr`); a *non-user* list content is
// flattened to its joined text so the engine's per-role `content.get_string()`
// never throws.  Every other member of the original message (role, tool_calls,
// tool_call_id, ...) is carried through verbatim.
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

// Translate a gen_config object (`g`, read-only) into a ParsedRequest: build the
// normalised messages array as a compact JSON string (for from_json_string), and
// detect the tools/kwargs sub-rule + the pre-apply add_generation_prompt.
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

// (b) the specials-preserving streamer: detokenizes with skip_special_tokens
// = false so the CLI accumulates the *full* decoded text (
//  etc. intact); the engine's own result.texts is stripped (see the module
// header).  The lambda returns StreamingStatus::RUNNING so the CLI never stops
// generation itself -- the engine's own eos/stop drives the (c) finish reason.
std::shared_ptr<ov::genai::StreamerBase> makeAccumulatingStreamer(
    const ov::genai::Tokenizer& tok, std::string& acc) {
  auto cb = [&acc](std::string chunk) -> ov::genai::CallbackTypeVariant {
    acc += std::move(chunk);
    return ov::genai::StreamingStatus::RUNNING;
  };
  ov::AnyMap detok{ ov::genai::skip_special_tokens(false) };
  return std::make_shared<ov::genai::TextStreamer>(tok, cb, detok);
}

// (b) The parser-aware streamer for the (tools | kwargs) pre-apply path.  When a
// tool-call parser is selected (an explicit --tool-call-parser, else auto-detected
// from the model's OWN rendered template -- see selectToolCallParserName) we carry it
// in a ContentStreamer, a TextParserStreamer subclass: the engine both runs the parser
// loop AND copies its get_parsed_message() (which, on a completed call, holds .tool_calls)
// into res.parsed[0] -- it copies parsed[0] regardless of cfg.parsers, because it first
// dynamic-casts the streamer to TextParserStreamer -- so we run the parser IN the streamer
// and leave cfg.parsers empty.  With no selection it keeps today's plain accumulating
// TextStreamer: no .tool_calls, no TOOL_CALL_STOP, finish_reason stays stop.
std::shared_ptr<ov::genai::StreamerBase> makeContentStreamer(
    const ov::genai::Tokenizer& tok,
    const std::optional<std::shared_ptr<ov::genai::IncrementalParser>>& parser,
    std::string& acc) {
  if (!parser.has_value() || !*parser)
    return makeAccumulatingStreamer(tok, acc);
  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers{std::move(*parser)};
  return std::make_shared<::ovgenai::toolcall::ContentStreamer>(tok, std::move(parsers), acc);
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
//   { "v,omitempty" "type","result":"<specialspreserving text>",
//     "finish_reason":null|"stop"|"length"|"tool_calls", "metrics":{...},
//     "gen_config":{...echoed...} }  -- one compact line.
void emitResult(bool isWorker, std::int64_t reqId,
                const std::string& text,
                const ov::genai::VLMDecodedResults& res,
                const ovgenai::config::ModelRequestConfig& reqConfig) {
  // (b) The structured tool-call read from the streamer (a TextParserStreamer subclass) lands in
  // res.parsed[0]["tool_calls"], which the engine copies from get_parsed_message() (the parser
  // flips TOOL_CALL_STOP when a call is complete).  We key EVERYTHING off that, NOT off the
  // engine's own finish reason: a small model can under-rand before its OUTER ‹/tool_call› tag,
  // leaving the engine's stored finish as "stop" even though the PARSER (our authority) already
  // recorded a call.  So a non-empty tool_calls => we report finish_reason "tool_calls".
  // When a tool call is present we keep the specials-preserving `out` (NOT the engine's
  // specials-stripped res.texts[0], which would re-expose the raw ‹tool_call› tags); the
  // res.texts[0] fallback stays for the no-parser / no-call path.
  const bool hasToolCalls =
      !res.parsed.empty() && res.parsed[0].is_object() &&
      res.parsed[0].contains("tool_calls") &&
      res.parsed[0]["tool_calls"].is_array() &&
      res.parsed[0]["tool_calls"].size() > 0;
  std::string out = text;
  if (!hasToolCalls && out.empty() && !res.texts.empty()) out = res.texts[0];
  // A pure tool-call turn has no prose to show: if the only content in `out` is the model's own
  // leading whitespace before the block, blank it (so a pure call reports result:"" rather than a
  // bare newline).  A real prose-then-call turn carries non-whitespace text and is left intact.
  if (hasToolCalls) {
    bool blank = true;
    for (const char c : out)
      if (!std::isspace(static_cast<unsigned char>(c))) { blank = false; break; }
    if (blank)
      out.clear();
  }

  std::vector<ov::genai::GenerationFinishReason> fr = res.finish_reasons;
  ov::genai::GenerationFinishReason finish =
      fr.empty() ? ov::genai::GenerationFinishReason::NONE : fr[0];

  std::string line = "{";
  auto first = true;
  auto put = [&](const std::string& frag) {
    if (!first) line += ",";
    line += frag;
    first = false;
  };
  if (isWorker) {
    put(std::string("\"v\":1"));
    put(std::string("\"type\":\"RESULT\""));
    put("\"req_id\":" + std::to_string(reqId));
  }
  put("\"result\":\"" + jsonEscape(out) + "\"");

  // (b) the structured tool calls, if any -- one compact array so the worker's single-line
  // JSONL stays one physical line (to_json_string(-1) is the compact dumper).
  if (hasToolCalls)
    put(std::string("\"tool_calls\":") + res.parsed[0]["tool_calls"].to_json_string(-1));

  // The PARSER is the authority for a tool-call turn: a non-empty tool_calls wins over the
  // engine's finish, which may read "stop" when the model under-rands its ‹/tool_call› outer tag
  // (we finalize on the inner ‹/function›, so the turn is still reported as a tool call, not a stop).
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

// A worker control line (READY/PONG/CANCEL/DONE/ERROR), one compact JSON object
// per line -- the {worker->parent} side of the wire protocol.
void emitWorkerLine(const std::string& type, std::int64_t reqId = 0) {
  std::string line = std::string("{\"v\":1,\"type\":\"") + type + "\"";
  if (reqId) line += std::string(",\"req_id\":") + std::to_string(reqId);
  line += "}";
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}
void emitWorkerError(const std::string& msg, std::int64_t reqId = 0) {
  std::string line = std::string("{\"v\":1,\"type\":\"ERROR\",\"error\":\"") + jsonEscape(msg) + "\"";
  if (reqId) line += std::string(",\"req_id\":") + std::to_string(reqId);
  line += "}";
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// The core one-request driver shared by the one-shot and worker forms.
// `inputJson` carries the conversation (a full chat, gen_config shape): for a
// one-shot this is the acquiried input (the --input value when given, or -- when
// --input is omitted -- the chat read from stdin); for a worker GENERATE it is
// that request's gen_config, echoed compactly.  `reqConfig` is the *sampling* for
// this one request (CLI-derived for a one-shot; gen_config-derived for a worker
// GENERATE).  Input is always a chat the model's template is applied to -- there
// is no raw / no-template path.
int executeRequest(ov::genai::VLMPipeline& pipe, ov::genai::Tokenizer tokenizer,
                   const ovgenai::config::ModelRequestConfig& reqConfig,
                   const std::string& inputJson,
                   bool isWorker, std::int64_t reqId) {
  // Build the base GenerationConfig from the per-request sampling (the
  // apply_chat_template gate is set per-branch below -- the (tools|kwargs)
  // sub-rule decides whether the engine or the CLI applies the model template).
  ov::genai::GenerationConfig cfg = makeGenerationConfig(reqConfig);

  // The one input surface: parse the (possibly multi-line) chat as a gen_config
  // and translate it to a normalised ChatHistory + media tensors.  An empty input
  // leaves the messages "[]" so the engine path reports "no message".
  ParsedRequest pr;
  if (!inputJson.empty()) {
    rapidjson::Document g;
    if (g.Parse<0>(inputJson.c_str()).HasParseError())
      throw config::ConfigError("vlm: input is not valid JSON at position " +
                                 std::to_string(g.GetErrorOffset()));
    pr = translateGenConfig(g);
  }

  bool useEngineTemplate = !(pr.hasTools || pr.hasKwargs);   // the (tools|kwargs) sub-rule

  if (!useEngineTemplate) {
    // --- (tools | kwargs) pre-apply (string path, apply_chat_template = false).
    // Mirror OpenArc: apply the template here (the model's own template -- "" =
    // keep the loaded one -- plus tools / extra_context), then hand the resulting
    // string to the string generate with the gate OFF so the engine does not
    // template a second time.  The engine's own normalize_media_tags then maps
    // the (model-native) tags the template emitted onto the images by order.
    cfg.apply_chat_template = false;
    ov::genai::ChatHistory history(ov::genai::JsonContainer::from_json_string(pr.normMessagesJson));
    std::optional<ov::genai::JsonContainer> tools = pr.hasTools
        ? std::optional<ov::genai::JsonContainer>(ov::genai::JsonContainer::from_json_string(pr.toolsJson))
        : std::nullopt;
    std::optional<ov::genai::JsonContainer> extra = pr.hasKwargs
        ? std::optional<ov::genai::JsonContainer>(ov::genai::JsonContainer::from_json_string(pr.kwargsJson))
        : std::nullopt;
    std::string rendered = tokenizer.apply_chat_template(
        history, pr.addGenerationPrompt, std::string(), tools, extra);
    // (b) Resolve the tool-call parser for THIS request: an explicit --tool-call-parser
    // wins; else auto-detect from the model's OWN rendered template (the model-intrinsic
    // sentinel); else none.  A selected-but-unwired name resolves to no parser and degrades
    // to plain text; an auto-detected (or explicitly selected + wired) format is reported on
    // stderr (never the wire stdout, so the protocol stays one-JSON-per-line).
    std::optional<std::string> detectedName;
    const auto parserName = ::ovgenai::toolcall::selectToolCallParserName(
        reqConfig.tool_call_parser, tokenizer, &detectedName);
    std::optional<std::shared_ptr<ov::genai::IncrementalParser>> parser;
    if (parserName.has_value() && !parserName->empty()) {
      auto p = ::ovgenai::toolcall::makeToolCallParserByName(*parserName);
      if (p) {
        parser = std::move(p);
        std::fprintf(stderr, "ovgenai: vlm: using tool-call parser '%s'\n", parserName->c_str());
      } else {
        std::fprintf(stderr, "ovgenai: vlm: --tool-call-parser '%s' is advertised but not yet "
                             "wired in the CLI; falling back to plain text (no tool_calls)\n",
                     parserName->c_str());
      }
    } else if (detectedName.has_value()) {
      std::fprintf(stderr, "ovgenai: vlm: auto-detected tool-call format '%s' from the model's "
                           "chat template\n", detectedName->c_str());
    }
    std::string acc;
    auto streamer = makeContentStreamer(tokenizer, parser, acc);
    auto res = pipe.generate(rendered, pr.images, cfg, ov::genai::StreamerVariant(streamer));
    emitResult(isWorker, reqId, acc, res, reqConfig);
    return 0;
  }

  // --- engine path (default, no tools/kwargs): the HISTORY form, apply_chat_
  // template = true (the history path always applies the template in the engine
  // -- the gate only affects the string form -- and splices
  // <ov_genai_image_N>/<ov_genai_video_N>/<ov_genai_audio_N>, which the translator
  // already put in the content parts), so we just pass the history + media.
  cfg.apply_chat_template = true;
  if (pr.normMessagesJson.empty() || pr.normMessagesJson == "[]")
    throw config::ConfigError("vlm: no message to generate -- give a chat via --input,"
                              " pipe one on stdin, or type one at a terminal (a lone"
                              " '.' ends the input, as in the mail command)");
  ov::genai::ChatHistory history(ov::genai::JsonContainer::from_json_string(pr.normMessagesJson));
  std::string acc;
  auto streamer = makeAccumulatingStreamer(tokenizer, acc);
  auto res = pipe.generate(history, pr.images, pr.videos, pr.audios, {}, cfg,
                           ov::genai::StreamerVariant(streamer));
  emitResult(isWorker, reqId, acc, res, reqConfig);
  return 0;
}

// The one-shot form: acquire the chat input (acquireInput -- --input, or, when
// omitted, piped stdin / an interactive terminal read to a lone '.'), then drive
// the single executeRequest path.  A present --input is used verbatim (req 2, no
// stdin read); an over-limit piped line or a missing input surfaces as a
// ConfigError that propagates up to runVlm's catch.
int runOneShotInner(ov::genai::VLMPipeline& pipe, ov::genai::Tokenizer tokenizer,
                    const ovgenai::config::ModelRequestConfig& reqConfig,
                    const std::optional<std::string>& cliInput) {
  const std::string input = acquireInput(cliInput);
  return executeRequest(pipe, tokenizer, reqConfig, input,
                        /*isWorker=*/false, /*reqId=*/0);
}

// --- the --worker long-lived form (one JSON op per stdin line) ----------------
// Parent -> worker: GENERATE / PING / CANCEL / UNLOAD.  Worker -> parent:
// READY (once, at start) / RESULT (per GENERATE) / PONG / CANCEL / DONE /
// ERROR.  A GENERATE's gen_config carries that request's full sampling +
// messages; a malformed line is answered with an ERROR and the loop keeps
// running (one bad line must not kill the worker).  The (b)/(c) hardening rides
// the same executeRequest path as the one-shot, so streamed + native-
// finish-reason output is identical in both forms.
int runWorker(ov::genai::VLMPipeline& pipe, ov::genai::Tokenizer tokenizer,
              const ovgenai::config::ModelRequestConfig& baseReq) {
  emitWorkerLine("READY");
  const std::size_t limit = protocolLineLimit();
  for (;;) {
    std::string line;
    try {
      line = readOneLine(std::cin, limit);
    } catch (const config::ConfigError& e) {
      emitWorkerError(e.what());
      continue;
    }
    if (line.empty() && std::cin.eof()) break;   // clean EOF -> exit
    if (line.empty()) continue;                    // blank line: skip
    rapidjson::Document msg;
    if (msg.Parse<0>(line.c_str()).HasParseError()) {
      emitWorkerError("malformed line");
      continue;
    }
    std::string op = (msg.HasMember("op") && msg["op"].IsString()) ? std::string(msg["op"].GetString()) : "";
    if (op == "UNLOAD") {
      emitWorkerLine("DONE");
      break;
    } else if (op == "PING") {
      emitWorkerLine("PONG");
      continue;
    } else if (op == "CANCEL") {
      emitWorkerLine("CANCEL");
      continue;
    } else if (op == "GENERATE") {
      std::int64_t reqId = 0;
      if (msg.HasMember("req_id") && msg["req_id"].IsInt64()) reqId = msg["req_id"].GetInt64();
      else if (msg.HasMember("request_id") && msg["request_id"].IsInt64()) reqId = msg["request_id"].GetInt64();
      // Sampling for THIS request comes from the gen_config (sent per request);
      // start from the CLI defaults for any field it omits.
      ovgenai::config::ModelRequestConfig reqConfig = baseReq;
      std::string genConfigStr;
      auto gc = msg.FindMember("gen_config");
      if (gc != msg.MemberEnd() && gc->value.IsObject())
        genConfigStr = config::compactJsonOf(gc->value);
      if (!genConfigStr.empty())
        reqConfig = config::modelRequestConfigFromJson(genConfigStr);
      try {
        executeRequest(pipe, tokenizer, reqConfig, genConfigStr,
                        /*isWorker=*/true, reqId);
      } catch (const config::ConfigError& e) {
        emitWorkerError(e.what(), reqId);
      } catch (const std::exception& e) {
        emitWorkerError(e.what(), reqId);
      }
      continue;
    }
    emitWorkerError("unknown op: " + op);
  }
  return 0;
}

}  // close the anonymous namespace

// --- vlm-specific --help: the shared sections + the vlm-only options + a note --
po::options_description vlmOptionsDesc(bool withWorker, bool& worker);
std::string vlmHelp(const std::string& prog, bool withWorker) {
  std::string d;
  bool worker = false;
  d += "Usage: " + prog + " vlm [model-dir] <options>\n";
  d += "  Run a vision-language pipeline (VLMPipeline). One-shot by default;\n";
  d += "  add --worker for the long-lived piped-worker form.\n\n";
  d += "A positional <model-dir> is a friend for --model when it is the only\n";
  d += "model. The input is always a chat the model's template is applied to; it is\n";
  d += "  taken from --input when given, otherwise from stdin: a pipe -> one line,\n";
  d += "a terminal -> typed line-by-line, a lone '.' ending it (as in mail).\n";
  d += "An image part may come from a file (\"file://..\" / ~ / absolute / relative to\n";
  d += "the CWD) or the built-in base64 / data: form. --input is for one-shot (not\n";
  d += "the --worker form).\n\n";
  std::ostringstream os;
  os << vlmOptionsDesc(withWorker, worker);   // boost renders the section
  d += os.str();
  return d;
}

po::options_description vlmOptionsDesc(bool withWorker, bool& worker) {
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  s.add_options()
    ("input,i", po::value<std::string>(),
        "The chat to run, as a direct (chat-templated) input -- a gen_config JSON"
        " object ({\"messages\":[...], optionally \"tools\" / \"chat_template_kwargs\"}"
        " in the OpenArc shape). When given, no stdin is read. When omitted, the chat"
        " is read from stdin: a pipe yields one (possibly large) line, an interactive"
        " terminal is read a line at a time until a lone '.', the mail command's way."
        " An image part may carry a url that is a file (\"file://..\" / ~ / absolute"
        " / relative to the CWD) or the built-in data: / base64 form. Mutually"
        " exclusive with --worker: --input is a one-shot direct chat (no stdin), the"
        " --worker form reads its protocol from stdin -- they are not both given.");
  if (withWorker)
    s.add_options()("worker", po::bool_switch(&worker),
                     "Run as a long-lived worker subprocess: stay up and read/write"
                     " the line-JSON protocol on stdin/stdout (GENERATE/PING/CANCEL/"
                     " UNLOAD -> READY/RESULT/PONG/CANCEL/DONE/ERROR). Omit for a"
                     " one-shot run. Mutually exclusive with --input: a --worker reads"
                     " its protocol from stdin, where --input is a one-shot chat that"
                     " reads no stdin.");
  else
    worker = false;
  s.add_options()("help", po::bool_switch(), "Show this help message");
  return s;
}

// The one public entry point, registered for the `vlm` subcommand (see
// commands.cpp).  Signature is (prog, args) (not the stub's (name,desc,prog,a)):
// the body knows it is "vlm", so the table just forwards (prog, args).
int runVlm(const std::string& prog, const std::vector<std::string>& args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the "vlm" token
  if (anyHelp(sa)) {
    std::printf("%s", vlmHelp(prog, true).c_str());
    return 0;
  }

  // Parse the vlm-specific options (+ worker bound) as a first pass over the full
  // argv.  The four shared sections are in the same description so their flags
  // don't read as "unknown" here; they are re-read (typed) below from a copy with
  // the vlm-only flags stripped out.
  bool worker = false;
  po::options_description desc = vlmOptionsDesc(/*withWorker=*/true, worker);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(sa).options(desc)
                  .style(po::command_line_style::unix_style).run(),
              vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::fprintf(stderr, "ovgenai: vlm: %s\n", config::rephraseBoost(e).what());
    return 2;
  }
  // model/device straight from pass 1 (for the label emitted before loading).
  std::string model = vm.count("model") ? vm["model"].as<std::string>() : std::string();
  std::string device = vm.count("device") ? vm["device"].as<std::string>() : std::string();
  // --input (optional): the direct chat for the one-shot form; when absent a
  // chat is read from stdin instead (see acquireInput).
  std::optional<std::string> cliInput =
      vm.count("input") ? std::optional<std::string>(vm["input"].as<std::string>()) : std::nullopt;

  // --input and --worker are two exclusive input surfaces: --input is a one-shot
  // direct chat (read with no stdin), --worker is the long-lived form that reads
  // its line-JSON protocol from stdin.  Giving both is a usage error, so reject
  // it on stderr and exit 1 -- before loading a model, so the rejection needs no
  // fixture and never starts a run.  (The explicit requirement is exit 1, unlike
  // the exit 2 that the ConfigError boost/parse errors use further down.)
  if (worker && cliInput.has_value()) {
    std::fprintf(stderr, "ovgenai: vlm: --input and --worker are mutually exclusive "
                        "(give the chat via --input for the one-shot form, or on "
                        "stdin for the --worker form; not both)\n");
    return 1;
  }

  // Emit the mode+model label up front (mirrors the other pipelines' stub), so
  // the mode is visible on stderr even if a later step (e.g. load) then fails.
  std::fprintf(stderr, "ovgenai: vlm%s: loading model '%s' on %s\n",
               worker ? " --worker" : " (one-shot)",
               model.c_str(), device.c_str());

  try {
    // The typed shared configs: only their own fields are read (the vlm-only
    // flags are stripped so these parsers -- which do not know them -- accept
    // the line).
    auto strip = stripVlmArgs(sa);
    ovgenai::config::ModelRequestConfig reqConfig     = config::parseModelRequestConfig(strip);
    ovgenai::config::ModelLoadConfig     loadConfig   = config::parseModelLoadConfig(strip);
    ovgenai::config::ModelSchedulerConfig schedConfig = config::parseModelSchedulerConfig(strip);
    ovgenai::config::ModelRuntimeConfig  runtimeConfig = config::parseModelRuntimeConfig(strip);

    // The up-front label carries the *requested* model (an `hf://` ref, a repo
    // name); for a resolved ref that is not the path actually loaded, report where
    // it resolved (its snapshot dir) so a long load is identifiable.  A plain model
    // path is returned verbatim by the fold, so this stays silent for the common case.
    if (model.rfind("hf://", 0) == 0 && model != loadConfig.model)
      std::fprintf(stderr, "ovgenai: vlm: resolved %s -> %s\n",
                   model.c_str(), loadConfig.model.c_str());

    // -- Load the pipeline (throws on a missing / non-VLM model -- a run failure,
    // surfaced below).
    GenAIPipelineConfig pcfg(loadConfig, schedConfig, runtimeConfig);
    ov::genai::VLMPipeline pipeline(loadConfig.model, loadConfig.device, pcfg.toPipelineProperties());
    ov::genai::Tokenizer tokenizer = pipeline.get_tokenizer();

    if (worker)
      return runWorker(pipeline, tokenizer, reqConfig);
    return runOneShotInner(pipeline, tokenizer, reqConfig, cliInput);
  } catch (const config::ConfigError& e) {
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    // A failed model load / generation is a run failure (not a usage error); the
    // label was already printed, so just report and exit non-zero.
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 1;
  }
}

}  // namespace ovgenai
