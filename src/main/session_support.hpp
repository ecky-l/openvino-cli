// Session support -- the GENERAL half of the former vlm.cpp input handling, pulled out
// of the vlm pipeline so it is not vlm-specific: a `SessionDispatcher` that owns the
// conversation state a run/worker needs and the only thing that reads input.
//
//   * it keeps a map of every open session -- `std::map<std::string, ...>` keyed by
//     session id.  The value is a `std::shared_ptr<ov::genai::ChatHistory>` rather than
//     the `ChatHistory` itself: `ChatHistory` has a user-declared destructor (it is a
//     pimpl), so it is neither copyable nor move-constructible and a bare
//     `std::map<std::string, ChatHistory>` would not even compile -- a shared_ptr to it
//     is the natural way to hold such a type in a container, and the dispatcher hands a
//     `ChatHistory&` (the dereferenced one) to the per-pipeline generate.
//   * it moves the input readers out of vlm.cpp in as its own private methods
//     (readOneLine / readAllInput / readMailInput / acquireInput / protocolLineLimit),
//     the per-session select/create, and the {worker -> parent} control-line emitters.
//   * it exposes two forms, the same two subcommands:
//        run()    the one-shot -- exactly ONE session, given a freshly generated id that
//                 is always the selected one; its input is --input when given, else the
//                 whole stdin (a pipe read to EOF, or a terminal typed to a lone '.');
//        worker() the long-lived process -- hook stdin, read the line-JSON protocol a line
//                 at a time; every GENERATE carries a session_id, which is looked up in the
//                 map to select that session, CREATING a fresh empty ChatHistory for it on
//                 first sight (so a brand-new session is just "the id is not there yet");
//                 PING / CANCEL / UNLOAD are the global control ops.
//
// It is intentionally pipeline-agnostic.  The per-pipeline `generate()` the dispatcher runs
// against a selected session is supplied as a `GenerateRequestFn` callback (the vlm binding
// feeds its own pipeline-specific driver); only the session book-keeping + the input reading
// + the run/worker loop are here, so a later pipeline reuses this unchanged and plugs in its
// own generate.
#ifndef OVGENAI_SESSION_SUPPORT_HPP
#define OVGENAI_SESSION_SUPPORT_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "config.hpp"              // ovgenai::config::ConfigError
#include "model_request_config.hpp" // the full ModelRequestConfig (the dispatcher holds a copy as its per-request base)

// Forward-declared (the full type arrives in the .cpp): the engine's conversation it owns per
// session.  A `ChatHistory&` (returned by selectSession) and a `std::shared_ptr<ChatHistory>`
// (the map value) both name an incomplete type fine, so no header is pulled in here.
namespace ov { namespace genai { class ChatHistory; } }

namespace ovgenai {

// The per-pipeline `generate()` the dispatcher executes against a selected session.  It is
// supplied by the pipeline binding that created this dispatcher (the vlm binding captures the
// loaded VLMPipeline + its Tokenizer in it); per-request sampling arrives as `reqConfig` (the
// CLI base for a one-shot, the line's gen_config folded over it for a worker GENERATE), the
// request payload (`inputJson`, a gen_config chat) is translated + appended to `history`, the
// accumulated `history` is what the engine renders, and the produced reply is appended back so
// the NEXT turn on the same session carries it.  May throw (ConfigError / std::exception),
// which the one-shot form lets propagate to the body that built it (for the "ovgenai: <pipe>:"
// report) while the worker catches it per-line and keeps running.
using GenerateRequestFn =
    std::function<int(const std::string& /*sessionId*/,
                      ov::genai::ChatHistory& /*history*/,
                      const config::ModelRequestConfig& /*reqConfig*/,
                      const std::string& /*inputJson*/,
                      bool /*isWorker*/,
                      std::int64_t /*reqId*/)>;

// Holds open sessions, reads input, and runs one session in the form its subcommand named.
class SessionDispatcher {
 public:
  // `generate` -- the injected per-pipeline driver (a std::function, hence by value);
  // `baseReq` -- the CLI per-request config, the per-GENERATE default the worker folds its
  // line's gen_config over (and the one-shot's only request config); `cliInput` -- the
  // one-shot's direct --input (nullopt for the worker, and for a --input-less one-shot,
  // at which point run() reads stdin instead).
  SessionDispatcher(GenerateRequestFn generate,
                    config::ModelRequestConfig baseReq,
                    std::optional<std::string> cliInput);

  // The one-shot `run` form: one session, a freshly generated id that is always the selected
  // one.  Fetches the input (--input, else the whole stdin) and drives one generate on it.
  int run();

  // The long-lived `worker` form: hooks stdin and runs the line-JSON protocol.  READY is
  // emitted once up front; each line is parsed for its op (GENERATE / PING / CANCEL / UNLOAD)
  // and -- for a GENERATE -- its session_id, which selects the matching session (creating it on
  // first sight) and runs generate() on the (pre-existing) pipeline with that ChatHistory + this
  // request's input.  One bad line answers with an ERROR and the loop keeps running; a clean EOF
  // (or an UNLOAD) exits.
  int worker();

 private:
  // --- the input readers (moved in from vlm.cpp, now the dispatcher's own methods) ---

  // Read one (possibly very large) line from `in`, bounded by `limit` bytes; an over-limit
  // line is a ConfigError (the limit is protocolLineLimit() -- large, an inlined-base64
  // conversation is tens of MiB on one line).
  std::string readOneLine(std::istream& in, std::size_t limit);
  // The whole-stream reader the one-shot's piped / redirected (a non-tty) case needs: a
  // multi-line / pretty-printed chat is ONE document, so read it to EOF (no per-line stop).
  std::string readAllInput(std::istream& in, std::size_t limit);
  // The protocol line cap (256 MiB default).
  static std::size_t protocolLineLimit();
  // The terminal (interactive) reader: a line at a time, a lone '.' ends the input (mail(1)).
  std::string readMailInput();
  // The one-shot input surface: --input when given (verbatim, no stdin read); else a terminal
  // (a lone '.') or a pipe (one document read to EOF).  An empty result is fine -- the
  // generate reports "no message" for an empty input.
  std::string acquireInput();

  // A freshly generated one-shot session id (the run form has exactly one, always selected).
  // Non-static: it draws on the per-dispatch instance counter m_seq below (one fresh
  // id per dispatch -- a one-shot opens exactly one such session).
  std::string newSessionId();

  // Look up the session by id, creating a new empty ChatHistory on first sight -- the
  // worker's "select, or create if there is no session with this id".  Returns the
  // (managed) ChatHistory& the per-pipeline generate drives.
  ov::genai::ChatHistory& selectSession(const std::string& sessionId);

  // The {worker -> parent} control lines: READY / PONG / CANCEL / DONE / ERROR -- one compact
  // JSON object per physical line; a non-empty sessionId is carried on the line so the consumer
  // can correlate (READY, emitted with none, has none).
  void emitWorkerLine(const std::string& type, std::int64_t reqId, const std::string& sessionId);
  void emitWorkerError(const std::string& msg, std::int64_t reqId, const std::string& sessionId);

  // --- state ---
  GenerateRequestFn m_generate;
  config::ModelRequestConfig m_baseReq;
  std::optional<std::string> m_cliInput;

  // session id -> that session's accumulated conversation (shared_ptr: ChatHistory is a
  // pimpl, see the top-of-file note), keyed so the worker can select / create by id.
  std::map<std::string, std::shared_ptr<ov::genai::ChatHistory>> m_sessions;
  std::int64_t m_seq = 0;   // the per-dispatch counter newSessionId() draws on
};

// A JSON string-member escaper -- ", \ and control chars escaped, non-ASCII passed through as
// UTF-8.  Shared by the dispatcher's control / RESULT lines AND the vlm pipeline's RESULT body
// (defined once here; the free function a second TU reuses instead of re-typing the same case
// table -- single source of truth, as the rest of the project favours).
std::string jsonEscape(const std::string& in);

}  // namespace ovgenai

#endif  // OVGENAI_SESSION_SUPPORT_HPP
