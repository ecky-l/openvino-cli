// --- Per-request generation config ------------------------------------------
//
// For ovcli run:  populated from CLI flags (parseModelRequestConfig).
// For ovcli worker (future):  populated from a protocol command per request.
// Maps the per-request generation knobs; the JSON round-trip uses snake_case
// keys (see modelRequestConfigFromJson below), so the wire form is stable.
//
// Fields with a non-nil default carry that default here;  fields that are
// std::optional are "not supplied in this request" when nullopt, and then fall
// back to the model-load default or, failing that, the engine default.
#ifndef OVCLI_MODEL_REQUEST_CONFIG_HPP
#define OVCLI_MODEL_REQUEST_CONFIG_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // for po::options_description in requestOptionsDesc
#include "config.hpp"  // ConfigError + the parse helpers both sections share

namespace ovcli {
namespace config {

struct ModelRequestConfig {
  // Non-optional; a struct default applies when the request omits them.
  std::size_t max_tokens            = 16384;
  double      temperature           = 1.0;
  std::size_t top_k               = 50;
  double      top_p               = 1.0;
  double      repetition_penalty  = 1.0;

  // Optional (nullopt == "not in this request")
  std::optional<std::size_t> seed;
  std::optional<double>      frequency_penalty;
  std::optional<double>      presence_penalty;
  std::optional<std::size_t> num_assistant_tokens;
  std::optional<double>      assistant_confidence_threshold;
  std::optional<std::string> tool_call_parser;

  // Chat template kwargs; a JSON object (e.g. {"add_generation_prompt": true})
  std::optional<std::string> chat_template_kwargs;

  bool operator==(const ModelRequestConfig&) const = default;

  // --- The "Model request parameters" boost section this class owns ---------
  //
  // The generation-config flags (the ones the model-load section does not
  // carry), each with a single-letter short where its first letter is free
  // (see the note beside the section below and requestOptionsDesc()). Feeds
  // both usage() and parseModelRequestConfig (spliced alongside the load
  // section the other class owns).
  static po::options_description requestOptionsDesc();

  // This class's slice of the --help listing (its "Model request parameters"
  // section only); the combined two-section view is config.cpp's
  // combinedConfigUsage.
  static std::string usage();
};

// Parses the per-request generation config (for `ovcli run`) via boost over the
// full argv (its section plus the load section the other class owns, so the
// load flags a caller also passes are "known", not "unknown option"); only the
// generation-config fields are read. Unknown/missing flags leave the
// corresponding field at its struct default (matching pydantic's defaults for
// non-optional fields; nullopt for optional ones). Throws ConfigError on an
// unknown / value-invalid flag.
ModelRequestConfig parseModelRequestConfig(const std::vector<std::string>& args);

// JSON round-trip for the per-request generation config.
// Uses snake_case keys, so the wire form is identical on both sides.
ModelRequestConfig modelRequestConfigFromJson(const std::string& json);
std::string modelRequestConfigToJson(const ModelRequestConfig& config);

}  // namespace config
}  // namespace ovcli

#endif  // OVCLI_MODEL_REQUEST_CONFIG_HPP
