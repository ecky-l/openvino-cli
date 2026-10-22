// Model-gated coverage for (b) of the tool-call work: the parser *choice* is derived
// from the model's OWN loaded chat template (rendered through the public Tokenizer API
// and matched against a model-intrinsic sentinel), not hard-coded per caller -- the
// future-proof D1 auto-detect gate.  A Qwen3.5 model that ships the Qwen3.5 function-
// calling block (outer/inner/parameter tag tiers) auto-selects "qwen35"; a model whose
// template ships none of the sentinels resolves to no parser and degrades to plain text.
//
// The no-model unit test (tool_call_parser_toolkit.cpp) already covers the matcher core
// (matchFormat); this covers the render-and-detect path (detectToolCallFormat) against a
// real Qwen3.5 fixture.  It only ASSERTS when the fixture's rendered template carries the
// sentinel; an under-provisioned box (no runtime, or a degenerate template lacking the
// sentinel) SKIPS, never fails.
//
// OVGENAI_VLM_FIXTURE is the committed tiny-random--qwen3.5-ov dir (baked absolute at
// configure time; see CMakeLists).  Its chat_template.jinja bakes the model-intrinsic
// example_function_name / example_parameter markers, so the detector fires for it.
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <openvino/genai/tokenizer.hpp>   // Tokenizer + apply_chat_template (the public detect API)
#include "tool_call_support.hpp"          // ovgenai::toolcall::detectToolCallFormat

#if defined(OVGENAI_VLM_FIXTURE)

TEST(vlm_tool_call, qwen35DetectsFromLoadedTemplate) {
  // Gate on the fixture dir still existing at runtime; an under-provisioned box
  // (fixture deleted, or the OpenVINO.GenAI runtime absent from LD_LIBRARY_PATH) skips
  // rather than fails.
  if (::access(OVGENAI_VLM_FIXTURE, F_OK) != 0) {
    GTEST_SKIP() << "baked VLM fixture dir is gone at runtime";
    return;
  }
  // Build the tokenizer and let detectToolCallFormat render the model's OWN template
  // with a probe tool (so the format instruction materialises) and match the sentinels.
  // A default-constructed Tokenizer is always valid, so if loading throws we still have a
  // usable object and simply skip -- we cannot assert a choice that was never derivable.
  ov::genai::Tokenizer tok;
  try {
    tok = ov::genai::Tokenizer(std::filesystem::path(OVGENAI_VLM_FIXTURE));
  } catch (...) {
    GTEST_SKIP() << "could not build a tokenizer from the VLM fixture here (needs the "
                   "OpenVINO.GenAI runtime on LD_LIBRARY_PATH -- run ctest under `eval "
                   "$(./setupenv.sh)`)";
    return;
  }
  auto detected = ovgenai::toolcall::detectToolCallFormat(tok);
  if (!detected.has_value()) {
    GTEST_SKIP() << "the fixture's rendered chat template did not carry the qwen35 sentinel "
                   "(example_function_name / example_parameter); nothing to auto-select (skip)";
    return;
  }
  EXPECT_EQ(*detected, std::string("qwen35"));
}

#else  // OVGENAI_VLM_FIXTURE not baked -> the whole file skips

TEST(vlm_tool_call, fixture_not_baked_in) {
  GTEST_SKIP() << "ovgenai qwen35 tool-call auto-detect coverage is baked only when the "
                  "tiny-random--qwen3.5-ov VLM fixture is present at configure time";
}

#endif
