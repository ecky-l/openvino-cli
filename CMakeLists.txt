# ovcli -- self-contained C++ drop-in for OpenArc's Python inference worker.
#
# One binary, two front-ends sharing an engine core (see docs/design.md):
#   ovcli worker <model>   long-lived line-JSON protocol (stdin/out); the drop-in
#                           the supervisor respawns; config arrives in the LOAD
#                           message, process knobs in the environment.
#   ovcli run ...          one-shot; self-configures from file/flags/env, reads one
#                           request (base64 input for now), writes the response, exits.
#
# Design notes:
#   * Dependency-free at P0 (only RapidJSON + GTest) so it builds/links clean now.
#   * No fully static build: GPU/NPU device plugins dlopen their .so at runtime, so
#     the engine links the INSTALLED dynamic runtime (find_package(OpenVINOGenAI))
#     behind OVCLI_BUILD_OVGENAI, defaulting OFF (the P1 seam).
#   * No git submodules for deps: use the package manager.

cmake_minimum_required(VERSION 3.16)

project(ovcli VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Binaries (ovcli + tests) land directly under <build>/ so the integration test
# can reference the worker by a stable, configure-time absolute path.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# find_package(RapidJSON) resolves to the installed config-mode package first:
# rapidjson-devel ships /usr/lib64/cmake/RapidJSON/RapidJSONConfig.cmake exporting
# an INTERFACE target `RapidJSON` (and an alias `rapidjson`). Only when NO such
# config package is found do we include the bundled headers-only fallback finder
# (cmake/FindRapidJSON.cmake) so the project still configures on a box that
# installed RapidJSON without the CMake config. Config mode is searched before
# module mode, so the installed package always wins when present.
find_package(RapidJSON CONFIG QUIET)
if(NOT RapidJSON_FOUND)
  message(STATUS "openvino-cli: no installed RapidJSON config package; using bundled headers-only fallback.")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRapidJSON.cmake")
endif()
option(OVCLI_BUILD_TESTS  "Build the GoogleTest suite (worker/protocol/config)" ON)

find_package(OpenVINO COMPONENTS Runtime CONFIG HINTS "${OPENVINO_CONFIG}")
if(NOT OpenVINO_FOUND)
    message(FATAL_ERROR "Could not find OpenVINO. Set OPENVINO_CONFIG to the directory containing OpenVINOConfig.cmake")
endif()

find_package(OpenVINOGenAI CONFIG HINTS ${OPENVINO_GENAI_CONFIG})
if(NOT OpenVINOGenAI_FOUND)
    message(FATAL_ERROR "Could not find OpenVINOGenAI (required by LLM/embeddings/rerank/image_gen). Set OPENVINO_GENAI_CONFIG to the directory that contains OpenVINOGenAIConfig.cmake")
endif()

# Generated constants header consumed by ovcli_core; build output under
# src/generated/ (gitignored, regenerated on every configure).
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/generated")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ovcli_config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/ovcli_config.h"
  @ONLY)

# --- Engine core (static lib): everything but main.cpp ----------------------
file(GLOB OVCLI_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/*.cpp")
list(REMOVE_ITEM OVCLI_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")

add_library(ovcli_core STATIC ${OVCLI_CORE_SOURCES})
target_include_directories(ovcli_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_link_libraries(ovcli_core PUBLIC RapidJSON)
target_link_libraries(ovcli_core PUBLIC openvino::genai)
target_compile_features(ovcli_core PUBLIC cxx_std_20)

# --- The ovcli executable: core + the CLI front-end ------------------------
add_executable(ovcli "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")
target_link_libraries(ovcli PRIVATE ovcli_core)
set_target_properties(ovcli PROPERTIES OUTPUT_NAME ovcli)

# --- Tests ----------------------------------------------------------------
if(OVCLI_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)           # installed config exports GTest::GTest / GTest::Main

  file(GLOB OVCLI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cpp")

  # The worker's absolute path is a configure-time fact (the exe is emitted to
  # CMAKE_RUNTIME_OUTPUT_DIRECTORY = <build>/), so bake it as a quoted macro.
  add_executable(ovcli_tests
    ${OVCLI_TEST_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest_main.cpp")
  target_link_libraries(ovcli_tests PRIVATE
    ovcli_core GTest::GTest GTest::Main)
  set_target_properties(ovcli_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(ovcli_tests PRIVATE
    OVCLI_WORKER_BIN="${CMAKE_CURRENT_BINARY_DIR}/ovcli")

  add_test(NAME ovcli_tests COMMAND ovcli_tests)
  set_tests_properties(ovcli_tests PROPERTIES
    TIMEOUT 120
    ENVIRONMENT "OPENARC_WORKER_LOGFILE_STUB=/tmp/ovcli-integration-worker.log")
endif()
