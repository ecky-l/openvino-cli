# ovgenai -- frontend cli tool for openvino/openvino.genai interference
#
# One binary whose subcommand surface (see src/main/commands.{hpp,cpp}) is one
# entry per OpenVINO.GenAI C++ pipeline -- llm / vlm / whisper / asr / tts / omni
# / embedding / text-embedding / rerank / the image- and video-generation
# pipelines -- plus the `show-config` helper, each rendered by
# boost::program_options as two categories in `ovgenai --help`
# (Pipeline subcommands / Helper subcommands). Every pipeline subcommand also
# takes a valueless --worker switch for the long-lived piped worker form -- a
# subprocess that speaks a line-JSON protocol; omit it for a one-shot run.
# This phase is the infrastructure:
# the per-pipeline bodies arrive later, one at a time.
cmake_minimum_required(VERSION 3.28)

project(ovgenai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Binaries (ovgenai + tests) land directly under <build>/ so the integration test
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

# Boost.program_options for the shared load-model CLI front-end. Prefer the
# static variant (b2 --layout=system --link=static install, packaged by
# distro as boost-static); setting Boost_USE_STATIC_LIBS=ON selects the .a
# variants when the .so ones would otherwise be chosen. Boost::program_options
# pulls in Boost::container + Boost::headers through the config file, so we
# only need to name program_options here.
set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Prefer static boost libraries" FORCE)
find_package(Boost 1.90 CONFIG REQUIRED COMPONENTS program_options)

option(OVGENAI_BUILD_TESTS  "Build the GoogleTest suite (worker/protocol/config)" ON)

find_package(OpenVINO COMPONENTS Runtime CONFIG REQUIRED)
message(STATUS "OpenVINO: ${OpenVINO_VERSION} (${OpenVINO_DIR})")

find_package(OpenVINOGenAI CONFIG REQUIRED)
message(STATUS "OpenVINO.GenAI: ${OpenVINOGenAI_VERSION} (${OpenVINOGenAI_DIR})")

# Generated constants header consumed by ovgenai_core; build output under
# src/generated/ (gitignored, regenerated on every configure).
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/generated")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ovgenai_config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/ovgenai_config.h"
  @ONLY)

# --- Engine core (static lib): everything but main.cpp ----------------------
file(GLOB OVGENAI_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/*.cpp")
list(REMOVE_ITEM OVGENAI_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")

add_library(ovgenai_core STATIC ${OVGENAI_CORE_SOURCES})
target_include_directories(ovgenai_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_link_libraries(ovgenai_core PUBLIC RapidJSON)
target_link_libraries(ovgenai_core PUBLIC Boost::program_options)
target_link_libraries(ovgenai_core PUBLIC openvino::genai)
target_compile_features(ovgenai_core PUBLIC cxx_std_20)

# --- The ovgenai executable: core + the CLI front-end ------------------------
add_executable(ovgenai "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")
target_link_libraries(ovgenai PRIVATE ovgenai_core)
set_target_properties(ovgenai PROPERTIES OUTPUT_NAME ovgenai)

# --- Tests ----------------------------------------------------------------
if(OVGENAI_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)           # installed config exports GTest::GTest / GTest::Main

  file(GLOB OVGENAI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cpp")

  # The worker's absolute path is a configure-time fact (the exe is emitted to
  # CMAKE_RUNTIME_OUTPUT_DIRECTORY = <build>/), so bake it as a quoted macro.
  add_executable(ovgenai_tests
    ${OVGENAI_TEST_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest_main.cpp")
  target_link_libraries(ovgenai_tests PRIVATE
    ovgenai_core GTest::GTest GTest::Main)
  set_target_properties(ovgenai_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(ovgenai_tests PRIVATE
    OVGENAI_WORKER_BIN="${CMAKE_CURRENT_BINARY_DIR}/ovgenai")

  # A real (random-weight) VLM pipeline to exercise the now-genuine `vlm` body
  # end-to-end (the committed tiny-random--qwen3.5-ov fixture): baked absolute,
  # like the worker path, so a subprocess test resolves it regardless of CWD; when
  # the fixture is absent at configure time the macro is simply not defined and the
  # test skips (as ovgenaiRuns() makes it do when the box cannot exec).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov/config.json")
    target_compile_definitions(ovgenai_tests PRIVATE
      OVGENAI_VLM_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov")
  endif()

  include(GoogleTest)
  gtest_discover_tests(ovgenai_tests
    DISCOVERY_MODE POST_BUILD
    PROPERTIES
      TIMEOUT 120
      ENVIRONMENT "OVGENAI_WORKER_LOGFILE_STUB=/tmp/ovgenai-integration-worker.log")
endif()
