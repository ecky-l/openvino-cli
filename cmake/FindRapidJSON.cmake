# Guarded module-mode fallback for find_package(RapidJSON).
#
# This box already ships a config-mode package at
#   /usr/lib64/cmake/RapidJSON/RapidJSONConfig.cmake
# exporting an IMPORTED target `RapidJSON` (plus an alias `rapidjson`).
# Config mode is searched BEFORE module mode, so that installed package wins
# on this machine and this file is never even loaded.
#
# On a host where rapidjson-devel was installed without the CMake config
# (older packaging), this module creates the same INTERFACE target so that
# `find_package(RapidJSON REQUIRED)` still succeeds and `target_link_libraries(
# ... RapidJSON)` keeps working. CMAKE_MODULE_PATH is pointed at this dir so
# it is discoverable.

if(RapidJSON_FOUND)
    # A config-mode package already satisfied the request; do nothing.
    return()
endif()

find_path(RAPIDJSON_INCLUDE_DIR
    NAMES rapidjson/document.h
    DOC "Include directory of RapidJSON headers (document.h, etc.).")

include(FindHandleStandardArgs)

find_package_handle_standard_args(RapidJSON
    REQUIRED_VARS RAPIDJSON_INCLUDE_DIR
    VERSION_VAR "")

if(RapidJSON_FOUND AND NOT TARGET RapidJSON)
    add_library(RapidJSON INTERFACE)
    target_include_directories(RapidJSON INTERFACE "${RAPIDJSON_INCLUDE_DIR}")
    add_library(RapidJSON::RapidJSON ALIAS RapidJSON)
endif()

mark_as_advanced(FORCE RAPIDJSON_INCLUDE_DIR)
