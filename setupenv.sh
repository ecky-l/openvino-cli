#!/bin/bash
#
# Evaluating $(./setupenv.sh) in the current shell exports the three
# variables ovcli's CMake and the test harness expect:
#   OpenVINO_DIR        = <root of openvino install>/runtime/cmake
#   OpenVINOGenAI_DIR   = <root of openvino.genai install>/runtime/cmake
#   LD_LIBRARY_PATH     = the runtime lib dirs (prepended, not overwriting)
#
# The defaults point at the distro install paths; override by pre-setting
# OPENVINO_DIR / OPENVINO_GENAI_DIR to the parent of your `runtime/` dir.

: "${OPENVINO_DIR:=/opt/intel/openvino}"
: "${OPENVINO_GENAI_DIR:=/opt/intel/openvino.genai}"
echo "export OpenVINO_DIR=\"${OPENVINO_DIR}/runtime/cmake\""
echo "export OpenVINOGenAI_DIR=\"${OPENVINO_GENAI_DIR}/runtime/cmake\""
echo "export LD_LIBRARY_PATH=\"${OPENVINO_GENAI_DIR}/runtime/lib/intel64:${OPENVINO_DIR}/runtime/3rdparty/tbb/lib:${OPENVINO_DIR}/runtime/lib/intel64:\$LD_LIBRARY_PATH\""
