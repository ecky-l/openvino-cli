#!/bin/bash

set -eo pipefail

test -z "OPENVINO_DIR" && OPENVINO_DIR=/opt/intel/openvino
test -z "OPENVINO_GENAI_DIR" && OPENVINO_GENAI_DIR=/opt/intel/openvino.genai

echo export OpenVINO_DIR=${OPENVINO_DIR}/runtime/cmake
echo export OpenVINOGenAI_DIR=${OPENVINO_GENAI_DIR}/runtime/cmake
echo export LD_LIBRARY_PATH=${OPENVINO_GENAI_DIR}/runtime/lib/intel64:${OPENVINO_DIR}/runtime/3rdparty/tbb/lib:${OPENVINO_DIR}/runtime/lib/intel64:$LD_LIBRARY_PATH
