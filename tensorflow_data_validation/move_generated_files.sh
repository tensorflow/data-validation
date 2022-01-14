#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Moves the bazel generated files needed for packaging the wheel to the source
# tree.

function _is_windows() {
  [[ "$(uname -s | tr 'A-Z' 'a-z')" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]
}

function tfdv::move_generated_files() {
  if _is_windows; then
    # Newer bazel does not create bazel-genfiles any more (
    # https://github.com/bazelbuild/bazel/issues/6761). It's merged with bazel-bin
    GENFILES=bazel-genfiles
    if [[ ! -d ${BUILD_WORKSPACE_DIRECTORY}/${GENFILES} ]]; then
      GENFILES=bazel-bin
    fi
    PYWRAP_TFDV="tensorflow_data_validation/pywrap/tensorflow_data_validation_extension.pyd"
    cp -f "${BUILD_WORKSPACE_DIRECTORY}/bazel-out/x64_windows-opt/bin/${PYWRAP_TFDV}" \
      "${BUILD_WORKSPACE_DIRECTORY}/${PYWRAP_TFDV}"

    cp -f ${BUILD_WORKSPACE_DIRECTORY}/${GENFILES}/tensorflow_data_validation/skew/protos/feature_skew_results_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/skew/protos
    cp -f ${BUILD_WORKSPACE_DIRECTORY}/${GENFILES}/tensorflow_data_validation/anomalies/proto/validation_config_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/anomalies/proto
    cp -f ${BUILD_WORKSPACE_DIRECTORY}/${GENFILES}/tensorflow_data_validation/anomalies/proto/validation_metadata_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/anomalies/proto
  else
    PYWRAP_TFDV="tensorflow_data_validation/pywrap/tensorflow_data_validation_extension.so"
    cp -f "${BUILD_WORKSPACE_DIRECTORY}/bazel-bin/${PYWRAP_TFDV}" \
      "${BUILD_WORKSPACE_DIRECTORY}/${PYWRAP_TFDV}"

    # If run by "bazel run", $(pwd) is the .runfiles dir that contains all the
    # data dependencies.
    RUNFILES_DIR=$(pwd)
    cp -f ${RUNFILES_DIR}/tensorflow_data_validation/skew/protos/feature_skew_results_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/skew/protos
    cp -f ${RUNFILES_DIR}/tensorflow_data_validation/anomalies/proto/validation_config_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/anomalies/proto
    cp -f ${RUNFILES_DIR}/tensorflow_data_validation/anomalies/proto/validation_metadata_pb2.py \
      ${BUILD_WORKSPACE_DIRECTORY}/tensorflow_data_validation/anomalies/proto
  fi
  chmod +w "${BUILD_WORKSPACE_DIRECTORY}/${PYWRAP_TFDV}"
}

tfdv::move_generated_files
