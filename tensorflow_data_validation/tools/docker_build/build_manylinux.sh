#!/bin/bash
# Copyright 2019 Google LLC
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

# This script is expected to run in the docker container defined in
# Dockerfile.manylinux2010
# Assumptions:
# - CentOS environment.
# - devtoolset-8 is installed.
# - $PWD is TFDV's project root.
# - Python of different versions are installed at /opt/python/.
# - patchelf, zip, bazel is installed and is in $PATH.

WORKING_DIR=$PWD

function setup_environment() {
  # Re-enable for CentOS container if needed.
  # source scl_source enable devtoolset-10
  # source scl_source enable rh-python38
  if [[ -z "${PYTHON_VERSION}" ]]; then
    echo "Must set PYTHON_VERSION env to 39|310|311"; exit 1;
  fi
  # Bazel will use PYTHON_BIN_PATH to determine the right python library.
  if [[ "${PYTHON_VERSION}" == 39 ]]; then
    PYTHON_DIR=${VIRTUAL_ENV}/cp39-cp39
  elif [[ "${PYTHON_VERSION}" == 310 ]]; then
    PYTHON_DIR=${VIRTUAL_ENV}/cp310-cp310
  elif [[ "${PYTHON_VERSION}" == 311 ]]; then
    PYTHON_DIR=${VIRTUAL_ENV}/cp311-cp311
  else
    echo "Must set PYTHON_VERSION env to 39|310|311"; exit 1;
  fi
  source "${PYTHON_DIR}/bin/activate"
  export PYTHON_BIN_PATH="${PYTHON_DIR}/bin/python"
  pip3 install --upgrade pip setuptools
  pip3 install wheel
  pip3 install "numpy~=1.22.0" --force
  pip3 install auditwheel
}

function build_wheel() {
  rm -rf dist
  "${PYTHON_BIN_PATH}" setup.py bdist_wheel
}

function stamp_wheel() {
  WHEEL_PATH="$(ls "$PWD"/dist/*.whl)"
  WHEEL_DIR=$(dirname "${WHEEL_PATH}")
  TMP_DIR="$(mktemp -d)"
  auditwheel repair --plat manylinux2014_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"
  rm "${WHEEL_PATH}"
}

set -x
bazel clean --expunge
setup_environment && \
build_wheel && \
# stamp_wheel
bazel clean --expunge
set +x
