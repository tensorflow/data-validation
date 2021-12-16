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
  source scl_source enable devtoolset-8
  source scl_source enable rh-python36
  if [[ -z "${PYTHON_VERSION}" ]]; then
    echo "Must set PYTHON_VERSION env to 37|38|39"; exit 1;
  fi
  # Bazel will use PYTHON_BIN_PATH to determine the right python library.
  if [[ "${PYTHON_VERSION}" == 37 ]]; then
    PYTHON_DIR=/opt/python/cp37-cp37m
  elif [[ "${PYTHON_VERSION}" == 38 ]]; then
    PYTHON_DIR=/opt/python/cp38-cp38
  elif [[ "${PYTHON_VERSION}" == 39 ]]; then
    PYTHON_DIR=/opt/python/cp39-cp39
  else
    echo "Must set PYTHON_VERSION env to 37|38|39"; exit 1;
  fi
  export PIP_BIN="${PYTHON_DIR}"/bin/pip
  export PYTHON_BIN_PATH="${PYTHON_DIR}"/bin/python
  echo "PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
  export WHEEL_BIN="${PYTHON_DIR}"/bin/wheel
  ${PIP_BIN} install --upgrade pip
  ${PIP_BIN} install wheel --upgrade
  # Auditwheel does not have a python2 version and auditwheel is just a binary.
  pip3 install auditwheel
}

function install_numpy() {
  ${PIP_BIN} install "numpy>=1.16,<2"
}

function build_wheel() {
  rm -rf dist
  "${PYTHON_BIN_PATH}" setup.py bdist_wheel
}

function stamp_wheel() {
  WHEEL_PATH="$(ls "$PWD"/dist/*.whl)"
  WHEEL_DIR=$(dirname "${WHEEL_PATH}")
  TMP_DIR="$(mktemp -d)"
  auditwheel repair --plat manylinux2010_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"
  rm "${WHEEL_PATH}"
}

set -x
setup_environment && \
install_numpy && \
build_wheel && \
stamp_wheel
