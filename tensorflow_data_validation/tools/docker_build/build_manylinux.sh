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
  if [[ -z "${PYTHON_VERSION}" ]]; then
    echo "Must set PYTHON_VERSION env to 35|36|37|27"; exit 1;
  fi
  # Bazel will use PYTHON_BIN_PATH to determine the right python library.
  if [[ "${PYTHON_VERSION}" == 27 ]]; then
    PYTHON_DIR=/opt/python/cp27-cp27mu
  elif [[ "${PYTHON_VERSION}" == 35 ]]; then
    PYTHON_DIR=/opt/python/cp35-cp35m
  elif [[ "${PYTHON_VERSION}" == 36 ]]; then
    PYTHON_DIR=/opt/python/cp36-cp36m
  elif [[ "${PYTHON_VERSION}" == 37 ]]; then
    PYTHON_DIR=/opt/python/cp37-cp37m
  else
    echo "Must set PYTHON_VERSION env to 35|36|37|27"; exit 1;
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

function install_pyarrow() {
  PYARROW_VERSION_FILE=$1
  PYARROW_REQUIREMENT=$("${PYTHON_BIN_PATH}" -c "fp = open('$PYARROW_VERSION_FILE', 'r'); d = {}; exec(fp.read(), d); fp.close(); print(d['PY_DEP'])")
  ${PIP_BIN} install "${PYARROW_REQUIREMENT}"
}

function bazel_build() {
  rm -f .bazelrc
  rm -rf dist
  ./configure.sh
  bazel run -c opt \
    --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
    tensorflow_data_validation:build_pip_package \
    --\
    --python_bin_path "${PYTHON_BIN_PATH}"
}

# This should have been simply an invocation of "auditwheel repair" but because
# of https://github.com/pypa/auditwheel/issues/76, Arrow's shared libraries that
# TFDV depends on are treated incorrectly by auditwheel. We have to do this
# trick to make auditwheel happily stamp on our wheel.
# Note that even though auditwheel would reject the wheel produced in the end,
# it's still manylinux2010 compliant according to the standard, because it only
# depends on the specified shared libraries, assuming pyarrow is installed.
function stamp_wheel() {
  WHEEL_PATH="$(ls "$PWD"/dist/*.whl)"
  WHEEL_DIR=$(dirname "${WHEEL_PATH}")
  TMP_DIR="$(mktemp -d)"
  pushd "${TMP_DIR}"
  unzip "${WHEEL_PATH}"
  SO_FILE_PATH=tensorflow_data_validation/pywrap/_pywrap_tensorflow_data_validation.so
  LIBARROW="$(patchelf --print-needed "${SO_FILE_PATH}" | fgrep libarrow.so)"
  LIBARROW_PYTHON="$(patchelf --print-needed "${SO_FILE_PATH}" | fgrep libarrow_python.so)"
  patchelf --remove-needed "${LIBARROW}" "${SO_FILE_PATH}"
  patchelf --remove-needed "${LIBARROW_PYTHON}" "${SO_FILE_PATH}"
  # update the .so file in the original wheel.
  zip "${WHEEL_PATH}" "${SO_FILE_PATH}"
  popd
  auditwheel repair --plat manylinux2010_x86_64 -w "${WHEEL_DIR}" "${WHEEL_PATH}"
  rm "${WHEEL_PATH}"
  MANY_LINUX_WHEEL_PATH=$(ls "${WHEEL_DIR}"/*manylinux*.whl)
  # Unzip the manylinux2010 wheel and pack it again with the original .so file.
  # We need to use "wheel pack" in order to compute the file hashes again.
  TMP_DIR="$(mktemp -d)"
  unzip "${MANY_LINUX_WHEEL_PATH}" -d "${TMP_DIR}"
  cp ${SO_FILE_PATH} "${TMP_DIR}/${SO_FILE_PATH}"
  rm "${MANY_LINUX_WHEEL_PATH}"
  ${WHEEL_BIN} version
  ${WHEEL_BIN} pack "${TMP_DIR}" --dest-dir "${WHEEL_DIR}"
}

setup_environment
set -e
set -x
install_pyarrow "third_party/pyarrow_version.bzl"
bazel_build
stamp_wheel

set +e
set +x
