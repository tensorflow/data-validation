// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Defines the tensorflow_data_validation extension module. We aim at having
// only one extension module (i.e. dynamic shared library), therefore all the
// TFDV C++ APIs must be added here.
// This C++ object has exception (-fexception) enabled (to work with
// pybind11). -fexception may harm performance and increase the binary size,
// therefore do not put any non-trivial logic here.

#include "tensorflow_data_validation/pywrap/validation_submodule.h"
#include "include/pybind11/pybind11.h"

namespace tensorflow {
namespace data_validation {

PYBIND11_MODULE(
    tensorflow_data_validation_extension,  // this must be kept the same as the
                                           // "extension_name" param in the
                                           // build rule
    m) {
  m.doc() = "TensorFlow Data Validation extension module";
  DefineValidationSubmodule(m);
}

}  // namespace data_validation
}  // namespace tensorflow
