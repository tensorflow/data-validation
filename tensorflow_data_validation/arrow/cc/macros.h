// Copyright 2018 Google LLC
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

#ifndef TENSORFLOW_DATA_VALIDATION_ARROW_CC_MACROS_H_
#define TENSORFLOW_DATA_VALIDATION_ARROW_CC_MACROS_H_

// Raises a python exception if `s` (a statement that evaluates to an
// arrow::Status) is not OK.
#define RAISE_IF_NOT_OK(s)                                       \
  do {                                                           \
    ::arrow::Status s_ = (s);                                    \
    if (!s_.ok()) {                                              \
      PyErr_SetString(PyExc_RuntimeError, s_.message().c_str()); \
      return nullptr;                                            \
    }                                                            \
  } while (0);

#endif  // TENSORFLOW_DATA_VALIDATION_ARROW_CC_MACROS_H_
