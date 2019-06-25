// Copyright 2019 Google LLC
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

// This header declares Numpy API entry points that can be shared by multiple
// translation units.
// To use Numpy C-APIs (PyArray_*), include this header instead of the numpy
// headers, then call ImportNumpy() once before the first use of the API.
// See https://github.com/numpy/numpy/issues/9309 for detailed explaination.
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_INIT_NUMPY_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_INIT_NUMPY_H_

#define PY_ARRAY_UNIQUE_SYMBOL TFDV_ARROW_ARRAY_API
#ifndef NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

namespace tensorflow {
namespace data_validation {
// Initializes Numpy C-API. This must be called before calling any numpy C-APIs.
void ImportNumpy();
}  // namespace data_validation
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_INIT_NUMPY_H_
