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

// This header contains function stubs that forward to corresponding functions
// under ::arrow::py. They are needed because their arrow counter-parts are
// declared in headers where Arrow's own Numpy C-API stubs are also declared.
// Including those headers will pollute our own source files, as we are defining
// our own Numpy C-API stubs.
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_PYARROW_NUMPY_STUB_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_PYARROW_NUMPY_STUB_H_

#include <memory>

#include "Python.h"
// This include is needed to avoid C2528. See https://bugs.python.org/issue24643
// Must be included before other arrow headers.
#include "arrow/python/platform.h"
#include "arrow/api.h"

namespace tensorflow {
namespace data_validation {
namespace pyarrow_numpy_stub {
// See ::arrow::py::NumPyDtypeToArrow.
arrow::Status NumPyDtypeToArrow(PyObject* dtype,
                                std::shared_ptr<arrow::DataType>* out);

// See ::arrow::py::NdarrayToArrow.
arrow::Status NdarrayToArrow(arrow::MemoryPool* pool, PyObject* ao,
                             PyObject* mo, bool from_pandas,
                             const std::shared_ptr<arrow::DataType>& type,
                             std::shared_ptr<arrow::ChunkedArray>* out);
}  // namespace pyarrow_numpy_stub
}  // namespace data_validation
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_PYARROW_NUMPY_STUB_H_
