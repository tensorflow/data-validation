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

#include "tensorflow_data_validation/arrow/cc/pyarrow_numpy_stub.h"

#include "arrow/python/common.h"
#include "arrow/python/numpy_convert.h"
#include "arrow/python/numpy_to_arrow.h"

namespace tensorflow {
namespace data_validation {
namespace pyarrow_numpy_stub {
arrow::Status NumPyDtypeToArrow(PyObject* dtype,
                                std::shared_ptr<arrow::DataType>* out) {
  return arrow::py::NumPyDtypeToArrow(dtype, out);
}

arrow::Status NdarrayToArrow(arrow::MemoryPool* pool, PyObject* ao,
                             PyObject* mo, bool from_pandas,
                             const std::shared_ptr<arrow::DataType>& type,
                             std::shared_ptr<arrow::ChunkedArray>* out) {
  return arrow::py::NdarrayToArrow(pool, ao, mo, from_pandas, type, out);
}
}  // namespace pyarrow_numpy_stub
}  // namespace data_validation
}  // namespace tensorflow
