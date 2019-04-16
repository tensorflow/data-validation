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

#include "tensorflow_data_validation/arrow/cc/arrow_util.h"

#include <memory>

// This include is needed to avoid C2528. See https://bugs.python.org/issue24643
#include "arrow/python/platform.h"
#include "arrow/python/pyarrow.h"
#include "arrow/api.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_data_validation/arrow/cc/macros.h"

namespace {
using ::arrow::Status;
using ::arrow::Array;
using ::arrow::ListArray;
using ::arrow::Type;

Status GetListArray(const Array& array, const ListArray** list_array) {
  if (array.type()->id() != Type::LIST) {
    return Status::Invalid(absl::StrCat("Expected ListArray but got type id: ",
                                        array.type()->ToString()));
  }
  *list_array = static_cast<const ListArray*>(&array);
  return Status::OK();
}

Status FlattenListArray(const Array& array, std::shared_ptr<Array>* flattened) {
  const ListArray* list_array;
  RETURN_NOT_OK(GetListArray(array, &list_array));
  *flattened = list_array->values();
  return Status::OK();
}

Status GetFlattenedArrayParentIndices(
    const Array& array,
    std::shared_ptr<Array>* parent_indices_array) {
  const ListArray* list_array;
  RETURN_NOT_OK(GetListArray(array, &list_array));
  arrow::Int32Builder indices_builder;
  RETURN_NOT_OK(
      indices_builder.Reserve(list_array->value_offset(list_array->length()) -
                              list_array->value_offset(0)));
  for (int i = 0; i < list_array->length(); ++i) {
    const int range_begin = list_array->value_offset(i);
    const int range_end = list_array->value_offset(i + 1);
    for (int j = range_begin; j < range_end; ++j) {
      RETURN_NOT_OK(indices_builder.Append(i));
    }
  }
  return indices_builder.Finish(parent_indices_array);
}

Status ListLengthsFromListArray(
    const Array& array,
    std::shared_ptr<arrow::Array>* list_lengths_array) {
  const ListArray* list_array;
  RETURN_NOT_OK(GetListArray(array, &list_array));

  arrow::Int32Builder lengths_builder;
  RETURN_NOT_OK(lengths_builder.Reserve(list_array->length()));
  for (int i = 0; i < list_array->length(); ++i) {
    RETURN_NOT_OK(lengths_builder.Append(list_array->value_length(i)));
  }
  return lengths_builder.Finish(list_lengths_array);
}

Status GetArrayNullBitmapAsByteArray(
    const Array& array,
    std::shared_ptr<Array>* byte_array) {
  arrow::UInt8Builder masks_builder;
  RETURN_NOT_OK(masks_builder.Reserve(array.length()));
  // array.null_count() might be O(n). However array.data()->null_count
  // is just a number (although it can be kUnknownNullCount in which case
  // the else branch is followed).
  if (array.null_bitmap_data() == nullptr || array.data()->null_count == 0) {
    // the masks array should not contain null, and its contents should be
    // all zero. AppendNulls() does exactly that.
    RETURN_NOT_OK(
        masks_builder.AppendNulls(/*valid_bytes=*/nullptr, array.length()));
  } else {
    for (int i = 0; i < array.length(); ++i) {
      RETURN_NOT_OK(
          masks_builder.Append(static_cast<uint8_t>(array.IsNull(i))));
    }
  }
  return masks_builder.Finish(byte_array);
}

// This must be called before calling any functions under arrow::py::.
void ImportPyArrow() {
  static const int kUnused = arrow::py::import_pyarrow();
  // This suppresses the "unused variable" warning.
  (void)kUnused;
}

}  // namespace

PyObject* TFDV_Arrow_FlattenListArray(PyObject* py_list_array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(py_list_array, &unwrapped));
  std::shared_ptr<arrow::Array> flattened;
  TFDV_RAISE_IF_NOT_OK(FlattenListArray(*unwrapped, &flattened));
  return arrow::py::wrap_array(flattened);
}

PyObject* TFDV_Arrow_ListLengthsFromListArray(PyObject* py_list_array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(py_list_array, &unwrapped));
  std::shared_ptr<arrow::Array> list_lengths_array;
  TFDV_RAISE_IF_NOT_OK(
      ListLengthsFromListArray(*unwrapped, &list_lengths_array));
  return arrow::py::wrap_array(list_lengths_array);
}

PyObject* TFDV_Arrow_GetFlattenedArrayParentIndices(PyObject* py_list_array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(py_list_array, &unwrapped));
  std::shared_ptr<arrow::Array> parent_indices_array;
  TFDV_RAISE_IF_NOT_OK(
      GetFlattenedArrayParentIndices(*unwrapped, &parent_indices_array));
  return arrow::py::wrap_array(parent_indices_array);
}

PyObject* TFDV_Arrow_GetArrayNullBitmapAsByteArray(PyObject* array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(array, &unwrapped));
  std::shared_ptr<arrow::Array> null_bitmap_byte_array;
  TFDV_RAISE_IF_NOT_OK(
      GetArrayNullBitmapAsByteArray(*unwrapped, &null_bitmap_byte_array));
  return arrow::py::wrap_array(null_bitmap_byte_array);
}
