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
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "tensorflow_data_validation/arrow/cc/common.h"
#include "tensorflow_data_validation/arrow/cc/init_numpy.h"

namespace {
using ::arrow::Array;
using ::arrow::Buffer;
using ::arrow::ListArray;
using ::arrow::Status;
using ::arrow::Type;
using ::arrow::TypedBufferBuilder;
using ::tensorflow::data_validation::ImportNumpy;
using ::tensorflow::data_validation::ImportPyArrow;

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
      indices_builder.UnsafeAppend(i);
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
    lengths_builder.UnsafeAppend(list_array->value_length(i));
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
    for (int i = 0; i < array.length(); ++i) {
      masks_builder.UnsafeAppend(0);
    }
  } else {
    for (int i = 0; i < array.length(); ++i) {
      masks_builder.UnsafeAppend(static_cast<uint8_t>(array.IsNull(i)));
    }
  }
  return masks_builder.Finish(byte_array);
}

Status MakeListArrayFromParentIndicesAndValues(
    const int64_t num_parents,
    absl::Span<const int64_t> parent_indices,
    const std::shared_ptr<Array>& values,
    std::shared_ptr<Array>* out) {
  if (values->length() != parent_indices.size()) {
    return arrow::Status::Invalid(
        "values array and parent indices array must be of the same length: ",
        values->length(), " v.s. ", parent_indices.size());
  }
  if (!parent_indices.empty() && num_parents < parent_indices.back() + 1) {
    return arrow::Status::Invalid("Found a parent index ",
                                  parent_indices.back(),
                                  " while num_parents was ", num_parents);
  }

  TypedBufferBuilder<bool> null_bitmap_builder;
  RETURN_NOT_OK(null_bitmap_builder.Reserve(num_parents));
  TypedBufferBuilder<int32_t> offsets_builder;
  RETURN_NOT_OK(offsets_builder.Reserve(num_parents + 1));

  offsets_builder.UnsafeAppend(0);
  for (int i = 0, current_pi = 0; i < num_parents; ++i) {
    if (current_pi >= parent_indices.size() ||
        parent_indices[current_pi] != i) {
      null_bitmap_builder.UnsafeAppend(false);
    } else {
      while (current_pi < parent_indices.size() &&
             parent_indices[current_pi] == i) {
        ++current_pi;
      }
      null_bitmap_builder.UnsafeAppend(true);
    }
    offsets_builder.UnsafeAppend(current_pi);
  }

  const int64_t null_count = null_bitmap_builder.false_count();
  std::shared_ptr<Buffer> null_bitmap_buffer;
  RETURN_NOT_OK(null_bitmap_builder.Finish(&null_bitmap_buffer));
  std::shared_ptr<Buffer> offsets_buffer;
  RETURN_NOT_OK(offsets_builder.Finish(&offsets_buffer));

  *out = std::make_shared<ListArray>(arrow::list(values->type()), num_parents,
                                     offsets_buffer, values, null_bitmap_buffer,
                                     null_count, /*offset=*/0);
  return Status::OK();
}

Status PythonIntToInt64(PyObject* py_int, int64_t* out) {
  if (PyLong_Check(py_int)) *out = PyLong_AsLongLong(py_int);
#if PY_MAJOR_VERSION < 3
  else if (PyInt_Check(py_int)) *out = PyInt_AsLong(py_int);
#endif
  else
    return Status::Invalid("Expected integer.");
  if (PyErr_Occurred()) {
    return Status::Invalid("Integer overflow during conversion.");
  }
  return Status::OK();
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

PyObject* TFDV_Arrow_GetBinaryArrayTotalByteSize(PyObject* py_binary_array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(py_binary_array, &unwrapped));
  // StringArray is a subclass of BinaryArray.
  if (!(unwrapped->type_id() == arrow::Type::BINARY ||
        unwrapped->type_id() == arrow::Type::STRING)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        absl::StrCat("Expected BinaryArray (or StringArray) but got: ",
                     unwrapped->type()->ToString())
            .c_str());
    return nullptr;
  }
  const arrow::BinaryArray* binary_array =
      static_cast<const arrow::BinaryArray*>(unwrapped.get());
  const size_t total_byte_size =
      binary_array->value_offset(binary_array->length()) -
      binary_array->value_offset(0);
  return PyLong_FromSize_t(total_byte_size);
}

PyObject* TFDV_Arrow_ValueCounts(PyObject* array) {
  ImportPyArrow();
  std::shared_ptr<arrow::Array> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_array(array, &unwrapped));
  arrow::compute::FunctionContext ctx;
  std::shared_ptr<arrow::Array> result;
  TFDV_RAISE_IF_NOT_OK(arrow::compute::ValueCounts(&ctx, unwrapped, &result));
  return arrow::py::wrap_array(result);
}

PyObject* TFDV_Arrow_MakeListArrayFromParentIndicesAndValues(
    PyObject* num_parents, PyObject* parent_indices, PyObject* values_array) {
  ImportPyArrow();
  ImportNumpy();
  int64_t num_parents_int;
  TFDV_RAISE_IF_NOT_OK(PythonIntToInt64(num_parents, &num_parents_int));

  PyArrayObject* parent_indices_np;
  if (!PyArray_Check(parent_indices)) {
    PyErr_SetString(PyExc_TypeError,
                    "MakeListArrayFromParentIndicesAndValues expected "
                    "parent_indices to be a numpy array.");
    return nullptr;
  }

  parent_indices_np = reinterpret_cast<PyArrayObject*>(parent_indices);
  if (PyArray_TYPE(parent_indices_np) != NPY_INT64 ||
      PyArray_NDIM(parent_indices_np) != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "MakeListArrayFromParentIndicesAndValues expected "
                    "parent_indices to be a 1-D int64 numpy array.");
    return nullptr;
  }

  std::shared_ptr<Array> unwrapped_values_array;
  TFDV_RAISE_IF_NOT_OK(
      arrow::py::unwrap_array(values_array, &unwrapped_values_array));

  absl::Span<const int64_t> parent_indices_span(
      static_cast<const int64_t*>(PyArray_DATA(parent_indices_np)),
      PyArray_SIZE(parent_indices_np));
  std::shared_ptr<Array> list_array;
  TFDV_RAISE_IF_NOT_OK(MakeListArrayFromParentIndicesAndValues(
      num_parents_int, parent_indices_span, unwrapped_values_array,
      &list_array));
  return arrow::py::wrap_array(list_array);
}
