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

#include "tensorflow_data_validation/arrow/cc/decoded_examples_to_arrow.h"

#include <iostream>
#include <cstring>
#include <memory>

#include "arrow/python/pyarrow.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "arrow/api.h"
#include "tensorflow_data_validation/arrow/cc/macros.h"

namespace {
using arrow::DataType;
using arrow::Type;
using arrow::ArrayBuilder;
using arrow::ArrayVector;
using arrow::BinaryBuilder;
using arrow::Field;
using arrow::FloatBuilder;
using arrow::Int64Builder;
using arrow::ListBuilder;
using arrow::Status;
using arrow::Table;

// Appends `num_nulls` of nulls to `list_builder`.
Status AppendNulls(const int64_t num_nulls, ListBuilder* list_builder) {
  // If the API allows, we should do this in bulk.
  for (int i = 0; i < num_nulls; ++i) {
    RETURN_NOT_OK(list_builder->AppendNull());
  }
  return Status::OK();
}

// Makes a string_view out of a Python bytes object. Note that the Python object
// still owns the underlying memory.
Status PyBytesToStringView(PyObject* py_bytes, absl::string_view* sv) {
  if (!PyBytes_Check(py_bytes)) {
    return Status::Invalid("Expected bytes.");
  }
  char* data;
  Py_ssize_t length;
  PyBytes_AsStringAndSize(py_bytes, &data, &length);
  *sv = absl::string_view(data, length);
  return Status::OK();
}

// A feature name can be either unicode or bytes. This function converts
// a unicode feature name into utf-8 bytes and stores in `result`.
Status FeatureNameToString(PyObject* feature_name, std::string* result) {
  if (PyUnicode_Check(feature_name)) {
    PyObject* utf8_bytes = PyUnicode_AsUTF8String(feature_name);
    absl::string_view sv;
    RETURN_NOT_OK(PyBytesToStringView(utf8_bytes, &sv));
    *result = std::string(sv);
    Py_XDECREF(utf8_bytes);
    return Status::OK();
  }

  if (PyBytes_Check(feature_name)) {
    absl::string_view sv;
    RETURN_NOT_OK(PyBytesToStringView(feature_name, &sv));
    *result = std::string(sv);
    return Status::OK();
  }

  return Status::Invalid("Feature names must be either bytes or unicode.");
}

// Casts `py_obj` to PyArrayObject* if it's a numpy array. Otherwise returns
// an error.
Status GetNumPyArray(PyObject* py_obj, PyArrayObject** py_array_obj) {
  if (!PyArray_Check(py_obj)) {
    return Status::Invalid("Expected a numpy ndarray.");
  }
  *py_array_obj = reinterpret_cast<PyArrayObject*>(py_obj);
  return Status::OK();
}

// Makes an Arrow ListArray builder based on the type of `np_array`.
// Returns an error if `np_array` is of unsupported type. Currently only
// np.int64, np.float32 and np.object (for strings) are supported.
Status MakeListBuilderFromFeature(
    PyArrayObject* np_array, std::unique_ptr<ListBuilder>* builder) {
  int np_type = PyArray_TYPE(np_array);
  if (np_type == NPY_INT64) {
    *builder = absl::make_unique<ListBuilder>(
        arrow::default_memory_pool(),
        std::make_shared<Int64Builder>(arrow::default_memory_pool()));
  } else if (np_type == NPY_FLOAT32) {
    *builder = absl::make_unique<ListBuilder>(
        arrow::default_memory_pool(),
        std::make_shared<FloatBuilder>(arrow::default_memory_pool()));
  } else if (np_type == NPY_OBJECT) {
    *builder = absl::make_unique<ListBuilder>(
        arrow::default_memory_pool(),
        std::make_shared<BinaryBuilder>(arrow::default_memory_pool()));
  } else {
    return Status::Invalid(absl::StrCat("Unsupported ndarray type: ", np_type));
  }
  return Status::OK();
}

Status ArrowTypeToNumpyType(const DataType& arrow_type, int* np_type) {
  switch (arrow_type.id()) {
    case Type::INT64:
      *np_type = NPY_INT64;
      break;
    case Type::FLOAT:
      *np_type = NPY_FLOAT32;
      break;
    case Type::BINARY:
      *np_type = NPY_OBJECT;
      break;
    default:
      return Status::NotImplemented("Internal error: unsupported type");
  }
  return Status::OK();
}

Status AppendNpArrayToBuilder(PyArrayObject* np_array, ListBuilder* builder) {
  const int np_type = PyArray_TYPE(np_array);
  ArrayBuilder* value_builder = builder->value_builder();
  int expected_np_type = -1;
  RETURN_NOT_OK(
      ArrowTypeToNumpyType(*value_builder->type(), &expected_np_type));
  if (np_type != expected_np_type) {
    return Status::Invalid(absl::StrCat(
        "Expected ndarray type ", expected_np_type, " but got ", np_type));
  }

  // Now that src and dst types match, we can safely static_cast.
  RETURN_NOT_OK(builder->Append());
  if (np_type == NPY_INT64) {
    auto* int64_builder = static_cast<Int64Builder*>(value_builder);
    RETURN_NOT_OK(int64_builder->AppendValues(
        static_cast<int64_t*>(PyArray_DATA(np_array)), PyArray_SIZE(np_array)));
  } else if (np_type == NPY_FLOAT32) {
    auto* float_builder = static_cast<FloatBuilder*>(value_builder);
    RETURN_NOT_OK(float_builder->AppendValues(
        static_cast<float*>(PyArray_DATA(np_array)), PyArray_SIZE(np_array)));
  } else if (np_type == NPY_OBJECT) {
    auto* binary_builder = static_cast<BinaryBuilder*>(value_builder);
    PyObject** py_strings = static_cast<PyObject**>(PyArray_DATA(np_array));
    for (int i = 0; i < PyArray_SIZE(np_array); ++i) {
      absl::string_view sv;
      RETURN_NOT_OK(PyBytesToStringView(py_strings[i], &sv));
      RETURN_NOT_OK(binary_builder->Append(sv.data(), sv.size()));
    }
  } else {
    return Status::Invalid("Unsupported npy type");
  }
  return Status::OK();
}

// Helper class to convert decoded examples to an Arrow Table.
// One converter should be created for each Table to be created. Do not re-use.
class DecodedExamplesToTableConverter {
 public:
  DecodedExamplesToTableConverter() = default;
  ~DecodedExamplesToTableConverter() = default;

  // Disallow copy and move.
  DecodedExamplesToTableConverter(
      const DecodedExamplesToTableConverter&) = delete;
  DecodedExamplesToTableConverter& operator=(
      const DecodedExamplesToTableConverter&) = delete;

  // Adds a decoded example (a Dict[str, Union[None, ndarray]]), i.e. one "row",
  // to the Table.
  // Returns an error if `decoded_example` is invalid. If an error is returned,
  // this class is in undefined state and no further calls to it should be made.
  Status AddDecodedExample(PyObject* decoded_example) {
    if (!PyDict_Check(decoded_example)) {
      return Status::Invalid("Expected a dict.");
    }
    PyObject* py_key;
    PyObject* py_value;
    Py_ssize_t pos = 0;
    std::vector<bool> feature_seen(array_builder_by_index_.size(), false);
    while (PyDict_Next(decoded_example, &pos, &py_key, &py_value)) {
      std::string feature_name;
      RETURN_NOT_OK(FeatureNameToString(py_key, &feature_name));
      auto status = AddFeature(feature_name, py_value, &feature_seen);
      if (!status.ok()) {
        return Status(status.code(),
                      absl::StrCat("Failed to append feature ", feature_name,
                                   " from example ", example_count_,
                                   " error message: ", status.message()));
      }
    }

    // For any unseen feature, append null if there is a builder for it,
    // otherwise skip, as when the builder is being created when a first
    // not-None values are seen, the builder will be appened with nulls of
    // number of examples seen so far.
    for (int j = 0; j < feature_seen.size(); ++j) {
      if (!feature_seen[j]) {
        if (array_builder_by_index_[j]) {
          RETURN_NOT_OK(array_builder_by_index_[j]->AppendNull());
        }
      }
    }
    ++example_count_;
    return Status::OK();
  }

  // This should be called after all AddDecodedExample() calls are made.
  Status BuildTable(std::shared_ptr<Table>* table) {
    ArrayVector arrays;
    std::vector<std::shared_ptr<Field>> fields;
    for (const auto& pair : feature_name_to_index_) {
      const std::string& feature_name = pair.first;
      const int feature_index = pair.second;
      const auto& builder = array_builder_by_index_[feature_index];
      arrays.emplace_back();
      std::shared_ptr<DataType> type;
      if (builder) {
        RETURN_NOT_OK(builder->Finish(&arrays.back()));
        type = builder->type();
      } else {
        // This feature is None for all the examples we've seen so far. To
        // distinguish the case where the feature did not appear in any of the
        // examples from it being None for any of the examples, we use a
        // NullArray to represent this feature.
        arrays.back() = std::make_shared<arrow::NullArray>(example_count_);
        type = std::make_shared<arrow::NullType>();
      }
      fields.push_back(std::make_shared<Field>(feature_name, type));
    }
    *table = Table::Make(schema(std::move(fields)), arrays);
    return Status::OK();
  }

 private:
  // Adds `value`, a feature, i.e. a "cell", to the Table. Returns an error if
  // `value` is not a valid feature.
  Status AddFeature(const std::string& feature_name, PyObject* value,
                    std::vector<bool>* feature_seen) {
    auto iter = feature_name_to_index_.find(feature_name);

    // We haven't seen this feature before. Make room for it in various
    // parallel arrays.
    if (iter == feature_name_to_index_.end()) {
      std::tie(iter, std::ignore) = feature_name_to_index_.insert(
          std::make_pair(feature_name, array_builder_by_index_.size()));
      array_builder_by_index_.emplace_back();
      feature_seen->push_back(true);
    }

    // We've seen this feature, but it might not have a builder yet (
    // all its previous values are None), and we might
    // not be able to create a builder (its current value is still None) for it
    // at this time.
    // If we do create a builder for it at this time, we need to append nulls
    // to account for its previous Nones, if any.
    const int feature_index = iter->second;
    (*feature_seen)[feature_index] = true;
    auto& array_builder = array_builder_by_index_[feature_index];
    if (value == Py_None) {
      if (array_builder) {
        RETURN_NOT_OK(array_builder->AppendNull());
      }
    } else {
      PyArrayObject* np_array = nullptr;
      RETURN_NOT_OK(GetNumPyArray(value, &np_array));
      if (!array_builder) {
        RETURN_NOT_OK(MakeListBuilderFromFeature(np_array, &array_builder));
        RETURN_NOT_OK(AppendNulls(example_count_, array_builder.get()));
      }
      RETURN_NOT_OK(AppendNpArrayToBuilder(np_array, array_builder.get()));
    }
    return Status::OK();
  }

  int64_t example_count_ = 0;
  std::vector<std::unique_ptr<ListBuilder>> array_builder_by_index_;
  absl::flat_hash_map<std::string, int> feature_name_to_index_;
};

Status DecodedExamplesToTable(PyObject* list_of_decoded_examples,
                              std::shared_ptr<Table>* table) {
  DecodedExamplesToTableConverter converter;
  const Py_ssize_t list_size = PyList_Size(list_of_decoded_examples);
  if (list_size == 0) {
    // A table must have at list one column, but we can't know what that column
    // should be given an empty list.
    return Status::Invalid("Could not convert an empty list to a Table.");
  }
  for (int i = 0; i < list_size; ++i) {
    RETURN_NOT_OK(converter.AddDecodedExample(
        PyList_GetItem(list_of_decoded_examples, i)));
  }
  return converter.BuildTable(table);
}

}  // namespace

PyObject* TFDV_Arrow_DecodedExamplesToTable(
    PyObject* list_of_decoded_examples) {
  static const int kUnused = arrow::py::import_pyarrow();
  // This suppresses the "unused variable" warning.
  (void)kUnused;
  // Import numpy. (This is actually a macro, and "ret" is the return value
  // if import fails.)
  import_array1(/*ret=*/nullptr);

  if (!PyList_Check(list_of_decoded_examples)) {
    PyErr_SetString(PyExc_TypeError, "DecodedExamplesToTable Expected a list.");
    return nullptr;
  }
  std::shared_ptr<Table> table;
  TFDV_RAISE_IF_NOT_OK(
      DecodedExamplesToTable(list_of_decoded_examples, &table));
  return arrow::py::wrap_table(table);
}
