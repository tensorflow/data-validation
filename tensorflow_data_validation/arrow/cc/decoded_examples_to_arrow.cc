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
// CAUTION: Arrow API and NumPy API are both used in this file. Care must be
// taken to make sure we use our own NumPy API stubs instead of Arrow's:
// - Do not include any arrow header that pulls in numpy headers. If you have
//   to use certain APIs in such a header, you need to create a stub (
//   see pyarrow_numpy_stub.h)
// - Do not directly include numpy headers. Include init_numpy.h instead.

// This include is needed to avoid C2528. See https://bugs.python.org/issue24643
// Must be included before other arrow headers.
#include "arrow/python/platform.h"
#include "arrow/python/pyarrow.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "arrow/api.h"
#include "tensorflow_data_validation/arrow/cc/common.h"
#include "tensorflow_data_validation/arrow/cc/init_numpy.h"
#include "tensorflow_data_validation/arrow/cc/pyarrow_numpy_stub.h"

namespace {
using ::arrow::ArrayVector;
using ::arrow::ChunkedArray;
using ::arrow::DataType;
using ::arrow::Field;
using ::arrow::Status;
using ::arrow::Table;
namespace pyarrow_numpy_stub =
    ::tensorflow::data_validation::pyarrow_numpy_stub;

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


// Returns the arrow DataType of ListArray<x> where x corresponds to `descr`.
Status ArrowTypeFromNumpyDescr(PyArray_Descr* descr,
                               std::shared_ptr<DataType>* arrow_type) {
  std::shared_ptr<DataType> value_type;
  if (descr->type_num == NPY_OBJECT) {
    value_type = std::make_shared<arrow::BinaryType>();
  } else {
    RETURN_NOT_OK(pyarrow_numpy_stub::NumPyDtypeToArrow(
        reinterpret_cast<PyObject*>(descr), &value_type));
  }
  *arrow_type = std::make_shared<arrow::ListType>(std::move(value_type));
  return Status::OK();
}

// PyRef holds a PyObject and automatically decreases its refcount upon
// destruction.
class PyRef {
 public:
  PyRef() : PyRef(nullptr) {}
  explicit PyRef(PyObject* obj) : obj_(obj) {}
  ~PyRef() {
    Py_XDECREF(obj_);
  }
  // Not copyable but movable.
  PyRef(const PyRef&) = delete;
  PyRef& operator=(const PyRef&) = delete;

  PyRef(PyRef&& other) {
    obj_ = other.release();
  }
  PyRef& operator=(PyRef&& other) {
    obj_ = other.release();
    return *this;
  }

  PyObject* get() const {
    return obj_;
  }

  void reset(PyObject* obj) {
    Py_XDECREF(obj_);
    obj_ = obj;
  }

  PyObject* release() {
    PyObject* result = obj_;
    obj_ = nullptr;
    return result;
  }

 private:
  PyObject* obj_;
};

// Helper class to convert decoded examples to an Arrow Table.
// One converter should be created for each Table to be created. Do not re-use.
// Implementation note:
// The implementation builds a numpy array of numpy arrays for each feature,
// and uses arrow's routine to convert that to an Arrow ListArray. The routine
// is exactly what's used in pyarrow, so we could have built the numpy array
// of arrays in python and called the same routine (python wrapped), however
// that is ~8x slower.
// We also could have gone directly with Arrow's ArrayBuilders without the
// numpy array indirection but then we would have to implement the type bridge
// from numpy to arrow.
class DecodedExamplesToTableConverter {
 public:
  DecodedExamplesToTableConverter(const int64_t num_examples)
      : num_examples_(num_examples) {}
  ~DecodedExamplesToTableConverter() {}

  // Disallow copy and move.
  DecodedExamplesToTableConverter(
      const DecodedExamplesToTableConverter&) = delete;
  DecodedExamplesToTableConverter& operator=(
      const DecodedExamplesToTableConverter&) = delete;

  // Adds a decoded example (a Dict[str, Union[None, ndarray]]), i.e. one "row",
  // to the Table.
  // Returns an error if `decoded_example` is invalid. If an error is returned,
  // this class is in undefined state and no further calls to it should be made.
  // Note that this must be called exactly `num_examples` times before
  // BuildTable() is called otherwise BuildTable() will return an error.
  Status AddDecodedExample(PyObject* decoded_example) {
    if (!PyDict_Check(decoded_example)) {
      return Status::Invalid("Expected a dict.");
    }
    PyObject* py_key;
    PyObject* py_value;
    Py_ssize_t pos = 0;
    std::vector<bool> feature_seen(feature_rep_by_index_.size(), false);
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
    // For any unseen feature, set the value of this example to None if there is
    // a FeatureRep for it, otherwise skip, as when a FeatureRep is being
    // created when a first not-None value is seen, it will be populated with
    // Nones of number of examples seen so far.
    for (int j = 0; j < feature_seen.size(); ++j) {
      if (!feature_seen[j]) {
        if (feature_rep_by_index_[j]) {
          RETURN_NOT_OK(feature_rep_by_index_[j]->SetValueNdArray(
              example_count_, Py_None));
        }
      }
    }
    ++example_count_;
    return Status::OK();
  }

  // This should be called after all AddDecodedExample() calls are made.
  Status BuildTable(std::shared_ptr<Table>* table) {
    if (num_examples_ != example_count_) {
      return Status::Invalid(absl::StrCat(
          "Internal error: AddExample must be called ", num_examples_,
          " times but was called ", example_count_));
    }
    ArrayVector arrays;
    std::vector<std::shared_ptr<Field>> fields;
    for (const auto& pair : feature_name_to_index_) {
      const std::string& feature_name = pair.first;
      const int feature_index = pair.second;
      auto& maybe_feature_rep = feature_rep_by_index_[feature_index];
      arrays.emplace_back();
      std::shared_ptr<DataType> type;
      if (maybe_feature_rep) {
        RETURN_NOT_OK(ArrowTypeFromNumpyDescr(
            maybe_feature_rep->value_ndarray_descr(), &type));
        std::shared_ptr<ChunkedArray> chunked_array;
        RETURN_NOT_OK(pyarrow_numpy_stub::NdarrayToArrow(
            arrow::default_memory_pool(),
            /*ao=*/maybe_feature_rep->ndarray_of_ndarrays(),
            /*mo=*/nullptr, /*use_pandas_null_sentinels=*/false, type,
            &chunked_array));
        if (!chunked_array || chunked_array->num_chunks() != 1) {
          return Status::Invalid(absl::StrCat("Internal error."));
        }
        arrays.back() = chunked_array->chunk(0);
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
    *table = Table::Make(schema(std::move(fields)), arrays, num_examples_);
    return Status::OK();
  }

 private:
  // Groups internal states about a feature whose type has been determined
  // (which will be come a column in the resulting table).
  class FeatureRep {
   public:
    FeatureRep(const size_t num_examples, const size_t num_leading_nones,
               PyArray_Descr* value_ndarray_descr)
        : value_ndarray_descr_(value_ndarray_descr) {
      npy_intp values_dims[] = {static_cast<npy_intp>(num_examples)};
      ndarray_of_ndarrays_.reset(PyArray_SimpleNew(1, values_dims, NPY_OBJECT));
      ndarray_of_ndarrays_data_ = reinterpret_cast<PyObject**>(PyArray_DATA(
          reinterpret_cast<PyArrayObject*>(ndarray_of_ndarrays_.get())));
      for (int i = 0; i < num_leading_nones; ++i) {
        ndarray_of_ndarrays_data_[i] = Py_None;
        Py_INCREF(Py_None);
      }
    }

    ~FeatureRep() = default;

    // Disallow copy (otherwise the refcount is messed up) but allow move (
    // otherwise cannot be put inside a vector).
    FeatureRep(const FeatureRep&) = delete;
    FeatureRep& operator=(const FeatureRep&) = delete;
    FeatureRep(FeatureRep&&) = default;
    FeatureRep& operator=(FeatureRep&&) = default;

    // Type of the numpy array of the feature value.
    int value_npy_type() const {
      return value_ndarray_descr()->type_num;
    }

    // The numpy descriptor of one of the numpy arrays in `ndarray_of_ndarrays`.
    // Used for getting the type only since all the numpy arrays contained
    // are of the same type.
    PyArray_Descr* value_ndarray_descr() const {
      return value_ndarray_descr_;
    }

    // Sets `ndarray_of_ndarrays`[`index`] to `py_object_np_array`.
    // `py_object_np_array` could be Py_None or a numpy array. If it's a numpy
    // array, returns an error if its type does not match `value_npy_type`().
    Status SetValueNdArray(const size_t index, PyObject* py_obj_np_array) {
      if (py_obj_np_array != Py_None) {
        PyArrayObject* np_array;
        RETURN_NOT_OK(GetNumPyArray(py_obj_np_array, &np_array));
        const int this_npy_type = PyArray_TYPE(np_array);
        if (this_npy_type != value_npy_type()) {
          return Status::Invalid(
              absl::StrCat("Mismatch feature numpy array types. Previously: ",
                           value_npy_type(), " , now: ", this_npy_type));
        }
      }
      ndarray_of_ndarrays_data_[index] = py_obj_np_array;
      Py_INCREF(py_obj_np_array);
      return Status::OK();
    }

    PyObject* ndarray_of_ndarrays() const {
      return ndarray_of_ndarrays_.get();
    }

   private:
    // A numpy array of type np.object that contains feature values
    // (as numpy arrays).
    PyRef ndarray_of_ndarrays_;
    // The numpy descriptor of one of the numpy arrays in `ndarray_of_ndarrays_`
    // .
    PyArray_Descr* value_ndarray_descr_;
    // Points to the data buffer of `ndarray_of_ndarrays`.
    PyObject** ndarray_of_ndarrays_data_;
  };

  // Adds `value`, a feature, i.e. a "cell", to the Table. Returns an error if
  // `value` is not a valid feature.
  Status AddFeature(const std::string& feature_name, PyObject* value,
                    std::vector<bool>* feature_seen) {
    auto iter = feature_name_to_index_.find(feature_name);

    // We haven't seen this feature before. Make room for it in various
    // parallel arrays.
    if (iter == feature_name_to_index_.end()) {
      std::tie(iter, std::ignore) = feature_name_to_index_.insert(
          std::make_pair(feature_name, feature_name_to_index_.size()));
      feature_seen->push_back(true);
      feature_rep_by_index_.emplace_back();
    }

    // We've seen this feature, but it might not have a FeatureRep yet (
    // all its previous values are None), and we might
    // not be able to create a FeatureRep (its current value is still None) for
    // it at this time.
    const int feature_index = iter->second;
    (*feature_seen)[feature_index] = true;
    auto& maybe_feature_rep = feature_rep_by_index_[feature_index];
    if (!maybe_feature_rep) {
      if (value != Py_None) {
        PyArrayObject* numpy_array;
        RETURN_NOT_OK(GetNumPyArray(value, &numpy_array));
        maybe_feature_rep = FeatureRep(num_examples_, example_count_,
                                       PyArray_DESCR(numpy_array));
      }
    }
    // maybe_feature_rep might be created at above.
    if (maybe_feature_rep) {
      RETURN_NOT_OK(maybe_feature_rep->SetValueNdArray(example_count_, value));
    }

    return Status::OK();
  }

  int64_t num_examples_;
  int64_t example_count_ = 0;
  std::vector<absl::optional<FeatureRep>> feature_rep_by_index_;
  absl::flat_hash_map<std::string, int> feature_name_to_index_;
};

Status DecodedExamplesToTable(PyObject* list_of_decoded_examples,
                              std::shared_ptr<Table>* table) {
  const Py_ssize_t list_size = PyList_Size(list_of_decoded_examples);
  if (list_size == 0) {
    // A table must have at list one column, but we can't know what that column
    // should be given an empty list.
    return Status::Invalid("Could not convert an empty list to a Table.");
  }
  DecodedExamplesToTableConverter converter(list_size);
  for (int i = 0; i < list_size; ++i) {
    RETURN_NOT_OK(converter.AddDecodedExample(
        PyList_GetItem(list_of_decoded_examples, i)));
  }
  return converter.BuildTable(table);
}

}  // namespace

PyObject* TFDV_Arrow_DecodedExamplesToTable(
    PyObject* list_of_decoded_examples) {
  tensorflow::data_validation::ImportPyArrow();
  tensorflow::data_validation::ImportNumpy();

  if (!PyList_Check(list_of_decoded_examples)) {
    PyErr_SetString(PyExc_TypeError, "DecodedExamplesToTable Expected a list.");
    return nullptr;
  }
  std::shared_ptr<Table> table;
  TFDV_RAISE_IF_NOT_OK(
      DecodedExamplesToTable(list_of_decoded_examples, &table));
  return arrow::py::wrap_table(table);
}
