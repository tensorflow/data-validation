/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_data_validation/coders/cc/fast_example_decoder.h"

#include <Python.h>
#include <string>

#include "numpy/arrayobject.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace data_validation {

using ::tensorflow::Example;
using ::tensorflow::Feature;

PyObject* TFDV_DecodeExample(PyObject* serialized_proto) {
  // Import numpy. (This is actually a macro, and "ret" is the return value
  // if import fails.)
  import_array1(/*ret=*/nullptr);
  if (!PyBytes_Check(serialized_proto)) {
    PyErr_Format(PyExc_ValueError, "Invalid input type: expected bytes.");
    return nullptr;
  }
  // Parse example proto.
  char* data = nullptr;
  Py_ssize_t size;
  if (PyBytes_AsStringAndSize(serialized_proto, &data, &size) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert bytes to string.");
    return nullptr;
  }
  Example example;
  if (!example.ParseFromArray(data, size)) {
    PyErr_Format(PyExc_ValueError, "Failed to parse input proto.");
    return nullptr;
  }

  // Initialize Python result dict.
  PyObject* result = PyDict_New();

  // Iterate over the features and add it to the dict.
  for (const auto& p : example.features().feature()) {
    const string& feature_name = p.first;
    const Feature& feature = p.second;

    PyObject* feature_values_ndarray;

    switch (feature.kind_case()) {
      case Feature::kBytesList: {
        const tensorflow::protobuf::RepeatedPtrField<string>& values =
            feature.bytes_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray =
            PyArray_SimpleNew(1, values_dims, PyArray_OBJECT);
        PyObject** buffer =
            reinterpret_cast<PyObject**>(PyArray_DATA(feature_values_ndarray));
        for (int i = 0; i < values.size(); ++i) {
          const string& v = values[i];
          buffer[i] = PyBytes_FromStringAndSize(v.data(), v.size());
        }
        break;
      }
      case Feature::kFloatList: {
        const tensorflow::protobuf::RepeatedField<float>& values =
            feature.float_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray =
            PyArray_SimpleNew(1, values_dims, PyArray_FLOAT32);
        memcpy(reinterpret_cast<void*>(PyArray_DATA(feature_values_ndarray)),
               values.data(), values.size() * sizeof(float));
        break;
      }
      case Feature::kInt64List: {
        const tensorflow::protobuf::RepeatedField<
            tensorflow::protobuf_int64>& values = feature.int64_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray =
            PyArray_SimpleNew(1, values_dims, PyArray_INT64);
        memcpy(reinterpret_cast<void*>(PyArray_DATA(feature_values_ndarray)),
               values.data(), values.size() * sizeof(int64));
        break;
      }
      case Feature::KIND_NOT_SET: {
        // If we have a feature with no value list, we consider it to be a
        // missing value.
        feature_values_ndarray = Py_None;
        Py_INCREF(Py_None);
        break;
      }
      default: {
        CHECK(false) << "Invalid value list in input proto.";
      }
    }
    int err = PyDict_SetItemString(
        result, feature_name.data(), feature_values_ndarray);
    Py_XDECREF(feature_values_ndarray);
    if (err == -1) {
      PyErr_Format(PyExc_ValueError, "Failed to insert item into Dict.");
      return nullptr;
    }
  }
  return result;
}

}  // namespace data_validation
}  // namespace tensorflow
