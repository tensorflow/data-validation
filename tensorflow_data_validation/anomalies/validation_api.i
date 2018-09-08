/* Copyright 2018 Google LLC

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

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_data_validation/anomalies/feature_statistics_validator.h"

#ifdef HAS_GLOBAL_STRING
  using ::string;
#else
  using std::string;
#endif

PyObject* ConvertToPythonString(const string& input_str) {
  return PyBytes_FromStringAndSize(input_str.data(), input_str.size());
}
%}

%{
PyObject* InferSchema(const string& statistics_proto_string,
                      int max_string_domain_size) {
  string schema_proto_string;
  const tensorflow::Status status = tensorflow::data_validation::InferSchema(
    statistics_proto_string, max_string_domain_size, &schema_proto_string);
  if (!status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, status.error_message().c_str());
    return NULL;
  }
  return ConvertToPythonString(schema_proto_string);
}


PyObject* ValidateFeatureStatistics(
  const string& statistics_proto_string,
  const string& schema_proto_string,
  const string& environment,
  const string& previous_statistics_proto_string,
  const string& serving_statistics_proto_string) {
  string anomalies_proto_string;
  const tensorflow::Status status = tensorflow::data_validation::ValidateFeatureStatistics(
    statistics_proto_string, schema_proto_string, environment,
    previous_statistics_proto_string, serving_statistics_proto_string,
    &anomalies_proto_string);
  if (!status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, status.error_message().c_str());
    return NULL;
  }
  return ConvertToPythonString(anomalies_proto_string);
}
%}

// Typemap to convert an input argument from Python object to C++ string.
%typemap(in) const string& (string temp) {
  char *buf;
  Py_ssize_t len;
  if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) SWIG_fail;
  temp.assign(buf, len);
  $1 = &temp;
}

PyObject* InferSchema(const string& statistics_proto_string,
                      int max_string_domain_size);

PyObject* ValidateFeatureStatistics(
  const string& statistics_proto_string,
  const string& schema_proto_string,
  const string& environment,
  const string& previous_statistics_proto_string,
  const string& serving_statistics_proto_string);
