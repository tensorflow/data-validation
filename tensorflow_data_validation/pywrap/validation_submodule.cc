// Copyright 2020 Google LLC
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
#include "tensorflow_data_validation/pywrap/validation_submodule.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_data_validation/anomalies/feature_statistics_validator.h"
#include "include/pybind11/pybind11.h"


namespace tensorflow {
namespace data_validation {
namespace py = pybind11;

void DefineValidationSubmodule(py::module main_module) {
  auto m = main_module.def_submodule("validation");
  m.doc() = "Validation API.";

  m.def("InferSchema",
        [](const std::string& statistics_proto_string,
           int max_string_domain_size, bool infer_feature_shape) -> py::object {
          std::string schema_proto_string;
          const tensorflow::Status status =
              InferSchema(statistics_proto_string, max_string_domain_size,
                          infer_feature_shape, &schema_proto_string);
          if (!status.ok()) {
            throw std::runtime_error(status.ToString());
          }
          return py::bytes(schema_proto_string);
        });

  m.def("UpdateSchema",
        [](const std::string& schema_proto_string,
           const std::string& statistics_proto_string,
           int max_string_domain_size) -> py::object {
          std::string output_schema_proto_string;
          const tensorflow::Status status =
              UpdateSchema(
                  schema_proto_string, statistics_proto_string,
                  max_string_domain_size, &output_schema_proto_string);
          if (!status.ok()) {
            throw std::runtime_error(status.ToString());
          }
          return py::bytes(output_schema_proto_string);
        });

  m.def("ValidateFeatureStatistics",
        [](const std::string& statistics_proto_string,
           const std::string& schema_proto_string,
           const std::string& environment,
           const std::string& previous_span_statistics_proto_string,
           const std::string& serving_statistics_proto_string,
           const std::string& previous_version_statistics_proto_string,
           const std::string& feature_needed_string,
           const std::string& validation_config_string,
           const bool enable_diff_regions) -> py::object {
          std::string anomalies_proto_string;
          const tensorflow::Status status = \
              ValidateFeatureStatisticsWithSerializedInputs(
                  statistics_proto_string, schema_proto_string, environment,
                  previous_span_statistics_proto_string,
                  serving_statistics_proto_string,
                  previous_version_statistics_proto_string,
                  feature_needed_string, validation_config_string,
                  enable_diff_regions, &anomalies_proto_string);
          if (!status.ok()) {
            throw std::runtime_error(status.ToString());
          }
          return py::bytes(anomalies_proto_string);
        });
}

}  // namespace data_validation
}  // namespace tensorflow
