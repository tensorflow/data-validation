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

#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
int NumericalSeverity(tensorflow::metadata::v0::AnomalyInfo::Severity a) {
  switch (a) {
    case tensorflow::metadata::v0::AnomalyInfo::UNKNOWN:
      return 0;
    case tensorflow::metadata::v0::AnomalyInfo::WARNING:
      return 1;
    case tensorflow::metadata::v0::AnomalyInfo::ERROR:
      return 2;
    default:
      LOG(FATAL) << "Unknown severity: " << a;
  }
}
}  // namespace
// For internal use only.
tensorflow::metadata::v0::AnomalyInfo::Severity MaxSeverity(
    tensorflow::metadata::v0::AnomalyInfo::Severity a,
    tensorflow::metadata::v0::AnomalyInfo::Severity b) {
  return (NumericalSeverity(a) > NumericalSeverity(b)) ? a : b;
}

}  // namespace data_validation
}  // namespace tensorflow
