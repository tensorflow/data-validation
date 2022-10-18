/* Copyright 2022 Google LLC

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
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_VALIDATION_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_VALIDATION_H_

#include "tensorflow_data_validation/anomalies/proto/custom_validation_config.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

// Validates `test_statistics` (either alone or by comparing it to
// `base_statistics`) by running the SQL queries specified in `validations`. If
// a validation query returns False, a corresponding anomaly is added to
// `result`.
Status CustomValidateStatistics(
    const metadata::v0::DatasetFeatureStatisticsList& test_statistics,
    const metadata::v0::DatasetFeatureStatisticsList* base_statistics,
    const CustomValidationConfig& validations,
    const absl::optional<string> environment, metadata::v0::Anomalies* result);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_VALIDATION_H_
