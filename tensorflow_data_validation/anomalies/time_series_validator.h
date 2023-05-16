/* Copyright 2023 Google LLC

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

// Monitor data in time series by exporting statistics and computed anomalies
// in time series format.

#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_TIME_SERIES_VALIDATOR_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_TIME_SERIES_VALIDATOR_H_

#include <set>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow_data_validation/google/protos/time_series_metrics.pb.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

using ::tensorflow::data_validation::ValidationMetrics;
using ::tensorflow::metadata::v0::DatasetFeatureStatisticsList;

enum class SliceComparisonMode {kSame};

class SliceComparisonConfig {
 public:
  static inline SliceComparisonConfig Corresponding() {
    return SliceComparisonConfig(SliceComparisonMode::kSame);
  }

  SliceComparisonMode mode_;

 private:
  SliceComparisonConfig(const SliceComparisonMode& mode) : mode_(mode) {}
};

absl::StatusOr<std::vector<ValidationMetrics>> ValidateTimeSeriesStatistics(
    const metadata::v0::DatasetFeatureStatisticsList& statistics,
    const gtl::optional<metadata::v0::DatasetFeatureStatisticsList>&
        reference_statistics,
    const SliceComparisonConfig& slice_config);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_TIME_SERIES_VALIDATOR_H_
