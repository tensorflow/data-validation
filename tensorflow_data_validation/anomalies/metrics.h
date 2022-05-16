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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_METRICS_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_METRICS_H_

#include <map>
#include <string>
#include <utility>

#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

// Computes the L-infinity distance between the (weighted) histograms of the
// features.
// Only takes into account how many times the feature are present,
// and scales the histograms so that they sum to 1.
// The first value returned is the element with highest deviation, and
// the second value returned is the L infinity distance itself.
std::pair<string, double> LInftyDistance(const FeatureStatsView& a,
                                         const FeatureStatsView& b);

std::pair<string, double> LInftyDistance(
    const std::map<string, double>& counts_a,
    const std::map<string, double>& counts_b);

// Computes the approximate Jensen-Shannon divergence
// (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) between the
// (weighted) histograms of the features.
Status JensenShannonDivergence(const FeatureStatsView& a,
                                           const FeatureStatsView& b,
                                           double& result);

Status JensenShannonDivergence(
    ::tensorflow::metadata::v0::Histogram& histogram_1,
    ::tensorflow::metadata::v0::Histogram& histogram_2, double& result);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_METRICS_H_
