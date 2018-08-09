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

#include "tensorflow_data_validation/anomalies/metrics.h"

#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/platform/types.h"

using std::map;

namespace tensorflow {
namespace data_validation {

namespace {

// Gets the L-infty norm of a vector, represented as a map.
// This is the largest absolute value of any value.
// For convenience, the associated key is also returned.
std::pair<string, double> GetLInftyNorm(const map<string, double>& vec) {
  std::pair<string, double> best_so_far;
  for (const auto& pair : vec) {
    const string& key = pair.first;
    const double value = std::abs(pair.second);
    if (value >= best_so_far.second) {
      best_so_far = {key, value};
    }
  }
  return best_so_far;
}

}  // namespace

std::pair<string, double> LInftyDistance(const map<string, double>& counts_a,
                                         const map<string, double>& counts_b) {
  return GetLInftyNorm(GetDifference(Normalize(counts_a), Normalize(counts_b)));
}

std::pair<string, double> LInftyDistance(const FeatureStatsView& a,
                                         const FeatureStatsView& b) {
  const map<string, double> prob_a = Normalize(a.GetStringValuesWithCounts());
  const map<string, double> prob_b = Normalize(b.GetStringValuesWithCounts());

  return GetLInftyNorm(GetDifference(prob_a, prob_b));
}

}  // namespace data_validation
}  // namespace tensorflow
