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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_BOOL_DOMAIN_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_BOOL_DOMAIN_UTIL_H_

#include <vector>

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// Update a BoolDomain by itself. Namely, if the string values corresponding to
// true and false in the domain are the same, clear the value for false.
std::vector<Description> UpdateBoolDomainSelf(
    tensorflow::metadata::v0::BoolDomain* bool_domain);

// This updates bool_domain. Should only be called if bool_domain is set.
// If the type is INT and the min and max are out of the range {0,1},
// this will set int_domain.
std::vector<Description> UpdateBoolDomain(
    const FeatureStatsView& feature_stats,
    tensorflow::metadata::v0::Feature* feature);

// Determine if this could be a BoolDomain.
// Note this takes precedence over IntDomain and StringDomain.
bool IsBoolDomainCandidate(const FeatureStatsView& feature_stats);

// Generate a BoolDomain from the stats.
// The behavior is undefined if IsBoolDomainCandidate(stats) is false.
tensorflow::metadata::v0::BoolDomain BoolDomainFromStats(
    const FeatureStatsView& stats);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_BOOL_DOMAIN_UTIL_H_
