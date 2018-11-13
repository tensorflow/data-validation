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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_STRING_DOMAIN_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_STRING_DOMAIN_UTIL_H_

#include <vector>

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/schema.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// True if two domains are similar. If they are "small" according to the
// config.min_count, then they must be identical. Otherwise, they must
// have a large jaccard similarity.
bool IsSimilarStringDomain(const tensorflow::metadata::v0::StringDomain& a,
                           const tensorflow::metadata::v0::StringDomain& b,
                           const EnumsSimilarConfig& config);

// Returns true if this feature_stats has less than enum_threshold number of
// unique string values.
bool IsStringDomainCandidate(const FeatureStatsView& feature_stats,
                             const int enum_threshold);


// If there are any values that are repeated, remove them.
std::vector<Description> UpdateStringDomainSelf(
    tensorflow::metadata::v0::StringDomain* string_domain);

// Update a string domain.
// updater: configuration used to determine if the string domain needs to be
//   deleted.
// stats: the statistics of the string domain.
// max_off_domain: the maximum fraction of mass allowed to be off the domain.
// string_domain: string_domain to be modified.
UpdateSummary UpdateStringDomain(
    const Schema::Updater& updater,
    const FeatureStatsView& stats, double max_off_domain,
    tensorflow::metadata::v0::StringDomain* string_domain);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_STRING_DOMAIN_UTIL_H_
