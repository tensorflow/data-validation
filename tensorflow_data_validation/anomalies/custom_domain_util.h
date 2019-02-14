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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_DOMAIN_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_DOMAIN_UTIL_H_

#include <vector>

#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

// Semantic domains like image_domain, url_domain, ... can be detected by
// heuristics in stats generation. If such a domain is detected the feature
// stats are associated with a CustomStatistic with name: 'domain_info' and a
// str value with the text representation of the detected domain, e.g:
// custom_stats: {name: 'domain_info' str: 'mid_domain {}'}
//
// This method provides a best-effort update of the semantic type of `feature`
// based on `custom_stats` and returns true iff a valid custom domain was
// detected and successfully updated `feature`. The logic is currently
// conservative:
// - Never modify `feature` if it has an existing domain
// - If a feature is associated with multiple custom_stats for 'domain_info'
//   they are ignored
// - If the value of the 'domain_info' custom stat is invalid or does not set
//   exactly one field of the Feature.domain_info oneof it is ignored
bool BestEffortUpdateCustomDomain(
    const std::vector<tensorflow::metadata::v0::CustomStatistic>& custom_stats,
    tensorflow::metadata::v0::Feature* feature);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_CUSTOM_DOMAIN_UTIL_H_
