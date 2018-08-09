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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FLOAT_DOMAIN_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FLOAT_DOMAIN_UTIL_H_

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// Updates the float_domain based upon the range of values in <stats>, be they
// STRING or FLOAT.
// Will recommend the field be cleared if the type is STRING or BYTES but
// the strings do not represent floats. Undefined behavior if the data is INT.
UpdateSummary UpdateFloatDomain(
    const FeatureStatsView& stats,
    tensorflow::metadata::v0::FloatDomain* float_domain);

// Returns true if feature_stats is a STRING field has only floats and no
// non-UTF8 strings.
bool IsFloatDomainCandidate(const FeatureStatsView& feature_stats);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FLOAT_DOMAIN_UTIL_H_
