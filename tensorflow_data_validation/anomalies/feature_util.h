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

// Utilities to modify a feature in the schema.
// TODO(martinz): add UpdateFixedShape too.
#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_

#include <vector>

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// If the value count constraints are not satisfied, adjust them.
std::vector<Description> UpdateValueCount(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::ValueCount* value_count);

// If a feature occurs in too few examples, or a feature occurs in too small
// a fraction of the examples, adjust the presence constraints to account for
// this.
std::vector<Description> UpdatePresence(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::FeaturePresence* presence);

bool FeatureHasComparator(const tensorflow::metadata::v0::Feature& feature,
                          ComparatorType comparator_type);

// Gets the feature comparator, creating it if it does not exist.
tensorflow::metadata::v0::FeatureComparator* GetFeatureComparator(
    tensorflow::metadata::v0::Feature* feature, ComparatorType comparator_type);

// Updates comparator from the feature stats.
// Note that if the "control" was missing, we have deprecated the column.
std::vector<Description> UpdateFeatureComparatorDirect(
    const FeatureStatsView& stats, const ComparatorType comparator_type,
    tensorflow::metadata::v0::FeatureComparator* comparator);

// Initializes the value count and presence given a feature_stats_view.
// This is called when a Feature is first created from a FeatureStatsView.
// It infers OPTIONAL, REPEATED, REQUIRED (in the proto sense),
// and REPEATED_REQUIRED (a repeated field that is always present), and
// sets value count and presence analogously.
void InitValueCountAndPresence(const FeatureStatsView& feature_stats_view,
                               tensorflow::metadata::v0::Feature* feature);

// Deprecate a feature. Currently sets deprecated==true, but later will
// set the lifecycle_stage==DEPRECATED. The contract of this method is that
// FeatureIsDeprecated is set to true after it is called.
void DeprecateFeature(tensorflow::metadata::v0::Feature* feature);

// Same as above for SparseFeature.
void DeprecateSparseFeature(
    tensorflow::metadata::v0::SparseFeature* sparse_feature);

// Tell if a feature is deprecated (i.e., ignored for data validation).
// Note that a deprecated feature is a more relaxed constraint than a feature
// not being present in the schema, as it also suppresses the unexpected column
// anomaly.
// If neither deprecated is set nor lifecycle_stage is set, it is not
// deprecated.
// If deprecated==true, it is deprecated.
// Otherwise, if lifecycle_stage is in {ALPHA, PLANNED, DEPRECATED, DEBUG_ONLY}
//   it is deprecated.
// If lifecycle_stage is in {UNKNOWN_STAGE, BETA, PRODUCTION}
//   it is not deprecated.
// Setting deprecated==false has no effect.
// TODO(martinz): consider if PLANNED is really deprecated.
bool FeatureIsDeprecated(const tensorflow::metadata::v0::Feature& feature);

// Same as above for SparseFeature.
bool SparseFeatureIsDeprecated(
    const tensorflow::metadata::v0::SparseFeature& sparse_feature);

// Get the maximum allowed off the domain.
double GetMaxOffDomain(const tensorflow::metadata::v0::DistributionConstraints&
                           distribution_constraints);

// Clear the domain of the feature.
void ClearDomain(tensorflow::metadata::v0::Feature* feature);
}  // namespace data_validation
}  // namespace tensorflow
#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_
