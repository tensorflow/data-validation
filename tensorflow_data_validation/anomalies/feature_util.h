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
// TODO(b/148429931): add UpdateFixedShape too.
#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_

#include <vector>

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/derived_feature.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// If the value_count(s) constraints are not satisfied, adjust them.
std::vector<Description> UpdateFeatureValueCounts(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::Feature* feature);

// If the shape constraints are not satisfied, adjust them, or propose
// to switch to value_counts constraints.
std::vector<Description> UpdateFeatureShape(
    const FeatureStatsView& feature_stats_view,
    bool generate_legacy_feature_spec,
    tensorflow::metadata::v0::Feature* feature);

// If a feature occurs in too few examples, or a feature occurs in too small
// a fraction of the examples, adjust the presence constraints to account for
// this.
std::vector<Description> UpdatePresence(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::FeaturePresence* presence);

bool FeatureHasComparator(const tensorflow::metadata::v0::Feature& feature,
                          FeatureComparatorType comparator_type);

// Gets the feature comparator, creating it if it does not exist.
tensorflow::metadata::v0::FeatureComparator* GetFeatureComparator(
    tensorflow::metadata::v0::Feature* feature,
    FeatureComparatorType comparator_type);

// Translates a feature lifecycle enum into a boolean of whether a feature with
// that lifecycle stage is considered depreacted.
bool LifecycleStageIsDeprecated(const metadata::v0::LifecycleStage stage);

struct FeatureComparisonResult {
  std::vector<Description> descriptions;
  std::vector<tensorflow::metadata::v0::DriftSkewInfo::Measurement>
      measurements;
};

// Updates comparator from the feature stats.
// Note that if the "control" was missing, we have deprecated the column.
FeatureComparisonResult UpdateFeatureComparatorDirect(
    const FeatureStatsView& stats, const FeatureComparatorType comparator_type,
    tensorflow::metadata::v0::FeatureComparator* comparator);

// Initializes presence and shape constraints (value counts or fixed shape)
// given stats.
// If `infer_fixed_shape` is true, try inferring a fixed shape for the feature,
// otherwise, always infers value count.
void InitPresenceAndShape(const FeatureStatsView& feature_stats_view,
                          bool infer_fixed_shape,
                          tensorflow::metadata::v0::Feature* feature);

// Deprecate a feature. Currently sets deprecated==true, but later will
// set the lifecycle_stage==DEPRECATED. The contract of this method is that
// FeatureIsDeprecated is set to true after it is called.
void DeprecateFeature(tensorflow::metadata::v0::Feature* feature);

// Same as above for SparseFeature.
void DeprecateSparseFeature(
    tensorflow::metadata::v0::SparseFeature* sparse_feature);

// Same as above for WeightedFeature.
void DeprecateWeightedFeature(
    tensorflow::metadata::v0::WeightedFeature* weighted_feature);

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
// TODO(b/148429846): consider if PLANNED is really deprecated.
bool FeatureIsDeprecated(const tensorflow::metadata::v0::Feature& feature);

// Same as above for SparseFeature.
bool SparseFeatureIsDeprecated(
    const tensorflow::metadata::v0::SparseFeature& sparse_feature);

// Same as above for WeightedFeature.
bool WeightedFeatureIsDeprecated(
    const tensorflow::metadata::v0::WeightedFeature& weighted_feature);

// Get the maximum allowed off the domain.
double GetMaxOffDomain(const tensorflow::metadata::v0::DistributionConstraints&
                           distribution_constraints);

// Clear the domain of the feature.
void ClearDomain(tensorflow::metadata::v0::Feature* feature);

// Updates the UniqueConstraints specified for the feature.
std::vector<Description> UpdateUniqueConstraints(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::Feature* feature);

// Mark a feature as derived by setting its lifecycle to DERIVED and copying
// a derived source.
void MarkFeatureDerived(
    const tensorflow::metadata::v0::DerivedFeatureSource& derived_source,
    tensorflow::metadata::v0::Feature* feature);

}  // namespace data_validation
}  // namespace tensorflow
#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_UTIL_H_
