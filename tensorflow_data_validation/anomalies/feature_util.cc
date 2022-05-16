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

#include "tensorflow_data_validation/anomalies/feature_util.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/metrics.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using absl::optional;
using ::tensorflow::metadata::v0::AnomalyInfo;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureComparator;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::ValueCount;
using ::tensorflow::metadata::v0::WeightedFeature;

constexpr char kSuperfluousValues[] = "Superfluous values";
constexpr char kMissingValues[] = "Missing values";
constexpr char kDropped[] = "Column dropped";
constexpr char kValueNestednessMismatch[] = "Mismatched value nest level";

ComparatorContext GetContext(FeatureComparatorType comparator_type) {
  switch (comparator_type) {
    case FeatureComparatorType::SKEW:
      return {"serving", "training"};
    case FeatureComparatorType::DRIFT:
      return {"previous", "current"};
  }
}

bool HasControlDataset(const FeatureStatsView& stats,
                       FeatureComparatorType comparator_type) {
  switch (comparator_type) {
    case FeatureComparatorType::SKEW:
      return stats.parent_view().GetServing() != absl::nullopt;
    case FeatureComparatorType::DRIFT:
      return stats.parent_view().GetPreviousSpan() != absl::nullopt;
  }
}

absl::optional<FeatureStatsView> GetControlStats(
    const FeatureStatsView& stats, FeatureComparatorType comparator_type) {
  switch (comparator_type) {
    case FeatureComparatorType::SKEW:
      return stats.GetServing();
    case FeatureComparatorType::DRIFT:
      return stats.GetPreviousSpan();
  }
}

void InitValueCount(const FeatureStatsView& feature_stats_view,
                    Feature* feature) {
  // Set value_counts or value_count, depending on whether the feature's values
  // are nested.
  const std::vector<std::pair<int, int>> min_max_num_values =
      feature_stats_view.GetMinMaxNumValues();
  auto set_value_count = [](int min_num_values, int max_num_values,
                            ValueCount* value_count) {
    if (min_num_values > 0) {
      if (min_num_values == max_num_values) {
        // Set min and max value count in the schema if they are same. This
        // would allow required features with same valency to be parsed as dense
        // tensors in TFT.
        value_count->set_min(min_num_values);
        value_count->set_max(max_num_values);
      } else {
        value_count->set_min(1);
      }
    }
  };
  if (feature_stats_view.HasNestedValues()) {
    for (int i = 0; i < min_max_num_values.size(); i++) {
      set_value_count(min_max_num_values[i].first, min_max_num_values[i].second,
                      feature->mutable_value_counts()->add_value_count());
    }
  } else if (min_max_num_values.size() == 1 &&
             min_max_num_values[0].first > 0) {
    set_value_count(min_max_num_values[0].first, min_max_num_values[0].second,
                    feature->mutable_value_count());
  }
}

void InitFixedShape(const FeatureStatsView& feature_stats_view,
                    Feature* feature) {
  if (feature_stats_view.GetFeatureType() == metadata::v0::STRUCT) {
    return;
  }
  std::vector<int> shape;
  const std::vector<std::pair<int, int>> min_max_num_values =
      feature_stats_view.GetMinMaxNumValues();
  const std::vector<double> num_missings =
      feature_stats_view.GetNumMissingNested();
  CHECK_EQ(min_max_num_values.size(), num_missings.size());
  for (int i = 0; i < num_missings.size(); ++i) {
    if (num_missings[i] != 0) {
      return;
    }
    const int min_count = min_max_num_values[i].first;
    const int max_count = min_max_num_values[i].second;
    if (min_count != max_count || min_count <= 0) {
      return;
    }
    shape.push_back(min_count);
  }

  CHECK(feature->shape().dim().empty());
  for (int dim_size : shape) {
    feature->mutable_shape()->add_dim()->set_size(dim_size);
  }
}

std::vector<Description> UpdateValueCount(
    const FeatureStatsView& feature_stats_view, Feature* feature) {
  const std::vector<std::pair<int, int>> min_max_num_values =
      feature_stats_view.GetMinMaxNumValues();
  std::vector<Description> description;
  if (min_max_num_values.size() > 1) {
    description.push_back(
        {AnomalyInfo::VALUE_NESTEDNESS_MISMATCH, kValueNestednessMismatch,
         "This feature has a value_count, but the nestedness level of the "
         "feature > 1. For features with nestedness levels greater than 1, "
         "value_counts, not value_count, should be specified."});
    feature->clear_value_count();
    InitValueCount(feature_stats_view, feature);
    return description;
  }
  if (feature->value_count().has_min() &&
      min_max_num_values[0].first < feature->value_count().min()) {
    description.push_back({AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES,
                           kMissingValues,
                           "Some examples have fewer values than expected."});
    if (min_max_num_values[0].first == 0) {
      feature->mutable_value_count()->clear_min();
    } else {
      feature->mutable_value_count()->set_min(min_max_num_values[0].first);
    }
  }
  if (feature->value_count().has_max() &&
      min_max_num_values[0].second > feature->value_count().max()) {
    description.push_back({AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES,
                           kSuperfluousValues,
                           "Some examples have more values than expected."});
    feature->mutable_value_count()->set_max(min_max_num_values[0].second);
  }
  return description;
}

std::vector<Description> UpdateValueCounts(
    const FeatureStatsView& feature_stats_view, Feature* feature) {
  const std::vector<std::pair<int, int>> min_max_num_values =
      feature_stats_view.GetMinMaxNumValues();
  std::vector<Description> description;
  if (feature->value_counts().value_count_size() != min_max_num_values.size()) {
    description.push_back(
        {AnomalyInfo::VALUE_NESTEDNESS_MISMATCH, kValueNestednessMismatch,
         "The values have a different nest level than expected. Value counts "
         "will not be checked."});
    feature->clear_value_counts();
    InitValueCount(feature_stats_view, feature);
    return description;
  }
  for (int i = 0; i < feature->value_counts().value_count_size(); ++i) {
    if (feature->value_counts().value_count(i).has_min() &&
        min_max_num_values[i].first <
            feature->value_counts().value_count(i).min()) {
      description.push_back(
          {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES, kMissingValues,
           absl::StrCat("Some examples have fewer values than expected at "
                        "nestedness level ",
                        i, ".")});
      if (min_max_num_values[i].first == 0) {
        feature->mutable_value_counts()->mutable_value_count(i)->clear_min();
      } else {
        feature->mutable_value_counts()->mutable_value_count(i)->set_min(
            min_max_num_values[i].first);
      }
    }
    if (feature->mutable_value_counts()->mutable_value_count(i)->has_max() &&
        min_max_num_values[i].second >
            feature->value_counts().value_count(i).max()) {
      description.push_back(
          {AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES, kSuperfluousValues,
           absl::StrCat("Some examples have more values than expected at "
                        "nestedness level ",
                        i, ".")});
      feature->mutable_value_counts()->mutable_value_count(i)->set_max(
          min_max_num_values[i].second);
    }
  }
  return description;
}

}  // namespace

bool FeatureHasComparator(const Feature& feature,
                          FeatureComparatorType comparator_type) {
  switch (comparator_type) {
    case FeatureComparatorType::DRIFT:
      return feature.has_drift_comparator();
    case FeatureComparatorType::SKEW:
      return feature.has_skew_comparator();
  }
}

tensorflow::metadata::v0::FeatureComparator* GetFeatureComparator(
    Feature* feature, FeatureComparatorType comparator_type) {
  switch (comparator_type) {
    case FeatureComparatorType::DRIFT:
      return feature->mutable_drift_comparator();
    case FeatureComparatorType::SKEW:
      return feature->mutable_skew_comparator();
  }
}

bool LifecycleStageIsDeprecated(const metadata::v0::LifecycleStage stage) {
  switch (stage) {
    case tensorflow::metadata::v0::PLANNED:
    case tensorflow::metadata::v0::ALPHA:
    case tensorflow::metadata::v0::DEPRECATED:
    case tensorflow::metadata::v0::DEBUG_ONLY:
    case tensorflow::metadata::v0::DISABLED:
      return true;
    case tensorflow::metadata::v0::UNKNOWN_STAGE:
    case tensorflow::metadata::v0::BETA:
    case tensorflow::metadata::v0::PRODUCTION:
    case tensorflow::metadata::v0::VALIDATION_DERIVED:
    default:
      return false;
  }
}


namespace {

// Templated implementations for [Feature, SparseFeature].

template <class T>
void DeprecateFeatureType(T* feature) {
  DCHECK_NE(feature, nullptr);
  feature->set_lifecycle_stage(tensorflow::metadata::v0::DEPRECATED);
}

template <class T>
bool FeatureTypeIsDeprecated(const T& feature) {
  if (feature.deprecated()) {  // NOLINT
    return true;
  }
  if (feature.has_lifecycle_stage()) {
    return LifecycleStageIsDeprecated(feature.lifecycle_stage());
  }
  return false;
}

struct SingleFeatureComparisonResult {
  absl::optional<Description> description;
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo::Measurement>
      measurement;
};

// If the comparator contains an infinity norm threshold, checks whether the
// L-infinity distance between the stats and control stats is within that
// threshold. If not, updates the comparator and returns a description of the
// anomaly.
SingleFeatureComparisonResult UpdateInfinityNormComparator(
    const FeatureStatsView& stats, const FeatureStatsView& control_stats,
    const ComparatorContext& context,
    tensorflow::metadata::v0::FeatureComparator* comparator) {
  SingleFeatureComparisonResult result;
  if (!comparator->infinity_norm().has_threshold()) {
    return result;
  }
  const double linf_threshold = comparator->infinity_norm().threshold();
  const std::pair<std::string, double> linf_distance =
      LInftyDistance(stats, control_stats);
  const std::string max_difference_value = linf_distance.first;
  const double stats_infinity_norm = linf_distance.second;
  result.measurement.emplace();
  result.measurement->set_value(stats_infinity_norm);
  result.measurement->set_threshold(linf_threshold);
  result.measurement->set_type(
      metadata::v0::DriftSkewInfo_Measurement_Type_L_INFTY);
  if (stats_infinity_norm <= linf_threshold) {
    return result;
  }
  // TODO(b/68711199): Add support for Linf with numeric features, or log a
  // warning where the user has specified an infinity_norm threshold for a
  // numeric feature.
  comparator->mutable_infinity_norm()->set_threshold(stats_infinity_norm);
  result.description = Description(
      {tensorflow::metadata::v0::AnomalyInfo::COMPARATOR_L_INFTY_HIGH,
       absl::StrCat("High Linfty distance between ", context.treatment_name,
                    " and ", context.control_name),
       absl::StrCat("The Linfty distance between ", context.treatment_name,
                    " and ", context.control_name, " is ",
                    absl::SixDigits(stats_infinity_norm),
                    " (up to six significant digits), above the threshold ",
                    absl::SixDigits(linf_threshold),
                    ". The feature value with maximum difference is: ",
                    max_difference_value)});
  return result;
}

// If the comparator contains a Jensen-Shannon Divergence threshold, checks
// whether the approximate Jensen-Shannon Divergence between the stats and
// control stats is within that threshold. If not, updates the comparator and
// returns a description of the anomaly.
SingleFeatureComparisonResult UpdateJensenShannonDivergenceComparator(
    const FeatureStatsView& stats, const FeatureStatsView& control_stats,
    const ComparatorContext& context,
    tensorflow::metadata::v0::FeatureComparator* comparator) {
  SingleFeatureComparisonResult result;
  if (!comparator->jensen_shannon_divergence().has_threshold()) {
    return result;
  }
  const double jensen_shannon_threshold =
      comparator->jensen_shannon_divergence().threshold();
  double jensen_shannon_divergence;
  if (JensenShannonDivergence(stats, control_stats,
                                          jensen_shannon_divergence)
          .ok()) {
    result.measurement.emplace();
    result.measurement->set_value(jensen_shannon_divergence);
    result.measurement->set_threshold(jensen_shannon_threshold);
    result.measurement->set_type(
        metadata::v0::DriftSkewInfo_Measurement_Type_JENSEN_SHANNON_DIVERGENCE);
    if (jensen_shannon_divergence > jensen_shannon_threshold) {
      result.description = Description(
          {tensorflow::metadata::v0::AnomalyInfo::
               COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH,
           absl::StrCat("High approximate Jensen-Shannon divergence between ",
                        context.treatment_name, " and ", context.control_name),
           absl::StrCat("The approximate Jensen-Shannon divergence between ",
                        context.treatment_name, " and ", context.control_name,
                        " is ", absl::SixDigits(jensen_shannon_divergence),
                        " (up to six significant digits), above the threshold ",
                        absl::SixDigits(jensen_shannon_threshold), ".")});
      comparator->mutable_jensen_shannon_divergence()->set_threshold(
          jensen_shannon_divergence);
    }
  } else {
    // TODO(b/68711199): Add support for using JSD with categorical features.
    LOG(WARNING) << "A jensen_shannon_divergence threshold for feature "
                 << stats.GetPath().Serialize()
                 << ", but the stats for this feature do not include a "
                    "histogram from which the divergence can be analyzed. The "
                    "jensen_shannon_divergence can be specified for a "
                    "numeric feature only.";
  }
  return result;
}

}  // namespace

void DeprecateFeature(Feature* feature) {
  return DeprecateFeatureType(feature);
}

void DeprecateSparseFeature(SparseFeature* sparse_feature) {
  return DeprecateFeatureType(sparse_feature);
}

void DeprecateWeightedFeature(WeightedFeature* weighted_feature) {
  return DeprecateFeatureType(weighted_feature);
}

bool FeatureIsDeprecated(const Feature& feature) {
  return FeatureTypeIsDeprecated(feature);
}

bool SparseFeatureIsDeprecated(const SparseFeature& sparse_feature) {
  return FeatureTypeIsDeprecated(sparse_feature);
}

bool WeightedFeatureIsDeprecated(const WeightedFeature& weighted_feature) {
  return LifecycleStageIsDeprecated(weighted_feature.lifecycle_stage());
}

FeatureComparisonResult UpdateFeatureComparatorDirect(
    const FeatureStatsView& stats, const FeatureComparatorType comparator_type,
    tensorflow::metadata::v0::FeatureComparator* comparator) {
  FeatureComparisonResult result;
  if (!comparator->infinity_norm().has_threshold() &&
      !comparator->jensen_shannon_divergence().has_threshold()) {
    // There is nothing to check.
    return result;
  }
  const ComparatorContext& context = GetContext(comparator_type);
  const absl::optional<FeatureStatsView> control_stats =
      GetControlStats(stats, comparator_type);
  if (control_stats) {
    const SingleFeatureComparisonResult linfty_result =
        UpdateInfinityNormComparator(stats, control_stats.value(), context,
                                     comparator);
    if (linfty_result.description) {
      result.descriptions.push_back(*linfty_result.description);
    }
    if (linfty_result.measurement) {
      result.measurements.push_back(*linfty_result.measurement);
    }
    const SingleFeatureComparisonResult jensen_shannon_result =
        UpdateJensenShannonDivergenceComparator(stats, control_stats.value(),
                                                context, comparator);
    if (jensen_shannon_result.description) {
      result.descriptions.push_back(*jensen_shannon_result.description);
    }
    if (jensen_shannon_result.measurement) {
      result.measurements.push_back(*jensen_shannon_result.measurement);
    }
    return result;

  } else if (HasControlDataset(stats, comparator_type)) {
    // If there is a control dataset, but that dataset does not contain
    // statistics for the feature at issue, generate a missing control data
    // anomaly, and clear the comparator threshold(s).
    if (comparator->infinity_norm().has_threshold()) {
      comparator->mutable_infinity_norm()->clear_threshold();
    }
    if (comparator->jensen_shannon_divergence().has_threshold()) {
      comparator->mutable_jensen_shannon_divergence()->clear_threshold();
    }
    result.descriptions = {
        {AnomalyInfo::COMPARATOR_CONTROL_DATA_MISSING,
         absl::StrCat(context.control_name, " data missing"),
         absl::StrCat(context.control_name, " data is missing.")}};
    return result;
  }
  // If there is no control dataset at all, return without generating an
  // anomaly.
  return result;
}

double GetMaxOffDomain(const tensorflow::metadata::v0::DistributionConstraints&
                           distribution_constraints) {
  return distribution_constraints.has_min_domain_mass()
             ? (1.0 - distribution_constraints.min_domain_mass())
             : 0.0;
}

void ClearDomain(Feature* feature) {
  feature->clear_domain_info();
}

void MarkFeatureDerived(
    const tensorflow::metadata::v0::DerivedFeatureSource& derived_source,
    tensorflow::metadata::v0::Feature* feature) {
  DCHECK_NE(feature, nullptr);
  feature->set_lifecycle_stage(tensorflow::metadata::v0::VALIDATION_DERIVED);
  *feature->mutable_validation_derived_source() = derived_source;
}

void InitPresenceAndShape(const FeatureStatsView& feature_stats_view,
                          const bool infer_fixed_shape,
                          Feature* feature) {
  double num_present = feature_stats_view.GetNumPresent();
  if (num_present < 1.0) {
    // Note that we also set min_count to be zero when num_present is between
    // (0.0, 1.0)
    feature->mutable_presence()->set_min_count(0);
  } else {
    feature->mutable_presence()->set_min_count(1);
  }

  // There are no examples containing this feature counts, do not infer anything
  // else.
  if (num_present <= 0) {
    return;
  }

  if (feature_stats_view.GetNumMissing() == 0.0) {
    feature->mutable_presence()->set_min_fraction(1.0);
  }
  if (infer_fixed_shape) {
    InitFixedShape(feature_stats_view, feature);
  }
  // Infer value count if shape is not inferred.
  if (!feature->has_shape()) {
    InitValueCount(feature_stats_view, feature);
  }
}

std::vector<Description> UpdateFeatureValueCounts(
    const FeatureStatsView& feature_stats_view, Feature* feature) {
  CHECK_NE(feature, nullptr);
  if (!feature->has_value_count() && !feature->has_value_counts()) {
    return {};
  }
  if (feature->has_value_count()) {
    return UpdateValueCount(feature_stats_view, feature);
  }
  if (feature->has_value_counts()) {
    return UpdateValueCounts(feature_stats_view, feature);
  }
  return {};
}

std::vector<Description> UpdateFeatureShape(
    const FeatureStatsView& feature_stats_view,
    const bool generate_legacy_feature_spec, Feature* feature) {
  if (!feature->has_shape()) {
    return {};
  }
  std::vector<int> schema_shape;
  int expected_fixed_value_counts = 1;
  for (const auto& dim : feature->shape().dim()) {
    expected_fixed_value_counts *= dim.size();
  }
  int actual_fixed_value_counts = 1;
  for (auto min_and_max_count : feature_stats_view.GetMinMaxNumValues()) {
    const int min_count = min_and_max_count.first;
    const int max_count = min_and_max_count.second;
    if (min_count == max_count && min_count != 0) {
      actual_fixed_value_counts *= min_count;
    } else {
      // Set to a negative number to signal that the feature has a variable
      // number of values (thus no shape should be inferred).
      actual_fixed_value_counts = -1;
      break;
    }
  }

  bool has_missing = false;
  // If Schema.generate_legacy_feature_spec is true, feature absence is allowed.
  // See b/180761541.
  if (!generate_legacy_feature_spec) {
    for (const double num_missing : feature_stats_view.GetNumMissingNested()) {
      if (num_missing != 0) {
        has_missing = true;
        break;
      }
    }
  }

  if (actual_fixed_value_counts <= 0 || has_missing) {
    feature->clear_shape();
    // Note that we don't instead infer value count constraints for this feature
    // because the inferred value count may fail to be compatible with the stats
    // used to derive the original schema (which we don't have access to).
    return {{AnomalyInfo::INVALID_FEATURE_SHAPE,
             "Feature shape dropped",
             "The feature has a shape, but it's not always present (if the "
             "feature is nested, then it should always be present at each "
             "nested level) or its value lengths vary."}};
  } else if (actual_fixed_value_counts != expected_fixed_value_counts) {
    feature->clear_shape();
    return {
        {AnomalyInfo::INVALID_FEATURE_SHAPE,
         "Feature shape dropped",
         absl::StrCat("The feature has fixed value length ",
                      actual_fixed_value_counts,
                      " but it's not compatible with the specified shape.")}};
  }

  return {};
}

std::vector<Description> UpdatePresence(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::FeaturePresence* presence) {
  std::vector<Description> descriptions;
  const optional<double> num_present = feature_stats_view.GetNumPresent();
  if (presence->has_min_count() && num_present) {
    if (*num_present < presence->min_count()) {
      int64 original_min_count = presence->min_count();
      presence->set_min_count(*num_present);
      descriptions.push_back(
          {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_PRESENT, kDropped,
           absl::StrFormat("The feature was present in fewer examples than "
                           "expected: minimum count = %d, actual = %d",
                           original_min_count, presence->min_count())});
    }
  }
  const optional<double> fraction_present =
      feature_stats_view.GetFractionPresent();
  if (presence->has_min_fraction() && fraction_present) {
    if (*fraction_present < presence->min_fraction()) {
      float original_min_fraction = presence->min_fraction();
      presence->set_min_fraction(*fraction_present);
      descriptions.push_back(
          {AnomalyInfo::FEATURE_TYPE_LOW_FRACTION_PRESENT, kDropped,
           absl::StrFormat("The feature was present in fewer examples than "
                           "expected: minimum fraction = %f, actual = %f",
                           original_min_fraction, presence->min_fraction())});
    }
    if (presence->min_fraction() == 1.0) {
      if (feature_stats_view.GetNumMissing() != 0.0) {
        // In this case, there is a very small fraction of examples missing,
        // such that floating point error can hide it. We treat this case
        // separately, and set a threshold that is numerically distant from
        // 1.0.
        // TODO(b/148429185): update the anomaly type here to be unique.
        presence->set_min_fraction(0.9999);
        descriptions.push_back(
            {AnomalyInfo::FEATURE_TYPE_LOW_FRACTION_PRESENT, kDropped,
             absl::StrCat(
                 "The feature was expected everywhere, but was missing in ",
                 feature_stats_view.GetNumMissing(), " examples.")});
      }
    }
  }
  return descriptions;
}

std::vector<Description> UpdateUniqueConstraints(
    const FeatureStatsView& feature_stats_view,
    tensorflow::metadata::v0::Feature* feature) {
  std::vector<Description> descriptions;
  const absl::optional<int> num_unique = feature_stats_view.GetNumUnique();
  if (num_unique) {
    if (num_unique < feature->unique_constraints().min()) {
      descriptions.push_back(
          {AnomalyInfo::FEATURE_TYPE_LOW_UNIQUE, "Low number of unique values",
           absl::StrCat(
               "Expected at least ", feature->unique_constraints().min(),
               " unique values but found only ", num_unique.value(), ".")});
      feature->mutable_unique_constraints()->set_min(num_unique.value());
    }
    if (num_unique > feature->unique_constraints().max()) {
      descriptions.push_back(
          {AnomalyInfo::FEATURE_TYPE_HIGH_UNIQUE,
           "High number of unique values",
           absl::StrCat("Expected no more than ",
                        feature->unique_constraints().max(),
                        " unique values but found ", num_unique.value(), ".")});
      feature->mutable_unique_constraints()->set_max(num_unique.value());
    }
  } else {
    descriptions.push_back(
        {AnomalyInfo::FEATURE_TYPE_NO_UNIQUE, "No unique values",
         absl::StrCat(
             "UniqueConstraints specified for the feature, but unique values "
             "were not counted (i.e., feature is not string or "
             "categorical).")});
    feature->clear_unique_constraints();
  }
  return descriptions;
}

}  // namespace data_validation
}  // namespace tensorflow
