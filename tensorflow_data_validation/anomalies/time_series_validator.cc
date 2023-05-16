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

#include "tensorflow_data_validation/anomalies/time_series_validator.h"

#include <string>

#include "absl/status/statusor.h"
#include "tensorflow_data_validation/anomalies/metrics.h"
#include "tensorflow_data_validation/google/protos/time_series_metrics.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

using tensorflow::metadata::v0::DatasetFeatureStatisticsList;
using tensorflow::metadata::v0::FeatureNameStatistics;
using tensorflow::metadata::v0::CommonStatistics;

constexpr char kDefaultSlice[] = "All Examples";

namespace tensorflow {
namespace data_validation {

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<std::string, FeatureNameStatistics>>
BuildNamedStatisticsMap(
    const gtl::optional<DatasetFeatureStatisticsList>& statistics) {
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, FeatureNameStatistics>>
      named_statistics;
  if (!statistics) {
    return named_statistics;
  }
  for (const auto& dataset : statistics->datasets()) {
    std::string dataset_name;
    if (dataset.name().empty()) {
      dataset_name = kDefaultSlice;
    } else {
      dataset_name = dataset.name();
    }
    for (const auto& feature : dataset.features()) {
      const metadata::v0::Path& feature_path = feature.path();
      const std::string serialized_feature_path =
          Path(feature_path).Serialize();
      named_statistics[dataset_name][serialized_feature_path] = feature;
    }
  }
  return named_statistics;
}

const CommonStatistics& GetFeatureCommonStats(
    const FeatureNameStatistics& feature){
  if (feature.has_num_stats()) {
    return feature.num_stats().common_stats();
  } else if (feature.has_string_stats()) {
    return feature.string_stats().common_stats();
  } else if (feature.has_bytes_stats()) {
    return feature.bytes_stats().common_stats();
  } else {
    return feature.struct_stats().common_stats();
  }
}

absl::StatusOr<std::vector<ValidationMetrics>> ValidateTimeSeriesStatistics(
    const metadata::v0::DatasetFeatureStatisticsList& statistics,
    const gtl::optional<metadata::v0::DatasetFeatureStatisticsList>&
        reference_statistics,
    const SliceComparisonConfig& slice_config) {
  std::vector<ValidationMetrics> validation_metrics_vector;

  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, FeatureNameStatistics>>
      named_reference_statistics =
          BuildNamedStatisticsMap(reference_statistics);

  for (const auto& dataset : statistics.datasets()) {
    ValidationMetrics validation_metrics;

    std::string dataset_name;
    if (dataset.name().empty()) {
      dataset_name = kDefaultSlice;
    } else {
      dataset_name = dataset.name();
    }

    bool compute_drift_metrics = false;

    // TODO: Support calculate metrics between different slices
    // Find the statistics from the same slice in reference statistics
    auto reference_slice_statistics =
        named_reference_statistics.find(dataset_name);
    if (reference_slice_statistics != named_reference_statistics.end() &&
        slice_config.mode_ == SliceComparisonMode::kSame) {
      compute_drift_metrics = true;
    }

    // Add slice information for source and reference
    validation_metrics.mutable_source()->mutable_slice()->set_slice_name(
        dataset_name);

    if (compute_drift_metrics) {
      validation_metrics.mutable_reference_source()
          ->mutable_slice()
          ->set_slice_name(dataset_name);
    }

    for (const auto& feature : dataset.features()) {
      FeatureMetric* feature_metric = validation_metrics.add_feature_metric();

      // Add feature_name
      const std::string serialized_feature_path =
          Path(feature.path()).Serialize();
      feature_metric->mutable_feature_name()->set_name(serialized_feature_path);

      // Add num_examples metric
      Metric* num_examples_metric = feature_metric->add_metric();
      num_examples_metric->set_metric_name("num_examples");
      num_examples_metric->set_value(dataset.num_examples());

      // Get Common Stats
      tensorflow::metadata::v0::CommonStatistics common_stats =
          GetFeatureCommonStats(feature);

      // Add num_not_missing metric
      Metric* num_not_missing_metric = feature_metric->add_metric();
      num_not_missing_metric->set_metric_name("num_not_missing");
      num_not_missing_metric->set_value(common_stats.num_non_missing());

      // Check if compute drift metrics, if not move to next feature.
      if (compute_drift_metrics == false) {
        continue;
      }

      // Find feature statistics for the same slice in reference source.
      auto reference_feature_statistics_obj =
          reference_slice_statistics->second.find(serialized_feature_path);

      // Check missing feature in reference source.
      if (reference_feature_statistics_obj ==
          reference_slice_statistics->second.end())
        continue;
      FeatureNameStatistics reference_feature_statistics =
          reference_feature_statistics_obj->second;

      // Get Common Stats from reference statistics
      tensorflow::metadata::v0::CommonStatistics reference_common_stats =
          GetFeatureCommonStats(reference_feature_statistics);

      // Add drift metrics.
      // TODO: add jensen-shannon-divergence metric for the actual values (
      // under num stats).
      // TODO: make histogram choice configurable to have the option to use
      // the quantiles histogram instead.
      Metric* jensen_shannon_divergence_metric = feature_metric->add_metric();
      jensen_shannon_divergence_metric->set_metric_name(
          "num_values_jensen_shannon_divergence");
      double result;

      tensorflow::metadata::v0::Histogram& num_values_histogram =
          *common_stats.mutable_num_values_histogram();
      tensorflow::metadata::v0::Histogram& reference_num_values_histogram =
          *reference_common_stats.mutable_num_values_histogram();

      tensorflow::Status status =
          tensorflow::data_validation::JensenShannonDivergence(
              num_values_histogram, reference_num_values_histogram, result);
      if (status.ok()) {
        jensen_shannon_divergence_metric->set_value(result);
      }
    }

    validation_metrics_vector.push_back(validation_metrics);
  }

  return validation_metrics_vector;
}

}  // namespace data_validation
}  // namespace tensorflow
