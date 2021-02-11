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

#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {
namespace testing {
namespace {
using tensorflow::metadata::v0::CommonStatistics;
using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::FeatureNameStatistics;
using tensorflow::metadata::v0::WeightedCommonStatistics;

CommonStatistics* GetCommonStatisticsPtr(FeatureNameStatistics* feature_stats) {
  if (feature_stats->has_num_stats()) {
    return feature_stats->mutable_num_stats()->mutable_common_stats();
  } else if (feature_stats->has_string_stats()) {
    return feature_stats->mutable_string_stats()->mutable_common_stats();
  } else if (feature_stats->has_bytes_stats()) {
    return feature_stats->mutable_bytes_stats()->mutable_common_stats();
  } else if (feature_stats->has_struct_stats()) {
    return feature_stats->mutable_struct_stats()->mutable_common_stats();
  }
  LOG(FATAL) << "Unknown statistics: " << feature_stats->DebugString();
}

FeatureStatsView GetByNameOrDie(const DatasetStatsView& dataset,
                                const string& name) {
  absl::optional<FeatureStatsView> result = dataset.GetByPath(Path({name}));
  CHECK(absl::nullopt != result) << "Unknown name: " << name;
  return *result;
}

FeatureStatsView GetFirstOrDie(const DatasetStatsView& dataset) {
  CHECK(!dataset.features().empty()) << "Must have a feature name statistics";
  return dataset.features()[0];
}

}  // namespace

FeatureNameStatistics AddWeightedStats(const FeatureNameStatistics& original) {
  FeatureNameStatistics result = original;
  CommonStatistics& common_stats = *GetCommonStatisticsPtr(&result);
  WeightedCommonStatistics& weighted_common_stats =
      *common_stats.mutable_weighted_common_stats();
  weighted_common_stats.set_num_non_missing(common_stats.num_non_missing());
  weighted_common_stats.set_num_missing(common_stats.num_missing());
  weighted_common_stats.set_avg_num_values(common_stats.avg_num_values());
  weighted_common_stats.set_tot_num_values(common_stats.tot_num_values());
  if (result.has_string_stats()) {
    *result.mutable_string_stats()
         ->mutable_weighted_string_stats()
         ->mutable_rank_histogram() = result.string_stats().rank_histogram();
  }
  for (const auto& p_and_v_stats : common_stats.presence_and_valency_stats()) {
    auto* weighted_p_and_v_stats =
        common_stats.add_weighted_presence_and_valency_stats();
    weighted_p_and_v_stats->set_num_missing(p_and_v_stats.num_missing());
    weighted_p_and_v_stats->set_num_non_missing(
        p_and_v_stats.num_non_missing());
    weighted_p_and_v_stats->set_tot_num_values(p_and_v_stats.tot_num_values());
  }
  return result;
}

DatasetFeatureStatistics GetDatasetFeatureStatisticsForTesting(
    const FeatureNameStatistics& feature_name_stats) {
  DatasetFeatureStatistics result;
  FeatureNameStatistics& new_stats = *result.add_features();
  new_stats = feature_name_stats;
  const CommonStatistics& common_stats = *GetCommonStatisticsPtr(&new_stats);
  result.set_num_examples(common_stats.num_missing() +
                          common_stats.num_non_missing());
  const WeightedCommonStatistics& weighted_common_stats =
      common_stats.weighted_common_stats();
  result.set_weighted_num_examples(weighted_common_stats.num_non_missing() +
                                   weighted_common_stats.num_missing());
  return result;
}

DatasetForTesting::DatasetForTesting(
    const FeatureNameStatistics& feature_name_stats)
    : dataset_feature_statistics_(
          GetDatasetFeatureStatisticsForTesting(feature_name_stats)),
      dataset_stats_view_(dataset_feature_statistics_),
      feature_stats_view_(
          GetByNameOrDie(dataset_stats_view_, feature_name_stats.name())) {}

DatasetForTesting::DatasetForTesting(
    const FeatureNameStatistics& feature_name_stats, bool by_weight)
    : dataset_feature_statistics_(
          GetDatasetFeatureStatisticsForTesting(feature_name_stats)),
      dataset_stats_view_(dataset_feature_statistics_, by_weight),
      feature_stats_view_(
          GetByNameOrDie(dataset_stats_view_, feature_name_stats.name())) {}

DatasetForTesting::DatasetForTesting(
    const DatasetFeatureStatistics& dataset_feature_stats, bool by_weight)
    : dataset_feature_statistics_(dataset_feature_stats),
      dataset_stats_view_(dataset_feature_statistics_, by_weight),
      feature_stats_view_(GetFirstOrDie(dataset_stats_view_)) {}

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow
