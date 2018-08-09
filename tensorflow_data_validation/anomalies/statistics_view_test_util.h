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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_TEST_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_TEST_UTIL_H_

#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace testing {

// Makes a dataset with one feature. Assumes global counts match the
// count for the feature.
tensorflow::metadata::v0::DatasetFeatureStatistics
GetDatasetFeatureStatisticsForTesting(
    const tensorflow::metadata::v0::FeatureNameStatistics& feature_name_stats);

// For testing, we often just have information for one feature.
// However, DatasetStatsView and FeatureStatsView point to other objects.
// This structure allows us to set all that up in one call.
// Here is a pattern:
// FuncToTest(DatasetForTesting(stats).feature_stats_view())
// Here is an anti-pattern. It will make the resulting object point to a
// destroyed object (very bad).
// const FeatureStatsView& MyShortcut(
//     const tensorflow::metadata::v0::FeatureNameStatistics& stats) {
//   return DatasetForTesting(stats).feature_stats_view();
// }
class DatasetForTesting {
 public:
  explicit DatasetForTesting(
      const tensorflow::metadata::v0::FeatureNameStatistics&
          feature_name_stats);
  DatasetForTesting(
      const tensorflow::metadata::v0::FeatureNameStatistics& feature_name_stats,
      bool by_weight);

  DatasetForTesting(const tensorflow::metadata::v0::DatasetFeatureStatistics&
                        dataset_feature_stats,
                    bool by_weight);

  // DatasetForTesting is neither copyable nor movable, as DatasetStatsView
  // is neither copyable nor movable.
  DatasetForTesting(const DatasetForTesting&) = delete;
  DatasetForTesting& operator=(const DatasetForTesting&) = delete;

  const DatasetStatsView& dataset_stats_view() const {
    return dataset_stats_view_;
  }

  const FeatureStatsView& feature_stats_view() const {
    return feature_stats_view_;
  }

 private:
  // Notice that the destructor will destroy the objects from bottom to top,
  // respecting the proper order of destruction.
  const tensorflow::metadata::v0::DatasetFeatureStatistics
      dataset_feature_statistics_;
  const DatasetStatsView dataset_stats_view_;
  const FeatureStatsView feature_stats_view_;
};

DatasetForTesting GetDatasetForTesting(
    const tensorflow::metadata::v0::FeatureNameStatistics& feature_name_stats);

tensorflow::metadata::v0::FeatureNameStatistics AddWeightedStats(
    const tensorflow::metadata::v0::FeatureNameStatistics& original);

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_TEST_UTIL_H_
