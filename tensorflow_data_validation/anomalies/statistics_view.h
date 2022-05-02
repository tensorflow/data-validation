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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/derived_feature.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

class FeatureStatsView;

class DatasetStatsViewImpl;

// Wrapper for statistics.
// Designed to be passed by const reference.
class DatasetStatsView {
 public:
  DatasetStatsView(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& data,
      bool by_weight);

  DatasetStatsView(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& data,
      bool by_weight, const absl::optional<string>& environment,
      std::shared_ptr<DatasetStatsView> previous_span,
      std::shared_ptr<DatasetStatsView> serving,
      std::shared_ptr<DatasetStatsView> previous_version);

  // Default value of by_weight is false, there is no environment,
  // previous_span, previous_version, or serving.
  explicit DatasetStatsView(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& data);

  // Perform shallow copies of object, sharing the same
  // DatasetStatsViewImpl through a shared_ptr.
  DatasetStatsView(const DatasetStatsView& other) = default;
  DatasetStatsView& operator=(const DatasetStatsView& other) = default;

  // Constructs a FeatureStatsView vector on the fly.
  std::vector<FeatureStatsView> features() const;

  // Only includes FeatureStatsViews without parents.
  std::vector<FeatureStatsView> GetRootFeatures() const;

  // If returns zero, it could just be the default value.
  double GetNumExamples() const;

  bool by_weight() const;

  // If the path does not exist, returns absl::nullopt.
  absl::optional<FeatureStatsView> GetByPath(const Path& path) const;

  // Only call from FeatureStatsView::data().
  // check-fails if index is out of range. However, should never fail if
  // called from FeatureStatsView::data().
  const tensorflow::metadata::v0::FeatureNameStatistics&
  feature_name_statistics(int index) const;

  // Returns true if the weighted statistics exist.
  // Weighted stats must have feature parity with unweighted stats.
  // Note: this is independent of by_weight_.
  bool WeightedStatisticsExist() const;

  // Gets the parent feature of a feature, if one exists. Otherwise, returns
  // absl::nullopt.
  // FeatureStatsView a is an ancestor of FeatureStatsView b if:
  // a.is_struct() (otherwise, it can't be a parent), and
  // a.name() is a strict prefix of b.name().
  // FeatureStatsView a is the parent of FeatureStatsView b if it is the
  // ancestor with the longest name.
  // TODO(b/112209670): fix the issues with structural data and paths.
  absl::optional<FeatureStatsView> GetParent(
      const FeatureStatsView& view) const;

  const Path& GetPath(const FeatureStatsView& view) const;

  // Gets the children of a FeatureStatsView.
  std::vector<FeatureStatsView> GetChildren(const FeatureStatsView& view) const;

  const absl::optional<string>& environment() const;

  const absl::optional<DatasetStatsView> GetPreviousSpan() const;

  const absl::optional<DatasetStatsView> GetServing() const;

  const absl::optional<DatasetStatsView> GetPreviousVersion() const;

 private:
  // Guaranteed not to be null. This being a shared_ptr makes this object
  // lightweight.
  std::shared_ptr<const DatasetStatsViewImpl> impl_;
};

// Provides a view into the DatasetFeatureStatistics from a particular column.
// Note that this class is effectively a pair of pointers, and should be
// treated as such.
// This class handles whether results are weighted and unweighted.
class FeatureStatsView {
 public:
  FeatureStatsView(const FeatureStatsView& other) = default;

  const Path& GetPath() const;

  const absl::optional<string>& environment() const {
    return parent_view_.environment();
  }

  tensorflow::metadata::v0::FeatureNameStatistics::Type type() const {
    return data().type();
  }

  bool HasValidationDerivedSource() const {
    return data().has_validation_derived_source();
  }

  const tensorflow::metadata::v0::DerivedFeatureSource&
  GetValidationDerivedSource() const {
    return data().validation_derived_source();
  }

  // Gets the FeatureType representing the physical type represented in
  // a tf.Example. This glosses over the distinction between BYTES and STRING
  // in FeatureNameStatistics::Type.
  tensorflow::metadata::v0::FeatureType GetFeatureType() const;

  // Gets the num_non_missing, or the (weighted) number of examples where a
  // feature is present.
  double GetNumPresent() const;

  // Returns the minimum and maximum number of values for the feature at each
  // nestedness level.
  std::vector<std::pair<int, int>> GetMinMaxNumValues() const;

  // Returns whether the feature contains values with a nestedness level that is
  // greater than 1.
  bool HasNestedValues() const {
    return GetCommonStatistics().presence_and_valency_stats_size() > 1;
  }

  // Get the total (weighted) number of examples, regardless of whether this
  // feature was present or absent (calls the parent_view_).
  double GetNumExamples() const;

  // Returns the strings that occur in the data, along with the (weighted)
  // counts. If there are no string stats, then it returns an empty map.
  std::map<string, double> GetStringValuesWithCounts() const;

  // Returns the (weighted) standard histogram, if it exists for the feature.
  absl::optional<tensorflow::metadata::v0::Histogram> GetStandardHistogram()
      const;

  // Returns the strings that occur in the data.
  // If there are no string stats, then it returns an empty map.
  std::vector<string> GetStringValues() const;

  // Returns true if the column is a string column and there are some invalid
  // UTF8 strings present.
  bool HasInvalidUTF8Strings() const;

  // Returns the numeric stats, or an empty object if no numeric stats exist.
  const tensorflow::metadata::v0::NumericStatistics& num_stats() const;

  // Returns the bytes stats, or an empty object if no bytes stats exist.
  const tensorflow::metadata::v0::BytesStatistics& bytes_stats() const;

  double GetNumMissing() const;
  // Returns num missing at each nestedness level. The first element of the
  // returned vector equals to GetNumMissing().
  // Note that the values could be weighted.
  std::vector<double> GetNumMissingNested() const;

  absl::optional<double> GetFractionPresent() const;

  // The total number of values of this feature that occurred.
  double GetTotalValueCountInExamples() const;

  // Returns true if weighted statistics exist for this column.
  // Weighted stats must have feature parity with unweighted stats.
  bool WeightedStatisticsExist() const;

  absl::optional<FeatureStatsView> GetServing() const;

  absl::optional<FeatureStatsView> GetPreviousSpan() const;

  const DatasetStatsView& parent_view() const { return parent_view_; }

  // Returns the list of custom_stats of the underlying FeatureNameStatistics.
  std::vector<tensorflow::metadata::v0::CustomStatistic> custom_stats() const {
    return {data().custom_stats().begin(), data().custom_stats().end()};
  }

  // Returns the custom stat for the underlying FeatureNameStatistics, of a
  // given custom_stat_name. Returns `nullptr` if the custom stat is not
  // found.
  const tensorflow::metadata::v0::CustomStatistic* GetCustomStatByName(
      const std::string& custom_stat_name) const;

  std::vector<FeatureStatsView> GetChildren() const;

  absl::optional<FeatureStatsView> GetParent() const;

  bool is_struct() const {
    return type() == tensorflow::metadata::v0::FeatureNameStatistics::STRUCT;
  }

  // Returns the number of unique values if a string or categorical feature.
  // Otherwise, returns nullopt.
  const absl::optional<uint64> GetNumUnique() const;

  // Object is assumed to be created from DatasetStatsView::features().
  FeatureStatsView(int index, const DatasetStatsView& parent_view)
      : parent_view_(parent_view), index_(index) {}

 private:
  // Made a friend to access private constructor.
  friend std::vector<FeatureStatsView> DatasetStatsView::features() const;

  friend DatasetStatsViewImpl;

  // Get a reference to the statistics from the parent.
  // This should never check-fail.
  const tensorflow::metadata::v0::FeatureNameStatistics& data() const {
    return parent_view_.feature_name_statistics(index_);
  }

  const tensorflow::metadata::v0::CommonStatistics& GetCommonStatistics() const;
  // Note that this field guarantees data_ is not a pointer to nowhere.
  const DatasetStatsView parent_view_;
  // An index into DatasetStatsView.
  const int index_;
};

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATISTICS_VIEW_H_
