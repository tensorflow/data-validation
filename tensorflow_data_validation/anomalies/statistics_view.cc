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

#include "tensorflow_data_validation/anomalies/statistics_view.h"

#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {

namespace {
using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::FeatureNameStatistics;
}  // namespace

DatasetStatsView::DatasetStatsView(const DatasetFeatureStatistics& data,
                                   bool by_weight,
                                   const absl::optional<string>& environment,
                                   std::shared_ptr<DatasetStatsView> previous,
                                   std::shared_ptr<DatasetStatsView> serving)
    : data_(new DatasetFeatureStatistics(data)),
      by_weight_(by_weight),
      environment_(environment),
      previous_(std::move(previous)),
      serving_(std::move(serving)) {}

DatasetStatsView::DatasetStatsView(const DatasetFeatureStatistics& data,
                                   bool by_weight)
    : data_(new DatasetFeatureStatistics(data)),
      by_weight_(by_weight),
      environment_(),
      previous_(),
      serving_() {}

DatasetStatsView::DatasetStatsView(
    const tensorflow::metadata::v0::DatasetFeatureStatistics& data)
    : data_(new DatasetFeatureStatistics(data)),
      by_weight_(false),
      environment_(),
      previous_(),
      serving_() {}

std::vector<FeatureStatsView> DatasetStatsView::features() const {
  std::vector<FeatureStatsView> result;
  for (int i = 0; i < data_->features_size(); ++i) {
    result.push_back(FeatureStatsView(i, *this));
  }
  return result;
}

const tensorflow::metadata::v0::FeatureNameStatistics&
DatasetStatsView::feature_name_statistics(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, data_->features_size());
  return data_->features(index);
}

double DatasetStatsView::GetNumExamples() const {
  if (by_weight_) {
    return data_->weighted_num_examples();
  } else {
    return data_->num_examples();
  }
}

absl::optional<FeatureStatsView> DatasetStatsView::GetByName(
    const string& name) const {
  for (const FeatureStatsView& feature_stats_view : features()) {
    if (feature_stats_view.name() == name) {
      return feature_stats_view;
    }
  }
  return absl::nullopt;
}

absl::optional<FeatureStatsView> DatasetStatsView::GetParent(
    const string& name) const {
  std::unique_ptr<FeatureStatsView> best_so_far;
  for (const FeatureStatsView& feature_stats_view : features()) {
    if (!feature_stats_view.is_struct()) {
      continue;
    }
    const string& candidate_name = feature_stats_view.name();
    if (candidate_name.length() < name.length() &&
        name.substr(0, candidate_name.length()) == candidate_name) {
      // candidate_name is a strict substring.
      if (!best_so_far ||
          (best_so_far->name().length() < candidate_name.length())) {
        best_so_far = absl::make_unique<FeatureStatsView>(feature_stats_view);
      }
    }
  }
  if (best_so_far) {
    return *best_so_far;
  } else {
    return absl::nullopt;
  }
}

std::vector<FeatureStatsView> DatasetStatsView::GetRootFeatures() const {
  std::vector<FeatureStatsView> result;
  for (const FeatureStatsView& feature : features()) {
    if (!GetParent(feature.name())) {
      result.push_back(feature);
    }
  }
  return result;
}

// Returns true if the weighted statistics exist.
bool DatasetStatsView::WeightedStatisticsExist() const {
  if (data_->weighted_num_examples() == 0.0) {
    return false;
  }
  for (const FeatureStatsView& feature_stats_view : features()) {
    if (!feature_stats_view.WeightedStatisticsExist()) {
      return false;
    }
  }
  return true;
}

double FeatureStatsView::GetNumExamples() const {
  return parent_view_.GetNumExamples();
}

double FeatureStatsView::GetNumMissing() const {
  if (parent_view_.by_weight()) {
    return GetCommonStatistics().weighted_common_stats().num_missing();
  }
  return GetCommonStatistics().num_missing();
}

absl::optional<double> FeatureStatsView::GetFractionPresent() const {
  double num_examples = GetNumExamples();
  double num_present = GetNumPresent();
  if (GetNumMissing() == 0.0) {
    // Avoids numerical issues.
    return 1.0;
  }
  if (num_examples > 0.0) {
    return num_present / num_examples;
  }
  return absl::nullopt;
}

const tensorflow::metadata::v0::CommonStatistics&
FeatureStatsView::GetCommonStatistics() const {
  if (data().has_num_stats()) {
    return data().num_stats().common_stats();
  } else if (data().has_string_stats()) {
    return data().string_stats().common_stats();
  } else if (data().has_bytes_stats()) {
    return data().bytes_stats().common_stats();
  } else if (data().has_struct_stats()) {
    return data().struct_stats().common_statistics();
  }
  LOG(FATAL) << "Unknown statistics: " << data().DebugString();
}

// Get the number of examples in which this field is present.
double FeatureStatsView::GetNumPresent() const {
  if (parent_view_.by_weight()) {
    return GetCommonStatistics().weighted_common_stats().num_non_missing();
  }
  return GetCommonStatistics().num_non_missing();
}

std::map<string, double> FeatureStatsView::GetStringValuesWithCounts() const {
  std::map<string, double> result;
  const tensorflow::metadata::v0::RankHistogram& histogram =
      (parent_view_.by_weight())
          ? data().string_stats().weighted_string_stats().rank_histogram()
          : data().string_stats().rank_histogram();
  for (const tensorflow::metadata::v0::RankHistogram::Bucket& bucket :
       histogram.buckets()) {
    result.insert({bucket.label(), bucket.sample_count()});
  }
  return result;
}

std::vector<string> FeatureStatsView::GetStringValues() const {
  std::vector<string> result;
  std::map<string, double> counts = GetStringValuesWithCounts();
  for (const auto& pair : counts) {
    const string& string_value = pair.first;
    result.push_back(string_value);
  }
  return result;
}

bool FeatureStatsView::HasInvalidUTF8Strings() const {
  // Instead of writing non-UTF8 strings to the statistics summary, the
  // generator writes __BYTES_VALUE__.
  const string kInvalidString = "__BYTES_VALUE__";
  return ContainsKey(GetStringValuesWithCounts(), kInvalidString);
}

const tensorflow::metadata::v0::NumericStatistics& FeatureStatsView::num_stats()
    const {
  return data().num_stats();
}

// Returns the count of values appearing in the feature across all examples,
// based on <stats>.
double FeatureStatsView::GetTotalValueCountInExamples() const {
  const tensorflow::metadata::v0::CommonStatistics& common_stats =
      GetCommonStatistics();

  if (parent_view_.by_weight()) {
    const tensorflow::metadata::v0::WeightedCommonStatistics&
        weighted_common_stats = common_stats.weighted_common_stats();
    return weighted_common_stats.tot_num_values();
  } else {
    if (common_stats.tot_num_values() == 0) {
      // tot_num_values wasn't populated in the original statistics.
      // In case this is the case in the proto we are reading, return this
      // product instead.
      return common_stats.num_non_missing() * common_stats.avg_num_values();
    }
    return common_stats.tot_num_values();
  }
}

// Returns true if the weighted statistics exist.
bool FeatureStatsView::WeightedStatisticsExist() const {
  return GetCommonStatistics().has_weighted_common_stats();
}

absl::optional<FeatureStatsView> FeatureStatsView::GetServing() const {
  absl::optional<DatasetStatsView> dataset_stats_view =
      parent_view_.GetServing();
  if (dataset_stats_view) {
    return dataset_stats_view->GetByName(name());
  }
  return absl::nullopt;
}

absl::optional<FeatureStatsView> FeatureStatsView::GetPrevious() const {
  absl::optional<DatasetStatsView> dataset_stats_view =
      parent_view_.GetPrevious();
  if (dataset_stats_view) {
    return dataset_stats_view->GetByName(name());
  }
  return absl::nullopt;
}

tensorflow::metadata::v0::FeatureType FeatureStatsView::GetFeatureType() const {
  switch (type()) {
    case FeatureNameStatistics::BYTES:
    case FeatureNameStatistics::STRING:
      return tensorflow::metadata::v0::BYTES;
    case FeatureNameStatistics::INT:
      return tensorflow::metadata::v0::INT;
    case FeatureNameStatistics::FLOAT:
      return tensorflow::metadata::v0::FLOAT;
    default:
      return tensorflow::metadata::v0::TYPE_UNKNOWN;
  }
}

absl::optional<FeatureStatsView> FeatureStatsView::GetParent() const {
  return parent_view_.GetParent(name());
}

std::vector<FeatureStatsView> FeatureStatsView::GetChildren() const {
  std::vector<FeatureStatsView> result;
  for (const FeatureStatsView& feature : parent_view_.features()) {
    absl::optional<FeatureStatsView> parent = feature.GetParent();
    if (parent && parent->name() == name()) {
      result.push_back(feature);
    }
  }
  return result;
}

}  // namespace data_validation
}  // namespace tensorflow
