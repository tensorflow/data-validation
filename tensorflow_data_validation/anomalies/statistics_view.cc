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
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace data_validation {

namespace {
using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::FeatureNameStatistics;

// Returns true if a is a strict prefix of b.
const bool IsStrictPrefix(const string& a, const string& b) {
  return a.length() < b.length() && b.substr(0, a.length()) == a;
}

}  // namespace

// Context of a feature.
struct FeatureContext {
  // Index of the parent feature.
  absl::optional<int> parent_index;
  // Index of children of the feature.
  std::vector<int> child_indices;
  Path path;
};

// A class that summarizes the information from the DatasetFeatureStatistics.
// Takes O(#features log #features) time to initialize,
// O(# features) space, and:
// GetRootFeatures() takes O(# features) time
// GetChildren() takes O(# children) time
// GetParent() takes O(1) time
// GetByPath() takes O(log # features) time.
class DatasetStatsViewImpl {
 public:
  DatasetStatsViewImpl(const DatasetFeatureStatistics& data, bool by_weight,
                       const absl::optional<string>& environment,
                       const std::shared_ptr<DatasetStatsView>& previous,
                       const std::shared_ptr<DatasetStatsView>& serving)
      : data_(data),
        by_weight_(by_weight),
        environment_(environment),
        previous_(previous),
        serving_(serving) {
    // It takes O(n log n) time to construct location, a BST from the name
    // of a feature to its location in data_.features().
    // Map from name to location.
    // data_.features(location[foo]).name() == foo
    std::map<string, int> location;

    for (int i = 0; i < data_.features_size(); ++i) {
      location[data_.features(i).name()] = i;
      context_.push_back(FeatureContext());
    }

    // After we construct the map, we iterate over the names of features
    // alphabetically. Note that:
    // If feature a is right after feature b alphabetically, the ancestors
    // of feature b are a subset of the ancestors of feature a and possibly
    // feature a itself.
    // Since current_ancestors stores the ancestors by increasing name length,
    // then the last of the current ancestors is the parent of the next field.
    // Moreover, since every ancestor is added from current_ancestors once,
    // and removed from the list once, the runtime of this whole operation
    // is O(# features)
    std::vector<int> current_ancestors;

    for (const auto& pair : location) {
      const string& name = pair.first;
      int index = pair.second;
      while (!current_ancestors.empty() &&
             !IsStrictPrefix(data_.features()[current_ancestors.back()].name(),
                             name)) {
        current_ancestors.pop_back();
      }
      if (!current_ancestors.empty()) {
        int parent_index = current_ancestors.back();
        const string& parent_name = data_.features(parent_index).name();
        const string& name = data_.features(index).name();
        context_[index].parent_index = parent_index;
        context_[index].path = context_[parent_index].path.GetChild(
            name.substr(parent_name.size() + 1));
        context_[parent_index].child_indices.push_back(index);
      } else {
        context_[index].path = Path({data_.features(index).name()});
      }
      path_location_[context_[index].path] = index;
      if (data_.features(index).type() ==
          tensorflow::metadata::v0::FeatureNameStatistics::STRUCT) {
        current_ancestors.push_back(index);
      }
    }
  }

  const DatasetFeatureStatistics& data() const { return data_; }

  absl::optional<FeatureStatsView> GetByPath(const DatasetStatsView& view,
                                             const Path& path) const {
    auto ref = path_location_.find(path);
    if (ref == path_location_.end()) {
      VLOG(0) << "DatasetStatsViewImpl::GetByPath() can't find: "
              << path.Serialize();
      for (const FeatureStatsView& feature_view : view.features()) {
        VLOG(0) << "  DatasetStatsViewImpl::GetByPath(): path: "
                << feature_view.GetPath().Serialize();
      }
      return absl::nullopt;
    } else {
      return FeatureStatsView(ref->second, view);
    }
  }

  const Path& GetPath(const FeatureStatsView& view) const {
    return context_[view.index_].path;
  }

  absl::optional<FeatureStatsView> GetParent(
      const FeatureStatsView& view) const {
    absl::optional<int> opt_parent_index = context_[view.index_].parent_index;
    if (opt_parent_index) {
      return FeatureStatsView(*opt_parent_index, view.parent_view_);
    } else {
      return absl::nullopt;
    }
  }

  std::vector<FeatureStatsView> GetChildren(
      const FeatureStatsView& view) const {
    std::vector<FeatureStatsView> result;
    for (int i : context_[view.index_].child_indices) {
      result.emplace_back(i, view.parent_view_);
    }
    return result;
  }

 private:
  friend DatasetStatsView;
  // Underlying data.
  const DatasetFeatureStatistics data_;

  // Whether DatasetFeatureStatistics is accessed by weight or not.
  const bool by_weight_;

  // Environment.
  const absl::optional<string> environment_;

  // The previous dataset stats (if available).
  // Note that DatasetStatsView objects are very lightweight, so this
  // cost is minimal.
  const std::shared_ptr<DatasetStatsView> previous_;

  // The serving dataset stats (if available).
  const std::shared_ptr<DatasetStatsView> serving_;

  /*********** Cached information below, derivable from data_ *****************/

  // Context of each feature: parents and children.
  // parallel to features() array in data.
  std::vector<FeatureContext> context_;

  // Map from path to the index of the FeatureStatistics containing the
  // statistics for that path.
  std::map<Path, int> path_location_;
};

DatasetStatsView::DatasetStatsView(const DatasetFeatureStatistics& data,
                                   bool by_weight,
                                   const absl::optional<string>& environment,
                                   std::shared_ptr<DatasetStatsView> previous,
                                   std::shared_ptr<DatasetStatsView> serving)
    : impl_(new DatasetStatsViewImpl(data, by_weight, environment, previous,
                                     serving)) {}

DatasetStatsView::DatasetStatsView(const DatasetFeatureStatistics& data,
                                   bool by_weight)
    : impl_(new DatasetStatsViewImpl(data, by_weight, absl::nullopt,
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>())) {}

DatasetStatsView::DatasetStatsView(
    const tensorflow::metadata::v0::DatasetFeatureStatistics& data)
    : impl_(new DatasetStatsViewImpl(data, false, absl::nullopt,
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>())) {}

std::vector<FeatureStatsView> DatasetStatsView::features() const {
  std::vector<FeatureStatsView> result;
  for (int i = 0; i < impl_->data().features_size(); ++i) {
    result.push_back(FeatureStatsView(i, *this));
  }
  return result;
}

const tensorflow::metadata::v0::FeatureNameStatistics&
DatasetStatsView::feature_name_statistics(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, impl_->data().features_size());
  return impl_->data().features(index);
}

double DatasetStatsView::GetNumExamples() const {
  if (impl_->by_weight_) {
    return impl_->data().weighted_num_examples();
  } else {
    return impl_->data().num_examples();
  }
}

absl::optional<FeatureStatsView> DatasetStatsView::GetByPath(
    const Path& path) const {
  return impl_->GetByPath(*this, path);
}

absl::optional<FeatureStatsView> DatasetStatsView::GetParent(
    const FeatureStatsView& view) const {
  return impl_->GetParent(view);
}

const Path& DatasetStatsView::GetPath(const FeatureStatsView& view) const {
  return impl_->GetPath(view);
}

std::vector<FeatureStatsView> DatasetStatsView::GetChildren(
    const FeatureStatsView& view) const {
  return impl_->GetChildren(view);
}

std::vector<FeatureStatsView> DatasetStatsView::GetRootFeatures() const {
  std::vector<FeatureStatsView> result;
  for (const FeatureStatsView& feature : features()) {
    if (!feature.GetParent()) {
      result.push_back(feature);
    }
  }
  return result;
}

// Returns true if the weighted statistics exist.
bool DatasetStatsView::WeightedStatisticsExist() const {
  if (impl_->data().weighted_num_examples() == 0.0) {
    return false;
  }
  for (const FeatureStatsView& feature_stats_view : features()) {
    if (!feature_stats_view.WeightedStatisticsExist()) {
      return false;
    }
  }
  return true;
}

bool DatasetStatsView::by_weight() const { return impl_->by_weight_; }

const absl::optional<string>& DatasetStatsView::environment() const {
  return impl_->environment_;
}

const absl::optional<DatasetStatsView> DatasetStatsView::GetPrevious() const {
  if (impl_->previous_) {
    return *impl_->previous_;
  }
  return absl::nullopt;
}

const absl::optional<DatasetStatsView> DatasetStatsView::GetServing() const {
  if (impl_->serving_) {
    return *impl_->serving_;
  }
  return absl::nullopt;
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
    return data().struct_stats().common_stats();
  }
  LOG(FATAL) << "Unknown statistics (or missing stats): "
             << data().DebugString();
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
  // generator writes __BYTES_VALUE__. See
  // tensorflow_data_validation/statistics/generators/top_k_stats_generator.py
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
    return dataset_stats_view->GetByPath(GetPath());
  }
  return absl::nullopt;
}

absl::optional<FeatureStatsView> FeatureStatsView::GetPrevious() const {
  absl::optional<DatasetStatsView> dataset_stats_view =
      parent_view_.GetPrevious();
  if (dataset_stats_view) {
    return dataset_stats_view->GetByPath(GetPath());
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
    case FeatureNameStatistics::STRUCT:
      return tensorflow::metadata::v0::STRUCT;
    default:
      return tensorflow::metadata::v0::TYPE_UNKNOWN;
  }
}

absl::optional<FeatureStatsView> FeatureStatsView::GetParent() const {
  return parent_view_.GetParent(*this);
}

std::vector<FeatureStatsView> FeatureStatsView::GetChildren() const {
  return parent_view_.GetChildren(*this);
}

const Path& FeatureStatsView::GetPath() const {
  return parent_view_.GetPath(*this);
}

}  // namespace data_validation
}  // namespace tensorflow
