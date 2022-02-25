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

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::Histogram;
using ::tensorflow::protobuf::RepeatedPtrField;

// Returns true if a is a strict prefix of b.
const bool IsStrictPrefix(const string& a, const string& b) {
  return a.length() < b.length() && b.substr(0, a.length()) == a;
}

// Returns true if the feature has empty stats.
const bool HasEmptyStats(const FeatureNameStatistics& feature) {
  if (feature.stats_case() == FeatureNameStatistics::STATS_NOT_SET &&
      feature.custom_stats_size() == 0) {
    return true;
  }
  return false;
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
  DatasetStatsViewImpl(
      const DatasetFeatureStatistics& data, bool by_weight,
      const absl::optional<string>& environment,
      const std::shared_ptr<DatasetStatsView>& previous_span,
      const std::shared_ptr<DatasetStatsView>& serving,
      const std::shared_ptr<DatasetStatsView>& previous_version)
      : data_(data),
        by_weight_(by_weight),
        environment_(environment),
        previous_span_(previous_span),
        serving_(serving),
        previous_version_(previous_version) {
    const auto& features = data_.features();
    if (std::all_of(features.begin(), features.end(),
                    [](const FeatureNameStatistics& f) {
                      // The case of empty feature name is covered by
                      // FIELD_ID_NOT_SET.
                      return f.field_id_case() ==
                                 FeatureNameStatistics::kName ||
                             f.field_id_case() ==
                                 FeatureNameStatistics::FIELD_ID_NOT_SET;
                    })) {
      InitializeWithFeatureName();
    } else if (std::all_of(features.begin(), features.end(),
                           [](const FeatureNameStatistics& f) {
                             return f.field_id_case() ==
                                    FeatureNameStatistics::kPath;
                           })) {
      InitializeWithFeaturePath();
    } else {
      LOG(QFATAL) << "Some features had .name and some features had .path. "
                     "This is unexpected. "
                  << data_.DebugString();
    }
  }

  void InitializeWithFeaturePath() {
    for (int i = 0; i < data_.features_size(); ++i) {
      const FeatureNameStatistics& feature = data_.features(i);
      path_location_[Path(feature.path())] = i;
      context_[i] = FeatureContext();
    }
    for (const auto& path_and_index : path_location_) {
      const Path& path = path_and_index.first;
      const int index = path_and_index.second;
      FeatureContext& context = context_[index];
      context.path = path;
      if (!path.empty()) {
        const Path parent_path = path.GetParent();
        auto iter = path_location_.find(parent_path);
        if (iter != path_location_.end()) {
          context.parent_index = iter->second;
          context_[iter->second].child_indices.push_back(index);
        }
      }
    }
  }

  void InitializeWithFeatureName() {
    // It takes O(n log n) time to construct location, a BST from the name
    // of a feature to its location in data_.features().
    // Map from name to location.
    // data_.features(location[foo]).name() == foo
    std::map<string, int> location;

    for (int i = 0; i < data_.features_size(); ++i) {
      // TODO(b/124192588): This is a short term fix to ignore features with
      // empty stats. Remove this once we have added a unknown_stats message
      // in the stats proto which would keep track of common_stats for
      // completely missing features.
      if (HasEmptyStats(data_.features(i))) {
        continue;
      }
      location[data_.features(i).name()] = i;
      context_[i] = FeatureContext();
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
        context_.at(index).parent_index = parent_index;
        context_.at(index).path = context_.at(parent_index).path.GetChild(
            name.substr(parent_name.size() + 1));
        context_.at(parent_index).child_indices.push_back(index);
      } else {
        context_.at(index).path = Path({data_.features(index).name()});
      }
      path_location_[context_.at(index).path] = index;
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
    return context_.at(view.index_).path;
  }

  absl::optional<FeatureStatsView> GetParent(
      const FeatureStatsView& view) const {
    absl::optional<int> opt_parent_index
        = context_.at(view.index_).parent_index;
    if (opt_parent_index) {
      return FeatureStatsView(*opt_parent_index, view.parent_view_);
    } else {
      return absl::nullopt;
    }
  }

  std::vector<FeatureStatsView> GetChildren(
      const FeatureStatsView& view) const {
    std::vector<FeatureStatsView> result;
    for (int i : context_.at(view.index_).child_indices) {
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

  // The previous span dataset stats (if available).
  // Note that DatasetStatsView objects are very lightweight, so this
  // cost is minimal.
  const std::shared_ptr<DatasetStatsView> previous_span_;

  // The serving dataset stats (if available).
  const std::shared_ptr<DatasetStatsView> serving_;

  // The previous version dataset stats (if available).
  const std::shared_ptr<DatasetStatsView> previous_version_;

  /*********** Cached information below, derivable from data_ *****************/

  // Context of each feature: parents and children.
  // parallel to features() array in data.
  std::map<int, FeatureContext> context_;

  // Map from path to the index of the FeatureStatistics containing the
  // statistics for that path.
  std::map<Path, int> path_location_;
};

DatasetStatsView::DatasetStatsView(
    const DatasetFeatureStatistics& data, bool by_weight,
    const absl::optional<string>& environment,
    std::shared_ptr<DatasetStatsView> previous_span,
    std::shared_ptr<DatasetStatsView> serving,
    std::shared_ptr<DatasetStatsView> previous_version)
    : impl_(new DatasetStatsViewImpl(data, by_weight, environment,
                                     previous_span, serving,
                                     previous_version)) {}

DatasetStatsView::DatasetStatsView(const DatasetFeatureStatistics& data,
                                   bool by_weight)
    : impl_(new DatasetStatsViewImpl(data, by_weight, absl::nullopt,
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>())) {}

DatasetStatsView::DatasetStatsView(
    const tensorflow::metadata::v0::DatasetFeatureStatistics& data)
    : impl_(new DatasetStatsViewImpl(data, false, absl::nullopt,
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>(),
                                     std::shared_ptr<DatasetStatsView>())) {}

std::vector<FeatureStatsView> DatasetStatsView::features() const {
  std::vector<FeatureStatsView> result;
  for (int i = 0; i < impl_->data().features_size(); ++i) {
    // TODO(b/124192588): This is a short term fix to ignore features with
    // empty stats. Remove this once we have added a unknown_stats message
    // in the stats proto which would keep track of common_stats for
    // completely missing features.
    if (HasEmptyStats(impl_->data().features(i))) {
      continue;
    }
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

const absl::optional<DatasetStatsView> DatasetStatsView::GetPreviousSpan()
    const {
  if (impl_->previous_span_) {
    return *impl_->previous_span_;
  }
  return absl::nullopt;
}

const absl::optional<DatasetStatsView> DatasetStatsView::GetServing() const {
  if (impl_->serving_) {
    return *impl_->serving_;
  }
  return absl::nullopt;
}

const absl::optional<DatasetStatsView> DatasetStatsView::GetPreviousVersion()
    const {
  if (impl_->previous_version_) {
    return *impl_->previous_version_;
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

std::vector<double> FeatureStatsView::GetNumMissingNested() const {
  std::vector<double> result;
  if (parent_view_.by_weight()) {
    for (const auto& presence_and_valency_stats :
         GetCommonStatistics().weighted_presence_and_valency_stats()) {
      result.push_back(presence_and_valency_stats.num_missing());
    }
  } else {
    for (const auto& presence_and_valency_stats :
         GetCommonStatistics().presence_and_valency_stats()) {
      result.push_back(presence_and_valency_stats.num_missing());
    }
  }
  if (result.empty()) {
    result.push_back(GetNumMissing());
  }
  return result;
}

absl::optional<double> FeatureStatsView::GetFractionPresent() const {
  const double num_examples = GetNumExamples();
  if (num_examples <= 0) {
    return absl::nullopt;
  }
  if (GetNumMissing() == 0.0) {
    // Avoids numerical issues.
    return 1.0;
  }
  return GetNumPresent() / num_examples;
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

std::vector<std::pair<int, int>> FeatureStatsView::GetMinMaxNumValues() const {
  std::vector<std::pair<int, int>> min_max_num_values;
  for (const auto& presence_and_valency_stats :
       GetCommonStatistics().presence_and_valency_stats()) {
    min_max_num_values.push_back(
        // The number of values should never be negative: instead of
        // propagating such an error, we treat it as zero.
        {std::max<int>(presence_and_valency_stats.min_num_values(), 0),
         presence_and_valency_stats.max_num_values()});
  }
  if (min_max_num_values.empty()) {
    min_max_num_values.push_back(
        {std::max<int>(GetCommonStatistics().min_num_values(), 0),
         GetCommonStatistics().max_num_values()});
  }
  return min_max_num_values;
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

absl::optional<Histogram> FeatureStatsView::GetStandardHistogram() const {
  if (!data().has_num_stats()) {
    return absl::nullopt;
  }
  RepeatedPtrField<Histogram> histograms =
      (parent_view_.by_weight())
          ? data().num_stats().weighted_numeric_stats().histograms()
          : data().num_stats().histograms();
  for (const auto& histogram : histograms) {
    if (histogram.type() ==
        Histogram::HistogramType::Histogram_HistogramType_STANDARD) {
      return histogram;
    }
  }
  return absl::nullopt;
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
  return data().string_stats().invalid_utf8_count() > 0;
}

const tensorflow::metadata::v0::NumericStatistics& FeatureStatsView::num_stats()
    const {
  return data().num_stats();
}

const tensorflow::metadata::v0::BytesStatistics& FeatureStatsView::bytes_stats()
    const {
  return data().bytes_stats();
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
  const auto& common_stats = GetCommonStatistics();
  return common_stats.has_weighted_common_stats() &&
         (common_stats.presence_and_valency_stats().size() ==
          common_stats.weighted_presence_and_valency_stats().size());
}

absl::optional<FeatureStatsView> FeatureStatsView::GetServing() const {
  absl::optional<DatasetStatsView> dataset_stats_view =
      parent_view_.GetServing();
  if (dataset_stats_view) {
    return dataset_stats_view->GetByPath(GetPath());
  }
  return absl::nullopt;
}

absl::optional<FeatureStatsView> FeatureStatsView::GetPreviousSpan() const {
  absl::optional<DatasetStatsView> dataset_stats_view =
      parent_view_.GetPreviousSpan();
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

const absl::optional<uint64> FeatureStatsView::GetNumUnique() const {
  if (data().has_string_stats()) {
    return data().string_stats().unique();
  }
  return absl::nullopt;
}

const tensorflow::metadata::v0::CustomStatistic*
FeatureStatsView::GetCustomStatByName(
    const std::string& custom_stat_name) const {
  auto it = absl::c_find_if(
      data().custom_stats(),
      [&custom_stat_name](
          const tensorflow::metadata::v0::CustomStatistic& stat) {
        return stat.name() == custom_stat_name;
      });
  return it == data().custom_stats().end() ? nullptr : &(*it);
}

}  // namespace data_validation
}  // namespace tensorflow
