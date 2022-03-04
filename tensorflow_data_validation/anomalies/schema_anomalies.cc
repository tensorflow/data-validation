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

#include "tensorflow_data_validation/anomalies/schema_anomalies.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_data_validation/anomalies/diff_util.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/schema.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
using ::tensorflow::protobuf::TextFormat;
using ::tensorflow::Status;

// LINT.IfChange
constexpr char kMultipleErrors[] = "Multiple errors";
// LINT.ThenChange(../utils/anomalies_util.py)
constexpr char kColumnDropped[] = "Column dropped";

// For internal use only.
int NumericalSeverity(tensorflow::metadata::v0::AnomalyInfo::Severity a) {
  switch (a) {
    case tensorflow::metadata::v0::AnomalyInfo::UNKNOWN:
      return 0;
    case tensorflow::metadata::v0::AnomalyInfo::WARNING:
      return 1;
    case tensorflow::metadata::v0::AnomalyInfo::ERROR:
      return 2;
    default:
      LOG(FATAL) << "Unknown severity: " << a;
  }
}

// LINT.IfChange
bool AllSchemaNewColumn(const std::vector<Description>& descriptions) {
  for (const Description& description : descriptions) {
    if (description.type != metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN) {
      return false;
    }
  }
  return true;
}

// Handles multiple SchemaNewColumn descriptions as a single one.
// Otherwise, leaves descriptions unchanged.
std::vector<Description> FilterDescriptions(
    const std::vector<Description>& descriptions) {
  if (!descriptions.empty() && AllSchemaNewColumn(descriptions)) {
    return {descriptions[0]};
  }
  return descriptions;
}

// Aggregates the descriptions into a single description.
// Eventually, unification can happen in the front end.
Description UnifyDescriptions(const std::vector<Description>& descriptions) {
  return std::accumulate(
      descriptions.begin(), descriptions.end(), Description(),
      [](const Description& description_a, const Description& description_b) {
        if (description_a.long_description.empty()) {
          return description_b;
        } else if (description_b.long_description.empty()) {
          return description_a;
        } else {
          Description output;
          // The actual type of the aggregate anomaly is ignored.
          output.type = tensorflow::metadata::v0::AnomalyInfo::MULTIPLE_REASONS;
          absl::StrAppend(&output.long_description,
                          description_a.long_description, " ",
                          description_b.long_description);
          output.short_description = kMultipleErrors;
          return output;
        }
      });
}
// LINT.ThenChange(../utils/anomalies_util.py)

bool ShouldCreateFeature(const absl::optional<std::set<Path>>& features_needed,
                         const FeatureStatsView& feature) {
  return !features_needed ||
         (features_needed->find(feature.GetPath()) != features_needed->end());
}

}  // namespace

SchemaAnomalyBase::SchemaAnomalyBase()
    : severity_(tensorflow::metadata::v0::AnomalyInfo::UNKNOWN) {}

SchemaAnomalyBase::SchemaAnomalyBase(SchemaAnomalyBase&& schema_anomaly_base) {
  schema_ = std::move(schema_anomaly_base.schema_);
  descriptions_ = std::move(schema_anomaly_base.descriptions_);
  severity_ = schema_anomaly_base.severity_;
}

SchemaAnomalyBase& SchemaAnomalyBase::operator=(
    SchemaAnomalyBase&& schema_anomaly_base) {
  schema_ = std::move(schema_anomaly_base.schema_);
  descriptions_ = std::move(schema_anomaly_base.descriptions_);
  severity_ = schema_anomaly_base.severity_;
  return *this;
}

tensorflow::Status SchemaAnomalyBase::InitSchema(
    const tensorflow::metadata::v0::Schema& schema) {
  schema_ = absl::make_unique<Schema>();
  return schema_->Init(schema);
}

void SchemaAnomalyBase::UpgradeSeverity(
    tensorflow::metadata::v0::AnomalyInfo::Severity new_severity) {
  severity_ = MaxSeverity(severity_, new_severity);
}

tensorflow::metadata::v0::AnomalyInfo SchemaAnomalyBase::GetAnomalyInfo(
    const tensorflow::metadata::v0::Schema& baseline,
    bool enable_diff_regions) const {
  const tensorflow::metadata::v0::Schema new_schema = schema_->GetSchema();
  tensorflow::metadata::v0::AnomalyInfo anomaly_info;
  if (enable_diff_regions) {
    // Find diff regions.
    string baseline_text;
    TextFormat::PrintToString(baseline, &baseline_text);
    const std::vector<absl::string_view> existing_schema_lines =
        absl::StrSplit(baseline_text, '\n');

    string new_schema_text;
    TextFormat::PrintToString(new_schema, &new_schema_text);
    const std::vector<absl::string_view> new_schema_lines =
        absl::StrSplit(new_schema_text, '\n');

    std::vector<tensorflow::metadata::v0::DiffRegion> diff_regions_vec =
        ComputeDiff(existing_schema_lines, new_schema_lines);

    copy(diff_regions_vec.begin(), diff_regions_vec.end(),
         ::tensorflow::protobuf::RepeatedPtrFieldBackInserter(
             anomaly_info.mutable_diff_regions()));
  }
  const std::vector<Description> filtered_descriptions =
      FilterDescriptions(descriptions_);
  for (const Description& description : filtered_descriptions) {
    tensorflow::metadata::v0::AnomalyInfo::Reason& reason =
        *anomaly_info.add_reason();
    reason.set_type(description.type);
    reason.set_short_description(description.short_description);
    reason.set_description(description.long_description);
  }
  {
    // Set description of entire anomaly.
    const Description unified_description =
        UnifyDescriptions(filtered_descriptions);
    anomaly_info.set_description(unified_description.long_description);
    anomaly_info.set_short_description(unified_description.short_description);
    anomaly_info.set_severity(severity_);
  }
  return anomaly_info;
}

SchemaAnomaly::SchemaAnomaly() : SchemaAnomalyBase() {}

SchemaAnomaly::SchemaAnomaly(SchemaAnomaly&& schema_anomaly)
    : SchemaAnomalyBase(std::move(schema_anomaly)) {
  path_ = std::move(schema_anomaly.path_);
  drift_skew_info_ = std::move(schema_anomaly.drift_skew_info_);
}

SchemaAnomaly& SchemaAnomaly::operator=(SchemaAnomaly&& schema_anomaly) {
  path_ = std::move(schema_anomaly.path_);
  drift_skew_info_ = std::move(schema_anomaly.drift_skew_info_);
  SchemaAnomalyBase::operator=(std::move(schema_anomaly));
  return *this;
}

void SchemaAnomaly::ObserveMissing(const Schema::Updater& updater) {
  const Description description = {
      tensorflow::metadata::v0::AnomalyInfo::SCHEMA_MISSING_COLUMN,
      kColumnDropped, "Column is completely missing"};
  descriptions_.push_back(description);
  updater.UpdateSeverityForAnomaly(descriptions_, &severity_);
  schema_->DeprecateFeature(path_);
}

tensorflow::Status SchemaAnomaly::Update(
    const Schema::Updater& updater,
    const FeatureStatsView& feature_stats_view) {
  std::vector<Description> new_descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity new_severity;
  TF_RETURN_IF_ERROR(schema_->UpdateFeature(updater, feature_stats_view,
                                            &new_descriptions,
                                            &drift_skew_info_, &new_severity));
  descriptions_.insert(descriptions_.end(), new_descriptions.begin(),
                       new_descriptions.end());
  UpgradeSeverity(new_severity);
  return tensorflow::Status::OK();
}

tensorflow::Status SchemaAnomaly::CreateNewField(
    const Schema::Updater& updater,
    const absl::optional<std::set<Path>>& features_to_update,
    const FeatureStatsView& feature_stats_view) {
  tensorflow::metadata::v0::AnomalyInfo::Severity new_severity;
  std::vector<Description> new_descriptions;

  TF_RETURN_IF_ERROR(schema_->UpdateRecursively(
      updater, feature_stats_view, features_to_update, &new_descriptions,
      &new_severity));
  UpgradeSeverity(new_severity);
  // Having a recursive column creates multiple descriptions.
  // Instead, we just push the first one.
  descriptions_.insert(descriptions_.end(), new_descriptions.begin(),
                       new_descriptions.end());
  return Status::OK();
}

void SchemaAnomaly::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  auto feature_comparison_result =
      schema_->UpdateSkewComparator(feature_stats_view);
  if (!feature_comparison_result.descriptions.empty()) {
    UpgradeSeverity(tensorflow::metadata::v0::AnomalyInfo::ERROR);
  }
  descriptions_.insert(descriptions_.end(),
                       feature_comparison_result.descriptions.begin(),
                       feature_comparison_result.descriptions.end());
  if (!feature_comparison_result.measurements.empty()) {
    drift_skew_info_.emplace();
    for (const auto& measurement : feature_comparison_result.measurements) {
      *drift_skew_info_->add_skew_measurements() = measurement;
    }
  }
}

bool SchemaAnomaly::FeatureIsDeprecated(const Path& path) {
  if (schema_) {
    return schema_->FeatureIsDeprecated(path);
  }
  return false;
}

tensorflow::metadata::v0::AnomalyInfo SchemaAnomaly::GetAnomalyInfo(
    const tensorflow::metadata::v0::Schema& baseline,
    bool enable_diff_regions) const {
  tensorflow::metadata::v0::AnomalyInfo anomaly_info =
      SchemaAnomalyBase::GetAnomalyInfo(baseline, enable_diff_regions);
  *anomaly_info.mutable_path() = path_.AsProto();
  return anomaly_info;
}

DatasetSchemaAnomaly::DatasetSchemaAnomaly() : SchemaAnomalyBase() {}

void DatasetSchemaAnomaly::Update(const DatasetStatsView& dataset_stats_view) {
  const std::vector<Description> new_descriptions =
      schema_->UpdateDatasetConstraints(dataset_stats_view);
  if (!new_descriptions.empty()) {
    UpgradeSeverity(tensorflow::metadata::v0::AnomalyInfo::ERROR);
  }
  descriptions_.insert(descriptions_.end(), new_descriptions.begin(),
                       new_descriptions.end());
}

tensorflow::metadata::v0::Anomalies SchemaAnomalies::GetSchemaDiff(
    bool enable_diff_regions) const {
  const tensorflow::metadata::v0::Schema& schema_proto = serialized_baseline_;
  tensorflow::metadata::v0::Anomalies result;
  result.set_anomaly_name_format(
      tensorflow::metadata::v0::Anomalies::SERIALIZED_PATH);
  *result.mutable_baseline() = schema_proto;
  ::tensorflow::protobuf::Map<string, tensorflow::metadata::v0::AnomalyInfo>&
      result_schemas = *result.mutable_anomaly_info();
  for (const auto& pair : anomalies_) {
    const Path& feature_path = pair.first;
    const SchemaAnomaly& anomaly = pair.second;
    result_schemas[feature_path.Serialize()] =
        anomaly.GetAnomalyInfo(schema_proto, enable_diff_regions);
  }
  if (dataset_anomalies_) {
    *result.mutable_dataset_anomaly_info() =
        dataset_anomalies_->GetAnomalyInfo(schema_proto, enable_diff_regions);
  }
  for (const auto& pair : drift_skew_infos_) {
    auto* drift_skew_info = result.add_drift_skew_info();
    *drift_skew_info = pair.second;
    *drift_skew_info->mutable_path() = pair.first.AsProto();
  }
  return result;
}

tensorflow::Status SchemaAnomalies::InitSchema(Schema* schema) const {
  return schema->Init(serialized_baseline_);
}

tensorflow::Status SchemaAnomalies::GenericUpdate(
    const std::function<tensorflow::Status(SchemaAnomaly* anomaly)>& update,
    const Path& path) {
  auto iter = anomalies_.find(path);
  // schema_anomaly always points to the SchemaAnomaly passed to `update`.
  // It may not exist or be inserted in anomalies_. But we always want to
  // get the drift_skew_info from it.
  SchemaAnomaly* schema_anomaly;
  SchemaAnomaly new_schema_anomaly;
  if (iter != anomalies_.end()) {
    schema_anomaly = &iter->second;
    TF_RETURN_IF_ERROR(update(schema_anomaly));
  } else {
    schema_anomaly = &new_schema_anomaly;
    TF_RETURN_IF_ERROR(schema_anomaly->InitSchema(serialized_baseline_));
    schema_anomaly->set_path(path);
    TF_RETURN_IF_ERROR(update(schema_anomaly));
    if (schema_anomaly->is_problem()) {
      auto iter_and_inserted = anomalies_.insert(
          std::make_pair(path, std::move(*schema_anomaly)));
      schema_anomaly = &iter_and_inserted.first->second;
    }
  }
  if (schema_anomaly->drift_skew_info()) {
    drift_skew_infos_[path].MergeFrom(*schema_anomaly->drift_skew_info());
  }
  return Status::OK();
}

tensorflow::Status SchemaAnomalies::GenericDatasetUpdate(
    const std::function<tensorflow::Status(DatasetSchemaAnomaly* anomaly)>&
        update) {
  DatasetSchemaAnomaly dataset_schema_anomaly;
  TF_RETURN_IF_ERROR(dataset_schema_anomaly.InitSchema(serialized_baseline_));
  TF_RETURN_IF_ERROR(update(&dataset_schema_anomaly));
  if (dataset_schema_anomaly.is_problem()) {
    dataset_anomalies_ = std::move(dataset_schema_anomaly);
  }
  return Status::OK();
}

tensorflow::Status SchemaAnomalies::FindChangesRecursively(
    const FeatureStatsView& feature_stats_view,
    const absl::optional<std::set<Path>>& features_needed,
    const Schema::Updater& updater) {
  Schema baseline;
  TF_RETURN_IF_ERROR(InitSchema(&baseline));
  if (baseline.FeatureExists(feature_stats_view.GetPath())) {
    // TODO(b/148407751): Treat PLANNED separately.
    if (baseline.FeatureIsDeprecated(feature_stats_view.GetPath())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(GenericUpdate(
        [&feature_stats_view, &updater](SchemaAnomaly* schema_anomaly) {
          return schema_anomaly->Update(updater, feature_stats_view);
        },
        feature_stats_view.GetPath()));
    if (ContainsKey(anomalies_, feature_stats_view.GetPath()) &&
        anomalies_[feature_stats_view.GetPath()].FeatureIsDeprecated(
            feature_stats_view.GetPath())) {
      return Status::OK();
    }
    for (const FeatureStatsView& child : feature_stats_view.GetChildren()) {
      TF_RETURN_IF_ERROR(
          FindChangesRecursively(child, features_needed, updater));
    }
  } else if (ShouldCreateFeature(features_needed, feature_stats_view)) {
    // Feature doesn't exist. Need to recursively create it.

    if (!ContainsKey(anomalies_, feature_stats_view.GetPath())) {
      SchemaAnomaly anomaly;
      TF_RETURN_IF_ERROR(anomaly.InitSchema(serialized_baseline_));
      anomaly.set_path(feature_stats_view.GetPath());
      anomalies_[feature_stats_view.GetPath()] = std::move(anomaly);
    }
    // Since these features are all new,
    // features_needed == features_to_update.
    TF_RETURN_IF_ERROR(anomalies_[feature_stats_view.GetPath()].CreateNewField(
        updater, features_needed, feature_stats_view));
  }
  return Status::OK();
}

tensorflow::Status SchemaAnomalies::FindChanges(
    const DatasetStatsView& statistics,
    const absl::optional<FeaturesNeeded>& features_needed,
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config) {
  Schema::Updater updater(feature_statistics_to_proto_config);
  absl::optional<std::set<Path>> feature_set_to_create;
  if (features_needed) {
    feature_set_to_create = std::set<Path>();
    for (const auto& p : *features_needed) {
      const Path& path = p.first;
      feature_set_to_create->insert(path);
    }
  }

  for (const FeatureStatsView& feature_stats_view :
       statistics.GetRootFeatures()) {
    TF_RETURN_IF_ERROR(FindChangesRecursively(feature_stats_view,
                                              feature_set_to_create, updater));
  }
  Schema baseline;
  TF_RETURN_IF_ERROR(InitSchema(&baseline));
  for (const Path& path : baseline.GetMissingPaths(statistics)) {
    TF_RETURN_IF_ERROR(GenericUpdate(
        [&updater](SchemaAnomaly* schema_anomaly) {
          schema_anomaly->ObserveMissing(updater);
          return Status::OK();
        },
        path));
  }
  if (features_needed) {
    for (const auto& p : *features_needed) {
      const Path& path = p.first;
      if (!statistics.GetByPath(path) && !baseline.FeatureExists(path)) {
        // TODO(114770329): come up with a special error here, as this
        // indicates that a feature is required that is not in the data.
        LOG(ERROR) << "Required feature missing from data and schema: "
                   << path.Serialize();
      }
    }
  }

  TF_RETURN_IF_ERROR(FindDatasetChanges(statistics));

  return Status::OK();
}

tensorflow::Status SchemaAnomalies::FindSkew(
    const DatasetStatsView& dataset_stats_view) {
  for (const FeatureStatsView& feature_stats_view :
       dataset_stats_view.features()) {
    // This is a simplified version of finding skew, that ignores the feature
    // if there is no training data for it.
    TF_RETURN_IF_ERROR(GenericUpdate(
        [&feature_stats_view](SchemaAnomaly* schema_anomaly) {
          schema_anomaly->UpdateSkewComparator(feature_stats_view);
          return Status::OK();
        },
        feature_stats_view.GetPath()));
  }
  return Status::OK();
}

tensorflow::Status SchemaAnomalies::FindDatasetChanges(
    const DatasetStatsView& dataset_stats_view) {
  TF_RETURN_IF_ERROR(GenericDatasetUpdate(
      [&dataset_stats_view](DatasetSchemaAnomaly* dataset_schema_anomaly) {
        dataset_schema_anomaly->Update(dataset_stats_view);
        return Status::OK();
      }));
  return Status::OK();
}

}  // namespace data_validation
}  // namespace tensorflow
