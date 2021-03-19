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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_ANOMALIES_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_ANOMALIES_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow_data_validation/anomalies/features_needed.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/schema.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// A base class for individual schema anomalies, which can identify problems
// at the dataset or feature level.
class SchemaAnomalyBase {
 public:
  SchemaAnomalyBase();

  virtual ~SchemaAnomalyBase() {}

  SchemaAnomalyBase(SchemaAnomalyBase&& schema_anomaly_base);

  SchemaAnomalyBase& operator=(SchemaAnomalyBase&& schema_anomaly_base);

  // Initializes schema_.
  tensorflow::Status InitSchema(const tensorflow::metadata::v0::Schema& schema);

  // If new_severity is more severe that current severity, increases
  // severity. Otherwise, does nothing.
  void UpgradeSeverity(
      tensorflow::metadata::v0::AnomalyInfo::Severity new_severity);

  // Identifies if there is an issue.
  bool is_problem() {
    return severity_ != tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
  }

  // Returns an AnomalyInfo representing the change.
  // baseline is the original schema.
  virtual tensorflow::metadata::v0::AnomalyInfo GetAnomalyInfo(
      const tensorflow::metadata::v0::Schema& baseline,
      bool enable_diff_regions) const;

 protected:
  // A new schema that will make the anomaly go away.
  std::unique_ptr<Schema> schema_;
  // Descriptions of what caused the anomaly.
  std::vector<Description> descriptions_;
  // The severity of the anomaly.
  tensorflow::metadata::v0::AnomalyInfo::Severity severity_;
};

// DatasetSchemaAnomaly represents all dataset-level issues.
class DatasetSchemaAnomaly : public SchemaAnomalyBase {
 public:
  DatasetSchemaAnomaly();

  void Update(const DatasetStatsView& dataset_stats_view);
};

// SchemaAnomaly represents all the issues related to a single column.
class SchemaAnomaly : public SchemaAnomalyBase {
 public:
  SchemaAnomaly();

  SchemaAnomaly(SchemaAnomaly&& schema_anomaly);

  SchemaAnomaly& operator=(SchemaAnomaly&& schema_anomaly);

  // Updates based upon the relevant current feature statistics.
  tensorflow::Status Update(const Schema::Updater& updater,
                            const FeatureStatsView& feature_stats_view);

  // Updates recursively upon the relevant current feature statistics.
  // This is used to have all of the fields of a new sub-message appear
  // in the same anomaly.
  // If features_to_update is not nullopt, only updates fields in
  // features_to_update.
  tensorflow::Status CreateNewField(
      const Schema::Updater& updater,
      const absl::optional<std::set<Path>>& features_to_update,
      const FeatureStatsView& feature_stats_view);

  // Update the skew.
  void UpdateSkewComparator(const FeatureStatsView& feature_stats_view);

  // Makes a note that the feature is missing. Deprecates the feature,
  // and leaves a description.
  void ObserveMissing(const Schema::Updater& updater);

  void set_path(const Path& path) { path_ = path; }

  const absl::optional<tensorflow::metadata::v0::DriftSkewInfo>&
  drift_skew_info() const {
    return drift_skew_info_;
  }

  // Returns true iff the feature is deprecated after changes in this anomaly
  // have been applied.
  bool FeatureIsDeprecated(const Path& path);

  // Returns an AnomalyInfo representing the change for a feature-level anomaly
  // (and so populates the path in AnomalyInfo).
  // baseline is the original schema.
  tensorflow::metadata::v0::AnomalyInfo GetAnomalyInfo(
      const tensorflow::metadata::v0::Schema& baseline,
      bool enable_diff_regions) const override;

 private:
  // The name of the feature being fixed.
  Path path_;
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo> drift_skew_info_;
};

// A class for tracking all anomalies that occur based upon the feature that
// created the anomaly.
class SchemaAnomalies {
 public:
  explicit SchemaAnomalies(const tensorflow::metadata::v0::Schema& schema)
      : dataset_anomalies_(absl::nullopt), serialized_baseline_(schema) {}

  // Finds any column- and dataset-level issues. For column-level issues,
  // creates a map where the key is the key of the column with an anomaly, and
  // the value is a SchemaAnomaly object that contains a changed schema proto
  // that would allow the column at issue to be valid. For dataset-level issues,
  // creates a DatasetSchemaAnomaly object that contains a changed schema proto
  // that would allow the dataset to be valid. If features_needed is set, then a
  // field that is not present in the schema will only be created if it is
  // present in that set.
  // TODO(b/148408297): If a field is in features_needed, but not in statistics
  // or in the schema, then come up with a special kind of anomaly.
  tensorflow::Status FindChanges(
      const DatasetStatsView& statistics,
      const absl::optional<FeaturesNeeded>& features_needed,
      const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config);

  tensorflow::Status FindSkew(const DatasetStatsView& dataset_stats_view);

  // Records current anomalies as a schema diff.
  tensorflow::metadata::v0::Anomalies GetSchemaDiff(
      bool enable_diff_regions) const;

 private:
  // Checks a particular column for any issues, and:
  // 1. If the column is not in the schema, creates a new Schema proto
  //    where the column and all its descendants are added.
  // 2. If the column is in the schema, check if it is OK.
  //    A. If it is deprecated after repair, do nothing.
  //    B. Otherwise, recursively check all its children, returning separate
  //       anomalies for each child.
  tensorflow::Status FindChangesRecursively(
      const FeatureStatsView& feature_stats_view,
      const absl::optional<std::set<Path>>& features_needed,
      const Schema::Updater& updater);

  // 1. If there is a SchemaAnomaly for feature_name, applies update,
  // 2. otherwise, creates a new SchemaAnomaly for the feature_name and
  // initializes it using the serialized_baseline_. Then, it tries the
  // update(...) function. If there is a problem, then the new SchemaAnomaly
  // gets added.
  tensorflow::Status GenericUpdate(
      const std::function<tensorflow::Status(SchemaAnomaly* anomaly)>& update,
      const Path& path);

  // Find dataset-level anomalies.
  tensorflow::Status FindDatasetChanges(
      const DatasetStatsView& dataset_stats_view);

  // Creates a new DatasetSchemaAnomaly and initializes it using the
  // serialized_baseline_. Then, it tries the update(...) function. If there is
  // a problem, then the new DatasetSchemaAnomaly gets added to
  // dataset_anomalies_.
  tensorflow::Status GenericDatasetUpdate(
      const std::function<tensorflow::Status(DatasetSchemaAnomaly* anomaly)>&
          update);

  // Initialize a schema from the serialized_baseline_.
  tensorflow::Status InitSchema(Schema* schema) const;

  // A map from feature columns to anomalies in that column.
  std::map<Path, SchemaAnomaly> anomalies_;

  std::map<Path, tensorflow::metadata::v0::DriftSkewInfo> drift_skew_infos_;

  // Dataset-level anomalies.
  absl::optional<DatasetSchemaAnomaly> dataset_anomalies_;

  // The initial schema. Each SchemaAnomaly is initialized from this.
  tensorflow::metadata::v0::Schema serialized_baseline_;
};

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_ANOMALIES_H_
