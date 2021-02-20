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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/proto/validation_config.pb.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

// This class is used to generate schemas, and to check the validity of data,
// and to update schemas.
// See https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto
// Example:
// DatasetStatsView statistics; // Original statistics.
// FeatureStatisticsToProtoConfig config;
// // Create a new schema.
// Schema schema;
// TF_RETURN_IF_ERROR(schema.Update(statistics, config));
// tensorflow::metadata::v0::Schema schema_proto = schema.GetSchema();
// ...save proto somewhere...
// Schema schema2;
// TF_RETURN_IF_ERROR(schema2.Init(schema_proto));
// DatasetStatsView next_statistics = ...
// Update the schema again.
// TF_RETURN_IF_ERROR(schema2.Update(next_statistics, config));
// tensorflow::metadata::v0::Schema schema_proto2 = schema2.GetSchema();
class Schema {
 public:
  // Holds the configuration for updating the schema, based on
  // FeatureStatisticsToProtoConfig. Used in SchemaAnomaly and SchemaAnomalies.
  class Updater {
   public:
    // Creates a factory for new FeatureTypes, based on a config.
    explicit Updater(const FeatureStatisticsToProtoConfig& config);
    // Creates a column from the statistics object, based upon the
    // configuration in the factory.
    // Updates the severity of the change.
    tensorflow::Status CreateColumn(
        const FeatureStatsView& feature_stats_view, Schema* schema,
        tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const;

    // Sets the severity based on anomaly descriptions, possibly using severity
    // overrides.
    void UpdateSeverityForAnomaly(
        const std::vector<Description>& descriptions,
        tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const;

    // Returns true if there is a limit on the size of a string domain and it
    // should be deleted.
    bool string_domain_too_big(int size) const;

   private:
    // The config being used to create the schema.
    const FeatureStatisticsToProtoConfig config_;
    // The columns to ignore, extracted from config_.
    const std::set<string> columns_to_ignore_;
    // A map from a key to an enum, extracted from config_.
    std::map<Path, string> grouped_enums_;
  };

  // This creates an empty schema. In order to populate it, either call
  // Init(...) or Update(...).
  Schema() = default;

  // Initializes a schema from a protocol buffer.
  // Schema must be empty (i.e. it was just created), or the method will return
  // an InvalidArgumentException.
  // If the SchemaProto is not valid, the method will return an
  // InvalidArgumentException.
  tensorflow::Status Init(const tensorflow::metadata::v0::Schema& input);

  // Updates Schema given new data. If you have a new, previously unseen column,
  // then config is used to create it.
  tensorflow::Status Update(const DatasetStatsView& dataset_stats,
                            const FeatureStatisticsToProtoConfig& config);

  // Updates Schema given new data, but only on the columns specified.
  // If you have a new, previously unseen column on the list of columns to,
  // consider, then config is used to create it.
  tensorflow::Status Update(const DatasetStatsView& dataset_stats,
                            const FeatureStatisticsToProtoConfig& config,
                            const std::vector<Path>& paths_to_consider);

  // Check if there are any issues with a path or its children.
  tensorflow::Status UpdateRecursively(
      const Updater& updater, const FeatureStatsView& feature_stats_view,
      const absl::optional<std::set<Path>>& paths_to_consider,
      std::vector<Description>* descriptions,
      tensorflow::metadata::v0::AnomalyInfo::Severity* severity);

  // Returns true iff there is a feature corresponding to the path.
  bool FeatureExists(const Path& path);

  // Returns true if the feature corresponding to the view is deprecated,
  // false if it is not. If there is no feature corresponding to the
  // view, the result is undefined.
  bool FeatureIsDeprecated(const Path& path);

  // Deprecates a feature.
  void DeprecateFeature(const Path& path);

  // Gets the schema that represents the proto.
  tensorflow::metadata::v0::Schema GetSchema() const;

  // Populates FeatureStatisticsToProtoConfig with groups of enums that seem
  // similar. config is the original config, and dataset_stats has
  // the relevant data.
  static tensorflow::Status GetRelatedEnums(
      const DatasetStatsView& dataset_stats,
      FeatureStatisticsToProtoConfig* config);

  // Returns true if there are no enum types and no feature types.
  bool IsEmpty() const;

  // Check if there are any issues with a single column.
  tensorflow::Status UpdateFeature(
      const Updater& updater, const FeatureStatsView& feature_stats_view,
      std::vector<Description>* descriptions,
      absl::optional<tensorflow::metadata::v0::DriftSkewInfo>* drift_skew_info,
      tensorflow::metadata::v0::AnomalyInfo::Severity* severity);

  // A method for updating the skew comparator.
  FeatureComparisonResult UpdateSkewComparator(
      const FeatureStatsView& feature_stats_view);

  // Clears the schema, so that IsEmpty()==true.
  void Clear();

  // Returns columns that are required to be present but are absent
  // (i.e., no FeatureNameStatistics).
  std::vector<Path> GetMissingPaths(const DatasetStatsView& dataset_stats);

  // Updates dataset-level constraints.
  std::vector<Description> UpdateDatasetConstraints(
      const DatasetStatsView& dataset_stats_view);

 private:
  using Feature = tensorflow::metadata::v0::Feature;
  using SparseFeature = tensorflow::metadata::v0::SparseFeature;
  using StringDomain = tensorflow::metadata::v0::StringDomain;
  using WeightedFeature = tensorflow::metadata::v0::WeightedFeature;
  // Updates Schema given new data, but only on the columns specified.
  // If you have a new, previously unseen column on the list of columns to
  // consider, then config is used to create it.
  // If paths_to_consider is unspecified, then it updates all columns.
  tensorflow::Status Update(
      const DatasetStatsView& dataset_stats, const Updater& updater,
      const absl::optional<std::set<Path>>& paths_to_consider);

  // Gets a map from a simple enum name to the columns that are using it.
  // Used in GetRelatedEnums().
  std::map<string, std::set<Path>> EnumNameToPaths() const;

  // Deletes a StringDomain and all references to it.
  void ClearStringDomain(const string& domain_name);

  // Returns simple names of similar enum types.
  // Definition of similar (will be) configured in the
  // FeatureStatisticsToProtoConfig.
  // Used in GetRelatedEnums().
  std::vector<std::set<string>> SimilarEnumTypes(
      const EnumsSimilarConfig& config) const;

  // Gets an existing StringDomain. If it does not already exist, returns null.
  StringDomain* GetExistingStringDomain(const string& name);

  // Finds all names and of features in the environment.
  std::vector<Path> GetAllRequiredFeatures(
      const Path& prefix,
      const ::tensorflow::protobuf::RepeatedPtrField<Feature>& features,
      const absl::optional<string>& environment) const;

  // Returns the structure of the schema, organized by full name.
  // For each full name, returns an existing feature with that name,
  // and the parent name (or the empty string if there is no parent).
  void GetAllExistingFeaturesByName(
      tensorflow::protobuf::RepeatedPtrField<Feature>* features,
      const string& parent,
      std::map<string, std::pair<string, Feature*>>* result);

  bool IsExistenceRequired(const Feature& feature,
                           const absl::optional<string>& environment) const;

  bool IsFeatureInEnvironment(const Feature& feature,
                              const absl::optional<string>& environment) const;

  // Gets a new enum type. If the candidate name is already taken, the enum
  // returned has a different name. E.g., if there exists enums "foo" and
  // "foo2", then GetNewEnum("foo")->SimpleName() == "foo3".
  StringDomain* GetNewStringDomain(const string& candidate_name);

  // Check if a feature is internally consistent. If not, fix it and return a
  // description of what is wrong.
  std::vector<Description> UpdateFeatureSelf(Feature* feature);

  // Gets an EnumType, adding it to enum_types_ and/or appending
  // values if necessary.
  StringDomain* GetStringDomain(const string& name);

  // Gets an existing feature, or returns null if it doesn't exist.
  Feature* GetExistingFeature(const Path& path);

  // Gets an existing sparse feature, or returns null if it doesn't exist.
  SparseFeature* GetExistingSparseFeature(const Path& path);

  // Gets an existing weighted feature, or returns null if it doesn't exist.
  WeightedFeature* GetExistingWeightedFeature(const Path& path);

  // Gets a new feature. Assumes that the feature does not already exist.
  Feature* GetNewFeature(const Path& path);

  void UpdateFeatureInternal(
      const Updater& updater, const FeatureStatsView& view, Feature* feature,
      std::vector<Description>* descriptions,
      absl::optional<tensorflow::metadata::v0::DriftSkewInfo>* drift_skew_info);

  // Validates the dataset_stats of a sparse feature:
  // - Ensures that referred features are either all present, or all absent.
  // - If they are present, they have the same length.
  // If any of the validations fails it deprecates the sparse_feature.
  std::vector<Description> UpdateSparseFeature(const FeatureStatsView& view,
                                               SparseFeature* sparse_feature);

  // Validates the dataset_stats of a weighted feature:
  // - Ensures that referred features are either all present, or all absent.
  // - If they are present, they have the same length.
  // If any of the validations fails it deprecates the weighted_feature.
  std::vector<Description> UpdateWeightedFeature(
      const FeatureStatsView& view, WeightedFeature* weighted_feature);

  // Gets existing dataset constraints, and returns null if it doesn't exist.
  ::tensorflow::metadata::v0::DatasetConstraints*
  GetExistingDatasetConstraints();

  bool generate_legacy_feature_spec() const;

  // Note: do not manually add string_domains or features.
  // Call GetNewEnum() or GetNewFeature().
  tensorflow::metadata::v0::Schema schema_;
};

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_H_
