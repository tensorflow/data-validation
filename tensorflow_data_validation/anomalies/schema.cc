/* Copyright 2020 Google LLC

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

#include "tensorflow_data_validation/anomalies/schema.h"

#include <map>
#include <memory>
#include <set>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/bool_domain_util.h"
#include "tensorflow_data_validation/anomalies/custom_domain_util.h"
#include "tensorflow_data_validation/anomalies/dataset_constraints_util.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/float_domain_util.h"
#include "tensorflow_data_validation/anomalies/image_domain_util.h"
#include "tensorflow_data_validation/anomalies/int_domain_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/natural_language_domain_util.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_data_validation/anomalies/string_domain_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using ::absl::optional;
using ::tensorflow::Status;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::StringDomain;
using ::tensorflow::metadata::v0::WeightedFeature;
using PathProto = ::tensorflow::metadata::v0::Path;

constexpr char kTrainingServingSkew[] = "Training/Serving skew";

// LINT.IfChange(sparse_feature_custom_stat_names)
static constexpr char kMissingSparseValue[] = "missing_value";
static constexpr char kMissingSparseIndex[] = "missing_index";
static constexpr char kMaxLengthDiff[] = "max_length_diff";
static constexpr char kMinLengthDiff[] = "min_length_diff";
// LINT.ThenChange(../statistics/generators/sparse_feature_stats_generator.py:custom_stat_names)

// LINT.IfChange(weighted_feature_custom_stat_names)
static constexpr char kMissingWeightedValue[] = "missing_value";
static constexpr char kMissingWeight[] = "missing_weight";
static constexpr char kMaxWeightLengthDiff[] = "max_weight_length_diff";
static constexpr char kMinWeightLengthDiff[] = "min_weight_length_diff";
// LINT.ThenChange(../statistics/generators/weighted_feature_stats_generator.py:custom_stat_names)

template <typename Container>
bool ContainsValue(const Container& a, const string& value) {
  return absl::c_find(a, value) != a.end();
}

std::set<tensorflow::metadata::v0::FeatureType> AllowedFeatureTypes(
    Feature::DomainInfoCase domain_info_case) {
  switch (domain_info_case) {
    case Feature::kDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kBoolDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::FLOAT};
    case Feature::kIntDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::kFloatDomain:
      return {tensorflow::metadata::v0::FLOAT, tensorflow::metadata::v0::BYTES};
    case Feature::kStringDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kStructDomain:
      return {tensorflow::metadata::v0::STRUCT};
    case Feature::kNaturalLanguageDomain:
      return {tensorflow::metadata::v0::BYTES, tensorflow::metadata::v0::INT};
    case Feature::kImageDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kMidDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kUrlDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kTimeDomain:
      // Consider also supporting time as floats.
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::DOMAIN_INFO_NOT_SET:
      ABSL_FALLTHROUGH_INTENDED;
    default:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::FLOAT,
              tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::STRUCT};
  }
}

// Remove all elements from the input array for which the input predicate
// pred is true. Returns number of erased elements.
template <typename T, typename Predicate>
int RemoveIf(::tensorflow::protobuf::RepeatedPtrField<T>* array,
             const Predicate& pred) {
  int i = 0, end = array->size();
  while (i < end && !pred(&array->Get(i))) ++i;

  if (i == end) return 0;

  // 'i' is positioned at first element to be removed.
  for (int j = i + 1; j < end; ++j) {
    if (!pred(&array->Get(j))) array->SwapElements(j, i++);
  }

  array->DeleteSubrange(i, end - i);
  return end - i;
}

Feature* GetExistingFeatureHelper(
    const string& last_part,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.name() == last_part) {
      return &feature;
    }
  }
  return nullptr;
}

void ClearStringDomainHelper(
    const string& domain_name,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.domain() == domain_name) {
      ::tensorflow::data_validation::ClearDomain(&feature);
    }
    if (feature.has_struct_domain()) {
      ClearStringDomainHelper(
          domain_name, feature.mutable_struct_domain()->mutable_feature());
    }
  }
}

SparseFeature* GetExistingSparseFeatureHelper(
    const string& name,
    tensorflow::protobuf::RepeatedPtrField<
        tensorflow::metadata::v0::SparseFeature>* sparse_features) {
  for (SparseFeature& sparse_feature : *sparse_features) {
    if (sparse_feature.name() == name) {
      return &sparse_feature;
    }
  }
  return nullptr;
}

// absl::nullopt is the set of all paths.
bool ContainsPath(const absl::optional<std::set<Path>>& paths_to_consider,
                  const Path& path) {
  if (!paths_to_consider) {
    return true;
  }
  return ContainsKey(*paths_to_consider, path);
}

absl::string_view GetDomainInfoName(const Feature& feature) {
  const ::tensorflow::protobuf::OneofDescriptor* oneof =
      feature.GetDescriptor()->FindOneofByName("domain_info");
  const ::tensorflow::protobuf::FieldDescriptor* domain_info_field =
      feature.GetReflection()->GetOneofFieldDescriptor(feature, oneof);
  if (domain_info_field) {
    return domain_info_field->name();
  }
  return "UNSET";
}

}  // namespace

Status Schema::Init(const tensorflow::metadata::v0::Schema& input) {
  if (!IsEmpty()) {
    return InvalidArgument("Schema is not empty when Init() called.");
  }
  schema_ = input;
  return Status::OK();
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config) {
  return Update(dataset_stats, Updater(config), absl::nullopt);
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config,
                      const std::vector<Path>& paths_to_consider) {
  return Update(
      dataset_stats, Updater(config),
      std::set<Path>(paths_to_consider.begin(), paths_to_consider.end()));
}

tensorflow::Status Schema::UpdateFeature(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    std::vector<Description>* descriptions,
    absl::optional<tensorflow::metadata::v0::DriftSkewInfo>* drift_skew_info,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;

  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  SparseFeature* sparse_feature =
      GetExistingSparseFeature(feature_stats_view.GetPath());
  WeightedFeature* weighted_feature =
      GetExistingWeightedFeature(feature_stats_view.GetPath());
  if (weighted_feature != nullptr) {
    if ((feature != nullptr || sparse_feature != nullptr) &&
        !::tensorflow::data_validation::WeightedFeatureIsDeprecated(
            *weighted_feature)) {
      descriptions->push_back({tensorflow::metadata::v0::AnomalyInfo::
                                   WEIGHTED_FEATURE_NAME_COLLISION,
                               "Weighted feature name collision",
                               "Weighted feature name collision."});
      ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
      if (feature != nullptr) {
        ::tensorflow::data_validation::DeprecateFeature(feature);
      }
      if (sparse_feature != nullptr) {
        ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      }
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions =
          UpdateWeightedFeature(feature_stats_view, weighted_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }

  if (sparse_feature != nullptr &&
      !::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature)) {
    if (feature != nullptr &&
        !::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
      descriptions->push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_NAME_COLLISION,
           "Sparse feature name collision", "Sparse feature name collision."});
      ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      ::tensorflow::data_validation::DeprecateFeature(feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions = UpdateSparseFeature(feature_stats_view, sparse_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }

  if (feature != nullptr) {
    UpdateFeatureInternal(updater, feature_stats_view, feature, descriptions,
                          drift_skew_info);
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return Status::OK();
  } else {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN, "New column",
        "New column (column in data but not in schema)"};
    *descriptions = {description};
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return updater.CreateColumn(feature_stats_view, this, severity);
  }
  return Status::OK();
}

bool Schema::FeatureIsDeprecated(const Path& path) {
  Feature* feature = GetExistingFeature(path);
  if (feature == nullptr) {
    SparseFeature* sparse_feature = GetExistingSparseFeature(path);
    if (sparse_feature != nullptr) {
      return ::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature);
    }
    // Here, the result is undefined.
    return false;
  }
  return ::tensorflow::data_validation::FeatureIsDeprecated(*feature);
}

void Schema::DeprecateFeature(const Path& path) {
  ::tensorflow::data_validation::DeprecateFeature(
      CHECK_NOTNULL(GetExistingFeature(path)));
}

Status Schema::UpdateRecursively(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    const absl::optional<std::set<Path>>& paths_to_consider,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
  if (!ContainsPath(paths_to_consider, feature_stats_view.GetPath())) {
    return Status::OK();
  }
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo>
      unused_drift_skew_info;
  TF_RETURN_IF_ERROR(UpdateFeature(updater, feature_stats_view, descriptions,
                                   &unused_drift_skew_info, severity));
  if (!FeatureIsDeprecated(feature_stats_view.GetPath())) {
    for (const FeatureStatsView& child : feature_stats_view.GetChildren()) {
      std::vector<Description> child_descriptions;
      tensorflow::metadata::v0::AnomalyInfo::Severity child_severity;
      TF_RETURN_IF_ERROR(UpdateRecursively(updater, child, paths_to_consider,
                                           &child_descriptions,
                                           &child_severity));
      descriptions->insert(descriptions->end(), child_descriptions.begin(),
                           child_descriptions.end());
      *severity = MaxSeverity(child_severity, *severity);
    }
  }
  updater.UpdateSeverityForAnomaly(*descriptions, severity);
  return Status::OK();
}

Schema::Updater::Updater(const FeatureStatisticsToProtoConfig& config)
    : config_(config),
      columns_to_ignore_(config.column_to_ignore().begin(),
                         config.column_to_ignore().end()) {
  for (const ColumnConstraint& constraint : config.column_constraint()) {
    for (const PathProto& column_path : constraint.column_path()) {
      grouped_enums_[Path(column_path)] = constraint.enum_name();
    }
  }
}

// Sets the severity based on anomaly descriptions, possibly using severity
// overrides.
void Schema::Updater::UpdateSeverityForAnomaly(
    const std::vector<Description>& descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  for (const auto& description : descriptions) {
    // By default, all anomalies are ERROR level.
    tensorflow::metadata::v0::AnomalyInfo::Severity severity_for_anomaly =
        tensorflow::metadata::v0::AnomalyInfo::ERROR;

    if (config_.new_features_are_warnings() &&
        (description.type ==
         tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN)) {
      LOG(WARNING) << "new_features_are_warnings is deprecated. Use "
                      "severity_overrides";
      severity_for_anomaly = tensorflow::metadata::v0::AnomalyInfo::WARNING;
    }
    for (const auto& severity_override : config_.severity_overrides()) {
      if (severity_override.type() == description.type) {
        severity_for_anomaly = severity_override.severity();
      }
    }
    *severity = MaxSeverity(*severity, severity_for_anomaly);
  }
}

Status Schema::Updater::CreateColumn(
    const FeatureStatsView& feature_stats_view, Schema* schema,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  if (schema->GetExistingFeature(feature_stats_view.GetPath()) != nullptr) {
    return InvalidArgument("Schema already contains \"",
                           feature_stats_view.GetPath().Serialize(), "\".");
  }

  Feature* feature = schema->GetNewFeature(feature_stats_view.GetPath());

  feature->set_type(feature_stats_view.GetFeatureType());
  InitPresenceAndShape(feature_stats_view, config_.infer_feature_shape(),
                       feature);
  if (ContainsKey(columns_to_ignore_,
                  feature_stats_view.GetPath().Serialize())) {
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return Status::OK();
  }
  if (feature_stats_view.HasValidationDerivedSource()) {
    // TODO(b/227478330): Consider setting a lower severity.
    ::tensorflow::data_validation::MarkFeatureDerived(
        feature_stats_view.GetValidationDerivedSource(), feature);
  }

  if (BestEffortUpdateCustomDomain(feature_stats_view.custom_stats(),
                                   feature)) {
    return Status::OK();
  } else if (ContainsKey(grouped_enums_, feature_stats_view.GetPath())) {
    const string& enum_name = grouped_enums_.at(feature_stats_view.GetPath());
    StringDomain* result = schema->GetExistingStringDomain(enum_name);
    if (result == nullptr) {
      result = schema->GetNewStringDomain(enum_name);
    }
    UpdateStringDomain(*this, feature_stats_view, 0, result);
    return Status::OK();
  } else if (feature_stats_view.HasInvalidUTF8Strings() ||
             feature_stats_view.type() == FeatureNameStatistics::BYTES) {
    // If there are invalid UTF8 strings, or the field should not be further
    // interpreted, add no domain info.
    return Status::OK();
  } else if (IsBoolDomainCandidate(feature_stats_view)) {
    *feature->mutable_bool_domain() = BoolDomainFromStats(feature_stats_view);
    return Status::OK();
  } else if (IsIntDomainCandidate(feature_stats_view)) {
    // By default don't set any values.
    feature->mutable_int_domain();
    return Status::OK();
  } else if (IsStringDomainCandidate(feature_stats_view,
                                     config_.enum_threshold())) {
    StringDomain* string_domain =
        schema->GetNewStringDomain(feature_stats_view.GetPath().Serialize());
    UpdateStringDomain(*this, feature_stats_view, 0, string_domain);
    *feature->mutable_domain() = string_domain->name();
    return Status::OK();
  } else {
    // No domain info for this field.
    return Status::OK();
  }
}

// Returns true if there is a limit on the size of a string domain and it
// should be deleted.
bool Schema::Updater::string_domain_too_big(int size) const {
  return config_.has_enum_delete_threshold() &&
         config_.enum_delete_threshold() <= size;
}

bool Schema::IsEmpty() const {
  return schema_.feature().empty() && schema_.string_domain().empty();
}

void Schema::Clear() { schema_.Clear(); }

StringDomain* Schema::GetNewStringDomain(const string& candidate_name) {
  std::set<string> names;
  for (const StringDomain& string_domain : schema_.string_domain()) {
    names.insert(string_domain.name());
  }
  string new_name = candidate_name;
  int index = 1;
  while (ContainsKey(names, new_name)) {
    ++index;
    new_name = absl::StrCat(candidate_name, index);
  }
  StringDomain* result = schema_.add_string_domain();
  *result->mutable_name() = new_name;
  return result;
}

StringDomain* Schema::GetExistingStringDomain(const string& name) {
  for (int i = 0; i < schema_.string_domain_size(); ++i) {
    StringDomain* possible = schema_.mutable_string_domain(i);
    if (possible->name() == name) {
      return possible;
    }
  }

  // If there is no match, return nullptr.
  return nullptr;
}

std::vector<std::set<string>> Schema::SimilarEnumTypes(
    const EnumsSimilarConfig& config) const {
  std::vector<bool> used(schema_.string_domain_size(), false);
  std::vector<std::set<string>> result;
  for (int index_a = 0; index_a < schema_.string_domain_size(); ++index_a) {
    if (!used[index_a]) {
      const StringDomain& string_domain_a = schema_.string_domain(index_a);
      std::set<string> similar;
      for (int index_b = index_a + 1; index_b < schema_.string_domain_size();
           ++index_b) {
        if (!used[index_b]) {
          const StringDomain& string_domain_b = schema_.string_domain(index_b);
          if (IsSimilarStringDomain(string_domain_a, string_domain_b, config)) {
            similar.insert(string_domain_b.name());
          }
        }
      }
      if (!similar.empty()) {
        similar.insert(string_domain_a.name());
        result.push_back(similar);
      }
    }
  }
  return result;
}

std::vector<Path> Schema::GetAllRequiredFeatures(
    const Path& prefix,
    const tensorflow::protobuf::RepeatedPtrField<Feature>& features,
    const absl::optional<string>& environment) const {
  // This recursively walks through the structure. Sometimes, a feature is
  // not required because its parent is deprecated.
  std::vector<Path> result;
  for (const Feature& feature : features) {
    const Path child_path = prefix.GetChild(feature.name());
    if (IsExistenceRequired(feature, environment)) {
      result.push_back(child_path);
    }
    // There is an odd semantics here. Here, if a child feature is required,
    // but the parent is not, we could have an anomaly for the missing child
    // feature, even though it is the parent that is actually missing.
    if (!::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
      std::vector<Path> descendants = GetAllRequiredFeatures(
          child_path, feature.struct_domain().feature(), environment);
      result.insert(result.end(), descendants.begin(), descendants.end());
    }
  }
  return result;
}

std::vector<Path> Schema::GetMissingPaths(
    const DatasetStatsView& dataset_stats) {
  std::set<Path> paths_present;
  for (const FeatureStatsView& feature_stats_view : dataset_stats.features()) {
    paths_present.insert(feature_stats_view.GetPath());
  }
  std::vector<Path> paths_absent;

  for (const Path& path : GetAllRequiredFeatures(Path(), schema_.feature(),
                                                 dataset_stats.environment())) {
    if (!ContainsKey(paths_present, path)) {
      paths_absent.push_back(path);
    }
  }
  return paths_absent;
}

// TODO(b/148406484): currently, only looks at top-level features.
// Make this include lower level features as well.
// See also b/114757721.
std::map<string, std::set<Path>> Schema::EnumNameToPaths() const {
  std::map<string, std::set<Path>> result;
  for (const Feature& feature : schema_.feature()) {
    if (feature.has_domain()) {
      result[feature.domain()].insert(Path({feature.name()}));
    }
  }
  return result;
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const Updater& updater,
                      const absl::optional<std::set<Path>>& paths_to_consider) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity severity;

  for (const auto& feature_stats_view : dataset_stats.GetRootFeatures()) {
    TF_RETURN_IF_ERROR(UpdateRecursively(updater, feature_stats_view,
                                         paths_to_consider, &descriptions,
                                         &severity));
  }
  for (const Path& missing_path : GetMissingPaths(dataset_stats)) {
    if (ContainsPath(paths_to_consider, missing_path)) {
      DeprecateFeature(missing_path);
    }
  }
  return Status::OK();
}

// TODO(b/114757721): expose this.
Status Schema::GetRelatedEnums(const DatasetStatsView& dataset_stats,
                               FeatureStatisticsToProtoConfig* config) {
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Update(dataset_stats, *config));

  std::vector<std::set<string>> similar_enums =
      schema.SimilarEnumTypes(config->enums_similar_config());
  // Map the enum names to the paths.
  const std::map<string, std::set<Path>> enum_name_to_paths =
      schema.EnumNameToPaths();
  for (const std::set<string>& set : similar_enums) {
    if (set.empty()) {
      return Internal("Schema::SimilarEnumTypes returned an empty set.");
    }
    ColumnConstraint* column_constraint = config->add_column_constraint();
    for (const string& enum_name : set) {
      if (ContainsKey(enum_name_to_paths, enum_name)) {
        for (const auto& column : enum_name_to_paths.at(enum_name)) {
          *column_constraint->add_column_path() = column.AsProto();
        }
      }
    }
    // Choose the shortest name for the enum.
    string best_name = *set.begin();
    for (const string& current_name : set) {
      if (current_name.size() < best_name.size()) {
        best_name = current_name;
      }
    }
    *column_constraint->mutable_enum_name() = best_name;
  }
  return Status::OK();
}

tensorflow::metadata::v0::Schema Schema::GetSchema() const { return schema_; }

bool Schema::FeatureExists(const Path& path) {
  return GetExistingFeature(path) != nullptr ||
         GetExistingSparseFeature(path) != nullptr ||
         GetExistingWeightedFeature(path) != nullptr;
}

Feature* Schema::GetExistingFeature(const Path& path) {
  if (path.size() == 1) {
    return GetExistingFeatureHelper(path.last_step(),
                                    schema_.mutable_feature());
  } else {
    Path parent = path.GetParent();
    Feature* parent_feature = GetExistingFeature(parent);
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_feature());
  }
  return nullptr;
}

SparseFeature* Schema::GetExistingSparseFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() == 1) {
    return GetExistingSparseFeatureHelper(path.last_step(),
                                          schema_.mutable_sparse_feature());
  } else {
    Feature* parent_feature = GetExistingFeature(path.GetParent());
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingSparseFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_sparse_feature());
  }
}

WeightedFeature* Schema::GetExistingWeightedFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() != 1) {
    // Weighted features are always top-level features with single-step paths.
    return nullptr;
  }
  auto name = path.last_step();
  for (WeightedFeature& weighted_feature :
       *schema_.mutable_weighted_feature()) {
    if (weighted_feature.name() == name) {
      return &weighted_feature;
    }
  }
  return nullptr;
}

Feature* Schema::GetNewFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() > 1) {
    Path parent = path.GetParent();
    Feature* parent_feature = CHECK_NOTNULL(GetExistingFeature(parent));
    Feature* result = parent_feature->mutable_struct_domain()->add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  } else {
    Feature* result = schema_.add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  }
}

::tensorflow::metadata::v0::DatasetConstraints*
Schema::GetExistingDatasetConstraints() {
  if (schema_.has_dataset_constraints()) {
    return schema_.mutable_dataset_constraints();
  }
  return nullptr;
}

bool Schema::IsFeatureInEnvironment(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (environment) {
    if (ContainsValue(feature.in_environment(), *environment)) {
      return true;
    }
    if (ContainsValue(feature.not_in_environment(), *environment)) {
      return false;
    }
    if (ContainsValue(schema_.default_environment(), *environment)) {
      return true;
    }
    return false;
  }
  // If environment is not set, then the feature is considered in the
  // environment by default.
  return true;
}

bool Schema::IsExistenceRequired(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
    return false;
  }
  if (feature.presence().min_count() <= 0 &&
      feature.presence().min_fraction() <= 0.0) {
    return false;
  }
  // If a feature is in the environment, it is required.
  return IsFeatureInEnvironment(feature, environment);
}

// TODO(b/148406994): Handle missing FeatureType more elegantly, inferring it
// when necessary.
std::vector<Description> Schema::UpdateFeatureSelf(Feature* feature) {
  std::vector<Description> descriptions;
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
  if (!feature->has_name()) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::FEATURE_MISSING_NAME,
         absl::StrCat(
             "unspecified name (maybe meant to be the empty string): find "
             "name rather than deprecating.")});
    // Deprecating the feature is the only possible "fix" here.
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return descriptions;
  }

  if (!feature->has_type()) {
    if (feature->has_domain() || feature->has_string_domain()) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_MISSING_TYPE,
           absl::StrCat("unspecified type: inferring the type to "
                        "be BYTES, given the domain specified.")});
      feature->set_type(tensorflow::metadata::v0::BYTES);
    } else {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_MISSING_TYPE,
           absl::StrCat("unspecified type: determine the type and "
                        "set it, rather than deprecating.")});
      // Deprecating the feature is the only possible "fix" here.
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    }
  }
  if (feature->presence().min_fraction() < 0.0) {
    feature->mutable_presence()->clear_min_fraction();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
         "min_fraction should not be negative: clear is equal to zero"});
  }
  if (feature->presence().min_fraction() > 1.0) {
    feature->mutable_presence()->set_min_fraction(1.0);
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
         "min_fraction should not greater than 1"});
  }
  if (feature->value_count().min() < 0) {
    feature->mutable_value_count()->clear_min();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
         "ValueCount.min should not be negative"});
  }
  if (feature->value_count().has_max() &&
      feature->value_count().max() < feature->value_count().min()) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
         "ValueCount.max should not be less than min"});
    feature->mutable_value_count()->set_max(feature->value_count().min());
  }
  for (int i = 0; i < feature->value_counts().value_count_size(); ++i) {
    if (feature->value_counts().value_count(i).min() < 0) {
      feature->mutable_value_counts()->mutable_value_count(i)->clear_min();
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
           "ValueCounts.min should not be negative",
           absl::StrCat("ValueCounts.min at level ", i,
                        " should not be negative.")});
    }
    if (feature->value_counts().value_count(i).has_max() &&
        feature->value_counts().value_count(i).max() <
            feature->value_counts().value_count(i).min()) {
      feature->mutable_value_counts()->mutable_value_count(i)->set_max(
          feature->value_counts().value_count(i).min());
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
           "ValueCounts.max should not be less than min",
           absl::StrCat("ValueCounts.max at level ", i,
                        " should not be less than min.")});
    }
  }

  for (const auto& dim : feature->shape().dim()) {
    if (dim.size() <= 0) {
      feature->clear_shape();
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::INVALID_SCHEMA_SPECIFICATION,
           "Shape.dim.size must be a positive integer"});
      break;
    }
  }
  if (!ContainsKey(AllowedFeatureTypes(feature->domain_info_case()),
                   feature->type())) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::DOMAIN_INVALID_FOR_TYPE,
         "The domain does not match the type",
         absl::StrCat(
             "The domain \"", GetDomainInfoName(*feature),
             "\" does not match the type: ",
             tensorflow::metadata::v0::FeatureType_Name(feature->type()))});
    // Note that this clears the oneof field domain_info.
    ::tensorflow::data_validation::ClearDomain(feature);
  }

  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      if (GetExistingStringDomain(feature->domain()) == nullptr) {
        // Note that this clears the oneof field domain_info.
        feature->clear_domain();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_DOMAIN_SPECIFICATION,
             absl::StrCat("missing domain: ", feature->domain())});
      }
      break;
    case tensorflow::metadata::v0::Feature::kBoolDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints not supported for bool domains."});
      }
      UpdateBoolDomainSelf(feature->mutable_bool_domain());
      break;
    case tensorflow::metadata::v0::Feature::kIntDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints not supported for int domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints not supported for float domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      UpdateStringDomainSelf(feature->mutable_string_domain());
      break;
    case tensorflow::metadata::v0::Feature::kStructDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints not supported for struct domains."});
      }
      break;
    case Feature::kNaturalLanguageDomain:
    case Feature::kImageDomain:
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints not supported for semantic domains."});
      }
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 INVALID_SCHEMA_SPECIFICATION,
             "distribution constraints require domain or string domain."});
      }
      break;
    default:
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::INVALID_DOMAIN_SPECIFICATION,
           "internal issue: unknown domain_info type"});
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
  }

  return descriptions;
}

FeatureComparisonResult Schema::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  if (feature != nullptr &&
      FeatureHasComparator(*feature, FeatureComparatorType::SKEW)) {
    return UpdateFeatureComparatorDirect(
        feature_stats_view, FeatureComparatorType::SKEW,
        GetFeatureComparator(feature, FeatureComparatorType::SKEW));
  }
  return {};
}

void Schema::ClearStringDomain(const string& domain_name) {
  ClearStringDomainHelper(domain_name, schema_.mutable_feature());
  RemoveIf(schema_.mutable_string_domain(),
           [domain_name](const StringDomain* string_domain) {
             return (string_domain->name() == domain_name);
           });
}

void Schema::UpdateFeatureInternal(
    const Updater& updater, const FeatureStatsView& view, Feature* feature,
    std::vector<Description>* descriptions,
    absl::optional<tensorflow::metadata::v0::DriftSkewInfo>* drift_skew_info) {
  *descriptions = UpdateFeatureSelf(feature);

  // feature can be deprecated inside of UpdateFeatureSelf.
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return;
  }

  // This is to cover the rare case where there is actually no examples with
  // this feature, but there is still a dataset_stats object.
  const bool feature_missing = view.GetNumPresent() == 0;

  // If the feature is missing, but should be present, create an anomaly.
  // Otherwise, return without checking anything else.
  if (feature_missing) {
    if (IsExistenceRequired(*feature, view.environment())) {
      descriptions->push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_TYPE_NOT_PRESENT,
           "Column dropped", "The feature was not present in any examples."});
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return;
    } else {
      return;
    }
  }

  // If the feature is present in the dataset_stats and the schema, but is
  // excluded from the environment of the dataset_stats, then add it to that
  // environment.
  if (!feature_missing &&
      !IsFeatureInEnvironment(*feature, view.environment())) {
    // environment must be specified here, otherwise all features would be
    // present.
    CHECK(view.environment());
    const string view_environment = *view.environment();
    if (ContainsValue(feature->not_in_environment(), view_environment)) {
      RemoveIf(feature->mutable_not_in_environment(),
               [view_environment](const string* other) {
                 return *other == view_environment;
               });
    }
    // Even if we remove the feature from not in environment, we may need to
    // add it to in_environment.
    if (!IsFeatureInEnvironment(*feature, view.environment())) {
      feature->add_in_environment(view_environment);
    }
    descriptions->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN,
         "Column missing in environment",
         absl::StrCat("New column ", view.GetPath().Serialize(),
                      " found in data but not in the "
                      "environment ",
                      view_environment, " in the schema.")});
  }

  auto add_to_descriptions =
      [descriptions](const std::vector<Description>& other_descriptions) {
        descriptions->insert(descriptions->end(), other_descriptions.begin(),
                             other_descriptions.end());
      };

  // Clear domain_info if clear_field is set.
  // Either way, append descriptions.
  auto handle_update_summary = [feature, &add_to_descriptions](
                                   const UpdateSummary& update_summary) {
    add_to_descriptions(update_summary.descriptions);
    if (update_summary.clear_field) {
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
    }
  };

  if (feature->has_value_count() || feature->has_value_counts()) {
    add_to_descriptions(UpdateFeatureValueCounts(view, feature));
  }

  if (feature->has_shape()) {
    add_to_descriptions(
        UpdateFeatureShape(view, generate_legacy_feature_spec(), feature));
  }

  if (feature->has_presence()) {
    add_to_descriptions(::tensorflow::data_validation::UpdatePresence(
        view, feature->mutable_presence()));
  }

  if (view.GetFeatureType() != feature->type()) {
    // Basically, deprecate the feature. The rest is just getting a meaningful
    // message out.
    ::tensorflow::data_validation::DeprecateFeature(feature);
    const ::tensorflow::protobuf::EnumValueDescriptor* descriptor =
        tensorflow::metadata::v0::FeatureNameStatistics_Type_descriptor()
            ->FindValueByNumber(view.type());
    string data_type_name = (descriptor == nullptr)
                                ? absl::StrCat("unknown(", view.type(), ")")
                                : descriptor->name();

    const ::tensorflow::protobuf::EnumValueDescriptor* schema_descriptor =
        tensorflow::metadata::v0::FeatureType_descriptor()->FindValueByNumber(
            feature->type());
    string schema_type_name =
        (schema_descriptor == nullptr)
            ? absl::StrCat("unknown(", feature->type(), ")")
            : schema_descriptor->name();
    descriptions->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNEXPECTED_DATA_TYPE,
         "Unexpected data type",
         absl::StrCat("Expected data of type: ", schema_type_name, " but got ",
                      data_type_name)});
  }

  if (view.type() == FeatureNameStatistics::BYTES &&
      !ContainsKey(
          std::set<Feature::DomainInfoCase>(
              {Feature::DOMAIN_INFO_NOT_SET, Feature::kNaturalLanguageDomain,
               Feature::kImageDomain, Feature::kUrlDomain}),
          feature->domain_info_case())) {
    // Note that this clears the oneof field domain_info.
    ::tensorflow::data_validation::ClearDomain(feature);
    descriptions->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::DOMAIN_INVALID_FOR_TYPE,
         "Invalid domain type",
         absl::StrCat("Data is marked as BYTES with incompatible "
                      "domain_info: ",
                      feature->DebugString())});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain: {
      UpdateSummary update_summary =
          ::tensorflow::data_validation::UpdateStringDomain(
              updater, view,
              ::tensorflow::data_validation::GetMaxOffDomain(
                  feature->distribution_constraints()),
              CHECK_NOTNULL(GetExistingStringDomain(feature->domain())));

      add_to_descriptions(update_summary.descriptions);
      if (update_summary.clear_field) {
        // Note that this clears the oneof field domain_info.
        const string domain = feature->domain();
        ClearStringDomain(domain);
      }
    }

    break;
    case Feature::kBoolDomain:
      add_to_descriptions(
          ::tensorflow::data_validation::UpdateBoolDomain(view, feature));
      break;
    case Feature::kIntDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateIntDomain(
          view, feature->mutable_int_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateFloatDomain(
          view, feature->mutable_float_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateStringDomain(
          updater, view,
          ::tensorflow::data_validation::GetMaxOffDomain(
              feature->distribution_constraints()),
          feature->mutable_string_domain()));
      break;
    case Feature::kImageDomain:
      add_to_descriptions(
          ::tensorflow::data_validation::UpdateImageDomain(view, feature));
      break;
    case Feature::kNaturalLanguageDomain:
      add_to_descriptions(
          ::tensorflow::data_validation::UpdateNaturalLanguageDomain(view,
                                                                     feature));
      break;
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      // Updating existing semantic domains is not supported currently.
      break;
    case Feature::kStructDomain:
      // struct_domain is handled recursively.
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      // If the domain_info is not set, it is safe to try best-effort
      // semantic type update.
      if (BestEffortUpdateCustomDomain(view.custom_stats(), feature)) {
        descriptions->push_back(
            {tensorflow::metadata::v0::AnomalyInfo::SEMANTIC_DOMAIN_UPDATE,
             "Updated semantic domain",
             absl::StrCat("Updated semantic domain for feature: ",
                          feature->name())});
      }
      break;
    default:
      // In theory, default should have already been handled inside
      // UpdateFeatureSelf().
      LOG(ERROR) << "Internal error: unknown domains should be cleared inside "
                    "UpdateFeatureSelf.";
      DCHECK(false);
  }

  if (feature->has_unique_constraints()) {
    add_to_descriptions(UpdateUniqueConstraints(view, feature));
  }

  const std::vector<FeatureComparatorType> all_comparator_types = {
      FeatureComparatorType::DRIFT, FeatureComparatorType::SKEW};
  // Handle comparators here.
  for (const auto& comparator_type : all_comparator_types) {
    if (FeatureHasComparator(*feature, comparator_type)) {
      auto feature_comparison_result = UpdateFeatureComparatorDirect(
          view, comparator_type,
          GetFeatureComparator(feature, comparator_type));
      add_to_descriptions(feature_comparison_result.descriptions);
      if (!feature_comparison_result.measurements.empty()) {
        if (!drift_skew_info->has_value()) {
          drift_skew_info->emplace();
          *(*drift_skew_info)->mutable_path() = view.GetPath().AsProto();
        }
        if (comparator_type == FeatureComparatorType::DRIFT) {
          for (const auto& measurement :
               feature_comparison_result.measurements) {
            *(*drift_skew_info)->add_drift_measurements() = measurement;
          }
        } else if (comparator_type == FeatureComparatorType::SKEW) {
          for (const auto& measurement :
               feature_comparison_result.measurements) {
            *(*drift_skew_info)->add_skew_measurements() = measurement;
          }
        }
      }
    }
  }
  // Handle derived features for existing features.
  // If a feature has no derived source in the schema, but is derived in stats
  // then it should be marked derived in the schema.
  if (view.HasValidationDerivedSource() &&
      !feature->has_validation_derived_source()) {
    ::tensorflow::data_validation::MarkFeatureDerived(
        view.GetValidationDerivedSource(), feature);
    descriptions->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::DERIVED_FEATURE_INVALID_SOURCE,
         "Derived source not set in schema.",
         "Derived source not set in schema."});
  }
  // If a feature has a derived source in the schema but has an incorrectly
  // set lifecycle stage, set the stage.
  if (feature->has_validation_derived_source() &&
      (feature->lifecycle_stage() !=
           tensorflow::metadata::v0::VALIDATION_DERIVED ||
       feature->lifecycle_stage() != tensorflow::metadata::v0::DISABLED)) {
    feature->set_lifecycle_stage(tensorflow::metadata::v0::VALIDATION_DERIVED);
    descriptions->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::DERIVED_FEATURE_BAD_LIFECYCLE,
         "Derived feature has wrong lifecycle.",
         "Derived feature has wrong lifecycle."});
  }
}

std::vector<Description> Schema::UpdateSparseFeature(
    const FeatureStatsView& view, SparseFeature* sparse_feature) {
  std::vector<Description> descriptions;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the sparse_feature_stats_generator.
    if (stat_name == kMissingSparseValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature")});
    } else if (stat_name == kMissingSparseIndex) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        // This represents the index_feature name of this sparse feature.
        const string& index_feature_name = bucket.label();
        const int freq = bucket.sample_count();
        if (freq != 0) {
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_MISSING_INDEX,
               "Missing index feature",
               absl::StrCat("Found ", freq, " examples missing index feature: ",
                            index_feature_name)});
        }
      }
    } else if (stat_name == kMaxLengthDiff || stat_name == kMinLengthDiff) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        if (bucket.sample_count() != 0) {
          // This represents the index_feature name of this sparse feature.
          const string& index_feature_name = bucket.label();
          const int difference = bucket.sample_count();
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_LENGTH_MISMATCH,
               "Length mismatch between value and index feature",
               absl::StrCat(
                   "Mismatch between index feature: ", index_feature_name,
                   " and value column, with ", stat_name, " = ", difference)});
        }
      }
    }
    // Intentionally not generating anomalies for unknown custom stats for
    // forward compatibility.
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
  }
  return descriptions;
}

std::vector<Description> Schema::UpdateWeightedFeature(
    const FeatureStatsView& view, WeightedFeature* weighted_feature) {
  std::vector<Description> descriptions;
  int min_weight_length_diff = 0;
  int max_weight_length_diff = 0;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the weighted_feature_stats_generator.
    if (stat_name == kMissingWeightedValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature.")});
    } else if (stat_name == kMissingWeight && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_WEIGHT,
           "Missing weight feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing weight feature.")});
    } else if (stat_name == kMinWeightLengthDiff && custom_stat.num() != 0) {
      min_weight_length_diff = custom_stat.num();
    } else if (stat_name == kMaxWeightLengthDiff && custom_stat.num() != 0) {
      max_weight_length_diff = custom_stat.num();
    }
  }
  if (min_weight_length_diff != 0 || max_weight_length_diff != 0) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::
             WEIGHTED_FEATURE_LENGTH_MISMATCH,
         "Length mismatch between value and weight feature",
         absl::StrCat("Mismatch between weight and value feature with ",
                      kMinWeightLengthDiff, " = ", min_weight_length_diff,
                      " and ", kMaxWeightLengthDiff, " = ",
                      max_weight_length_diff, ".")});
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
  }
  return descriptions;
}

std::vector<Description> Schema::UpdateDatasetConstraints(
    const DatasetStatsView& dataset_stats_view) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::DatasetConstraints* dataset_constraints =
      GetExistingDatasetConstraints();
  if (dataset_constraints != nullptr) {
    const std::vector<DatasetComparatorType> all_comparator_types = {
        DatasetComparatorType::DRIFT, DatasetComparatorType::VERSION};
    for (const auto& comparator_type : all_comparator_types) {
      if (DatasetConstraintsHasComparator(*dataset_constraints,
                                          comparator_type)) {
        std::vector<Description> comparator_description_updates =
            UpdateNumExamplesComparatorDirect(
                dataset_stats_view, comparator_type,
                GetNumExamplesComparator(dataset_constraints, comparator_type));
        descriptions.insert(descriptions.end(),
                            comparator_description_updates.begin(),
                            comparator_description_updates.end());
      }
    }
    if (dataset_constraints->has_min_examples_count()) {
      std::vector<Description> min_examples_description_updates =
          UpdateExamplesCount(dataset_stats_view, dataset_constraints);
      descriptions.insert(descriptions.end(),
                          min_examples_description_updates.begin(),
                          min_examples_description_updates.end());
    }
  }
  return descriptions;
}

bool Schema::generate_legacy_feature_spec() const {
  // This field is not available in the OSS TFMD schema, so we use proto
  // reflection to get its value to avoid compilation errors.
  const auto* field_desc =
      schema_.GetDescriptor()->FindFieldByName("generate_legacy_feature_spec");
  if (!field_desc) return false;
  return schema_.GetReflection()->GetBool(schema_, field_desc);
}

}  // namespace data_validation
}  // namespace tensorflow
