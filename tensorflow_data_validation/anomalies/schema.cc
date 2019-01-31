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
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/float_domain_util.h"
#include "tensorflow_data_validation/anomalies/int_domain_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_data_validation/anomalies/string_domain_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using ::absl::make_unique;
using ::absl::optional;
using ::tensorflow::Status;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::StringDomain;

constexpr char kTrainingServingSkew[] = "Training/Serving skew";

// Statistics generated.
// Keep the following constants consistent with the constants used
// when generating sparse feature statistics.
static constexpr char kMissingValue[] = "missing_value";
static constexpr char kMissingIndex[] = "missing_index";
static constexpr char kMaxLengthDiff[] = "max_length_diff";
static constexpr char kMinLengthDiff[] = "min_length_diff";

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
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::kIntDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::kFloatDomain:
      return {tensorflow::metadata::v0::FLOAT, tensorflow::metadata::v0::BYTES};
    case Feature::kStringDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kStructDomain:
      return {tensorflow::metadata::v0::STRUCT};
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

}  // namespace

Status Schema::Init(const tensorflow::metadata::v0::Schema& input) {
  if (!IsEmpty()) {
    return InvalidArgument("Schema is not empty when Init() called.");
  }
  schema_ = input;
  return Status::OK();
}

tensorflow::Status Schema::Update(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;

  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());

  SparseFeature* sparse_feature =
      GetExistingSparseFeature(feature_stats_view.GetPath());
  if (sparse_feature != nullptr &&
      !::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature)) {
    if (feature != nullptr &&
        !::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
      descriptions->push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_NAME_COLLISION,
           "Sparse feature name collision", "Sparse feature name collision"});
      ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      ::tensorflow::data_validation::DeprecateFeature(feature);
      *severity = tensorflow::metadata::v0::AnomalyInfo::ERROR;
      return Status::OK();
    } else {
      *descriptions = UpdateSparseFeature(feature_stats_view, sparse_feature);
      if (!descriptions->empty()) {
        *severity = tensorflow::metadata::v0::AnomalyInfo::ERROR;
      }
      return Status::OK();
    }
  }

  if (feature != nullptr) {
    *descriptions = UpdateFeatureInternal(updater, feature_stats_view, feature);
    if (!descriptions->empty()) {
      *severity = tensorflow::metadata::v0::AnomalyInfo::ERROR;
    }
    return Status::OK();
  } else {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN, "New column",
        "New column (column in data but not in schema)"};
    *descriptions = {description};
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
  TF_RETURN_IF_ERROR(
      Update(updater, feature_stats_view, descriptions, severity));
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
  return Status::OK();
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config) {
  return Update(dataset_stats, Updater(config), absl::nullopt);
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const Updater& updater,
                      const absl::optional<std::set<Path>>& paths_to_consider) {
  std::vector<Description> dummy_descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity dummy_severity;

  for (const auto& feature_stats_view : dataset_stats.GetRootFeatures()) {
    TF_RETURN_IF_ERROR(UpdateRecursively(updater, feature_stats_view,
                                         paths_to_consider, &dummy_descriptions,
                                         &dummy_severity));
  }
  for (const Path& missing_path : GetMissingPaths(dataset_stats)) {
    if (ContainsPath(paths_to_consider, missing_path)) {
      DeprecateFeature(missing_path);
    }
  }
  return Status::OK();
}

Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config,
                      const std::vector<Path>& paths_to_consider) {
  return Update(
      dataset_stats, Updater(config),
      std::set<Path>(paths_to_consider.begin(), paths_to_consider.end()));
}

Schema::Updater::Updater(const FeatureStatisticsToProtoConfig& config)
    : config_(config),
      columns_to_ignore_(config.column_to_ignore().begin(),
                         config.column_to_ignore().end()) {
  for (const ColumnConstraint& constraint : config.column_constraint()) {
    for (const string& column_name : constraint.column_name()) {
      grouped_enums_[column_name] = constraint.enum_name();
    }
  }
}

Status Schema::Updater::CreateColumn(
    const FeatureStatsView& feature_stats_view, Schema* schema,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  if (schema->GetExistingFeature(feature_stats_view.GetPath()) != nullptr) {
    return InvalidArgument("Schema already contains \"",
                           feature_stats_view.name(), "\".");
  }

  *severity = config_.new_features_are_warnings()
                  ? tensorflow::metadata::v0::AnomalyInfo::WARNING
                  : tensorflow::metadata::v0::AnomalyInfo::ERROR;

  Feature* feature = schema->GetNewFeature(feature_stats_view.GetPath());

  feature->set_type(feature_stats_view.GetFeatureType());
  InitValueCountAndPresence(feature_stats_view, feature);
  if (ContainsKey(columns_to_ignore_,
                  feature_stats_view.GetPath().Serialize())) {
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return Status::OK();
  }
  if (ContainsKey(grouped_enums_, feature_stats_view.name())) {
    const string& enum_name = grouped_enums_.at(feature_stats_view.name());
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
        schema->GetNewStringDomain(feature_stats_view.name());
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

// TODO(martinz): currently, only looks at top-level features.
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

// TODO(114757721): expose this.
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
          *column_constraint->add_column_name() = column.Serialize();
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
         GetExistingSparseFeature(path) != nullptr;
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

// TODO(martinz): Switch AnomalyInfo::Type from UNKNOWN_TYPE.
// TODO(martinz): Handle missing FeatureType more elegantly, inferring it
// when necessary.
std::vector<Description> Schema::UpdateFeatureSelf(Feature* feature) {
  std::vector<Description> descriptions;
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
  if (!feature->has_name()) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat(
             "unspecified name (maybe meant to be the empty string): find "
             "name rather than deprecating.")});
    // Deprecating the feature is the only possible "fix" here.
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return descriptions;
  }

  if (!feature->has_type()) {
    // TODO(martinz): UNKNOWN_TYPE means the anomaly type is unknown.

    if (feature->has_domain() || feature->has_string_domain()) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           absl::StrCat("unspecified type: inferring the type to "
                        "be BYTES, given the domain specified.")});
      feature->set_type(tensorflow::metadata::v0::BYTES);
    } else {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
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
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         "min_fraction should not be negative: clear is equal to zero"});
  }
  if (feature->presence().min_fraction() > 1.0) {
    feature->mutable_presence()->set_min_fraction(1.0);
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min_fraction should not greater than 1"});
  }
  if (feature->value_count().min() < 0) {
    feature->mutable_value_count()->clear_min();
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min should not be negative"});
  }
  if (feature->value_count().has_max() &&
      feature->value_count().max() < feature->value_count().min()) {
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "max should not be less than min"});
    feature->mutable_value_count()->set_max(feature->value_count().min());
  }
  if (!ContainsKey(AllowedFeatureTypes(feature->domain_info_case()),
                   feature->type())) {
    // Note that this clears the oneof field domain_info.
    ::tensorflow::data_validation::ClearDomain(feature);
    // TODO(martinz): Give more detail here.
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "The domain does not match the type"});
  }

  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      if (GetExistingStringDomain(feature->domain()) == nullptr) {
        // Note that this clears the oneof field domain_info.
        feature->clear_domain();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             absl::StrCat("missing domain: ", feature->domain())});
      }
      break;
    case tensorflow::metadata::v0::Feature::kBoolDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for bool domains."});
      }
      UpdateBoolDomainSelf(feature->mutable_bool_domain());
      break;
    case tensorflow::metadata::v0::Feature::kIntDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for int domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
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
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for struct domains."});
      }
      break;
    case tensorflow::metadata::v0::Domain::DOMAIN_INFO_NOT_SET:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints require domain or string domain."});
      }
      break;
    default:
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           "internal issue: unknown domain_info type"});
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
  }

  return descriptions;
}

std::vector<Description> Schema::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  if (feature != nullptr &&
      FeatureHasComparator(*feature, ComparatorType::SKEW)) {
    return UpdateFeatureComparatorDirect(
        feature_stats_view, ComparatorType::SKEW,
        GetFeatureComparator(feature, ComparatorType::SKEW));
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

std::vector<Description> Schema::UpdateFeatureInternal(
    const Updater& updater, const FeatureStatsView& view, Feature* feature) {
  std::vector<Description> descriptions = UpdateFeatureSelf(feature);

  // feature can be deprecated inside of UpdateFeatureSelf.
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }

  // This is to cover the rare case where there is actually no examples with
  // this feature, but there is still a dataset_stats object.
  // TODO(martinz): consider not returning a feature dataset_stats view for a
  // path if the number present are zero.
  const bool feature_missing = view.GetNumPresent() == 0;

  // If the feature is missing, but should be present, create an anomaly.
  // Otherwise, return without checking anything else.
  if (feature_missing) {
    if (IsExistenceRequired(*feature, view.environment())) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_TYPE_NOT_PRESENT,
           "Column dropped", "The feature was not present in any examples."});
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    } else {
      return descriptions;
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
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN,
         "Column missing in environment",
         absl::StrCat("New column ", view.name(),
                      " found in data but not in the "
                      "environment ",
                      view_environment, " in the schema.")});
  }

  auto add_to_descriptions =
      [&descriptions](const std::vector<Description>& other_descriptions) {
        descriptions.insert(descriptions.end(), other_descriptions.begin(),
                            other_descriptions.end());
      };

  // Clear domain_info if clear_field is set.
  // Either way, append descriptions.
  auto handle_update_summary = [&descriptions,
                                feature](const UpdateSummary& update_summary) {
    descriptions.insert(descriptions.end(), update_summary.descriptions.begin(),
                        update_summary.descriptions.end());
    if (update_summary.clear_field) {
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
    }
  };

  if (feature->has_value_count()) {
    add_to_descriptions(::tensorflow::data_validation::UpdateValueCount(
        view, feature->mutable_value_count()));
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
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat("Expected data of type: ", schema_type_name, " but got ",
                      data_type_name)});
  }

  if (view.type() == FeatureNameStatistics::BYTES &&
      feature->domain_info_case() !=
          tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET) {
    // Note that this clears the oneof field domain_info.
    ::tensorflow::data_validation::ClearDomain(feature);
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "Data is marked as BYTES that indicates the data "
                            " should not be analyzed: this is incompatible "
                            "with domain info."});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain: {
      UpdateSummary update_summary =
          ::tensorflow::data_validation::UpdateStringDomain(
              updater, view,
              ::tensorflow::data_validation::GetMaxOffDomain(
                  feature->distribution_constraints()),
              CHECK_NOTNULL(GetExistingStringDomain(feature->domain())));

      descriptions.insert(descriptions.end(),
                          update_summary.descriptions.begin(),
                          update_summary.descriptions.end());
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
    case Feature::kStructDomain:
      // struct_domain is handled recursively.
      break;
    case tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET:
      // Nothing to check here.
      break;
    default:
      // In theory, default should have already been handled inside
      // UpdateFeatureSelf().
      LOG(ERROR) << "Internal error: unknown domains should be cleared inside "
                    "UpdateFeatureSelf.";
      DCHECK(false);
  }

  const std::vector<ComparatorType> all_comparator_types = {
      ComparatorType::DRIFT, ComparatorType::SKEW};
  // Handle comparators here.
  for (const auto& comparator_type : all_comparator_types) {
    if (FeatureHasComparator(*feature, comparator_type)) {
      add_to_descriptions(UpdateFeatureComparatorDirect(
          view, comparator_type,
          GetFeatureComparator(feature, comparator_type)));
    }
  }

  return descriptions;
}

std::vector<Description> Schema::UpdateSparseFeature(
    const FeatureStatsView& view, SparseFeature* sparse_feature) {
  std::vector<Description> descriptions;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the sparse_feature_stats_generator.
    if (stat_name == kMissingValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature")});
    } else if (stat_name == kMissingIndex) {
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

}  // namespace data_validation
}  // namespace tensorflow
