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
    case Feature::DOMAIN_INFO_NOT_SET:
      ABSL_FALLTHROUGH_INTENDED;
    default:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::FLOAT,
              tensorflow::metadata::v0::BYTES};
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

  Feature* feature = GetExistingFeature(feature_stats_view.name());


  if (feature != nullptr) {
    *descriptions = UpdateFeatureInternal(feature_stats_view, feature);
    if (!descriptions->empty()) {
      *severity = tensorflow::metadata::v0::AnomalyInfo::ERROR;
    }
    return Status::OK();
  } else {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN, "New column",
        absl::StrCat("New column (column in data but not in schema): ",
                     feature_stats_view.name())};
    *descriptions = {description};
    return updater.CreateColumn(feature_stats_view, this, severity);
  }
  return Status::OK();
}

void Schema::DeprecateFeature(const string& feature_name) {
  ::tensorflow::data_validation::DeprecateFeature(
      CHECK_NOTNULL(GetExistingFeature(feature_name)));
}

Status Schema::Update(const DatasetStatsView& statistics,
                      const FeatureStatisticsToProtoConfig& config) {
  const Updater factory(config);
  for (const auto& feature_stats_view : statistics.features()) {
    std::vector<Description> dummy_descriptions;
    tensorflow::metadata::v0::AnomalyInfo::Severity dummy_severity;
    // As a side-effect, this may be creating string_domains.
    TF_RETURN_IF_ERROR(Update(factory, feature_stats_view, &dummy_descriptions,
                              &dummy_severity));
  }
  for (const string& missing_column : GetMissingColumns(statistics)) {
    DeprecateFeature(missing_column);
  }
  return Status::OK();
}

Status Schema::Update(const DatasetStatsView& statistics,
                      const FeatureStatisticsToProtoConfig& config,
                      const std::vector<string>& columns_to_consider) {
  Updater factory(config);
  for (const string& column_name : columns_to_consider) {
    absl::optional<FeatureStatsView> feature_stats_view =
        statistics.GetByName(column_name);
    if (feature_stats_view) {
      std::vector<Description> dummy_descriptions;
      tensorflow::metadata::v0::AnomalyInfo::Severity dummy_severity;
      TF_RETURN_IF_ERROR(Update(factory, *feature_stats_view,
                                &dummy_descriptions, &dummy_severity));
    } else {
      Feature* feature = GetExistingFeature(column_name);
      if (feature != nullptr) {
        // A column present in the schema but absent from the statistics.
        // Deprecate it if it is required to be there.
        if (IsExistenceRequired(*feature, statistics.environment())) {
          ::tensorflow::data_validation::DeprecateFeature(feature);
        }
      } else {
        // There is a column specified that is neither present in the schema,
        // nor present in the statistics. For now, we'll ignore this case.
        LOG(ERROR) << "Warning: requested update of " << column_name
                   << " that is neither in the statistics nor in the schema.";
      }
    }
  }
  return Status::OK();
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
  if (schema->GetExistingFeature(feature_stats_view.name()) != nullptr) {
    return InvalidArgument("Schema already contains \"",
                           feature_stats_view.name(), "\".");
  }

  *severity = config_.new_features_are_warnings()
                  ? tensorflow::metadata::v0::AnomalyInfo::WARNING
                  : tensorflow::metadata::v0::AnomalyInfo::ERROR;

  Feature* feature = schema->GetNewFeature(feature_stats_view.name());

  feature->set_type(feature_stats_view.GetFeatureType());
  InitValueCountAndPresence(feature_stats_view, feature);
  if (ContainsKey(columns_to_ignore_, feature_stats_view.name())) {
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return Status::OK();
  }
  if (ContainsKey(grouped_enums_, feature_stats_view.name())) {
    const string& enum_name = grouped_enums_.at(feature_stats_view.name());
    StringDomain* result = schema->GetExistingStringDomain(enum_name);
    if (result == nullptr) {
      result = schema->GetNewStringDomain(enum_name);
    }
    UpdateStringDomain(feature_stats_view, 0, result);
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
    UpdateStringDomain(feature_stats_view, 0, string_domain);
    *feature->mutable_domain() = string_domain->name();
    return Status::OK();
  } else {
    // No domain info for this field.
    return Status::OK();
  }
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

std::vector<string> Schema::GetMissingColumns(
    const DatasetStatsView& statistics) const {
  std::set<string> columns_present;
  for (const FeatureStatsView& feature_stats_view : statistics.features()) {
    columns_present.insert(feature_stats_view.name());
  }
  std::vector<string> columns_absent;

  for (const Feature& feature : schema_.feature()) {
    if (IsExistenceRequired(feature, statistics.environment()) &&
        !ContainsKey(columns_present, feature.name())) {
      columns_absent.push_back(feature.name());
    }
  }
  return columns_absent;
}

std::map<string, std::set<string>> Schema::EnumNameToColumns() const {
  std::map<string, std::set<string>> result;
  for (const Feature& feature : schema_.feature()) {
    if (feature.has_domain()) {
      result[feature.domain()].insert(feature.name());
    }
  }
  return result;
}

Status Schema::GetRelatedEnums(const DatasetStatsView& statistics,
                               FeatureStatisticsToProtoConfig* config) {
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Update(statistics, *config));

  std::vector<std::set<string>> similar_enums =
      schema.SimilarEnumTypes(config->enums_similar_config());
  // Map the enum names to the column names.
  const std::map<string, std::set<string>> enum_name_to_columns =
      schema.EnumNameToColumns();
  for (const std::set<string>& set : similar_enums) {
    if (set.empty()) {
      return Internal("Schema::SimilarEnumTypes returned an empty set.");
    }
    ColumnConstraint* column_constraint = config->add_column_constraint();
    for (const string& enum_name : set) {
      if (ContainsKey(enum_name_to_columns, enum_name)) {
        for (const auto& column : enum_name_to_columns.at(enum_name)) {
          *column_constraint->add_column_name() = column;
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

Feature* Schema::GetExistingFeature(const string& name) {
  for (Feature& feature : *schema_.mutable_feature()) {
    if (feature.name() == name) {
      return &feature;
    }
  }
  return nullptr;
}

SparseFeature* Schema::GetExistingSparseFeature(const string& name) {
  for (SparseFeature& sparse_feature : *schema_.mutable_sparse_feature()) {
    if (sparse_feature.name() == name) {
      return &sparse_feature;
    }
  }
  return nullptr;
}

Feature* Schema::GetNewFeature(const string& name) {
  Feature* result = schema_.add_feature();
  *result->mutable_name() = name;
  return result;
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
    feature->clear_domain_info();
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "The domain does not match the type"});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      if (GetExistingStringDomain(feature->domain()) == nullptr) {
        feature->clear_domain_info();
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
      feature->clear_domain_info();
  }

  return descriptions;
}

std::vector<Description> Schema::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  Feature* feature = GetExistingFeature(feature_stats_view.name());
  if (feature != nullptr &&
      FeatureHasComparator(*feature, ComparatorType::SKEW)) {
    return UpdateFeatureComparatorDirect(
        feature_stats_view, ComparatorType::SKEW,
        GetFeatureComparator(feature, ComparatorType::SKEW));
  }
  return {};
}

std::vector<Description> Schema::UpdateFeatureInternal(
    const FeatureStatsView& view, Feature* feature) {
  std::vector<Description> descriptions = UpdateFeatureSelf(feature);

  // feature can be deprecated inside of UpdateFeatureSelf.
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }

  // This is to cover the rare case where there is actually no examples with
  // this feature, but there is still a statistics object.
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

  // If the feature is present in the statistics and the schema, but is
  // excluded from the environment of the statistics, then add it to that
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
      feature->clear_domain_info();
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
    feature->clear_domain_info();
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "Data is marked as BYTES that indicates the data "
                            " should not be analyzed: this is incompatible "
                            "with domain info."});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateStringDomain(
          view,
          ::tensorflow::data_validation::GetMaxOffDomain(
              feature->distribution_constraints()),
          CHECK_NOTNULL(GetExistingStringDomain(feature->domain()))));
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
          view,
          ::tensorflow::data_validation::GetMaxOffDomain(
              feature->distribution_constraints()),
          feature->mutable_string_domain()));
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


}  // namespace data_validation
}  // namespace tensorflow
