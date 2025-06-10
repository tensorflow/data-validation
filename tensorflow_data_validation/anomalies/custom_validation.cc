/* Copyright 2022 Google LLC

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
#include "tensorflow_data_validation/anomalies/custom_validation.h"

#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/status_util.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/path.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {

using ::tensorflow::metadata::v0::Anomalies;
using ::tensorflow::metadata::v0::AnomalyInfo;
using ::tensorflow::metadata::v0::DatasetFeatureStatisticsList;
using ::tensorflow::metadata::v0::FeatureNameStatistics;

constexpr char kDefaultSlice[] = "All Examples";

// TODO(b/208881543): Update this type alias if representation of slice keys
// changes.
using SliceKey = std::string;

absl::flat_hash_map<SliceKey,
                    absl::flat_hash_map<std::string, FeatureNameStatistics>>
BuildNamedStatisticsMap(const DatasetFeatureStatisticsList& statistics) {
  absl::flat_hash_map<SliceKey,
                      absl::flat_hash_map<std::string, FeatureNameStatistics>>
      named_statistics;
  for (const auto& dataset : statistics.datasets()) {
    for (const auto& feature : dataset.features()) {
      const metadata::v0::Path& feature_path = feature.path();
      const std::string serialized_feature_path =
          Path(feature_path).Serialize();
      named_statistics[dataset.name()][serialized_feature_path] = feature;
    }
  }
  return named_statistics;
}

absl::Status GetFeatureStatistics(
    const absl::flat_hash_map<
        SliceKey, absl::flat_hash_map<std::string, FeatureNameStatistics>>&
        named_statistics,
    const std::string& dataset_name, const metadata::v0::Path& feature_path,
    FeatureNameStatistics* statistics) {
  auto named_feature_statistics = named_statistics.find(dataset_name);
  if (named_feature_statistics == named_statistics.end()) {
    if (dataset_name.empty()) {
      // If no matching stats are found and no dataset name is specified, use
      // the default slice.
      named_feature_statistics = named_statistics.find(kDefaultSlice);
    }
    if (named_feature_statistics == named_statistics.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dataset ", dataset_name,
          " specified in validation config not found in statistics."));
    }
  }
  const std::string serialized_feature_path = Path(feature_path).Serialize();
  const auto& feature_statistics =
      named_feature_statistics->second.find(serialized_feature_path);
  if (feature_statistics == named_feature_statistics->second.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Feature ", serialized_feature_path,
        " specified in validation config not found in statistics."));
  }
  *statistics = feature_statistics->second;
  return absl::OkStatus();
}

absl::Status MergeAnomalyInfos(const AnomalyInfo& anomaly_info,
                               const std::string& key,
                               AnomalyInfo* existing_anomaly_info) {
  if (Path(anomaly_info.path()).Compare(Path(existing_anomaly_info->path())) !=
      0) {
    return absl::AlreadyExistsError(
        absl::StrCat("Anomaly info map includes entries for ", key,
                     " which do not have the same path."));
  }
  if (anomaly_info.severity() != existing_anomaly_info->severity()) {
    existing_anomaly_info->set_severity(MaxSeverity(
        anomaly_info.severity(), existing_anomaly_info->severity()));
    LOG(WARNING)
        << "Anomaly entry for " << key
        << " has conflicting severities. The higher severity will be used.";
  }
  for (const auto& reason : anomaly_info.reason()) {
    AnomalyInfo::Reason* new_reason = existing_anomaly_info->add_reason();
    new_reason->CopyFrom(reason);
  }
  return absl::OkStatus();
}

// TODO(b/239095455): Populate top-level descriptions if needed for
// visualization.
absl::Status UpdateAnomalyResults(
    const metadata::v0::Path& path, const std::string& test_dataset,
    const absl::optional<std::string> base_dataset,
    const absl::optional<metadata::v0::Path> base_path,
    const Validation& validation, Anomalies* results) {
  AnomalyInfo anomaly_info;
  AnomalyInfo::Reason reason;
  reason.set_type(AnomalyInfo::CUSTOM_VALIDATION);
  reason.set_short_description(validation.description());
  std::string anomaly_source_description =
      absl::StrCat("Query: ", validation.sql_expression(), " Test dataset: ");
  if (test_dataset.empty()) {
    absl::StrAppend(&anomaly_source_description, "default slice");
  } else {
    absl::StrAppend(&anomaly_source_description, test_dataset);
  }
  if (base_dataset.has_value()) {
    absl::StrAppend(&anomaly_source_description,
                    " Base dataset: ", base_dataset.value(), " ");
  }
  if (base_path.has_value()) {
    absl::StrAppend(&anomaly_source_description,
                    "Base path: ", Path(base_path.value()).Serialize());
  }
  reason.set_description(absl::StrCat("Custom validation triggered anomaly. ",
                                      anomaly_source_description));
  anomaly_info.mutable_path()->CopyFrom(path);
  anomaly_info.set_severity(validation.severity());
  anomaly_info.add_reason()->CopyFrom(reason);
  const std::string& feature_name = Path(path).Serialize();
  const auto& insert_result =
      results->mutable_anomaly_info()->insert({feature_name, anomaly_info});
  // feature_name already existed in anomaly_info.
  if (insert_result.second == false) {
    AnomalyInfo existing_anomaly_info =
        results->anomaly_info().at(feature_name);
    TFDV_RETURN_IF_ERROR(
        MergeAnomalyInfos(anomaly_info, feature_name, &existing_anomaly_info));
    results->mutable_anomaly_info()
        ->at(feature_name)
        .CopyFrom(existing_anomaly_info);
  }
  return absl::OkStatus();
}

bool InCurrentEnvironment(Validation validation,
                          const absl::optional<string>& environment) {
  if (validation.in_environment_size() == 0) {
    return true;
  }
  if (environment.has_value()) {
    const std::string& environment_value = environment.value();
    for (const auto& each : validation.in_environment()) {
      if (each == environment_value) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

absl::Status CustomValidateStatistics(
    const metadata::v0::DatasetFeatureStatisticsList& test_statistics,
    const metadata::v0::DatasetFeatureStatisticsList* base_statistics,
    const CustomValidationConfig& validations,
    const absl::optional<std::string> environment,
    metadata::v0::Anomalies* result) {
  absl::flat_hash_map<SliceKey,
                      absl::flat_hash_map<std::string, FeatureNameStatistics>>
      named_test_statistics = BuildNamedStatisticsMap(test_statistics);
  for (const auto& feature_validation : validations.feature_validations()) {
    FeatureNameStatistics test_statistics;
    TFDV_RETURN_IF_ERROR(GetFeatureStatistics(
        named_test_statistics, feature_validation.dataset_name(),
        feature_validation.feature_path(), &test_statistics));
  }
  if (validations.feature_pair_validations_size() > 0) {
    if (base_statistics == nullptr) {
      return absl::InvalidArgumentError(
          "Feature pair validations are included in the CustomValidationConfig "
          "but base_statistics have not been specified.");
    }
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, FeatureNameStatistics>>
        named_base_statistics = BuildNamedStatisticsMap(*base_statistics);
    for (const auto& feature_pair_validation :
         validations.feature_pair_validations()) {
      FeatureNameStatistics test_statistics;
      FeatureNameStatistics base_statistics;
      TFDV_RETURN_IF_ERROR(GetFeatureStatistics(
          named_test_statistics, feature_pair_validation.dataset_name(),
          feature_pair_validation.feature_test_path(), &test_statistics));
      TFDV_RETURN_IF_ERROR(GetFeatureStatistics(
          named_base_statistics, feature_pair_validation.base_dataset_name(),
          feature_pair_validation.feature_base_path(), &base_statistics));
    }
  }
  return absl::OkStatus();
}

absl::Status CustomValidateStatisticsWithSerializedInputs(
    const std::string& serialized_test_statistics,
    const std::string& serialized_base_statistics,
    const std::string& serialized_validations,
    const std::string& serialized_environment,
    std::string* serialized_anomalies_proto) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics;
  metadata::v0::DatasetFeatureStatisticsList base_statistics;
  metadata::v0::DatasetFeatureStatisticsList* base_statistics_ptr = nullptr;
  if (!test_statistics.ParseFromString(serialized_test_statistics)) {
    return absl::InvalidArgumentError(
        "Failed to parse DatasetFeatureStatistics proto.");
  }
  if (!serialized_base_statistics.empty()) {
    if (!base_statistics.ParseFromString(serialized_base_statistics)) {
      return absl::InvalidArgumentError(
          "Failed to parse DatasetFeatureStatistics proto.");
    }
    base_statistics_ptr = &base_statistics;
  }
  CustomValidationConfig validations;
  if (!validations.ParseFromString(serialized_validations)) {
    return absl::InvalidArgumentError(
        "Failed to parse CustomValidationConfig proto.");
  }
  absl::optional<std::string> environment = absl::nullopt;
  if (!serialized_environment.empty()) {
    environment = serialized_environment;
  }
  metadata::v0::Anomalies anomalies;
  const absl::Status status =
      CustomValidateStatistics(test_statistics, base_statistics_ptr,
                               validations, environment, &anomalies);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to run custom validations: ", status.message()));
  }
  if (!anomalies.SerializeToString(serialized_anomalies_proto)) {
    return absl::InternalError(
        "Failed to serialize Anomalies output proto to string.");
  }
  return absl::OkStatus();
}

}  // namespace data_validation
}  // namespace tensorflow
