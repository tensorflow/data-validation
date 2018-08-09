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

#include "tensorflow_data_validation/anomalies/feature_statistics_validator.h"

#include <memory>
#include <string>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/schema.h"
#include "tensorflow_data_validation/anomalies/schema_anomalies.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

using tensorflow::metadata::v0::DatasetFeatureStatistics;

namespace tensorflow {
namespace data_validation {

const int64 kDefaultEnumThreshold = 400;

tensorflow::Status UpdateSchema(
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config,
    const ValidationConfig& validation_config,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::gtl::optional<string>& environment,
    tensorflow::metadata::v0::Schema* schema_to_update) {
  const absl::optional<string> maybe_environment =
      environment ? absl::optional<string>(*environment) : absl::nullopt;
  const bool by_weight =
      DatasetStatsView(feature_statistics).WeightedStatisticsExist();
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Init(*schema_to_update));
  TF_RETURN_IF_ERROR(schema.Update(
      DatasetStatsView(feature_statistics, by_weight, maybe_environment,
                       /* previous= */ nullptr,
                       /* serving= */ nullptr),
      feature_statistics_to_proto_config));
  *schema_to_update = schema.GetSchema();
  return tensorflow::Status::OK();
}

tensorflow::Status InferSchema(const string& feature_statistics_proto_string,
                               const int max_string_domain_size,
                               string* schema_proto_string) {
  tensorflow::metadata::v0::DatasetFeatureStatistics feature_statistics;
  if (!feature_statistics.ParseFromString(feature_statistics_proto_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse DatasetFeatureStatistics proto.");
  }
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(max_string_domain_size);
  tensorflow::metadata::v0::Schema schema;
  TF_RETURN_IF_ERROR(UpdateSchema(feature_statistics_to_proto_config,
                                  ValidationConfig(), feature_statistics,
                                  /* environment= */ tensorflow::gtl::nullopt,
                                  &schema));
  if (!schema.SerializeToString(schema_proto_string)) {
    return tensorflow::errors::Internal(
        "Could not serialize Schema output proto to string.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateFeatureStatistics(
    const ValidationConfig& validation_config,
    const tensorflow::metadata::v0::Schema& schema_proto,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::gtl::optional<
        tensorflow::metadata::v0::DatasetFeatureStatistics>&
        prev_feature_statistics,
    const tensorflow::gtl::optional<string>& environment,
    tensorflow::metadata::v0::Anomalies* result) {
  const absl::optional<string> maybe_environment =
      environment ? absl::optional<string>(*environment)
                  : absl::optional<string>();
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  feature_statistics_to_proto_config.set_new_features_are_warnings(
      validation_config.new_features_are_warnings());
  const bool by_weight =
      DatasetStatsView(feature_statistics).WeightedStatisticsExist();
  if (feature_statistics.num_examples() == 0) {
    *result->mutable_baseline() = schema_proto;
    result->set_data_missing(true);
  } else {
    SchemaAnomalies schema_anomalies(schema_proto);
    std::shared_ptr<DatasetStatsView> previous =
        (prev_feature_statistics)
            ? std::make_shared<DatasetStatsView>(
                  prev_feature_statistics.value(), by_weight, maybe_environment,
                  /* previous= */ nullptr,
                  /* serving= */ nullptr)
            : nullptr;

    TF_RETURN_IF_ERROR(schema_anomalies.FindChanges(
        DatasetStatsView(feature_statistics, by_weight, maybe_environment,
                         previous,
                         /* serving= */ nullptr),
        feature_statistics_to_proto_config));
    *result = schema_anomalies.GetSchemaDiff();
  }

  return tensorflow::Status::OK();
}

tensorflow::Status ValidateFeatureStatistics(
    const string& schema_proto_string,
    const string& feature_statistics_proto_string,
    string* anomalies_proto_string) {
  tensorflow::metadata::v0::Schema schema;
  tensorflow::metadata::v0::DatasetFeatureStatistics feature_statistics;
  if (!schema.ParseFromString(schema_proto_string)) {
    return tensorflow::errors::InvalidArgument("Failed to parse Schema proto.");
  }
  if (!feature_statistics.ParseFromString(feature_statistics_proto_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse DatasetFeatureStatistics proto.");
  }
  tensorflow::metadata::v0::Anomalies anomalies;
  TF_RETURN_IF_ERROR(ValidateFeatureStatistics(
      ValidationConfig(), schema, feature_statistics,
      /* prev_feature_statistics= */ tensorflow::gtl::nullopt,
      /* environment= */ ::tensorflow::gtl::nullopt, &anomalies));
  if (!anomalies.SerializeToString(anomalies_proto_string)) {
    return tensorflow::errors::Internal(
        "Could not serialize Anomalies output proto to string.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status FeatureStatisticsValidator::UpdateSchema(
    const ValidationConfig& validation_config,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    tensorflow::metadata::v0::Schema* schema_to_update) {
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  return ::tensorflow::data_validation::UpdateSchema(
      feature_statistics_to_proto_config,
      validation_config, feature_statistics,
      /* environment= */ tensorflow::gtl::nullopt, schema_to_update);
}

tensorflow::Status FeatureStatisticsValidator::UpdateSchema(
    const tensorflow::metadata::v0::Schema& schema_to_update,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const std::vector<string>& columns_to_consider,
    tensorflow::metadata::v0::Schema* result) {
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  const bool by_weight =
      DatasetStatsView(feature_statistics).WeightedStatisticsExist();
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Init(schema_to_update));
  TF_RETURN_IF_ERROR(
      schema.Update(DatasetStatsView(feature_statistics, by_weight),
                    feature_statistics_to_proto_config, columns_to_consider));
  *result = schema.GetSchema();
  return tensorflow::Status::OK();
}

tensorflow::Status FeatureStatisticsValidator::ValidateFeatureStatistics(
    const ValidationConfig& validation_config,
    const tensorflow::metadata::v0::Schema& schema_proto,
    const DatasetFeatureStatistics& feature_statistics,
    const tensorflow::gtl::optional<DatasetFeatureStatistics>&
        prev_feature_statistics,
    const tensorflow::gtl::optional<string>& environment,
    tensorflow::metadata::v0::Anomalies* result) {
  return ::tensorflow::data_validation::ValidateFeatureStatistics(
      validation_config, schema_proto, feature_statistics,
      prev_feature_statistics, environment, result);
}

}  // namespace data_validation
}  // namespace tensorflow
