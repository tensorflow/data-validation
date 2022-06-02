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
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

using tensorflow::metadata::v0::DatasetFeatureStatistics;

namespace tensorflow {
namespace data_validation {

namespace {
const int64 kDefaultEnumThreshold = 400;
auto* anomaly_type_counts = tensorflow::monitoring::Counter<1>::New(
    "/tfx/example_validation/anomaly_type_counts", "Anomaly types found.",
    "anomaly_type");
}

FeatureStatisticsToProtoConfig GetDefaultFeatureStatisticsToProtoConfig() {
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  return feature_statistics_to_proto_config;
}


tensorflow::Status InferSchema(const string& feature_statistics_proto_string,
                               const int max_string_domain_size,
                               const bool infer_feature_shape,
                               string* schema_proto_string) {
  tensorflow::metadata::v0::DatasetFeatureStatistics feature_statistics;
  if (!feature_statistics.ParseFromString(feature_statistics_proto_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse DatasetFeatureStatistics proto.");
  }
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(max_string_domain_size);
  feature_statistics_to_proto_config.set_infer_feature_shape(
      infer_feature_shape);
  tensorflow::metadata::v0::Schema schema;
  TF_RETURN_IF_ERROR(
      UpdateSchema(feature_statistics_to_proto_config,
                   schema, feature_statistics,
                   /* paths_to_consider= */ gtl::nullopt,
                   /* environment= */ gtl::nullopt, &schema));
  if (!schema.SerializeToString(schema_proto_string)) {
    return tensorflow::errors::Internal(
        "Could not serialize Schema output proto to string.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status UpdateSchema(const string& schema_proto_string,
                                const string& feature_statistics_proto_string,
                                const int max_string_domain_size,
                                string* output_schema_proto_string) {
  tensorflow::metadata::v0::Schema schema;
  if (!schema.ParseFromString(schema_proto_string)) {
    return tensorflow::errors::InvalidArgument("Failed to parse Schema proto.");
  }
  tensorflow::metadata::v0::DatasetFeatureStatistics feature_statistics;
  if (!feature_statistics.ParseFromString(feature_statistics_proto_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse DatasetFeatureStatistics proto.");
  }
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(max_string_domain_size);
  tensorflow::metadata::v0::Schema output_schema;
  TF_RETURN_IF_ERROR(
      UpdateSchema(feature_statistics_to_proto_config,
                   schema, feature_statistics,
                   /* paths_to_consider= */ gtl::nullopt,
                   /* environment= */ gtl::nullopt, &output_schema));
  if (!output_schema.SerializeToString(output_schema_proto_string)) {
    return tensorflow::errors::Internal(
        "Could not serialize Schema output proto to string.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateFeatureStatistics(
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::metadata::v0::Schema& schema_proto,
    const absl::optional<string>& environment,
    const absl::optional<tensorflow::metadata::v0::DatasetFeatureStatistics>&
        prev_span_feature_statistics,
    const absl::optional<tensorflow::metadata::v0::DatasetFeatureStatistics>&
        serving_feature_statistics,
    const absl::optional<metadata::v0::DatasetFeatureStatistics>&
        prev_version_feature_statistics,
    const absl::optional<FeaturesNeeded>& features_needed,
    const ValidationConfig& validation_config, bool enable_diff_regions,
    tensorflow::metadata::v0::Anomalies* result) {
  // TODO(b/113295423): Clean up the optional conversions.
  const absl::optional<string> maybe_environment =
      environment ? absl::optional<string>(*environment)
                  : absl::optional<string>();
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  feature_statistics_to_proto_config.set_new_features_are_warnings(
      validation_config.new_features_are_warnings());
  *feature_statistics_to_proto_config.mutable_severity_overrides() =
      validation_config.severity_overrides();

  const bool by_weight =
      DatasetStatsView(feature_statistics).WeightedStatisticsExist();
  if (feature_statistics.num_examples() == 0) {
    *result->mutable_baseline() = schema_proto;
    result->set_data_missing(true);
  } else {
    SchemaAnomalies schema_anomalies(schema_proto);
    std::shared_ptr<DatasetStatsView> previous_span =
        (prev_span_feature_statistics)
            ? std::make_shared<DatasetStatsView>(
                  prev_span_feature_statistics.value(), by_weight,
                  maybe_environment,
                  /* previous_span= */ nullptr,
                  /* serving= */ nullptr,
                  /* previous_version= */ nullptr)
            : nullptr;

    std::shared_ptr<DatasetStatsView> serving =
        (serving_feature_statistics) ? std::make_shared<DatasetStatsView>(
                                           serving_feature_statistics.value(),
                                           by_weight, maybe_environment,
                                           /* previous_span= */ nullptr,
                                           /* serving= */ nullptr,
                                           /* previous_version= */ nullptr)
                                     : nullptr;

    std::shared_ptr<DatasetStatsView> previous_version =
        (prev_version_feature_statistics)
            ? std::make_shared<DatasetStatsView>(
                  prev_version_feature_statistics.value(), by_weight,
                  maybe_environment,
                  /* previous_span= */ nullptr,
                  /* serving= */ nullptr,
                  /* previous_version= */ nullptr)
            : nullptr;

    const DatasetStatsView training =
        DatasetStatsView(feature_statistics, by_weight, maybe_environment,
                         previous_span, serving, previous_version);
    TF_RETURN_IF_ERROR(
        schema_anomalies.FindChanges(training, features_needed,
                                     feature_statistics_to_proto_config));
    *result = schema_anomalies.GetSchemaDiff(enable_diff_regions);
  }

  for (const auto& anomaly : result->anomaly_info()) {
    for (const auto& reason : anomaly.second.reason()) {
      anomaly_type_counts
          ->GetCell(metadata::v0::AnomalyInfo::Type_Name(reason.type()))
          ->IncrementBy(1);
    }
  }
  if (result->has_dataset_anomaly_info()) {
    for (const auto& reason : result->dataset_anomaly_info().reason()) {
      anomaly_type_counts
          ->GetCell(metadata::v0::AnomalyInfo::Type_Name(reason.type()))
          ->IncrementBy(1);
    }
  }
  if (result->data_missing()) {
    anomaly_type_counts->GetCell("data_missing")->IncrementBy(1);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status ValidateFeatureStatisticsWithSerializedInputs(
    const string& feature_statistics_proto_string,
    const string& schema_proto_string, const string& environment,
    const string& previous_span_statistics_proto_string,
    const string& serving_statistics_proto_string,
    const string& previous_version_statistics_proto_string,
    const string& features_needed_string,
    const string& validation_config_string, const bool enable_diff_regions,
    string* anomalies_proto_string) {
  tensorflow::metadata::v0::Schema schema;
  if (!schema.ParseFromString(schema_proto_string)) {
    return tensorflow::errors::InvalidArgument("Failed to parse Schema proto.");
  }

  tensorflow::metadata::v0::DatasetFeatureStatistics feature_statistics;
  if (!feature_statistics.ParseFromString(feature_statistics_proto_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse DatasetFeatureStatistics proto.");
  }

  absl::optional<tensorflow::metadata::v0::DatasetFeatureStatistics>
      previous_span_statistics = tensorflow::gtl::nullopt;
  if (!previous_span_statistics_proto_string.empty()) {
    tensorflow::metadata::v0::DatasetFeatureStatistics tmp_stats;
    if (!tmp_stats.ParseFromString(previous_span_statistics_proto_string)) {
      return tensorflow::errors::InvalidArgument(
          "Failed to parse DatasetFeatureStatistics proto.");
    }
    previous_span_statistics = tmp_stats;
  }

  absl::optional<tensorflow::metadata::v0::DatasetFeatureStatistics>
      serving_statistics = tensorflow::gtl::nullopt;
  if (!serving_statistics_proto_string.empty()) {
    tensorflow::metadata::v0::DatasetFeatureStatistics tmp_stats;
    if (!tmp_stats.ParseFromString(serving_statistics_proto_string)) {
      return tensorflow::errors::InvalidArgument(
          "Failed to parse DatasetFeatureStatistics proto.");
    }
    serving_statistics = tmp_stats;
  }

  absl::optional<tensorflow::metadata::v0::DatasetFeatureStatistics>
      previous_version_statistics = tensorflow::gtl::nullopt;
  if (!previous_version_statistics_proto_string.empty()) {
    tensorflow::metadata::v0::DatasetFeatureStatistics tmp_stats;
    if (!tmp_stats.ParseFromString(previous_version_statistics_proto_string)) {
      return tensorflow::errors::InvalidArgument(
          "Failed to parse DatasetFeatureStatistics proto.");
    }
    previous_version_statistics = tmp_stats;
  }

  absl::optional<string> may_be_environment =
      tensorflow::gtl::nullopt;
  if (!environment.empty()) {
    may_be_environment = environment;
  }

  absl::optional<FeaturesNeeded> features_needed = gtl::nullopt;
  if (!features_needed_string.empty()) {
    FeaturesNeededProto parsed_proto;
    if (!parsed_proto.ParseFromString(features_needed_string)) {
      return tensorflow::errors::InvalidArgument(
          "Failed to parse FeaturesNeeded");
    }

    FeaturesNeeded parsed_feature_needed;
    TF_RETURN_IF_ERROR(
        FromFeaturesNeededProto(parsed_proto, &parsed_feature_needed));
    if (!parsed_feature_needed.empty()) {
      features_needed = parsed_feature_needed;
    }
  }

  data_validation::ValidationConfig validation_config;
  if (!validation_config.ParseFromString(validation_config_string)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse ValidationConfig");
  }

  tensorflow::metadata::v0::Anomalies anomalies;
  TF_RETURN_IF_ERROR(ValidateFeatureStatistics(
      feature_statistics, schema, may_be_environment, previous_span_statistics,
      serving_statistics, previous_version_statistics, features_needed,
      validation_config, enable_diff_regions, &anomalies));

  if (!anomalies.SerializeToString(anomalies_proto_string)) {
    return tensorflow::errors::Internal(
        "Could not serialize Anomalies output proto to string.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status UpdateSchema(
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config,
    const tensorflow::metadata::v0::Schema& schema_to_update,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const absl::optional<std::vector<Path>>& paths_to_consider,
    const absl::optional<string>& environment,
    tensorflow::metadata::v0::Schema* result) {
  // TODO(b/112762449): Add full support for multi-step
  // paths from paths_to_consider.
  // In the meantime, there are two ways you could imagine
  // translating to a string:
  // 1: As below, using the last step. Since all the paths at present are just
  //    a single step, this won't break anything. Moreover, since in
  //    schema_updater.cc, we are populating the string into last_step, this
  //    is the same as the existing implementation.
  // 2: As a serialized path. This will break current designs, as the Schema
  //    Updater will incorrectly handle paths like Path({"foo.bar"}).
  // TODO(b/113295423): Clean up the optional conversions.
  const absl::optional<string> maybe_environment =
      environment ? absl::optional<string>(*environment) : absl::nullopt;

  const bool by_weight =
      DatasetStatsView(feature_statistics).WeightedStatisticsExist();
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Init(schema_to_update));
  if (paths_to_consider) {
    TF_RETURN_IF_ERROR(schema.Update(
        DatasetStatsView(feature_statistics, by_weight, maybe_environment,
                         /* previous_span= */ nullptr,
                         /* serving= */ nullptr,
                         /* previous_version= */ nullptr),
        feature_statistics_to_proto_config, *paths_to_consider));
  } else {
    TF_RETURN_IF_ERROR(schema.Update(
        DatasetStatsView(feature_statistics, by_weight, maybe_environment,
                         /* previous_span= */ nullptr,
                         /* serving= */ nullptr,
                         /* previous_version= */ nullptr),
        feature_statistics_to_proto_config));
  }
  *result = schema.GetSchema();
  return tensorflow::Status::OK();
}

}  // namespace data_validation
}  // namespace tensorflow
