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

// Validates a dataset by identifying anomalies in statistics computed over
// data with respect to a known dataset schema.
#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_

#include <string>
#include <vector>

#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"

#include "tensorflow_data_validation/anomalies/proto/validation_config.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

// Gets the default FeatureStatisticsToProtoConfig.
FeatureStatisticsToProtoConfig GetDefaultFeatureStatisticsToProtoConfig();

// Updates an existing schema to match the data characteristics in
// <feature_statistics>. An empty schema_to_update is a valid input schema.
// If an environment is specified, only check the fields in that environment.
// Otherwise, check all fields.
ABSL_DEPRECATED("Use UpdateSchema below instead.")
tensorflow::Status UpdateSchema(
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config,
    const ValidationConfig& validation_config,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::gtl::optional<string>& environment,
    tensorflow::metadata::v0::Schema* schema_to_update);

// Generates a schema which matches the data characteristics in the input
// feature statistics. This method will take as input the serialized statistics
// proto string and will output the serialized schema proto string.
// max_string_domain_size argument refers to the maximum size of the domain of
// a string feature in order to be interpreted as a categorical feature.
tensorflow::Status InferSchema(const string& feature_statistics_proto_string,
                               const int max_string_domain_size,
                               string* schema_proto_string);

// Validates the feature statistics in <feature_statistics> with respect to
// the <schema_proto> and returns a schema diff proto which captures the
// changes that need to be made to <schema_proto> to make the statistics
// conform to it. If a drift comparator is specified in the schema and the
// stats for the previous span are provided, then the schema diff may also
// contains changes that need to be made the drift comparators to make the
// <schema_proto> conform. If a skew comparator is specified in the schema and
// the serving stats are provided, the validation will detect if there exists
// distribution skew between current data and serving data.
// If an environment is specified, only validate the feature statistics of the
// fields in that environment. Otherwise, validate all fields.
tensorflow::Status ValidateFeatureStatistics(
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::metadata::v0::Schema& schema_proto,
    const tensorflow::gtl::optional<string>& environment,
    const tensorflow::gtl::optional<
        tensorflow::metadata::v0::DatasetFeatureStatistics>&
        prev_feature_statistics,
    const tensorflow::gtl::optional<
        tensorflow::metadata::v0::DatasetFeatureStatistics>&
        serving_feature_statistics,
    const ValidationConfig& validation_config,
    tensorflow::metadata::v0::Anomalies* result);

// Validates the feature statistics with respect to the schema and returns an
// anomalies proto. This method will take as input serialized proto strings
// and will output the serialized anomalies proto string.
// If a drift comparator is specified in the schema and the previous stats are
// provided, the validation will detect if there exists drift between current
// data and previous data. If a skew comparator is specified in the schema and
// the serving stats are provided, the validation will detect if there exists
// distribution skew between current data and serving data. If an environment
// is specified, only validate the feature statistics of the fields in that
// environment. Otherwise, validate all fields.
tensorflow::Status ValidateFeatureStatistics(
    const string& feature_statistics_proto_string,
    const string& schema_proto_string,
    const string& environment,
    const string& previous_statistics_proto_string,
    const string& serving_statistics_proto_string,
    string* anomalies_proto_string);

// Updates an existing schema to match the data characteristics in
// <feature_statistics>, but only on the paths_to_consider.
// An empty schema_to_update is a valid input schema.
// Also, one can pass *result as schema_to_update, as the code does not assume
// that these are separate objects.
// If ValidationConfig is updated, this function should be revisited.
// Note: paths_to_consider only currently supports paths that have exactly
// one step (eg path.size() == 1).
tensorflow::Status UpdateSchema(
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config,
    const tensorflow::metadata::v0::Schema& schema_to_update,
    const tensorflow::metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const tensorflow::gtl::optional<std::vector<Path>>& paths_to_consider,
    const tensorflow::gtl::optional<string>& environment,
    tensorflow::metadata::v0::Schema* result);

class FeatureStatisticsValidator {
 public:
  FeatureStatisticsValidator() = default;

  FeatureStatisticsValidator(const FeatureStatisticsValidator&) = delete;
  FeatureStatisticsValidator& operator=(const FeatureStatisticsValidator&) =
      delete;

  // Updates an existing schema to match the data characteristics in
  // <feature_statistics>. An empty schema_to_update is a valid input schema.
  ABSL_DEPRECATED("Use UpdateSchema above instead.")
  static tensorflow::Status UpdateSchema(
      const ValidationConfig& validation_config,
      const tensorflow::metadata::v0::DatasetFeatureStatistics&
          feature_statistics,
      tensorflow::metadata::v0::Schema* schema_to_update);

  // Updates an existing schema to match the data characteristics in
  // <feature_statistics>, but only on the columns_to_consider.
  // An empty schema_to_update is a valid input schema.
  // If ValidationConfig is updated, this function should be revisited.
  ABSL_DEPRECATED("Use UpdateSchema above instead.")
  static tensorflow::Status UpdateSchema(
      const tensorflow::metadata::v0::Schema& schema_to_update,
      const tensorflow::metadata::v0::DatasetFeatureStatistics&
          feature_statistics,
      const std::vector<string>& columns_to_consider,
      tensorflow::metadata::v0::Schema* result);

  // Validates the feature statistics in <feature_statistics> with respect to
  // the <schema_proto> and returns a schema diff proto which captures the
  // changes that need to be made to <schema_proto> to make the statistics
  // conform to it. If a drift comparator is specified in the schema and the
  // stats for the previous span are provided, then the schema diff may also
  // contains changes that need to be made the drift comparators to make the
  // <schema_proto> conform.
  static tensorflow::Status ValidateFeatureStatistics(
      const tensorflow::metadata::v0::DatasetFeatureStatistics&
          feature_statistics,
      const tensorflow::metadata::v0::Schema& schema_proto,
      const tensorflow::gtl::optional<string>& environment,
      const tensorflow::gtl::optional<
          tensorflow::metadata::v0::DatasetFeatureStatistics>&
          prev_feature_statistics,
      const ValidationConfig& validation_config,
      tensorflow::metadata::v0::Anomalies* result);
};

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_
