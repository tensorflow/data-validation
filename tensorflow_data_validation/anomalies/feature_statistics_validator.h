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
// TODO(b/113284855): cleanup the multiple APIs.
#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_

#include <set>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/features_needed.h"
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


// Generates a schema which matches the data characteristics in the input
// feature statistics. This method will take as input the serialized statistics
// proto string and will output the serialized schema proto string.
// `max_string_domain_size` argument refers to the maximum size of the domain of
// a string feature in order to be interpreted as a categorical feature.
// if `infer_feature_shape` is true, then it will try inferring a fixed shape
// from a feature's statistics.
Status InferSchema(const string& feature_statistics_proto_string,
                   int max_string_domain_size, bool infer_feature_shape,
                   string* schema_proto_string);

// Updates the input schema to match the data characteristics in the input
// feature statistics. This method will take as input the serialized input
// schema proto string, the serialized statistics proto string and will output
// the serialized updated schema proto string.
// max_string_domain_size argument refers to the maximum size of the domain of
// a string feature in order to be interpreted as a categorical feature.
Status UpdateSchema(const string& schema_proto_string,
                    const string& feature_statistics_proto_string,
                    const int max_string_domain_size,
                    string* output_schema_proto_string);

// Validates the statistics in <feature_statistics> with respect to the
// <schema_proto> and returns a schema diff proto which captures the
// changes that need to be made to <schema_proto> to make the statistics
// conform to it. If a drift comparator is specified in the schema and the
// stats for the previous span are provided, then the schema diff may also
// contain changes that need to be made to the drift comparators to make the
// <schema_proto> conform. If a skew comparator is specified in the schema and
// the serving stats are provided, the validation will detect if there exists
// distribution skew between current data and serving data. If a dataset-level
// num examples comparator is specified in the schema and the relevant previous
// stats (span or version) are provided, then the validation will detect if
// there are changes in num examples beyond the specified thresholds. If an
// environment is specified, only validate the feature statistics of the fields
// in that environment. Otherwise, validate all fields.
Status ValidateFeatureStatistics(
    const metadata::v0::DatasetFeatureStatistics& feature_statistics,
    const metadata::v0::Schema& schema_proto,
    const absl::optional<string>& environment,
    const absl::optional<metadata::v0::DatasetFeatureStatistics>&
        prev_span_feature_statistics,
    const absl::optional<metadata::v0::DatasetFeatureStatistics>&
        serving_feature_statistics,
    const absl::optional<metadata::v0::DatasetFeatureStatistics>&
        prev_version_feature_statistics,
    const absl::optional<FeaturesNeeded>& features_needed,
    const ValidationConfig& validation_config, bool enable_diff_regions,
    metadata::v0::Anomalies* result);

// Similar to the above, but takes all the proto parameters as serialized
// strings. This method is called by the Python code using PyBind11.
Status ValidateFeatureStatisticsWithSerializedInputs(
    const string& feature_statistics_proto_string,
    const string& schema_proto_string, const string& environment,
    const string& previous_span_statistics_proto_string,
    const string& serving_statistics_proto_string,
    const string& previous_version_statistics_proto_string,
    const string& features_needed_string,
    const string& validation_config_string, const bool enable_diff_regions,
    string* anomalies_proto_string);

// Updates an existing schema to match the data characteristics in
// <feature_statistics>, but only on the paths_to_consider.
// An empty schema_to_update is a valid input schema.
// Also, one can pass *result as schema_to_update, as the code does not assume
// that these are separate objects.
// If ValidationConfig is updated, this function should be revisited.
// Note: paths_to_consider only currently supports paths that have exactly
// one step (eg path.size() == 1).
Status UpdateSchema(
    const FeatureStatisticsToProtoConfig& feature_statistics_to_proto_config,
    const metadata::v0::Schema& schema_to_update,
    const metadata::v0::DatasetFeatureStatistics&
        feature_statistics,
    const absl::optional<std::vector<Path>>& paths_to_consider,
    const absl::optional<string>& environment,
    metadata::v0::Schema* result);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURE_STATISTICS_VALIDATOR_H_
