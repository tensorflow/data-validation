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

#include <gtest/gtest.h>
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {

using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

TEST(CustomValidationTest, TestSingleStatisticsDefaultSliceValidation) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "All Examples"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: ERROR
                   description: 'Feature has too many zeros.'
                 }
                 validations {
                   sql_expression: 'feature.num_stats.max > 10'
                   severity: ERROR
                   description: 'Maximum value is too low.'
                 }
               })pb");
  metadata::v0::Anomalies expected_anomalies = ParseTextProtoOrDie<
      metadata::v0::Anomalies>(
      R"pb(anomaly_info {
             key: 'test_feature'
             value: {
               path { step: 'test_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Feature has too many zeros.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.num_zeros < 3 Test dataset: default slice'
               }
             }
           })pb");
  metadata::v0::Anomalies result;
  TF_CHECK_OK(CustomValidateStatistics(test_statistics,
                                       /*base_statistics=*/nullptr, validations,
                                       /*environment=*/absl::nullopt, &result));
  EXPECT_THAT(result, EqualsProto(expected_anomalies));
}

TEST(CustomValidationTest, TestSingleStatisticsSpecifiedSliceValidation) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "All Examples"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               }
               datasets {
                 name: "some_slice"
                 num_examples: 5
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 0 max: 5 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 dataset_name: 'some_slice'
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: ERROR
                   description: 'Feature has too many zeros.'
                 }
                 validations {
                   sql_expression: 'feature.num_stats.max > 10'
                   severity: ERROR
                   description: 'Maximum value is too low.'
                 }
               })pb");
  metadata::v0::Anomalies expected_anomalies = ParseTextProtoOrDie<
      metadata::v0::Anomalies>(
      R"pb(anomaly_info {
             key: 'test_feature'
             value: {
               path { step: 'test_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Maximum value is too low.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.max > 10 Test dataset: some_slice'
               }
             }
           })pb");
  metadata::v0::Anomalies result;
  TF_CHECK_OK(CustomValidateStatistics(test_statistics,
                                       /*base_statistics=*/nullptr, validations,
                                       /*environment=*/absl::nullopt, &result));
  EXPECT_THAT(result, EqualsProto(expected_anomalies));
}

TEST(CustomValidationTest, TestPairValidation) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "slice_1"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  metadata::v0::DatasetFeatureStatisticsList base_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "slice_2"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 1 max: 1 }
                 }
               })pb");
  CustomValidationConfig validations = ParseTextProtoOrDie<
      CustomValidationConfig>(
      R"pb(feature_pair_validations {
             dataset_name: 'slice_1'
             feature_test_path { step: 'test_feature' }
             base_dataset_name: 'slice_2'
             feature_base_path { step: 'test_feature' }
             validations {
               sql_expression: 'feature_test.num_stats.num_zeros < feature_base.num_stats.num_zeros'
               severity: ERROR
               description: 'Test feature has too many zeros.'
             }
           })pb");
  metadata::v0::Anomalies expected_anomalies = ParseTextProtoOrDie<
      metadata::v0::Anomalies>(
      R"pb(anomaly_info {
             key: 'test_feature'
             value: {
               path { step: 'test_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Test feature has too many zeros.'
                 description: 'Custom validation triggered anomaly. Query: feature_test.num_stats.num_zeros < feature_base.num_stats.num_zeros Test dataset: slice_1 Base dataset: slice_2 Base path: test_feature'
               }
             }
           })pb");
  metadata::v0::Anomalies result;
  TF_CHECK_OK(CustomValidateStatistics(test_statistics, &base_statistics,
                                       validations,
                                       /*environment=*/absl::nullopt, &result));
  EXPECT_THAT(result, EqualsProto(expected_anomalies));
}

TEST(CustomValidationTest, TestSpecifiedFeatureNotFound) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "All Examples"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 feature_path { step: 'other_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: ERROR
                   description: 'Feature has too many zeros.'
                 }
               })pb");
  metadata::v0::Anomalies result;
  auto error =
      CustomValidateStatistics(test_statistics,
                               /*base_statistics=*/nullptr, validations,
                               /*environment=*/absl::nullopt, &result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));
}

TEST(CustomValidationTest, TestSpecifiedDatasetNotFound) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "some_slice"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 dataset_name: 'other_slice'
                 feature_path { step: 'other_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: ERROR
                   description: 'Feature has too many zeros.'
                 }
               })pb");
  metadata::v0::Anomalies result;
  auto error =
      CustomValidateStatistics(test_statistics,
                               /*base_statistics=*/nullptr, validations,
                               /*environment=*/absl::nullopt, &result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));
}

TEST(CustomValidationTest, TestMultipleAnomalyReasons) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "All Examples"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: WARNING
                   description: 'Feature has too many zeros.'
                 }
                 validations {
                   sql_expression: 'feature.num_stats.max > 100'
                   severity: ERROR
                   description: 'Maximum value is too low.'
                 }
               })pb");
  metadata::v0::Anomalies expected_anomalies = ParseTextProtoOrDie<
      metadata::v0::Anomalies>(
      R"pb(anomaly_info {
             key: 'test_feature'
             value: {
               path { step: 'test_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Feature has too many zeros.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.num_zeros < 3 Test dataset: default slice'
               }
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Maximum value is too low.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.max > 100 Test dataset: default slice'
               }
             }
           })pb");
  metadata::v0::Anomalies result;
  TF_CHECK_OK(CustomValidateStatistics(test_statistics,
                                       /*base_statistics=*/nullptr, validations,
                                       /*environment=*/absl::nullopt, &result));
  EXPECT_THAT(result, EqualsProto(expected_anomalies));
}

TEST(CustomValidationTest, TestPairValidationsConfiguredButNoBaselineStats) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "some_slice"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 25 }
                 }
               })pb");
  CustomValidationConfig validations = ParseTextProtoOrDie<
      CustomValidationConfig>(
      R"pb(feature_pair_validations {
             dataset_name: 'some_slice'
             feature_test_path { step: 'test_feature' }
             base_dataset_name: 'slice_2'
             feature_base_path { step: 'test_feature' }
             validations {
               sql_expression: 'feature_test.num_stats.num_zeros < feature_base.num_stats.num_zeros'
               severity: ERROR
               description: 'Test feature has too many zeros.'
             }
           })pb");
  metadata::v0::Anomalies result;
  auto error =
      CustomValidateStatistics(test_statistics,
                               /*base_statistics=*/nullptr, validations,
                               /*environment=*/absl::nullopt, &result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));
}

TEST(CustomValidationTest, TestEnvironmentFiltering) {
  metadata::v0::DatasetFeatureStatisticsList test_statistics =
      ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
          R"pb(datasets {
                 name: "All Examples"
                 num_examples: 10
                 features {
                   path { step: 'test_feature' }
                   type: INT
                   num_stats { num_zeros: 5 max: 1 }
                 }
               })pb");
  CustomValidationConfig validations =
      ParseTextProtoOrDie<CustomValidationConfig>(
          R"pb(feature_validations {
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.num_zeros < 3'
                   severity: ERROR
                   description: 'Feature has too many zeros.'
                 }
               }
               feature_validations {
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.max > 2'
                   severity: ERROR
                   description: 'Maximum value is wrong.'
                   in_environment: 'not_this_environment'
                 }
               }
               feature_validations {
                 feature_path { step: 'test_feature' }
                 validations {
                   sql_expression: 'feature.num_stats.max > 10'
                   severity: ERROR
                   description: 'Maximum value is too low.'
                   in_environment: 'some_environment'
                 }
               })pb");
  metadata::v0::Anomalies expected_anomalies = ParseTextProtoOrDie<
      metadata::v0::Anomalies>(
      R"pb(anomaly_info {
             key: 'test_feature'
             value: {
               path { step: 'test_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Feature has too many zeros.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.num_zeros < 3 Test dataset: default slice'
               }
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Maximum value is too low.'
                 description: 'Custom validation triggered anomaly. Query: feature.num_stats.max > 10 Test dataset: default slice'
               }
             }
           })pb");
  metadata::v0::Anomalies result;
  TF_CHECK_OK(CustomValidateStatistics(test_statistics,
                                       /*base_statistics=*/nullptr, validations,
                                       "some_environment", &result));
  EXPECT_THAT(result, EqualsProto(expected_anomalies));
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
