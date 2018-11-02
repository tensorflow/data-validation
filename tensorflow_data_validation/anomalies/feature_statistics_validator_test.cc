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

#include <map>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/proto/validation_config.pb.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::Schema;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

void TestSchemaUpdate(const ValidationConfig& config,
                      const DatasetFeatureStatistics& statistics,
                      const Schema& old_schema, const Schema& expected) {
  Schema result;
  TF_ASSERT_OK(UpdateSchema(
      GetDefaultFeatureStatisticsToProtoConfig(), old_schema, statistics,
      /*paths_to_consider=*/ tensorflow::gtl::nullopt,
      /*environment=*/ tensorflow::gtl::nullopt, &result));
  EXPECT_THAT(result, EqualsProto(expected));
}

bool operator==(const tensorflow::metadata::v0::AnomalyInfo::Reason& a,
                const tensorflow::metadata::v0::AnomalyInfo::Reason& b) {
  return a.type() == b.type() &&
         a.short_description() == b.short_description() &&
         a.description() == b.description();
}

void TestFeatureStatisticsValidator(
    const Schema& old_schema, const ValidationConfig& validation_config,
    const DatasetFeatureStatistics& feature_statistics,
    const tensorflow::gtl::optional<DatasetFeatureStatistics>&
        prev_feature_statistics,
    const tensorflow::gtl::optional<string>& environment,
    const tensorflow::gtl::optional<FeaturesNeeded>& features_needed,
    const std::map<string, testing::ExpectedAnomalyInfo>& expected_anomalies) {
  tensorflow::metadata::v0::Anomalies result;
  TF_CHECK_OK(ValidateFeatureStatistics(
      feature_statistics, old_schema, environment, prev_feature_statistics,
      /*serving_feature_statistics=*/gtl::nullopt, features_needed,
      validation_config, &result));
  TestAnomalies(result, old_schema, expected_anomalies);
}

TEST(FeatureStatisticsValidatorTest, EndToEnd) {
  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
    string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
    feature {
      name: "annotated_enum"
      value_count: { min: 1 max: 1 }
      presence: { min_count: 1 }
      type: BYTES
      domain: "MyAloneEnum"
    }
    feature {
      name: "missing_column"
      type: BYTES
    }
    feature {
      name: "ignore_this"
      lifecycle_stage: DEPRECATED
      value_count: { min: 1 }
      presence: { min_count: 1 }
      type: BYTES
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 1 } }
          }
        },
        features: {
          name: 'missing_column'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 1000
            }
          }
        })");

  std::map<string, testing::ExpectedAnomalyInfo> anomalies;
  // In this case, there are two anomalies.
  anomalies["annotated_enum"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "MyAloneEnum"
      presence { min_count: 1 }
    }
    feature {
      name: "missing_column"
      type: BYTES
    }
    feature {
      name: "ignore_this"
      lifecycle_stage: DEPRECATED
      value_count { min: 1 }
      type: BYTES
      presence { min_count: 1 }
    }
    string_domain {
      name: "MyAloneEnum"
      value: "A"
      value: "B"
      value: "C"
      value: "D"
    })");
  anomalies["annotated_enum"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    description: "Examples contain values missing from the schema: D (?). "
    severity: ERROR
    short_description: "Unexpected string values"
    reason {
      type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
      short_description: "Unexpected string values"
      description: "Examples contain values missing from the schema: D (?). "
    }
    path { step: "annotated_enum" }
  )");

  TestFeatureStatisticsValidator(
      schema, ValidationConfig(), statistics,
      /*prev_feature_statistics=*/tensorflow::gtl::nullopt,
      /*environment=*/gtl::nullopt,
      /*features_needed=*/gtl::nullopt, anomalies);
}

TEST(FeatureStatisticsValidatorTest, MissingFeatureAndEnvironments) {
  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
    default_environment: "TRAINING"
    default_environment: "SERVING"
    feature {
      name: "label"
      not_in_environment: "SERVING"
      value_count: { min: 1 max: 1 }
      presence: { min_count: 1 }
      type: BYTES
    }
    feature {
      name: "feature"
      value_count: { min: 1 max: 1 }
      presence: { min_count: 1 }
      type: BYTES
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'feature'
          type: STRING
          string_stats: {
            common_stats: {
              num_non_missing: 1000
              min_num_values: 1
              max_num_values: 1
            }
            unique: 3
          }
        })");

  std::map<string, testing::ExpectedAnomalyInfo> anomalies;
  anomalies["label"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    default_environment: "TRAINING"
    default_environment: "SERVING"
    feature {
      name: "label"
      not_in_environment: "SERVING"
      lifecycle_stage: DEPRECATED
      value_count: { min: 1 max: 1 }
      presence: { min_count: 1 }
      type: BYTES
    }
    feature {
      name: "feature"
      value_count: { min: 1 max: 1 }
      presence: { min_count: 1 }
      type: BYTES
    }
  )");
  anomalies["label"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        description: "Column is completely missing"
        severity: ERROR
        short_description: "Column dropped"
        reason {
          type: SCHEMA_MISSING_COLUMN
          short_description: "Column dropped"
          description: "Column is completely missing"
        }
        path { step: "label" })");

  // Running for no environment, or "TRAINING" environment without feature
  // 'label' should deprecate the feature.
  TestFeatureStatisticsValidator(schema, ValidationConfig(), statistics,
                                 tensorflow::gtl::nullopt,
                                 /*environment=*/gtl::nullopt,
                                 /*features_needed=*/gtl::nullopt, anomalies);
  TestFeatureStatisticsValidator(schema, ValidationConfig(), statistics,
                                 tensorflow::gtl::nullopt, "TRAINING",
                                 /*features_needed=*/gtl::nullopt, anomalies);

  // Running for environment "SERVING" should not generate anomalies.
  TestFeatureStatisticsValidator(
      schema, ValidationConfig(), statistics,
      /*prev_feature_statistics=*/ tensorflow::gtl::nullopt, "SERVING",
      /*features_needed=*/gtl::nullopt,
      /*expected_anomalies=*/ {});
}

TEST(FeatureStatisticsValidatorTest, FeaturesNeeded) {
  const Schema empty_schema;
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 1 } }
          }
        },
        features: {
          name: 'feature2'
          type: STRING
          string_stats: { common_stats: { num_missing: 1000 } }
        })");

  std::map<string, testing::ExpectedAnomalyInfo> anomalies;
  anomalies["feature1"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        }
        path { step: "feature1" }
      )pb");
  anomalies["feature1"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "feature1"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "feature1"
      presence { min_count: 1 }
    }
    string_domain { name: "feature1" value: "D" }
  )");
  TestFeatureStatisticsValidator(
      empty_schema, ValidationConfig(), statistics, tensorflow::gtl::nullopt,
      /*environment=*/gtl::nullopt,
      /*features_needed=*/FeaturesNeeded({{Path({"feature1"}), {{"needed"}}}}),
      anomalies);
}

// If there are no examples, then we don't crazily fire every exception, we
// only fire an alert for missing data.
TEST(FeatureStatisticsValidatorTest, MissingExamples) {
  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
    string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
    feature { name: "annotated_enum" type: BYTES domain: "MyAloneEnum" }
    feature { name: "ignore_this" lifecycle_stage: DEPRECATED type: BYTES })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 0)");

  tensorflow::metadata::v0::Anomalies want;
  *want.mutable_baseline() = schema;
  want.set_data_missing(true);

  tensorflow::metadata::v0::Anomalies got;
  TF_ASSERT_OK(ValidateFeatureStatistics(
      statistics, schema, /*environment=*/tensorflow::gtl::nullopt,
      /*prev_feature_statistics=*/tensorflow::gtl::nullopt,
      /*serving_feature_statistics=*/gtl::nullopt,
      /*features_needed=*/absl::nullopt, ValidationConfig(), &got));
  EXPECT_THAT(got, EqualsProto(want));
}

TEST(FeatureStatisticsValidatorTest, UpdateEmptySchema) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 997
              max_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" } }
          }
        })");

  const Schema want = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "D" })");
  TestSchemaUpdate(ValidationConfig(), statistics, Schema(), want);
}

TEST(FeatureStatisticsValidatorTest, UpdateEmptySchemaWithMissingColumn) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'bar'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 1000
            }
          }
        })");

  const Schema want = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "bar"
      type: BYTES
      presence { min_count: 0 }
    })");
  TestSchemaUpdate(ValidationConfig(), statistics, Schema(), want);
}

TEST(FeatureStatisticsValidatorTest, UpdateSchema) {
  const Schema old_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "E" })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              max_num_values: 1
              num_non_missing: 2
              avg_num_values: 2
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 1 } }
          }
        })");

  const Schema want = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "E" value: "D" })");
  TestSchemaUpdate(ValidationConfig(), statistics, old_schema, want);
}

TEST(FeatureStatisticsValidatorTest, UpdateSchemaWithColumnsToConsider) {
  const Schema old_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "E" })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              max_num_values: 1
              num_non_missing: 2
              avg_num_values: 2
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 1 } }
          }
        })");

  const Schema want = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "E" value: "D" })");
  Schema got;
  std::vector<Path> paths_to_consider = {Path({"annotated_enum"})};
  TF_EXPECT_OK(UpdateSchema(GetDefaultFeatureStatisticsToProtoConfig(),
                            old_schema, statistics, paths_to_consider,
                            /*environment=*/gtl::nullopt, &got));
  EXPECT_THAT(got, EqualsProto(want));
}

TEST(FeatureStatisticsValidatorTest, UseWeightedStatistics) {
  // Those missing have weight zero.
  // Also, (impossibly) there is an E for the weighted and a D for the
  // unweighted. This helps disambiguate.
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        weighted_num_examples: 997.0
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 997
              max_num_values: 1
              weighted_common_stats: { num_missing: 0.0 num_non_missing: 997.0 }
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" } }
            weighted_string_stats: {
              rank_histogram: { buckets: { label: "E" } }
            }
          }
        })");

  const Schema want = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "annotated_enum"
      presence { min_count: 1 }
    }
    string_domain { name: "annotated_enum" value: "E" })");
  TestSchemaUpdate(ValidationConfig(), statistics, Schema(), want);
}

TEST(FeatureStatisticsValidatorTest, UpdateDriftComparatorInSchema) {
  const Schema old_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      type: BYTES
      domain: "annotated_enum"
      drift_comparator { infinity_norm { threshold: 0.01 } }
    }
    string_domain { name: "annotated_enum" value: "a" })");

  const DatasetFeatureStatistics statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    num_examples: 2
    features: {
      name: 'annotated_enum'
      type: STRING
      string_stats: {
        common_stats: { num_non_missing: 1 num_missing: 0 max_num_values: 1 }
        rank_histogram {
          buckets { label: "a" sample_count: 1 }
          buckets { label: "b" sample_count: 1 }
        }
      }
    })");

  const DatasetFeatureStatistics prev_statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    num_examples: 4
    features: {
      name: 'annotated_enum'
      type: STRING
      string_stats: {
        common_stats: { num_non_missing: 1 num_missing: 0 max_num_values: 1 }
        rank_histogram {
          buckets { label: "a" sample_count: 3 }
          buckets { label: "b" sample_count: 1 }
        }
      }
    })");

  const Schema want_fixed_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      type: BYTES
      domain: "annotated_enum"
      drift_comparator { infinity_norm { threshold: 0.25 } }
    }
    string_domain { name: "annotated_enum" value: "a" value: "b" })");

  std::map<string, testing::ExpectedAnomalyInfo> anomalies;
  anomalies["annotated_enum"].new_schema = want_fixed_schema;
  anomalies["annotated_enum"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    description: "Examples contain values missing from the schema: b (?).  The Linfty distance between current and previous is 0.25 (up to six significant digits), above the threshold 0.01. The feature value with maximum difference is: b"
    severity: ERROR
    short_description: "Multiple errors"
    reason {
      type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
      short_description: "Unexpected string values"
      description: "Examples contain values missing from the schema: b (?). "
    }
    reason {
      type: COMPARATOR_L_INFTY_HIGH
      short_description: "High Linfty distance between current and previous"
      description: "The Linfty distance between current and previous is 0.25 (up to six significant digits), above the threshold 0.01. The feature value with maximum difference is: b"
    }
    path: { step: "annotated_enum" }
  )");
  TestFeatureStatisticsValidator(old_schema, ValidationConfig(), statistics,
                                 prev_statistics,
                                 /*environment=*/gtl::nullopt,
                                 /*features_needed=*/gtl::nullopt, anomalies);
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
