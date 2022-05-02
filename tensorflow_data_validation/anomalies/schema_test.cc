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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/proto/validation_config.pb.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_schema_protos.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using testing::EqualsProto;
using testing::GetAnnotatedFieldsMessage;
using testing::GetTestAllTypesMessage;
using testing::GetTestSchemaAlone;
using testing::ParseTextProtoOrDie;

std::vector<ValidationConfig> GetValidationConfigs() {
  return std::vector<ValidationConfig>(
      {ValidationConfig(), ParseTextProtoOrDie<ValidationConfig>(
                               "new_features_are_warnings: true")});
}

// This schema was broken. The problem was default Initialization in IntType.
TEST(SchemaTest, CreateFromSchemaProtoBroken) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "few_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: INT
        })");
  Schema schema;

  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(feature {
                                       name: "few_int64"
                                       presence: { min_count: 1 }
                                       value_count: { min: 1 max: 1 }
                                       type: INT
                                     })"));
}

// Test that initializing from a schema proto, then exporting a schema proto,
// does not change the schema proto. See
// CreateFromProtoWithEmbeddedStringDomain for when this doesn't work.
TEST(SchemaTest, CreateFromSchemaProto) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        string_domain {
          name: "MyAloneEnum"
          value: "4"
          value: "5"
          value: "6"
          value: "ALONE_BUT_NORMAL"
        }
        feature {
          name: "annotated_enum"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          domain: "MyAloneEnum"
        }
        feature {
          name: "big_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "small_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { max: 123 }
        }
        feature {
          name: "string_int32"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain { min: -2147483648 max: 2147483647 }
        }
        feature {
          name: "string_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain {}
        }
        feature {
          name: "string_uint32"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain { min: 0 max: 4294967295 }
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(initial));
}

// The StringDomain is too large after a new value arrives.
TEST(SchemaTest, StringDomainTooLarge) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        string_domain {
          name: "MyAloneEnum"
          value: "4"
          value: "5"
          value: "6"
          value: "ALONE_BUT_NORMAL"
        }
        feature {
          name: "annotated_enum"
          value_count: { min: 1 max: 1 }
          type: BYTES
          domain: "MyAloneEnum"
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  FeatureStatisticsToProtoConfig config;
  config.set_enum_threshold(4);
  config.set_enum_delete_threshold(4);
  const DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(
          R"(
            num_examples: 10
            features {
              name: 'annotated_enum'
              type: STRING
              string_stats: {
                common_stats: {
                  num_missing: 0
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
                rank_histogram {
                  buckets { label: "a" sample_count: 1 }
                  buckets { label: "b" sample_count: 2 }
                  buckets { label: "c" sample_count: 7 }
                }
              }
            })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(stats), config));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "annotated_enum"
                  value_count: { min: 1 max: 1 }
                  type: BYTES
                })"));
}

// The StringDomain is too large after a new field arrives.
TEST(SchemaTest, EmbeddedStringDomainTooLarge) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "annotated_enum"
          value_count: { min: 1 max: 1 }
          type: BYTES
          string_domain: {
            value: "4"
            value: "5"
            value: "6"
            value: "ALONE_BUT_NORMAL"
          }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  FeatureStatisticsToProtoConfig config;
  config.set_enum_threshold(4);
  config.set_enum_delete_threshold(4);
  const DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(
          R"(
            num_examples: 10
            features {
              name: 'annotated_enum'
              type: STRING
              string_stats: {
                common_stats: {
                  num_missing: 0
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
                rank_histogram {
                  buckets { label: "a" sample_count: 1 }
                  buckets { label: "b" sample_count: 2 }
                  buckets { label: "c" sample_count: 7 }
                }
              }
            })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(stats), config));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "annotated_enum"
                  value_count: { min: 1 max: 1 }
                  type: BYTES
                })"));
}

// Test that initializing from a schema proto, then exporting a schema proto,
// does not change the schema proto. See
// CreateFromProtoWithEmbeddedStringDomain for when this doesn't work.
TEST(SchemaTest, CreateFromSchema) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        string_domain {
          name: "MyAloneEnum"
          value: "4"
          value: "5"
          value: "6"
          value: "ALONE_BUT_NORMAL"
        }
        feature {
          name: "annotated_enum"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          domain: "MyAloneEnum"
        }
        feature {
          name: "annotated_enum_with_constraint"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          distribution_constraints: { min_domain_mass: 0.75 }
          domain: "MyAloneEnum"
        }
        feature {
          name: "big_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "real_optional"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "small_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { max: 123 }
        }
        feature {
          name: "string_int32"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain { min: -2147483648 max: 2147483647 }
        }
        feature {
          name: "string_int64"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain {}
        }
        feature {
          name: "string_uint32"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
          int_domain { min: 0 max: 4294967295 }
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  Schema schema_2;
  TF_ASSERT_OK(schema_2.Init(actual));
  const tensorflow::metadata::v0::Schema actual_2 = schema_2.GetSchema();
  EXPECT_THAT(actual, EqualsProto(initial));
  EXPECT_THAT(actual_2, EqualsProto(initial));
}

// Runs all codepaths in UpdateSomeColumns, and makes sure we can update
// multiple columns.
// Lots of these test the logic where FeatureStats are present but the
// number of examples for a feature is zero.
TEST(SchemaTest, UpdateSomeColumns) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "standard_update"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "standard_update_2"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "untouched_update"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "missing_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: INT
          int_domain { min: 65 }
        }
        feature {
          name: "missing_feature_deprecated"
          lifecycle_stage: DEPRECATED
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: INT
          int_domain { min: 65 }
        }
      )");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  DatasetFeatureStatistics dataset_statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features {
      name: 'standard_update'
      type: INT
      num_stats: { common_stats: { tot_num_values: 0 num_missing: 3 } }
    }
    features {
      name: 'standard_update_2'
      type: INT
      num_stats: { common_stats: { tot_num_values: 0 num_missing: 3 } }
    }
    features {
      name: 'new_column'
      type: INT
      num_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 1
          min_num_values: 2
          max_num_values: 2
        }
      }
    }
    features {
      name: 'untouched_update'
      type: INT
      num_stats: {
        common_stats: {
          num_missing: 3
          min_num_values: 1
          max_num_values: 1
        }
      }
    })");
  DatasetStatsView stats(dataset_statistics, false);
  TF_ASSERT_OK(schema.Update(
      stats, FeatureStatisticsToProtoConfig(),
      {Path({"missing_feature"}), Path({"completely_missing_column"}),
       Path({"standard_update"}), Path({"standard_update_2"}),
       Path({"new_column"})}));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  EXPECT_THAT(actual, EqualsProto(
                          R"(
                            feature {
                              name: "standard_update"
                              lifecycle_stage: DEPRECATED
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: INT
                              int_domain { min: 65 }
                            }
                            feature {
                              name: "standard_update_2"
                              lifecycle_stage: DEPRECATED
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: INT
                              int_domain { min: 65 }
                            }
                            feature {
                              name: "untouched_update"
                              presence: { min_count: 1 }
                              value_count: { min: 1 max: 1 }
                              type: INT
                              int_domain { min: 65 }
                            }
                            feature {
                              name: "missing_feature"
                              lifecycle_stage: DEPRECATED
                              value_count: { min: 1 max: 1 }
                              type: INT
                              int_domain { min: 65 }
                              presence: { min_count: 1 }
                            }
                            feature {
                              name: "missing_feature_deprecated"
                              lifecycle_stage: DEPRECATED
                              value_count: { min: 1 max: 1 }
                              type: INT
                              int_domain { min: 65 }
                              presence: { min_count: 1 }
                            }
                            feature {
                              name: "new_column"
                              value_count { min: 2 max: 2 }
                              type: INT
                              presence { min_count: 1 }
                            }
                          )"));
}


TEST(SchemaTest, CreateFeatureWithSemanticType) {
  // For new features:
  // - If custom domain_info is valid, add it to the schema.
  // - If custom domain_info is not-valid ignore it.
  const tensorflow::metadata::v0::Schema initial =
      tensorflow::metadata::v0::Schema();

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  DatasetFeatureStatistics dataset_statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features {
      name: "nl_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "natural_language_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    }
    features {
      name: "another_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "This is invalid!" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    })");
  DatasetStatsView stats(dataset_statistics, false);
  TF_ASSERT_OK(schema.Update(
      stats, FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  EXPECT_THAT(actual, EqualsProto(
                          R"(
                            feature {
                              name: "nl_feature"
                              presence: { min_count: 1 min_fraction: 1}
                              value_count: { min: 1 max: 1 }
                              type: BYTES
                              natural_language_domain: {}
                            }
                            feature {
                              name: "another_feature"
                              presence: { min_count: 1 min_fraction: 1}
                              value_count: { min: 1 max: 1 }
                              type: BYTES
                            })"));
}

TEST(SchemaTest, UpdateFeatureWithSemanticType) {
  // Updated features with valid custom domain_info should only get semantic
  // type if they did not have an existing domain.
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "nl_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
          natural_language_domain: {}
        }
        feature {
          name: "image_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
          image_domain: {}
        }
        feature {
          name: "time_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
          time_domain: { string_format: "%Y/%m/%d" }
        }
        feature {
          name: "string_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
          string_domain: {}
        }
        feature {
          name: "nodomain_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
        }
      )");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  // Try to set url_domain to all features above.
  DatasetFeatureStatistics dataset_statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features {
      name: "nl_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "url_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    }
    features {
      name: "image_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "url_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    }
    features {
      name: "time_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "url_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    }
    features {
      name: "string_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "url_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    }
    features {
      name: "nodomain_feature"
      type: STRING
      custom_stats: { name: "domain_info" str: "url_domain {}" }
      bytes_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    })");
  DatasetStatsView stats(dataset_statistics, false);
  TF_ASSERT_OK(schema.Update(
      stats, FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  EXPECT_THAT(actual, EqualsProto(
                          R"(
                            feature {
                              name: "nl_feature"
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: BYTES
                              natural_language_domain: {}
                            }
                            feature {
                              name: "image_feature"
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: BYTES
                              image_domain: {}
                            }
                            feature {
                              name: "time_feature"
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: BYTES
                              time_domain: { string_format: "%Y/%m/%d" }
                            }
                            feature {
                              name: "string_feature"
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: BYTES
                              string_domain: {}
                            }
                            feature {
                              name: "nodomain_feature"
                              presence: { min_count: 1 }
                              value_count: { min: 1 }
                              type: BYTES
                              url_domain: {}
                            }
                          )"));
}

TEST(SchemaTest, UpdateColumnsWithEnvironments) {
  // Define all schema protos for the test cases.
  const auto schema_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "feature"
          type: INT
          value_count { min: 1 max: 1 }
          presence: { min_count: 1 min_fraction: 1 }
        })");
  const auto schema_feature_env1 = [&]() {
    auto result = schema_feature;
    result.mutable_feature(0)->add_in_environment("env1");
    return result;
  }();
  const auto schema_feature_env1_deprecated = [&]() {
    auto result = schema_feature_env1;
    result.mutable_feature(0)->set_lifecycle_stage(
        tensorflow::metadata::v0::DEPRECATED);
    return result;
  }();
  const auto schema_feature_env12 = [&]() {
    auto result = schema_feature_env1;
    result.mutable_feature(0)->add_in_environment("env2");
    return result;
  }();
  // Define a statistics feature.
  const auto statistics_feature =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: "feature"
          type: INT
          num_stats: {
            common_stats: {
              tot_num_values: 1
              num_non_missing: 1
              max_num_values: 1
              min_num_values: 1
            }
          }
        }
      )");
  // A lambda that checks that the schema given by 'schema_proto' when Updated
  // with the 'statistics_proto' with given 'environment' produces the
  // 'result_schema_proto'.
  auto check_update =
      [](const tensorflow::metadata::v0::Schema& schema_proto,
         const DatasetFeatureStatistics& statistics_proto,
         const string& environment,
         const tensorflow::metadata::v0::Schema& result_schema_proto) {
        Schema schema;
        TF_ASSERT_OK(schema.Init(schema_proto));
        DatasetStatsView stats(statistics_proto,
                               /* by_weight= */ false, environment,
                               /* previous_span= */ nullptr,
                               /* serving= */ nullptr,
                               /* previous_version= */ nullptr);
        TF_ASSERT_OK(schema.Update(stats, FeatureStatisticsToProtoConfig(),
                                   {Path({"feature"})}));
        EXPECT_THAT(schema.GetSchema(), EqualsProto(result_schema_proto));
      };
  // The following test cases cover all distinct posibilities of:
  // (In Schema?, In Statistic?, Environment Match?) => Result
  // ( No, No,  *) => Do Nothing
  check_update(tensorflow::metadata::v0::Schema(), DatasetFeatureStatistics(),
               "env1", tensorflow::metadata::v0::Schema());
  // ( No,Yes,  *) => Add Feature
  check_update(tensorflow::metadata::v0::Schema(), statistics_feature, "env1",
               schema_feature);
  // (Yes, No, No) => Do Nothing
  check_update(schema_feature_env1, DatasetFeatureStatistics(), "env2",
               schema_feature_env1);
  // (Yes, No,Yes) => Deprecate
  check_update(schema_feature_env1, DatasetFeatureStatistics(), "env1",
               schema_feature_env1_deprecated);
  // (Yes,Yes, No) => Add Environment and perhaps modify
  check_update(schema_feature_env1, statistics_feature, "env2",
               schema_feature_env12);
  // (Yes,Yes,Yes) => Perhaps modify
  check_update(schema_feature_env1, statistics_feature, "env1",
               schema_feature_env1);

  // Cover special case when a feature is in statistics but has no values.
  const auto statistics_feature_no_values =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: "feature"
          type: INT
          num_stats: {
            common_stats: {
              tot_num_values: 0
              num_non_missing: 0
              max_num_values: 0
              min_num_values: 0
            }
          }
        }
      )");
  // If environment matches, feature should be deprecated.
  check_update(schema_feature_env1, statistics_feature_no_values, "env1",
               schema_feature_env1_deprecated);

  // If environment does not match do nothing.
  check_update(schema_feature_env1, statistics_feature_no_values, "env2",
               schema_feature_env1);
}

TEST(SchemaTest, UpdateColumnsWithNewEnvironmentDescription) {
  // Define all schema protos for the test cases.
  const auto schema_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          in_environment: "TRAINING"
          name: "feature"
          type: INT
          value_count { min: 1 max: 1 }
          presence: { min_count: 1 min_fraction: 1 }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_feature));
  const auto statistics_feature =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: "feature"
          type: INT
          num_stats: {
            common_stats: {
              tot_num_values: 1
              num_non_missing: 1
              max_num_values: 1
              min_num_values: 1
            }
          }
        }
      )");
  DatasetStatsView dataset_stats_view(statistics_feature, false, "SERVING",
                                      nullptr, nullptr, nullptr);
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity severity;
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo> drift_skew_info;
  TF_ASSERT_OK(
      schema.UpdateFeature(Schema::Updater(FeatureStatisticsToProtoConfig()),
                           *dataset_stats_view.GetByPath(Path({"feature"})),
                           &descriptions, &drift_skew_info, &severity));
  ASSERT_EQ(descriptions.size(), 1);
  EXPECT_EQ(descriptions[0].type,
            tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN);
}

TEST(SchemaTest, DeprecateFeature) {
  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "feature_name"
          type: INT
          skew_comparator: { infinity_norm: { threshold: 0.1 } }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));
  schema.DeprecateFeature(Path({"feature_name"}));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "feature_name"
                  type: INT
                  skew_comparator: { infinity_norm: { threshold: 0.1 } }
                  lifecycle_stage: DEPRECATED
                })"));
}

// For now, just checks if the environments are passed through.
TEST(SchemaTest, DefaultEnvironments) {
  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "feature_name"
          type: INT
          in_environment: "FOO"
          in_environment: "BAR"
          not_in_environment: "TRAINING"
        }
        default_environment: "TRAINING"
        default_environment: "SERVING")");

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "feature_name"
                  type: INT
                  in_environment: "FOO"
                  in_environment: "BAR"
                  not_in_environment: "TRAINING"
                }
                default_environment: "TRAINING"
                default_environment: "SERVING")"));
}

TEST(SchemaTest, FindSkew) {
  const DatasetFeatureStatistics training =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(
              R"(name: 'foo'
                 type: STRING
                 string_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   rank_histogram {
                     buckets { label: "a" sample_count: 1 }
                     buckets { label: "b" sample_count: 2 }
                     buckets { label: "c" sample_count: 7 }
                   }
                 })"));
  const DatasetFeatureStatistics serving =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(
              R"(name: 'foo'
                 type: STRING
                 string_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   rank_histogram {
                     buckets { label: "a" sample_count: 3 }
                     buckets { label: "b" sample_count: 1 }
                     buckets { label: "c" sample_count: 6 }
                   }
                 })"));

  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: 'foo'
          type: BYTES
          skew_comparator { infinity_norm: { threshold: 0.1 } }
        })");
  std::shared_ptr<DatasetStatsView> serving_view =
      std::make_shared<DatasetStatsView>(serving);
  std::shared_ptr<DatasetStatsView> training_view =
      std::make_shared<DatasetStatsView>(
          training,
          /* by_weight= */ false,
          /* environment= */ absl::nullopt,
          /* previous_span= */ std::shared_ptr<DatasetStatsView>(),
          serving_view,
          /* previous_version= */ std::shared_ptr<DatasetStatsView>());

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));

  tensorflow::metadata::v0::Schema expected_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "foo"
          type: BYTES
          skew_comparator { infinity_norm: { threshold: 0.19999999999999998 } }
        })");
  FeatureComparisonResult result =
      schema.UpdateSkewComparator(*training_view->GetByPath(Path({"foo"})));

  EXPECT_THAT(schema.GetSchema(), EqualsProto(expected_schema));
  // We're not particular about the description, just that there be one.
  ASSERT_FALSE(result.descriptions.empty());
  ASSERT_EQ(result.measurements.size(), 1);
  EXPECT_THAT(
      result.measurements[0],
      EqualsProto(ParseTextProtoOrDie<
                  tensorflow::metadata::v0::DriftSkewInfo::Measurement>(R"(
        type: L_INFTY
        value: 0.19999999999999998
        threshold: 0.1
      )")));
}

TEST(SchemaTest, FindDrift) {
  const DatasetFeatureStatistics training =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(
              R"(name: 'foo'
                 type: STRING
                 string_stats: {
                   common_stats: {
                     num_missing: 0
                     num_non_missing: 1
                     max_num_values: 1
                   }
                   rank_histogram {
                     buckets { label: "a" sample_count: 1 }
                     buckets { label: "b" sample_count: 2 }
                     buckets { label: "c" sample_count: 7 }
                   }
                 })"));
  const DatasetFeatureStatistics previous =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(
              R"(name: 'foo'
                 type: STRING
                 string_stats: {
                   common_stats: {
                     num_missing: 0
                     num_non_missing: 1
                     max_num_values: 1
                   }
                   rank_histogram {
                     buckets { label: "a" sample_count: 3 }
                     buckets { label: "b" sample_count: 1 }
                     buckets { label: "c" sample_count: 6 }
                   }
                 })"));

  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: 'foo'
          type: BYTES
          drift_comparator { infinity_norm: { threshold: 0.1 } }
        })");
  std::shared_ptr<DatasetStatsView> previous_view =
      std::make_shared<DatasetStatsView>(previous);
  std::shared_ptr<DatasetStatsView> training_view =
      std::make_shared<DatasetStatsView>(training,
                                         /* by_weight= */ false,
                                         /* environment= */ absl::nullopt,
                                         previous_view, /* serving= */ nullptr,
                                         /* previous_version= */ nullptr);

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));

  tensorflow::metadata::v0::Schema expected_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "foo"
          type: BYTES
          drift_comparator { infinity_norm: { threshold: 0.19999999999999998 } }
        })");
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity severity;
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo> drift_skew_info;
  TF_ASSERT_OK(
      schema.UpdateFeature(Schema::Updater(FeatureStatisticsToProtoConfig()),
                           *training_view->GetByPath(Path({"foo"})),
                           &descriptions, &drift_skew_info, &severity));

  EXPECT_THAT(schema.GetSchema(), EqualsProto(expected_schema));
  // We're not particular about the description, just that there be one.
  EXPECT_FALSE(descriptions.empty());
  // Drift is always an error.
  EXPECT_EQ(tensorflow::metadata::v0::AnomalyInfo::ERROR, severity);
  ASSERT_TRUE(drift_skew_info.has_value());
  EXPECT_THAT(
      *drift_skew_info,
      EqualsProto(
          ParseTextProtoOrDie<tensorflow::metadata::v0::DriftSkewInfo>(R"(
            path { step: [ "foo" ] }
            drift_measurements {
              type: L_INFTY
              value: 0.19999999999999998
              threshold: 0.1
            }
          )")));
}

TEST(SchemaTest, FindNumExamplesDrift) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 2)");
  const DatasetFeatureStatistics previous_span =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 4)");

  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints {
               num_examples_drift_comparator {
                 min_fraction_threshold: 1.0,
                 max_fraction_threshold: 1.0
               }
             })");
  std::shared_ptr<DatasetStatsView> previous_span_view =
      std::make_shared<DatasetStatsView>(previous_span);
  DatasetStatsView training_view =
      DatasetStatsView(current,
                       /* by_weight= */ false,
                       /* environment= */ absl::nullopt, previous_span_view,
                       /* serving= */ nullptr,
                       /* previous_version= */ nullptr);

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));

  tensorflow::metadata::v0::Schema expected_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints {
               num_examples_drift_comparator {
                 min_fraction_threshold: 0.5,
                 max_fraction_threshold: 1.0
               }
             })");
  const std::vector<Description> result =
      schema.UpdateDatasetConstraints(training_view);

  EXPECT_THAT(schema.GetSchema(), EqualsProto(expected_schema));
  // We're not particular about the description, just that there be one.
  EXPECT_FALSE(result.empty());
}

TEST(SchemaTest, FindNumExamplesChangeAcrossVersions) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 2)");
  const DatasetFeatureStatistics previous_version =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 1)");

  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints {
               num_examples_version_comparator {
                 min_fraction_threshold: 1.0,
                 max_fraction_threshold: 1.0
               }
             })");
  std::shared_ptr<DatasetStatsView> previous_version_view =
      std::make_shared<DatasetStatsView>(previous_version);
  DatasetStatsView training_view = DatasetStatsView(
      current,
      /* by_weight= */ false,
      /* environment= */ absl::nullopt, /* previous_span= */ nullptr,
      /* serving= */ nullptr, previous_version_view);

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));

  tensorflow::metadata::v0::Schema expected_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints {
               num_examples_version_comparator {
                 min_fraction_threshold: 1.0,
                 max_fraction_threshold: 2.0
               }
             })");
  const std::vector<Description> result =
      schema.UpdateDatasetConstraints(training_view);

  EXPECT_THAT(schema.GetSchema(), EqualsProto(expected_schema));
  // We're not particular about the description, just that there be one.
  EXPECT_FALSE(result.empty());
}

TEST(SchemaTest, FindNumExamplesTooLow) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 2)");

  const tensorflow::metadata::v0::Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints { min_examples_count: 3 })");
  DatasetStatsView dataset_view = DatasetStatsView(
      current,
      /* by_weight= */ false,
      /* environment= */ absl::nullopt, /* previous_span= */ nullptr,
      /* serving= */ nullptr, /* previous_version= */ nullptr);

  Schema schema;
  TF_ASSERT_OK(schema.Init(schema_proto));

  tensorflow::metadata::v0::Schema expected_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
          R"(dataset_constraints { min_examples_count: 2 })");
  const std::vector<Description> result =
      schema.UpdateDatasetConstraints(dataset_view);

  EXPECT_THAT(schema.GetSchema(), EqualsProto(expected_schema));
  // We're not particular about the description, just that there be one.
  EXPECT_FALSE(result.empty());
}

// This test captures what happens today with embedded string domains. In the
// future, we want to make initial == Schema::Create(initial)->GetSchema().
TEST(SchemaTest, CreateFromProtoWithEmbeddedStringDomain) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "empty_domain"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: BYTES
          string_domain {}
        }
        feature {
          name: "one_value_domain"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: BYTES
          string_domain { value: "A" }
        }
        feature {
          name: "two_value_domain"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: BYTES
          string_domain { value: "A" value: "B" }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "empty_domain"
                  string_domain {}

                  presence: { min_count: 1 }
                  value_count { min: 1 max: 1 }
                  type: BYTES
                }
                feature {
                  name: "one_value_domain"
                  string_domain { value: "A" }

                  presence: { min_count: 1 }
                  value_count { min: 1 max: 1 }
                  type: BYTES
                }
                feature {
                  name: "two_value_domain"
                  string_domain { value: "A" value: "B" }
                  presence: { min_count: 1 }
                  value_count { min: 1 max: 1 }
                  type: BYTES
                })"));
}

// Test if the feature exists.
TEST(SchemaTest, FeatureExists) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "struct"
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar.baz" }
          }
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            feature: { name: "baz.bar" }
          }
        }
        feature {
          name: "(ext.field)"
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            sparse_feature: { name: "deep_sparse" }
          }
        }
        sparse_feature: { name: "shallow_sparse" }

        feature { name: "foo.bar" })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  EXPECT_TRUE(schema.FeatureExists(Path({"foo.bar"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"##SEQUENCE##", "foo"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"(ext.field)", "foo"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"(ext.field)", "deep_sparse"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"shallow_sparse"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"struct", "foo"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"struct", "bar.baz"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"##SEQUENCE##", "baz.bar"})));
  EXPECT_TRUE(schema.FeatureExists(Path({"##SEQUENCE##"})));
  EXPECT_FALSE(schema.FeatureExists(Path({"no_such_field"})));
  EXPECT_FALSE(schema.FeatureExists(Path({"##SEQUENCE##", "no_such_field"})));
}

// Tests the creation of a nested feature.
TEST(SchemaTest, CreateColumnsDeepAll) {
  Schema schema;
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features {
              name: "struct"
              type: STRUCT
              struct_stats {
                common_stats {
                  num_missing: 3
                  num_non_missing: 7
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.bar.baz"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  TF_ASSERT_OK(schema.Update(view, FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "struct"
                  value_count { min: 2 max: 2 }
                  type: STRUCT
                  presence { min_count: 1 }
                  struct_domain {
                    feature {
                      name: "bar.baz"
                      value_count { min: 2 max: 2 }
                      type: INT
                      presence { min_count: 1 }
                    }
                    feature {
                      name: "foo"
                      value_count { min: 2 max: 2 }
                      type: INT
                      presence { min_count: 1 }
                    }
                  }
                })"));
}

// Tests the creation of a nested feature.
TEST(SchemaTest, CreateColumnsDeep) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "struct"
          type: STRUCT
          struct_domain {}
        })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features {
              name: "struct"
              type: STRUCT
              struct_stats {
                common_stats {
                  num_missing: 3
                  num_non_missing: 7
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.bar.baz"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  TF_ASSERT_OK(schema.Update(view, FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "struct"
                  type: STRUCT
                  struct_domain {
                    feature {
                      name: "bar.baz"
                      value_count { min: 2 max: 2 }
                      type: INT
                      presence { min_count: 1 }
                    }
                    feature {
                      name: "foo"
                      value_count { min: 2 max: 2 }
                      type: INT
                      presence { min_count: 1 }
                    }
                  }
                })"));
}

// Tests the creation of a nested feature with the parent deprecated
// (all children are also considered deprecated).
TEST(SchemaTest, CreateColumnsDeepDeprecated) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "struct"
          type: STRUCT
          lifecycle_stage: DEPRECATED
          struct_domain { feature: { name: "foo" } }
        })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features {
              name: "struct"
              type: STRUCT
              struct_stats { common_stats { num_missing: 3 max_num_values: 2 } }
            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: { common_stats: { num_missing: 3 max_num_values: 2 } }
            }
            features {
              name: "struct.bar.baz"
              type: INT
              num_stats: { common_stats: { num_missing: 3 max_num_values: 2 } }
            })pb");

  DatasetStatsView view(stats);
  TF_ASSERT_OK(schema.Update(view, FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(initial));
}

// Test that FeatureIsDeprecated is correct when the output should be false.
TEST(SchemaTest, FeatureIsDeprecatedFalse) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "empty_domain"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: BYTES
          string_domain {}
        }
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar.baz" }
          }
        }
        feature {
          name: "##SEQUENCE##"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            feature: { name: "baz.bar" }
          }
        }
        feature {
          name: "(ext.field)"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            sparse_feature: { name: "deep_sparse" }
          }
        }
        sparse_feature: { name: "shallow_sparse" }

        feature { name: "foo.bar" })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"foo", "bar"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"##SEQUENCE##", "foo"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"(ext.field)", "foo"})));
  EXPECT_FALSE(
      schema.FeatureIsDeprecated(Path({"(ext.field)", "deep_sparse"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"shallow_sparse"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"struct", "foo"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"struct", "bar.baz"})));
  EXPECT_FALSE(schema.FeatureIsDeprecated(Path({"##SEQUENCE##", "baz.bar"})));
}

// Test when a bunch of paths are all present in the schema and missing in
// the data.
TEST(SchemaTest, GetMissingPathsAllMissing) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "struct"
          presence: { min_count: 1 }
          type: STRUCT
          struct_domain {
            feature: {
              presence: { min_count: 1 }
              name: "foo"
            }
            feature: {
              presence: { min_count: 1 }
              name: "bar.baz"
            }
          }
        }
        feature {
          name: "##SEQUENCE##"
          presence: { min_count: 1 }
          type: STRUCT
          struct_domain {
            feature: {
              presence: { min_count: 1 }
              name: "foo"
            }
            feature: {
              presence: { min_count: 1 }
              name: "bar"
            }
            feature: {
              presence: { min_count: 1 }
              name: "baz.bar"
            }
          }
        }
        feature {
          name: "(ext.field)"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: {
              presence: { min_count: 1 }
              name: "foo"
            }
            feature: {
              presence: { min_count: 1 }
              name: "bar"
            }
            sparse_feature: {
              presence: { min_count: 1 }
              name: "deep_sparse"
            }
          }
        }
        sparse_feature: {
          name: "shallow_sparse"
          presence: { min_count: 1 }
        }

        feature {
          name: "foo.bar"
          presence: { min_count: 1 }
        })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats;
  DatasetStatsView view(stats);
  // "(ext.field).deep_sparse", "shallow_sparse" are sparse and therefore
  // not required.
  EXPECT_THAT(schema.GetMissingPaths(view),
              ::testing::UnorderedElementsAre(
                  Path({"struct"}), Path({"struct", "foo"}),
                  Path({"struct", "bar.baz"}), Path({"##SEQUENCE##"}),
                  Path({"##SEQUENCE##", "foo"}), Path({"##SEQUENCE##", "bar"}),
                  Path({"##SEQUENCE##", "baz.bar"}), Path({"(ext.field)"}),
                  Path({"(ext.field)", "foo"}), Path({"(ext.field)", "bar"}),
                  Path({"foo.bar"})));
}

// Test when GetMissingPaths when all paths in a schema are in the data.
TEST(SchemaTest, GetMissingPathsAllPresent) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "empty_domain"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: BYTES
          string_domain {}
        }
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar.baz" }
          }
        }
        feature {
          name: "##SEQUENCE##"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            feature: { name: "baz.bar" }
          }
        }
        feature {
          name: "(ext.field)"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain {
            feature: { name: "foo" }
            feature: { name: "bar" }
            sparse_feature: { name: "deep_sparse" }
          }
        }
        sparse_feature: { name: "shallow_sparse" }

        feature { name: "foo.bar" })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features { name: "empty_domain" type: BYTES string_stats { } }
            features { name: "struct" type: STRUCT struct_stats { } }
            features { name: "struct.foo" struct_stats { } }
            features { name: "struct.bar.baz" struct_stats { } }
            features { name: "##SEQUENCE##" type: STRUCT struct_stats { } }
            features { name: "##SEQUENCE##.foo" struct_stats { } }
            features: { name: "##SEQUENCE##.bar" struct_stats { } }
            features: { name: "##SEQUENCE##.baz.bar" struct_stats { } }
            features: { name: "(ext.field)" type: STRUCT struct_stats { } }
            features: { name: "(ext.field).foo" struct_stats { } }
            features: { name: "(ext.field).bar" struct_stats { } }
            features: { name: "(ext.field).deep_sparse" struct_stats { } }
            features: { name: "shallow_sparse" struct_stats { } }
            features { name: "foo.bar" struct_stats { } })pb");

  DatasetStatsView view(stats);
  EXPECT_EQ(schema.GetMissingPaths(view).size(), 0);
}

// This tests if UpdateRecursively can create a deep feature.
TEST(SchemaTest, CreateDeepFieldUpdateRecursivelyStructFoo) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain { feature: { name: "bar" } }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"(
            features {
              name: "struct"
              type: STRUCT
              struct_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 7
                  min_num_values: 1
                  max_num_values: 1
                }
              }

            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            })");

  DatasetStatsView view(stats);
  std::vector<Description> descriptions;
  metadata::v0::AnomalyInfo::Severity severity;

  TF_ASSERT_OK(schema.UpdateRecursively(
      Schema::Updater(FeatureStatisticsToProtoConfig()),
      *view.GetByPath(Path({"struct", "foo"})), absl::nullopt, &descriptions,
      &severity));
  EXPECT_THAT(schema.GetSchema(),
              EqualsProto(R"(feature {
                                        name: "struct"
                                        value_count { min: 1 max: 1 }
                                        type: STRUCT
                                        presence { min_count: 1 }
                                        struct_domain {
                                          feature: { name: "bar" }
                                          feature {
                                            name: "foo"
                                            value_count { min: 2 max: 2 }
                                            type: INT
                                            presence { min_count: 1 }
                                          }
                                        }
                                      })"));
}

// This tests if UpdateRecursively can create a deep feature.
TEST(SchemaTest, CreateDeepFieldUpdateRecursivelyStruct) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { min: 1 max: 1 }
          type: STRUCT
          struct_domain { feature: { name: "bar" } }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"(
            features {
              name: "struct"
              type: STRUCT
              struct_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 7
                  min_num_values: 1
                  max_num_values: 1
                }
              }

            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            })");

  DatasetStatsView view(stats);
  std::vector<Description> descriptions;
  metadata::v0::AnomalyInfo::Severity severity;
  TF_ASSERT_OK(schema.UpdateRecursively(
      Schema::Updater(FeatureStatisticsToProtoConfig()),
      *view.GetByPath(Path({"struct"})), absl::nullopt, &descriptions,
      &severity));

  EXPECT_THAT(schema.GetSchema(),
              EqualsProto(R"(feature {
                                        name: "struct"
                                        value_count { min: 1 max: 1 }
                                        type: STRUCT
                                        presence { min_count: 1 }
                                        struct_domain {
                                          feature: { name: "bar" }
                                          feature {
                                            name: "foo"
                                            value_count { min: 2 max: 2 }
                                            type: INT
                                            presence { min_count: 1 }
                                          }
                                        }
                                      })"));
}

// applying Update to a struct should do nothing but confirm that it is a
// struct.
TEST(SchemaTest, UpdateStruct) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { max: 1 }
          type: STRUCT
          struct_domain { feature: { name: "bar" } }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"(
            features {
              name: "struct"
              type: STRUCT
              struct_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 7
                  max_num_values: 1
                }
              }

            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  max_num_values: 2
                }
              }
            })");

  DatasetStatsView view(stats);
  std::vector<Description> descriptions;
  metadata::v0::AnomalyInfo::Severity severity;
  absl::optional<tensorflow::metadata::v0::DriftSkewInfo> drift_skew_info;
  TF_ASSERT_OK(
      schema.UpdateFeature(Schema::Updater(FeatureStatisticsToProtoConfig()),
                           *view.GetByPath(Path({"struct"})), &descriptions,
                           &drift_skew_info, &severity));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(initial));
}

// This tests if Update can create a deep feature.
TEST(SchemaTest, CreateDeepFieldWithUpdate) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          presence: { min_count: 1 }
          value_count { min: 2 max: 2 }
          type: STRUCT
          struct_domain { feature: { name: "bar" } }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"(

            features {
              name: "struct"
              type: STRUCT
              struct_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 7
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.foo"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  min_num_values: 2
                  max_num_values: 2
                }
              }
            })");

  DatasetStatsView view(stats);

  TF_ASSERT_OK(schema.Update(view, FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "struct"
                  value_count { min: 2 max: 2 }
                  type: STRUCT
                  presence { min_count: 1 }
                  struct_domain {
                    feature { name: "bar" }
                    feature {
                      name: "foo"
                      value_count { min: 2 max: 2 }
                      type: INT
                      presence { min_count: 1 }
                    }
                  }
                }
              )"));
}

// Test if FeatureIsDeprecated using FeatureStatsView.
TEST(SchemaTest, FeatureIsDeprecatedTrue) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"pb(
        feature {
          name: "struct"
          type: STRUCT
          struct_domain {
            feature: {
              name: "foo"
              lifecycle_stage: DEPRECATED

            }
            feature: {
              name: "bar.baz"
              lifecycle_stage: DEPRECATED

            }
          }
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature: { name: "foo" lifecycle_stage: DEPRECATED }
            feature: { name: "bar" lifecycle_stage: DEPRECATED }
            feature: { name: "baz.bar" lifecycle_stage: DEPRECATED }
          }
        }
        feature {
          name: "(ext.field)"
          type: STRUCT
          struct_domain {
            feature: { name: "foo" lifecycle_stage: DEPRECATED }
            feature: { name: "bar" lifecycle_stage: DEPRECATED }
            sparse_feature: { name: "deep_sparse" lifecycle_stage: DEPRECATED }
          }
        }
        sparse_feature: { name: "shallow_sparse" lifecycle_stage: DEPRECATED }
        feature { name: "foo.bar" lifecycle_stage: DEPRECATED })pb");

  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features { name: "struct" type: STRUCT }
            features { name: "struct.foo" }
            features { name: "struct.bar.baz" }
            features { name: "##SEQUENCE##" type: STRUCT }
            features { name: "##SEQUENCE##.foo" }
            features: { name: "##SEQUENCE##.bar" }
            features: { name: "##SEQUENCE##.baz.bar" }
            features: { name: "(ext.field)" type: STRUCT }
            features: { name: "(ext.field).foo" }
            features: { name: "(ext.field).bar" }
            features: { name: "(ext.field).deep_sparse" }
            features: { name: "shallow_sparse" }
            features { name: "foo.bar" })pb");

  DatasetStatsView view(stats);
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"foo.bar"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"##SEQUENCE##", "foo"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"(ext.field)", "foo"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"(ext.field)", "deep_sparse"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"shallow_sparse"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"struct", "foo"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"struct", "bar.baz"})));
  EXPECT_TRUE(schema.FeatureIsDeprecated(Path({"##SEQUENCE##", "baz.bar"})));
}

TEST(SchemaTest, GetSchemaWithDash) {
  FeatureNameStatistics feature_statistics =
      ParseTextProtoOrDie<FeatureNameStatistics>(
          R"(name: 'name-with-dash'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 3
                 num_non_missing: 3
                 min_num_values: 2
                 max_num_values: 2
               }
               unique: 3
               rank_histogram: {
                 buckets: { label: "foo" }
                 buckets: { label: "bar" }
                 buckets: { label: "baz" }
               }
             })");
  FeatureStatisticsToProtoConfig config;
  DatasetFeatureStatistics dataset_statistics;
  *dataset_statistics.add_features() = feature_statistics;
  Schema schema;
  TF_ASSERT_OK(
      schema.Update(DatasetStatsView(dataset_statistics, false), config));
  tensorflow::metadata::v0::Schema result = schema.GetSchema();
  EXPECT_THAT(result, EqualsProto(R"(
                feature {
                  name: "name-with-dash"
                  presence: { min_count: 1 }
                  value_count { min: 2 max: 2 }
                  type: BYTES
                })"));
}

TEST(SchemaTest, GetSchema) {
  const tensorflow::metadata::v0::Schema test_schema_alone =
      GetTestSchemaAlone();
  Schema schema;
  TF_ASSERT_OK(schema.Init(test_schema_alone));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(GetTestSchemaAlone()));
}

TEST(SchemaTest, GetSchemaWithOptions) {
  Schema schema;
  TF_ASSERT_OK(schema.Init(GetAnnotatedFieldsMessage()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(GetAnnotatedFieldsMessage()));
}

TEST(SchemaTest, GetSchemaWithValueCounts) {
  Schema schema;
  TF_ASSERT_OK(schema.Init(tensorflow::metadata::v0::Schema()));
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "feature_with_one_nestedness_level"
          type: INT
          num_stats {
            common_stats {
              num_non_missing: 10
              num_missing: 10
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1.0
              tot_num_values: 10
            }
          }
        },
        features: {
          name: "feature_with_multiple_nestedness_levels"
          type: INT
          type: INT
          num_stats {
            common_stats {
              num_non_missing: 10
              num_missing: 10
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1.0
              tot_num_values: 10
              presence_and_valency_stats {
                num_non_missing: 10
                num_missing: 10
                min_num_values: 1
                max_num_values: 1
                tot_num_values: 10
              }
              presence_and_valency_stats {
                num_non_missing: 10
                min_num_values: 2
                max_num_values: 2
                tot_num_values: 20
              }
              presence_and_valency_stats {
                num_non_missing: 10
                min_num_values: 1
                max_num_values: 2
                tot_num_values: 15
              }
            }
          }
        })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, /*by_weight=*/false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "feature_with_one_nestedness_level"
                  type: INT
                  value_count { min: 1 max: 1 }
                  presence { min_count: 1 }
                },
                feature {
                  name: "feature_with_multiple_nestedness_levels"
                  type: INT
                  value_counts {
                    value_count { min: 1 max: 1 }
                    value_count { min: 2 max: 2 }
                    value_count { min: 1 }
                  }
                  presence { min_count: 1 }
                })"));
}

// Construct a schema from a DatasetFeatureStatistics, and then write it to a
// SchemaProto.
struct SchemaStatisticsConstructorTest {
  string name;
  FeatureNameStatistics statistics;
  tensorflow::metadata::v0::Schema expected;
};

TEST(FeatureTypeTest, ConstructFromSchemaStatistics) {
  std::vector<SchemaStatisticsConstructorTest> tests = {
      {"repeated_string",
       ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(name: 'bar1'
              type: STRING
              string_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 3
                  min_num_values: 2
                  max_num_values: 2
                }
                unique: 3
                rank_histogram: {
                  buckets: { label: "foo" }
                  buckets: { label: "bar" }
                  buckets: { label: "baz" }
                }
              })"),
       ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
         feature {
           value_count { min: 2  max: 2 }
           presence { min_count: 1 }
           name: "bar1"
           type: BYTES
         })")},
      {"optional_string", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar2'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 num_non_missing: 3 max_num_values: 1 }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"),
       ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(
           R"(feature {
                name: "bar2"
                presence: { min_count: 1 }
                type: BYTES
              })")},
      {"repeated_int", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar3'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 num_non_missing: 3 max_num_values: 2 }
         })"), ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
         feature {
           name: "bar3"
           presence: { min_count: 1 }
           type: INT
         })")},
      {"required_float", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar4'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 3
             max_num_values: 1
             min_num_values: 1
           }
         })"), ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
         feature {
           name: "bar4"
           value_count: { min: 1 max: 1 }
           presence: { min_fraction: 1 min_count: 1 }
           type: FLOAT
         })")}};
  for (const auto& test : tests) {
    FeatureStatisticsToProtoConfig config;
    DatasetFeatureStatistics dataset_statistics;
    *dataset_statistics.add_features() = test.statistics;

    Schema schema;
    TF_ASSERT_OK(schema.Update(DatasetStatsView(dataset_statistics), config));
    tensorflow::metadata::v0::Schema actual = schema.GetSchema();
    EXPECT_THAT(actual, EqualsProto(test.expected)) << "test: " << test.name;
  }
}

// Read a schema from a SchemaProto
// update using DatasetFeatureStatistics
// write it to a SchemaProto.
TEST(SchemaTest, Update) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "optional_bool"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features {
          name: "optional_enum"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
            unique: 3
            rank_histogram: {
              buckets: { label: "ALABAMA" }
              buckets: { label: "ALASKA" }
              buckets: { label: "CALIFORNIA" }
            }
          }
        }
        features: {
          name: 'optional_float'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'optional_int32'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'optional_int64'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'optional_string'
          type: STRING
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'optional_string_valid'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 2
            }
            unique: 3
            rank_histogram: {
              buckets: { label: "foo" }
              buckets: { label: "bar" }
              buckets: { label: "baz" }
            }
          }
        }
        features: {
          name: 'optional_uint32'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_bool"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_enum"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_float"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_int32"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_int64"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_string"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "repeated_uint32"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(GetTestAllTypesMessage()));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  tensorflow::metadata::v0::Schema expected = GetTestAllTypesMessage();
  // One message gets inserted at the end.
  *expected.add_feature() =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "optional_string_valid"
        value_count { min: 1 }
        type: BYTES
        presence { min_count: 1 })");
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(SchemaTest, UpdateWeightedNoAnomaly) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "weighted_feature"
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_weight" num: 0 }
          custom_stats { name: "min_weight_length_diff" num: 0 }
          custom_stats { name: "max_weight_length_diff" num: 0 }
        })");
  tensorflow::metadata::v0::Schema original =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'weighted_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(original));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  tensorflow::metadata::v0::Schema expected = original;
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(SchemaTest, UpdateWeightedNameCollision) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "existing_feature"
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_weight" num: 0 }
          custom_stats { name: "min_weight_length_diff" num: 0 }
          custom_stats { name: "max_weight_length_diff" num: 0 }
        })");
  tensorflow::metadata::v0::Schema original =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
        }
        feature: { name: 'existing_feature' type: INT })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(original));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  tensorflow::metadata::v0::Schema expected =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
          lifecycle_stage: DEPRECATED
        }
        feature: {
          name: 'existing_feature'
          type: INT
          lifecycle_stage: DEPRECATED
        })");
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(SchemaTest, UpdateWeightedSparseNameCollision) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "existing_feature"
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_weight" num: 0 }
          custom_stats { name: "min_weight_length_diff" num: 0 }
          custom_stats { name: "max_weight_length_diff" num: 0 }
          custom_stats {
            name: "missing_index"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
          custom_stats {
            name: "max_length_diff"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
          custom_stats {
            name: "min_length_diff"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
        })");
  tensorflow::metadata::v0::Schema original =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
        }
        sparse_feature {
          name: 'existing_feature'
          index_feature { name: 'index_feature' }
          value_feature { name: 'value_feature' }
        })");

  Schema schema;
  TF_ASSERT_OK(schema.Init(original));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  tensorflow::metadata::v0::Schema expected =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
          lifecycle_stage: DEPRECATED
        }
        sparse_feature {
          name: 'existing_feature'
          index_feature { name: 'index_feature' }
          value_feature { name: 'value_feature' }
          lifecycle_stage: DEPRECATED
        })");
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(SchemaTest, UpdateWeightedSparseRegularNameCollision) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "existing_feature"
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_value" num: 0 }
          custom_stats { name: "missing_weight" num: 0 }
          custom_stats { name: "min_weight_length_diff" num: 0 }
          custom_stats { name: "max_weight_length_diff" num: 0 }
          custom_stats {
            name: "missing_index"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
          custom_stats {
            name: "max_length_diff"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
          custom_stats {
            name: "min_length_diff"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
        })");
  tensorflow::metadata::v0::Schema original =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
        }
        sparse_feature {
          name: 'existing_feature'
          index_feature { name: 'index_feature' }
          value_feature { name: 'value_feature' }
        }
        feature: { name: 'existing_feature' type: INT })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(original));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();

  tensorflow::metadata::v0::Schema expected =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        weighted_feature {
          name: 'existing_feature'
          feature { step: 'value_feature' }
          weight_feature { step: 'weight_feature' }
          lifecycle_stage: DEPRECATED
        }
        sparse_feature {
          name: 'existing_feature'
          index_feature { name: 'index_feature' }
          value_feature { name: 'value_feature' }
          lifecycle_stage: DEPRECATED
        }
        feature: {
          name: 'existing_feature'
          type: INT
          lifecycle_stage: DEPRECATED
        })");
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(SchemaTest, RequiredFeatures) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "required_string"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: "required_int"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(tensorflow::metadata::v0::Schema()));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "required_string"
                  type: BYTES
                  value_count: { min: 1 max: 1 }
                  presence { min_fraction: 1 min_count: 1 }
                }
                feature {
                  name: "required_int"
                  type: INT
                  value_count: { min: 1 max: 1 }
                  presence { min_fraction: 1 min_count: 1 }
                })"));
}

// As requested in b/62826201, if the data is always present, then even if it
// is a repeated field, we infer that it will always be there in the future.
TEST(SchemaTest, RequiredRepeatedFeatures) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "required_repeated_string"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 10
            }
          }
        }
        features: {
          name: "required_repeated_int"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 3
              max_num_values: 10
            }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(tensorflow::metadata::v0::Schema()));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                             FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "required_repeated_string"
                  type: BYTES
                  value_count: { min: 1 }
                  presence { min_fraction: 1 min_count: 1 }
                }
                feature {
                  name: "required_repeated_int"
                  type: INT
                  value_count: { min: 1 }
                  presence { min_fraction: 1 min_count: 1 }
                })"));
}

TEST(SchemaTest, InferFeatureShape) {
  const auto statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "f1"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(tensorflow::metadata::v0::Schema()));
  FeatureStatisticsToProtoConfig config;
  config.set_infer_feature_shape(true);
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false), config));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "f1"
                  type: INT
                  shape { dim { size: 1 } }
                  presence { min_fraction: 1 min_count: 1 }
                })"));
}

TEST(SchemaTest, UpdateFeatureShape) {
  const auto statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "f1"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 2
            }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(
      schema.Init(ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "f1"
          type: INT
          shape { dim { size: 1 } }
          presence { min_fraction: 1 min_count: 1 }
        }
      )")));
  TF_ASSERT_OK(schema.Update(
      DatasetStatsView(statistics, false), FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                feature {
                  name: "f1"
                  type: INT
                  presence { min_fraction: 1 min_count: 1 }
                })"));
}

// b/180761541
TEST(SchemaTest, UpdateFeatureShapeInferLegacyFeatureSpecWithNumMissing) {
  const auto statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "f1"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 1
              num_non_missing: 9
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  auto schema_proto = ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
    feature {
      name: "f1"
      type: INT
      shape { dim { size: 1 } }
      presence { min_fraction: 1 min_count: 1 }
    }
  )");
  auto* field_desc = schema_proto.GetDescriptor()->FindFieldByName(
      "generate_legacy_feature_spec");
  if (!field_desc) {
    // Skip the test because the schema does not have the legacy field (OSS).
    return;
  }

  // The default value of that field is true.
  ASSERT_TRUE(schema_proto.GetReflection()->GetBool(schema_proto, field_desc));
  {
    Schema schema;
    TF_ASSERT_OK(schema.Init(schema_proto));
    TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                               FeatureStatisticsToProtoConfig()));
    EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                  feature {
                    name: "f1"
                    type: INT
                    shape { dim { size: 1 } }
                    presence { min_fraction: 0.9 min_count: 1 }
                  })"));
  }

  schema_proto.GetReflection()->SetBool(&schema_proto, field_desc, false);
  {
    Schema schema;
    TF_ASSERT_OK(schema.Init(schema_proto));
    TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false),
                               FeatureStatisticsToProtoConfig()));
    EXPECT_THAT(schema.GetSchema(), EqualsProto(R"(
                  generate_legacy_feature_spec: false
                  feature {
                    name: "f1"
                    type: INT
                    presence { min_fraction: 0.9 min_count: 1 }
                  })"));
  }
}

// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct ValidTest {
  const string name;
  const FeatureNameStatistics statistics;
  const bool expected_is_valid;
};

TEST(SchemaTest, ValidTestAllTypesMessage) {
  const std::vector<ValidTest> valid_tests = {
      {"optional_float_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: FLOAT
         num_stats: {
           common_stats: { num_missing: 0 min_num_values: 1 max_num_values: 1 }
         })"), true},
      {"optional_string_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"), false},
      {"optional_int64_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 0 min_num_values: 1 max_num_values: 1 }
         })"), true},
      {"optional_bool_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'optional_bool'
         type: INT
         num_stats: {
           common_stats: { num_missing: 0 min_num_values: 1 max_num_values: 1 }
           min: 0.0
           max: 1.0
         })"), true},
      {"repeated_float_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'repeated_float'
         type: FLOAT
         num_stats: {
           common_stats: { num_missing: 0 min_num_values: 1 max_num_values: 1 }
         })"), true},
      {"repeated_string_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'repeated_string'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"), true},
      {"repeated_int64_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: {
             num_missing: 10000
             min_num_values: 1
             max_num_values: 1012
           }
           min: 0.0
           max: 1.0
         })"), true},
      {"repeated_bool_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: {
             num_missing: 10000
             min_num_values: 1
             max_num_values: 1012
           }
           min: 0.0
           max: 1.0
         })"), true}};
}

DatasetFeatureStatistics GetDatasetFeatureStatisticsForAnnotated() {
  return ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
    num_examples: 10
    features: {
      name: 'bool_with_true'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 1
        rank_histogram: { buckets: { label: "my_true" } }
      }
    }
    features: {
      name: 'bool_with_false'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 1
        rank_histogram: { buckets: { label: "my_false" } }
      }
    }
    features: {
      name: 'bool_with_true_false'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 1
        rank_histogram: { buckets: { label: "my_false" } }
      }
    }
    features: {
      name: 'string_int64'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 3
        rank_histogram: {
          buckets: { label: "12" }
          buckets: { label: "39" }
          buckets: { label: "256" }
        }
      }
    }
    features: {
      name: 'string_int32'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 3
        rank_histogram: {
          buckets: { label: "12" }
          buckets: { label: "39" }
          buckets: { label: "256" }
        }
      }
    }
    features: {
      name: 'annotated_enum'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 3
        rank_histogram: {
          buckets: { label: "4" }
          buckets: { label: "5" }
          buckets: { label: "6" }
        }
      }
    }
    features: {
      name: 'string_uint32'
      type: STRING
      string_stats: {
        common_stats: {
          num_missing: 3
          num_non_missing: 7
          min_num_values: 1
          max_num_values: 1
        }
        unique: 3
        rank_histogram: {
          buckets: { label: "12" }
          buckets: { label: "39" }
          buckets: { label: "256" }
        }
      }
    }
    features: {
      name: 'few_int64'
      type: INT
      num_stats: {
        common_stats: {
          num_missing: 0
          num_non_missing: 10
          min_num_values: 1
          max_num_values: 3
        }
        min: 0.0
        max: 1.0
      }
    }
    features: {
      name: 'float_with_bounds'
      type: FLOAT
      num_stats: {
        common_stats: {
          num_missing: 9
          num_non_missing: 1
          min_num_values: 1
          max_num_values: 1
        }
        min: 0.0
        max: 1.0
      }
    }
    features: {
      name: 'float_very_common'
      type: FLOAT
      num_stats: {
        common_stats: {
          num_missing: 0
          num_non_missing: 10
          min_num_values: 1
          max_num_values: 1
        }
        min: 0.0
        max: 1.0
      }
    }
    features: {
      name: 'small_int64'
      type: INT
      num_stats: {
        common_stats: {
          num_missing: 0
          num_non_missing: 10
          min_num_values: 1
          max_num_values: 1
        }
      }
    }
    features: {
      name: 'big_int64'
      type: INT
      num_stats: {
        common_stats: {
          num_missing: 0
          num_non_missing: 10
          min_num_values: 1
          max_num_values: 1
        }
        min: 127.0
        max: 2456.0
      }
    })");
}

TEST(SchemaTest, ValidTestAnnotatedFieldsMessage) {
  const std::vector<ValidTest> tests = {
      {"string_int64_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_int64'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "12" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), true},
      {"string_int64_to_repeated",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_int64'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "12" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), false},
      {"string_int32_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_int32'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "12" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), true},
      {"string_int32_to_repeated_string",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_int32'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "FOO" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), false},
      {"string_uint32_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_uint32'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "12" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), true},
      {"string_uint32_negatives", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'string_uint32'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "-12" }
             buckets: { label: "39" }
             buckets: { label: "256" }
           }
         })"), false},
      {"few_int64_valid", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'few_int64'
         type: INT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 3
           }
           min: 0.0
           max: 1.0
         })"), true},
      {"float_with_bounds", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'float_with_bounds'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 12
             num_non_missing: 1
             min_num_values: 1
             max_num_values: 1
           }
           min: 0.0
           max: 1.0
         })"), true}};
  for (const auto& test : tests) {
    Schema schema;
    TF_ASSERT_OK(schema.Init(GetAnnotatedFieldsMessage()));
    DatasetFeatureStatistics statistics =
        GetDatasetFeatureStatisticsForAnnotated();
    for (FeatureNameStatistics& features : *statistics.mutable_features()) {
      if (features.name() == test.statistics.name()) {
        features = test.statistics;
      }
    }
    TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics),
                               FeatureStatisticsToProtoConfig()));
    const tensorflow::metadata::v0::Schema result = schema.GetSchema();
    if (test.expected_is_valid) {
      EXPECT_THAT(result, EqualsProto(GetAnnotatedFieldsMessage()));
    }
  }
}

TEST(Schema, GetRelatedEnums) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'field_a'
          type: STRING
          string_stats: {
            common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 1 }
            unique: 3
            rank_histogram: {
              buckets: { label: "A" }
              buckets: { label: "B" }
              buckets: { label: "C" }
            }
          }
        }
        features: {
          name: 'field_b'
          type: STRING
          string_stats: {
            common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 1 }
            unique: 3
            rank_histogram: {
              buckets: { label: "A" }
              buckets: { label: "B" }
              buckets: { label: "C" }
            }
          }
        }
      )");
  const FeatureStatisticsToProtoConfig proto_config =
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          R"(enum_threshold: 400)");
  FeatureStatisticsToProtoConfig actual = proto_config;
  TF_ASSERT_OK(
      Schema::GetRelatedEnums(DatasetStatsView(statistics, false), &actual));
  EXPECT_THAT(actual, EqualsProto(R"(
                enum_threshold: 400
                column_constraint {
                  column_path {
                    step: "field_a"
                  }
                  column_path  {
                    step: "field_b"
                  }
                  enum_name: "field_a"
                })"));
}

TEST(Schema, GetRelatedEnumsFeatureIdIsPath) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          path {
            step: 'field_a'
          }
          type: STRING
          string_stats: {
            common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 1 }
            unique: 3
            rank_histogram: {
              buckets: { label: "A" }
              buckets: { label: "B" }
              buckets: { label: "C" }
            }
          }
        }
        features: {
          path {
            step: 'field_b'
          }
          type: STRING
          string_stats: {
            common_stats: { num_missing: 3 min_num_values: 1 max_num_values: 1 }
            unique: 3
            rank_histogram: {
              buckets: { label: "A" }
              buckets: { label: "B" }
              buckets: { label: "C" }
            }
          }
        }
      )");
  const FeatureStatisticsToProtoConfig proto_config =
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          R"(enum_threshold: 400)");
  FeatureStatisticsToProtoConfig actual = proto_config;
  TF_ASSERT_OK(
      Schema::GetRelatedEnums(DatasetStatsView(statistics, false), &actual));
  EXPECT_THAT(actual, EqualsProto(R"(
                enum_threshold: 400
                column_constraint {
                  column_path {
                    step: "field_a"
                  }
                  column_path  {
                    step: "field_b"
                  }
                  enum_name: "field_a"
                })"));
}

TEST(Schema, MissingColumns) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
        feature {
          name: "missing_float_column"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: FLOAT
        }
        feature {
          name: "annotated_enum"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          domain: "MyAloneEnum"
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          presence: { min_count: 1 }
          value_count: { min: 1 }
          type: BYTES
        })");
  FeatureStatisticsToProtoConfig config;
  config.set_enum_threshold(100);

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 1 } }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false), config));
  const tensorflow::metadata::v0::Schema result = schema.GetSchema();
  EXPECT_THAT(result, EqualsProto(R"(
                feature {
                  name: "missing_float_column"
                  lifecycle_stage: DEPRECATED
                  presence: { min_count: 1 }
                  value_count: { min: 1 max: 1 }
                  type: FLOAT
                }
                feature {
                  name: "annotated_enum"
                  presence: { min_count: 1 }
                  value_count: { min: 1 max: 1 }
                  type: BYTES
                  domain: "MyAloneEnum"
                }
                feature {
                  name: "ignore_this"
                  lifecycle_stage: DEPRECATED
                  presence: { min_count: 1 }
                  value_count: { min: 1 }
                  type: BYTES
                }
                string_domain {
                  name: "MyAloneEnum"
                  value: "A"
                  value: "B"
                  value: "C"
                  value: "D"
                })"));
}

TEST(Schema, UnchangedProto) {
  const tensorflow::metadata::v0::Schema initial =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
        feature {
          name: "annotated_enum"
          type: BYTES
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          domain: "MyAloneEnum"
        })");
  FeatureStatisticsToProtoConfig config;
  config.set_enum_threshold(100);

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "A" sample_count: 1 } }
          }
        })");
  Schema schema;
  TF_ASSERT_OK(schema.Init(initial));
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, false), config));
  const tensorflow::metadata::v0::Schema result = schema.GetSchema();
  EXPECT_THAT(result, EqualsProto(R"(
                feature {
                  name: "annotated_enum"
                  presence: { min_count: 1 }
                  value_count: { min: 1 max: 1 }
                  type: BYTES
                  domain: "MyAloneEnum"
                }
                string_domain {
                  name: "MyAloneEnum"
                  value: "A"
                  value: "B"
                  value: "C"
                })"));
}

TEST(Schema, EmptySchemaProto) {
  Schema schema;
  TF_EXPECT_OK(schema.Init(tensorflow::metadata::v0::Schema()));
}

// Converted to a test that the schema does not change.
TEST(SchemaTest, ValidTestAnnotatedFieldsMessageBaseline) {
  const DatasetStatsView statistics(GetDatasetFeatureStatisticsForAnnotated());
  Schema schema;

  const tensorflow::metadata::v0::Schema original = GetAnnotatedFieldsMessage();
  TF_ASSERT_OK(schema.Init(original));
  TF_ASSERT_OK(schema.Update(statistics, FeatureStatisticsToProtoConfig()));
  EXPECT_THAT(original, EqualsProto(schema.GetSchema()));
}

TEST(SchemaTest, UpdateUniqueConstraints) {
  Schema schema;
  TF_ASSERT_OK(
      schema.Init(ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "categorical_feature"
          type: INT
          int_domain { is_categorical: true }
          unique_constraints { min: 2 max: 2 }
        },
        feature {
          name: "string_feature"
          type: BYTES
          unique_constraints { min: 2 max: 2 }
        })")));
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: INT
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 5
          }
        },
        features: {
          name: "string_feature"
          type: STRING
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 1
          }
        })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, /*by_weight=*/false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "categorical_feature"
                  type: INT
                  int_domain { is_categorical: true }
                  unique_constraints { min: 2 max: 5 }
                },
                feature {
                  name: "string_feature"
                  type: BYTES
                  unique_constraints { min: 1 max: 2 }
                })"));
}

TEST(SchemaTest, AddsDerivedFeature) {
  Schema schema;
  TF_ASSERT_OK(
      schema.Init(ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>("")));
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: STRING
          string_stats {}
          validation_derived_source: {
            deriver_name: 'deriver_name'
          }
        })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, /*by_weight=*/false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "categorical_feature"
                  type: BYTES
                  presence {
                    min_count: 0
                  }
                  lifecycle_stage: VALIDATION_DERIVED
                  validation_derived_source: {
                    deriver_name: 'deriver_name'
                  }
                })"));
}

TEST(SchemaTest, UpdatesDerivedFeatureWithoutSource) {
  Schema schema;
  TF_ASSERT_OK(
      schema.Init(ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
                feature {
                  name: "categorical_feature"
                  type: BYTES
                })")));
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: STRING
          string_stats {common_stats: {num_non_missing: 10}}
          validation_derived_source: {
            deriver_name: 'deriver_name'
          }
        })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, /*by_weight=*/false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "categorical_feature"
                  type: BYTES
                  lifecycle_stage: VALIDATION_DERIVED
                  validation_derived_source: {
                    deriver_name: "deriver_name"
                  }
                })"));
}

TEST(SchemaTest, UpdatesDerivedFeatureWithBadLifecycle) {
  Schema schema;
  TF_ASSERT_OK(
      schema.Init(ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
                feature {
                  name: "categorical_feature"
                  type: BYTES
                  lifecycle_stage: PRODUCTION
                  validation_derived_source: {
                    deriver_name: "deriver_name"
                  }
                })")));
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: STRING
          string_stats {common_stats: {num_non_missing: 10}}
          validation_derived_source: {
            deriver_name: 'deriver_name'
          }
        })");
  TF_ASSERT_OK(schema.Update(DatasetStatsView(statistics, /*by_weight=*/false),
                             FeatureStatisticsToProtoConfig()));
  const tensorflow::metadata::v0::Schema actual = schema.GetSchema();
  EXPECT_THAT(actual, EqualsProto(R"(
                feature {
                  name: "categorical_feature"
                  type: BYTES
                  lifecycle_stage: VALIDATION_DERIVED
                  validation_derived_source: {
                    deriver_name: "deriver_name"
                  }
                })"));
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
