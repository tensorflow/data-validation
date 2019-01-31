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

#include "tensorflow_data_validation/anomalies/schema_anomalies.h"

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::Schema;
using testing::ParseTextProtoOrDie;

void TestFindChanges(
    const Schema& schema, const DatasetStatsView& stats_view,
    const FeatureStatisticsToProtoConfig& config,
    const std::map<string, testing::ExpectedAnomalyInfo>& expected_anomalies) {
  SchemaAnomalies anomalies(schema);
  TF_CHECK_OK(anomalies.FindChanges(stats_view, absl::nullopt, config));
  TestAnomalies(anomalies.GetSchemaDiff(), schema, expected_anomalies);
}

std::vector<FeatureStatisticsToProtoConfig>
GetFeatureStatisticsToProtoConfigs() {
  return std::vector<FeatureStatisticsToProtoConfig>(
      {FeatureStatisticsToProtoConfig(),
       ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
           "new_features_are_warnings: true")});
}

// The dash in the name might cause issues.
TEST(SchemaAnomalies, FindChangesNoChanges) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "capital-gain"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "capital-gain"  # size=12
          type: FLOAT           # integer=1 (enum
                       # tensorflow.metadata.v0.FeatureNameStatistics.Type)
          num_stats: {
            # (tensorflow.metadata.v0.NumericStatistics) size=417B
            common_stats: {
              # (tensorflow.metadata.v0.CommonStatistics) size=13B
              num_non_missing: 0x0000000000007f31  # 32_561
              min_num_values: 0x0000000000000001
              max_num_values: 0x0000000000000001
              avg_num_values: 1.0
            }  # datasets[0].features[9].num_stats.common_stats
            mean: 1077.6488437087312
            std_dev: 7385.1786769476275
            num_zeros: 0x0000000000007499  # 29_849
            max: 99999.0                   # [if seconds]: 1 day 3 hours
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    TestFindChanges(initial, DatasetStatsView(statistics, false), config,
                    std::map<string, testing::ExpectedAnomalyInfo>());
  }
}

TEST(SchemaAnomalies, FindChangesCategoricalIntFeature) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "a_int"
      value_count: { min: 1 max: 1 }
      type: INT
      int_domain { is_categorical: true }
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'a_int'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 2
              avg_num_values: 1.5
            }
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["a_int"].new_schema = ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "a_int"
        value_count { min: 1 max: 2 }
        type: INT
        int_domain { is_categorical: true }
      })");
    expected_anomalies["a_int"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "a_int" }
          description: "Some examples have more values than expected."
          severity: ERROR
          short_description: "Superfluous values"
          reason {
            type: FEATURE_TYPE_HIGH_NUMBER_VALUES
            short_description: "Superfluous values"
            description: "Some examples have more values than expected."
          })");
    TestFindChanges(initial, DatasetStatsView(statistics, false), config,
                    expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindChanges) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
    feature {
      name: "annotated_enum"
      presence: { min_count: 1 }
      value_count: { min: 1 max: 1 }
      type: BYTES
      domain: "MyAloneEnum"
      annotation { tag: "some tag" comment: "some comment" }
    }
    feature {
      name: "ignore_this"
      lifecycle_stage: DEPRECATED
      presence: { min_count: 1 }
      value_count: { min: 1 }
      type: BYTES
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" sample_count: 2 } }
          }
        })");

  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["annotated_enum"].new_schema =
        ParseTextProtoOrDie<Schema>(R"(
          feature {
            name: "annotated_enum"
            value_count { min: 1 max: 1 }
            type: BYTES
            domain: "MyAloneEnum"
            presence { min_count: 1 }
            annotation { tag: "some tag" comment: "some comment" }
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
    expected_anomalies["annotated_enum"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "annotated_enum" }
      description: "Examples contain values missing from the schema: D (~50%). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D (~50%). "
      })");

    TestFindChanges(initial, DatasetStatsView(statistics, false), config,
                    expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindSkew) {
  const DatasetFeatureStatistics training =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
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
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
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

  const Schema schema_proto = ParseTextProtoOrDie<Schema>(R"(
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
          /* previous= */ std::shared_ptr<DatasetStatsView>(), serving_view);

  SchemaAnomalies skew(schema_proto);
  TF_CHECK_OK(skew.FindSkew(*training_view));
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "foo"
      type: BYTES
      skew_comparator { infinity_norm: { threshold: 0.19999999999999998 } }
    })");
  expected_anomalies["foo"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path { step: "foo" }
    description: "The Linfty distance between serving and training is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
    severity: ERROR
    short_description: "High Linfty distance between serving and training"
    reason {
      type: COMPARATOR_L_INFTY_HIGH
      short_description: "High Linfty distance between serving and training"
      description: "The Linfty distance between serving and training is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
    })");
  TestAnomalies(skew.GetSchemaDiff(), schema_proto, expected_anomalies);
}

TEST(Schema, FindChangesEmptySchemaProto) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
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
            rank_histogram: { buckets: { label: "D" } }
          }
        })");
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["annotated_enum"].new_schema =
      ParseTextProtoOrDie<Schema>(R"(
        feature {
          name: "annotated_enum"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
        })");
  expected_anomalies["annotated_enum"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path { step: "annotated_enum" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  TestFindChanges(Schema(), DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

// TODO(martinz): Figure out why there isn't a separate anomaly for the format
// of the annotated_enum.
TEST(Schema, FindChangesOnlyValidateSchemaFeatures) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: 'new_feature'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
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
            rank_histogram: { buckets: { label: "A" } }
          }
        })");

  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
    string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" }
    feature {
      name: "annotated_enum"
      presence: { min_count: 1 }
      value_count: { min: 1 max: 1 }
      type: BYTES
      domain: "MyAloneEnum"
    })");

  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["new_feature"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "annotated_enum"
      value_count { min: 1 max: 1 }
      type: BYTES
      domain: "MyAloneEnum"
      presence { min_count: 1 }
    }
    feature {
      name: "new_feature"
      value_count { min: 1 max: 1 }
      type: INT
      presence { min_count: 1 }
    }
    string_domain { name: "MyAloneEnum" value: "A" value: "B" value: "C" })");
  expected_anomalies["new_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: "new_feature" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  {
    const FeatureStatisticsToProtoConfig config =
        ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
            "new_features_are_warnings: false");

    TestFindChanges(schema, DatasetStatsView(statistics, false), config,
                    expected_anomalies);
  }
  expected_anomalies["new_feature"].expected_info_without_diff.set_severity(
      tensorflow::metadata::v0::AnomalyInfo::WARNING);

  TestFindChanges(schema, DatasetStatsView(statistics, false),
                  ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
                      "new_features_are_warnings: true"),
                  expected_anomalies);
}

TEST(GetSchemaDiff, BasicTest) {
  Schema initial =
      ParseTextProtoOrDie<Schema>(R"(feature { name: "bar" type: INT })");
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: 'bar'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'foo'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
      )");
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature { name: "bar" type: INT }
    feature {
      name: "foo"
      value_count { min: 1 max: 1 }
      type: INT
      presence { min_count: 1 }
    })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: "foo" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  TestFindChanges(initial, DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

// Make updates only to existing features and features_needed.
// foo is needed, so it is created.
// bar exists, so it is fixed.
// no_worries is neither needed nor existing, so it is unchanged.
TEST(GetSchemaDiff, FindSelectedChanges) {
  Schema initial = ParseTextProtoOrDie<Schema>(R"(feature {
                                                    name: "bar"
                                                    type: INT
                                                    value_count { max: 1 }
                                                  })");
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: 'bar'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 2
            }
          }
        }
        features: {
          name: 'foo'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'foo_noworries'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
      )"

      );
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "bar"
      value_count { max: 1 }
      type: INT
    }
    feature {
      name: "foo"
      value_count { min: 1 max: 1 }
      type: INT
      presence { min_count: 1 }
    })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path { step: "foo" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");
  expected_anomalies["bar"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "bar"
      value_count { max: 2 }
      type: INT
    })");
  expected_anomalies["bar"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path { step: "bar" }
        description: "Some examples have more values than expected."
        severity: ERROR
        short_description: "Superfluous values"
        reason {
          type: FEATURE_TYPE_HIGH_NUMBER_VALUES
          short_description: "Superfluous values"
          description: "Some examples have more values than expected."
        })");
  FeaturesNeeded features;
  // The next line creates a feature that is needed without a reason.
  features[Path({"foo"})];
  SchemaAnomalies anomalies(initial);
  TF_CHECK_OK(anomalies.FindChanges(DatasetStatsView(statistics), features,
                                    FeatureStatisticsToProtoConfig()));
  auto result = anomalies.GetSchemaDiff();

  TestAnomalies(result, initial, expected_anomalies);
}

TEST(GetSchemaDiff, ValidSparseFeature) {
  // Note: This schema is incomplete, as it does not fully define the index and
  // value features and we do not generate feature stats for them.
  const Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        sparse_feature: {
          name: 'sparse_feature'
          index_feature { name: 'index_feature1' }
          index_feature { name: 'index_feature2' }
          value_feature { name: 'value_feature' }
        })");

  const DatasetFeatureStatistics no_anomaly_stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "sparse_feature"
          custom_stats { name: "missing_value" num: 0 }
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
        }
      )");

  // No anomalies.
  TestFindChanges(schema_proto,
                  DatasetStatsView(no_anomaly_stats, /* by_weight= */ false),
                  FeatureStatisticsToProtoConfig(),
                  /*expected_anomalies=*/{});
}

// Same as above, but with name collision with existing feature.
TEST(GetSchemaDiff, SparseFeatureNameCollision) {
  // Note: This schema is incomplete, as it does not fully define the index and
  // value features and we do not generate feature stats for them.
  const Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        sparse_feature: {
          name: 'existing_feature'
          index_feature { name: 'index_feature' }
          value_feature { name: 'value_feature' }
        }
        feature: { name: 'existing_feature' type: INT })");

  const DatasetFeatureStatistics no_anomaly_stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "existing_feature"
          custom_stats { name: "missing_value" num: 0 }
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
  Schema schema_deprecated = schema_proto;
  DeprecateSparseFeature(schema_deprecated.mutable_sparse_feature(0));
  DeprecateFeature(schema_deprecated.mutable_feature(0));
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["existing_feature"].new_schema = schema_deprecated;
  expected_anomalies["existing_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path: { step: "existing_feature" }
        description: "Sparse feature name collision"
        severity: ERROR
        short_description: "Sparse feature name collision"
        reason {
          type: SPARSE_FEATURE_NAME_COLLISION
          short_description: "Sparse feature name collision"
          description: "Sparse feature name collision"
        })");
  TestFindChanges(schema_proto,
                  DatasetStatsView(no_anomaly_stats, /* by_weight= */ false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

TEST(SchemaAnomalyTest, CreateNewField) {
  // Empty schema proto.
  Schema baseline;
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
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  SchemaAnomaly anomaly;
  TF_ASSERT_OK(anomaly.InitSchema(baseline));

  TF_ASSERT_OK(
      anomaly.CreateNewField(::tensorflow::data_validation::Schema::Updater(
                                 FeatureStatisticsToProtoConfig()),
                             absl::nullopt, *view.GetByPath(Path({"struct"}))));

  testing::ExpectedAnomalyInfo expected_anomaly_info;
  expected_anomaly_info.new_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          value_count { min: 1 }
          type: STRUCT
          presence { min_count: 1 }
          struct_domain {
            feature {
              name: "bar.baz"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
            feature {
              name: "foo"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
          }
        })");
  expected_anomaly_info.expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path {}
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  testing::TestAnomalyInfo(anomaly.GetAnomalyInfo(baseline), baseline,
                           expected_anomaly_info, "CreateNewField failed");
}

TEST(SchemaAnomalyTest, CreateNewFieldSome) {
  // Empty schema proto.
  Schema baseline;
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
                  num_non_missing: 7
                  max_num_values: 2
                }
              }
            }
            features {
              name: "struct.'bar.baz'"
              type: INT
              num_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 4
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  SchemaAnomaly anomaly;
  TF_ASSERT_OK(anomaly.InitSchema(baseline));

  TF_ASSERT_OK(anomaly.CreateNewField(
      ::tensorflow::data_validation::Schema::Updater(
          FeatureStatisticsToProtoConfig()),
      std::set<Path>({Path({"struct"}), Path({"struct", "foo"})}),
      *view.GetByPath(Path({"struct"}))));

  testing::ExpectedAnomalyInfo expected_anomaly_info;
  expected_anomaly_info.new_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          value_count { min: 1 }
          type: STRUCT
          presence { min_count: 1 }
          struct_domain {
            feature {
              name: "foo"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
          }
        })");
  expected_anomaly_info.expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path {}
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        }
      )pb");

  testing::TestAnomalyInfo(anomaly.GetAnomalyInfo(baseline), baseline,
                           expected_anomaly_info, "CreateNewField failed");
}

// When there is a new structured feature, we create all of its children
// recursively and put all of them in one anomaly.
TEST(SchemaAnomaliesTest, FindChangesCreateDeep) {
  // Empty schema proto.
  Schema baseline;
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
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["struct"].new_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          value_count { min: 1 }
          type: STRUCT
          presence { min_count: 1 }
          struct_domain {
            feature {
              name: "bar.baz"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
            feature {
              name: "foo"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
          }
        })");
  expected_anomalies["struct"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path { step: "struct" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");
  TestFindChanges(baseline, view, FeatureStatisticsToProtoConfig(),
                  expected_anomalies);
}

// When a structured feature already exists, we create all of its children
// separately.
TEST(SchemaAnomaliesTest, FindChangesCreateDeepSeparately) {
  // Empty schema proto.
  Schema baseline = ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
    feature {
      name: "struct"
      value_count { min: 1 }
      type: STRUCT
      presence { min_count: 1 }
    })");

  const tensorflow::metadata::v0::DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<tensorflow::metadata::v0::DatasetFeatureStatistics>(
          R"pb(
            features {
              name: "struct"
              type: STRUCT
              struct_stats {
                common_stats {
                  num_missing: 3
                  num_non_missing: 6
                  min_num_values: 1
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
                  num_non_missing: 3
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
                  num_non_missing: 3
                  max_num_values: 2
                }
              }
            })pb");

  DatasetStatsView view(stats);
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["struct.foo"].new_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          value_count { min: 1 }
          type: STRUCT
          presence { min_count: 1 }
          struct_domain {
            feature {
              name: "foo"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
          }
        })");
  expected_anomalies["struct.foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: [ "struct", "foo" ] }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");
  expected_anomalies["struct.'bar.baz'"].new_schema =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "struct"
          value_count { min: 1 }
          type: STRUCT
          presence { min_count: 1 }
          struct_domain {
            feature {
              name: "bar.baz"
              value_count { min: 1 }
              type: INT
              presence { min_count: 1 }
            }
          }
        })");
  expected_anomalies["struct.'bar.baz'"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: [ "struct", "bar.baz" ] }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  TestFindChanges(baseline, view, FeatureStatisticsToProtoConfig(),
                  expected_anomalies);
}

// FindChanges should not find any errors if the data is not deprecated.
TEST(SchemaAnomaliesTest, FindChangesCreateDeepDeprecated) {
  // Empty schema proto.
  Schema baseline = ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
    feature {
      name: "struct"
      value_count { min: 1 }
      type: STRUCT
      lifecycle_stage: DEPRECATED
      presence { min_count: 1 }
    })");
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

  TestFindChanges(baseline, DatasetStatsView(stats),
                  FeatureStatisticsToProtoConfig(),
                  std::map<string, testing::ExpectedAnomalyInfo>());
}

TEST(SchemaAnomalyTest, FeatureIsDeprecated) {
  // Empty schema proto.
  Schema baseline = ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
    feature {
      name: "struct"
      value_count { min: 1 }
      type: STRUCT
      presence { min_count: 1 }
      struct_domain {
        feature {
          name: "foo"
          value_count { min: 1 }
          type: INT
          presence { min_count: 1 }
          lifecycle_stage: DEPRECATED
        }
        feature {
          name: "bar.baz"
          value_count { min: 1 }
          type: INT
          presence { min_count: 1 }
        }
      }
    })");

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
  SchemaAnomaly anomaly;
  TF_ASSERT_OK(anomaly.InitSchema(baseline));
  EXPECT_TRUE(anomaly.FeatureIsDeprecated(Path({"struct", "foo"})));
  EXPECT_FALSE(anomaly.FeatureIsDeprecated(Path({"struct"})));
  EXPECT_FALSE(anomaly.FeatureIsDeprecated(Path({"struct", "bar.baz"})));
}

TEST(GetSchemaDiff, MissingFeatureSparseFeature) {
  // Note: This schema is incomplete, as it does not fully define the index and
  // value features and we do not generate feature stats for them.
  const Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        sparse_feature: {
          name: 'sparse_feature'
          index_feature { name: 'index_feature1' }
          index_feature { name: 'index_feature2' }
          value_feature { name: 'value_feature' }
        })");

  // Missing value & missing index.
  const DatasetFeatureStatistics missing_features_stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "sparse_feature"
          custom_stats { name: "missing_value" num: 42 }
          custom_stats {
            name: "missing_index"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 10 }
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
        }
      )");

  Schema schema_deprecated = schema_proto;
  DeprecateSparseFeature(schema_deprecated.mutable_sparse_feature(0));
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["sparse_feature"].new_schema = schema_deprecated;
  expected_anomalies["sparse_feature"]
      .expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path: { step: "sparse_feature" }
    description: "Found 42 examples missing value feature Found 10 examples missing index feature: index_feature1"
    severity: ERROR
    short_description: "Multiple errors"
    reason {
      type: SPARSE_FEATURE_MISSING_VALUE
      short_description: "Missing value feature"
      description: "Found 42 examples missing value feature"
    }
    reason {
      type: SPARSE_FEATURE_MISSING_INDEX
      short_description: "Missing index feature"
      description: "Found 10 examples missing index feature: index_feature1"
    })");
  TestFindChanges(
      schema_proto,
      DatasetStatsView(missing_features_stats, /* by_weight= */ false),
      FeatureStatisticsToProtoConfig(), expected_anomalies);
}

TEST(GetSchemaDiff, LengthMismatchSparseFeature) {
  // Note: This schema is incomplete, as it does not fully define the index and
  // value features and we do not generate feature stats for them.
  const Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        sparse_feature: {
          name: 'sparse_feature'
          index_feature { name: 'index_feature1' }
          index_feature { name: 'index_feature2' }
          value_feature { name: 'value_feature' }
        })");
  // Length mismatch.
  const DatasetFeatureStatistics length_mismatch_stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "sparse_feature"
          custom_stats { name: "missing_value" num: 0 }
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
              buckets { label: "index_feature1" sample_count: 1 }
              buckets { label: "index_feature2" sample_count: 0 }
            }
          }
          custom_stats {
            name: "min_length_diff"
            rank_histogram {
              buckets { label: "index_feature1" sample_count: 0 }
              buckets { label: "index_feature2" sample_count: -2 }
            }
          }
        }
      )");

  Schema schema_deprecated = schema_proto;
  DeprecateSparseFeature(schema_deprecated.mutable_sparse_feature(0));

  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["sparse_feature"].new_schema = schema_deprecated;
  expected_anomalies["sparse_feature"]
      .expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path: { step: "sparse_feature" }
    description: "Mismatch between index feature: index_feature1 and value column, with max_length_diff = 1 Mismatch between index feature: index_feature2 and value column, with min_length_diff = -2"
    severity: ERROR
    short_description: "Multiple errors"
    reason {
      type: SPARSE_FEATURE_LENGTH_MISMATCH
      short_description: "Length mismatch between value and index feature"
      description: "Mismatch between index feature: index_feature1 and value column, with max_length_diff = 1"
    }
    reason {
      type: SPARSE_FEATURE_LENGTH_MISMATCH
      short_description: "Length mismatch between value and index feature"
      description: "Mismatch between index feature: index_feature2 and value column, with min_length_diff = -2"
    })");
  TestFindChanges(
      schema_proto,
      DatasetStatsView(length_mismatch_stats, /* by_weight= */ false),
      FeatureStatisticsToProtoConfig(), expected_anomalies);
}

// Two reasons in the same example.
// Replaces GetSchemaDiff::TwoReasons and SchemaAnomalyMerge::Basic
TEST(SchemaAnomalies, GetSchemaDiffTwoReasons) {
  Schema initial;
  FeatureStatisticsToProtoConfig config =
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          "new_features_are_warnings: true");
  // New feature introduced.
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: 'bar'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
      )");
  // Field made repeated.
  const DatasetFeatureStatistics statistics_2 =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features {
          name: "bar"
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 10
            }
          }
        }
      )");
  DatasetStatsView stats_view(statistics, false);
  DatasetStatsView stats_view_2(statistics_2, false);
  SchemaAnomalies anomalies(initial);
  TF_CHECK_OK(anomalies.FindChanges(stats_view, absl::nullopt, config));
  TF_CHECK_OK(anomalies.FindChanges(stats_view_2, absl::nullopt, config));
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;

  expected_anomalies["bar"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "bar"
      value_count { min: 1 max: 10 }
      type: INT
      presence { min_count: 1 }
    })");
  expected_anomalies["bar"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"pb(
    path: { step: "bar" }
    description: "New column (column in data but not in schema) Some examples have more values than expected."
    severity: ERROR
    short_description: "Multiple errors"
    reason {
      type: SCHEMA_NEW_COLUMN
      short_description: "New column"
      description: "New column (column in data but not in schema)"
    }
    reason {
      type: FEATURE_TYPE_HIGH_NUMBER_VALUES
      short_description: "Superfluous values"
      description: "Some examples have more values than expected."
    })pb");
  TestAnomalies(anomalies.GetSchemaDiff(), initial, expected_anomalies);
}

TEST(GetSchemaDiff, TwoChanges) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: 'bar'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        features: {
          name: 'foo'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
      )");

  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;

  expected_anomalies["bar"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "bar"
      value_count { min: 1 max: 1 }
      type: INT
      presence { min_count: 1 }
    })");
  expected_anomalies["bar"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: "bar" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "foo"
      value_count { min: 1 max: 1 }
      type: INT
      presence { min_count: 1 }
    })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path: { step: "foo" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");

  TestFindChanges(Schema(), DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

}  // namespace

}  // namespace data_validation
}  // namespace tensorflow
