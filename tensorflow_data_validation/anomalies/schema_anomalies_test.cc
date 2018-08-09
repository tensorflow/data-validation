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
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::Schema;
using testing::ParseTextProtoOrDie;

void TestFindChanges(
    const Schema& schema,
    const DatasetStatsView& stats_view,
    const FeatureStatisticsToProtoConfig& config,
    const std::map<string, testing::ExpectedAnomalyInfo>& expected_anomalies) {
  auto get_diff = [&](const Schema& schema_proto) {
    SchemaAnomalies anomalies(schema_proto);
    TF_CHECK_OK(anomalies.FindChanges(stats_view, config));
    return anomalies.GetSchemaDiff();
  };
  TestSchemaToAnomalies(schema, get_diff, expected_anomalies);
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
        presence: {min_count: 1 min_fraction: 1.0} value_count: {min: 1 max: 1}
        type: FLOAT
      })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          features: {
          name: "capital-gain"      # size=12
          type: FLOAT       # integer=1 (enum tensorflow.metadata.v0.FeatureNameStatistics.Type)
          num_stats: {      # (tensorflow.metadata.v0.NumericStatistics) size=417B
            common_stats: { # (tensorflow.metadata.v0.CommonStatistics) size=13B
              num_non_missing: 0x0000000000007f31   # 32_561
              min_num_values : 0x0000000000000001
              max_num_values : 0x0000000000000001
              avg_num_values : 1.0
            }       # datasets[0].features[9].num_stats.common_stats
            mean: 1077.6488437087312
            std_dev     : 7385.1786769476275
            num_zeros   : 0x0000000000007499        # 29_849
            max : 99999.0   # [if seconds]: 1 day 3 hours
          }})");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    TestFindChanges(initial, DatasetStatsView(statistics, false), config,
                    std::map<string, testing::ExpectedAnomalyInfo>());
  }
}

TEST(SchemaAnomalies, FindChangesCategoricalIntFeature) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "a_int"
        value_count: {min: 1 max: 1}
        type: INT
        int_domain {is_categorical: true}
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
          value_count {
            min: 1
            max: 2
          }
          type: INT
          int_domain {
            is_categorical: true
          }
        })");
    expected_anomalies["a_int"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
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
      string_domain {
        name: "MyAloneEnum"
        value: "A"
        value: "B"
        value: "C"
      }
      feature {
        name: "annotated_enum"
        presence: {min_count: 1} value_count: {min: 1 max: 1}
        type: BYTES
        domain: "MyAloneEnum"
        annotation {
          tag: "some tag"
          comment: "some comment"
        }
      }
      feature {
        name: "ignore_this"
        lifecycle_stage: DEPRECATED
        presence: {min_count: 1} value_count: {min: 1}
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
            rank_histogram: {
              buckets: {
                label: "D"
                sample_count: 2
              }
            }
          }
        })");

  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["annotated_enum"].new_schema =
        ParseTextProtoOrDie<Schema>(R"(
        feature {
          name: "annotated_enum"
          value_count {
            min: 1
            max: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
          presence {
            min_count: 1
          }
          annotation {
            tag: "some tag"
            comment: "some comment"
          }
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          value_count {
            min: 1
          }
          type: BYTES
          presence {
            min_count: 1
          }
        }
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
          value: "D"
        })");
    expected_anomalies["annotated_enum"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
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
                   common_stats: {
                     num_missing: 0
                     max_num_values: 1
                   }
                   rank_histogram {
                     buckets {
                       label: "a"
                       sample_count: 1
                     }
                     buckets {
                       label: "b"
                       sample_count: 2
                     }
                     buckets {
                       label: "c"
                       sample_count: 7
                     }}})"));
  const DatasetFeatureStatistics serving =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
              R"(name: 'foo'
                 type: STRING
                 string_stats: {
                   common_stats: {
                     num_missing: 0
                     max_num_values: 1
                   }
                   rank_histogram {
                     buckets {
                       label: "a"
                       sample_count: 3
                     }
                     buckets {
                       label: "b"
                       sample_count: 1
                     }
                     buckets {
                       label: "c"
                       sample_count: 6
                     }}})"));

  const Schema schema_proto = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: 'foo'
      type: BYTES
      skew_comparator {
        infinity_norm: {
          threshold: 0.1}}})");
  std::shared_ptr<DatasetStatsView> serving_view =
      std::make_shared<DatasetStatsView>(serving);
  std::shared_ptr<DatasetStatsView> training_view =
      std::make_shared<DatasetStatsView>(
          training,
          /* by_weight= */ false,
          /* environment= */ absl::nullopt,
          /* previous= */ std::shared_ptr<DatasetStatsView>(), serving_view);

  auto get_diff = [&](const Schema& schema_proto) {
    SchemaAnomalies skew(schema_proto);
    skew.FindSkew(*training_view);
    return skew.GetSchemaDiff();
  };
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "foo"
        type: BYTES
        skew_comparator {
          infinity_norm: {
            threshold: 0.19999999999999998
          }
        }
      })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "The Linfty distance between serving and training is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
      severity: ERROR
      short_description: "High Linfty distance between serving and training"
      reason {
        type: COMPARATOR_L_INFTY_HIGH
        short_description: "High Linfty distance between serving and training"
        description: "The Linfty distance between serving and training is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
      })");
  TestSchemaToAnomalies(schema_proto, get_diff, expected_anomalies);
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
            rank_histogram: {
              buckets: {
                label: "D"
              }
            }
          }
        })");
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["annotated_enum"].new_schema =
    ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "annotated_enum"
        presence: {min_count: 1} value_count: {min: 1 max: 1}
        type: BYTES
      })");
  expected_anomalies["annotated_enum"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): annotated_enum"
      severity: ERROR
      short_description: "New column"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): annotated_enum"
      })");

  TestFindChanges(Schema(),
                  DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

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
            rank_histogram: {
              buckets: {
                label: "A"
              }
            }
          }
        })");

  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
      string_domain {
        name: "MyAloneEnum"
        value: "A"
        value: "B"
        value: "C"
      }
      feature {
        name: "annotated_enum"
        presence: {min_count: 1} value_count: {min: 1 max: 1}
        type: BYTES
        domain: "MyAloneEnum"
      })");

  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["new_feature"].new_schema = ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "annotated_enum"
        value_count {
          min: 1
          max: 1
        }
        type: BYTES
        domain: "MyAloneEnum"
        presence {
          min_count: 1
        }
      }
      feature {
        name: "new_feature"
        value_count {
          min: 1
          max: 1
        }
        type: INT
        presence {
          min_count: 1
        }
      }
      string_domain {
        name: "MyAloneEnum"
        value: "A"
        value: "B"
        value: "C"
      })");
  expected_anomalies["new_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): new_feature"
      severity: ERROR
      short_description: "New column"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): new_feature"
      })");

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
  Schema initial = ParseTextProtoOrDie<Schema>(R"(feature { name:"bar" type: INT})");
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
      feature {
        name: "bar"
        type: INT
      }
      feature {
        name: "foo"
        value_count {
          min: 1
          max: 1
        }
        type: INT
        presence {
          min_count: 1
        }
      })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): foo"
      severity: ERROR
      short_description: "New column"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): foo"
      })");

  TestFindChanges(initial, DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}


// Two reasons in the same example.
// Replaces GetSchemaDiff::TwoReasons and SchemaAnomalyMerge::Basic
TEST(SchemaAnomalies, GetSchemaDiffTwoReasons) {
  Schema initial;
  FeatureStatisticsToProtoConfig config =
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          "new_features_are_warnings: true");
  // New field introduced.
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
  auto get_diff = [&](const Schema& schema_proto) {
    SchemaAnomalies anomalies(schema_proto);
    TF_CHECK_OK(anomalies.FindChanges(stats_view, config));
    TF_CHECK_OK(anomalies.FindChanges(stats_view_2, config));
    return anomalies.GetSchemaDiff();
  };
  std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;

  expected_anomalies["bar"].new_schema = ParseTextProtoOrDie<Schema>(R"(
        feature {
        name: "bar"
        value_count {
          min: 1
          max: 10
        }
        type: INT
        presence {
          min_count: 1
        }
      })");
  expected_anomalies["bar"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): bar Some examples have more values than expected."
      severity: ERROR
      short_description: "Multiple errors"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): bar"
      }
      reason {
        type: FEATURE_TYPE_HIGH_NUMBER_VALUES
        short_description: "Superfluous values"
        description: "Some examples have more values than expected."
      })");
  TestSchemaToAnomalies(initial, get_diff, expected_anomalies);
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
        value_count {
          min: 1
          max: 1
        }
        type: INT
        presence {
          min_count: 1
        }
      })");
  expected_anomalies["bar"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): bar"
      severity: ERROR
      short_description: "New column"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): bar"
      })");
  expected_anomalies["foo"].new_schema = ParseTextProtoOrDie<Schema>(R"(
      feature {
        name: "foo"
        value_count {
          min: 1
          max: 1
        }
        type: INT
        presence {
          min_count: 1
        }
      })");
  expected_anomalies["foo"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
      description: "New column (column in data but not in schema): foo"
      severity: ERROR
      short_description: "New column"
      reason {
        type: SCHEMA_NEW_COLUMN
        short_description: "New column"
        description: "New column (column in data but not in schema): foo"
      })");

  TestFindChanges(Schema(),
                  DatasetStatsView(statistics, false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

}  // namespace

}  // namespace data_validation
}  // namespace tensorflow
