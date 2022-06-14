/* Copyright 2020 Google LLC

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
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::Schema;
using testing::ParseTextProtoOrDie;

void TestFindChanges(const Schema& schema, const DatasetStatsView& stats_view,
                     const FeatureStatisticsToProtoConfig& config,
                     const std::map<std::string, testing::ExpectedAnomalyInfo>&
                         expected_anomalies) {
  SchemaAnomalies anomalies(schema);
  TF_CHECK_OK(anomalies.FindChanges(stats_view, absl::nullopt, config));
  TestAnomalies(anomalies.GetSchemaDiff(/*enable_diff_regions=*/false),
                schema, expected_anomalies);
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
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config,
                    std::map<std::string, testing::ExpectedAnomalyInfo>());
  }
}

// TODO(b/181962134): Add updated schema info for invalid_value_counts.
TEST(SchemaAnomalies, SimpleBadSchemaConfigurations) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"pb(
    feature {
      name: "no_type"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      float_domain: { disallow_nan: true }
    }
    feature {
      name: "invalid_value_count"
      presence: { min_count: 1 }
      value_count: { min: -1 max: 3 }
      type: BYTES
    }
    feature {
      name: "invalid_value_counts"
      presence: { min_count: 1 }
      value_counts {
        value_count: { min: -1 max: 3 }
        value_count: { min: 3 max: 1 }
      }
      type: BYTES
    }
    feature {
      name: "invalid_presence"
      presence: { min_fraction: 1.5 }
      type: BYTES
    }
    feature {
      name: "nl_float"
      presence: { min_fraction: 1.0 }
      natural_language_domain: {}
      type: FLOAT
    }
    feature {
      name: "struct_bytes"
      presence: { min_fraction: 1.0 }
      struct_domain: {}
      type: BYTES
    }
    feature {
      name: "distribution_constraints_bool"
      presence: { min_fraction: 1.0 }
      bool_domain: {}
      distribution_constraints: { min_domain_mass: .8 }
      type: FLOAT
    }
    feature {
      name: "int_domain_bytes"
      presence: { min_fraction: 1.0 }
      int_domain: {}
      type: BYTES
    }
  )pb");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"pb(
        features: {
          name: "no_type"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              num_nan: 5
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: 1.0
          }
        }
        features: {
          name: "int_domain_bytes"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              num_nan: 5
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: 1.0
          }
        }
        features: {
          name: "invalid_value_count"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
          }
        }
        features: {
          name: "invalid_value_counts"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
              presence_and_valency_stats {
                num_missing: 10
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 3
                tot_num_values: 8
              }
              presence_and_valency_stats {
                num_missing: 10
                num_non_missing: 4
                min_num_values: 3
                max_num_values: 3
                tot_num_values: 8
              }
            }
          }
        }
        features: {
          name: "invalid_presence"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
          }
        }
        features: {
          name: "nl_float"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
          }
        }
        features: {
          name: "struct_bytes"
          type: BYTES
          num_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
          }
        }
        features: {
          name: "distribution_constraints_bool"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
          }
        })pb");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["no_type"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "no_type" }
      description: ""
      severity: ERROR
      short_description: "unspecified type: determine the type and set it, rather than deprecating."
      reason {
        type: FEATURE_MISSING_TYPE
        short_description: "unspecified type: determine the type and set it, rather than deprecating."
        description: ""
      })");
    expected_anomalies["int_domain_bytes"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
          path { step: "int_domain_bytes" }
          description: "Expected data of type: BYTES but got FLOAT"
          severity: ERROR
          short_description: "Unexpected data type"
          reason {
            type: UNEXPECTED_DATA_TYPE
            short_description: "Unexpected data type"
            description: "Expected data of type: BYTES but got FLOAT"
          })pb");
    expected_anomalies["invalid_value_count"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "invalid_value_count" }
          description: ""
          severity: ERROR
          short_description: "ValueCount.min should not be negative"
          reason {
            type: INVALID_SCHEMA_SPECIFICATION
            short_description: "ValueCount.min should not be negative"
            description: ""
          })");
    expected_anomalies["invalid_value_counts"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "invalid_value_counts" }
      description: "ValueCounts.min at level 0 should not be negative. ValueCounts.max at level 1 should not be less than min."
      severity: ERROR
      short_description: "Multiple errors"
      reason {
        type: INVALID_SCHEMA_SPECIFICATION
        short_description: "ValueCounts.min should not be negative"
        description: "ValueCounts.min at level 0 should not be negative."
      }
      reason {
        type: INVALID_SCHEMA_SPECIFICATION
        short_description: "ValueCounts.max should not be less than min"
        description: "ValueCounts.max at level 1 should not be less than min."
      })");
    expected_anomalies["invalid_presence"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "invalid_presence" }
          description: ""
          severity: ERROR
          short_description: "min_fraction should not greater than 1"
          reason {
            type: INVALID_SCHEMA_SPECIFICATION
            short_description: "min_fraction should not greater than 1"
            description: ""
          })");
    expected_anomalies["distribution_constraints_bool"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "distribution_constraints_bool" }
      description: ""
      severity: ERROR
      short_description: "distribution constraints not supported for bool domains."
      reason {
        type: INVALID_SCHEMA_SPECIFICATION
        short_description: "distribution constraints not supported for bool domains."
        description: ""
      })");
    expected_anomalies["nl_float"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "nl_float" }
      short_description: "The domain does not match the type"
      severity: ERROR
      description: "The domain \"natural_language_domain\" does not match the type: FLOAT"
      reason {
        type: DOMAIN_INVALID_FOR_TYPE
        description: "The domain \"natural_language_domain\" does not match the type: FLOAT"
        short_description: "The domain does not match the type"
      })");
    expected_anomalies["struct_bytes"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "struct_bytes" }
      short_description: "The domain does not match the type"
      severity: ERROR
      description: "The domain \"struct_domain\" does not match the type: BYTES"
      reason {
        type: DOMAIN_INVALID_FOR_TYPE
        description: "The domain \"struct_domain\" does not match the type: BYTES"
        short_description: "The domain does not match the type"
      })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindNansInFloatDisallowNans) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "income"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
      float_domain: { disallow_nan: true }
    }
    feature {
      name: "string_encoded_float"
      presence: { min_count: 1 }
      value_count: { min: 1 max: 3 }
      type: BYTES
      float_domain: { disallow_nan: true }
    }
  )");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "income"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              num_nan: 5
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: 1.0
          }
        }
        features: {
          name: "string_encoded_float"
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
            rank_histogram: {
              buckets: { label: "1.5" sample_count: 5 }
              buckets: { label: "2.5" sample_count: 3 }
              buckets: { label: "-1.5" sample_count: 15 }
              buckets: { label: "NaN" sample_count: 20 }
              buckets: { label: "0.5" sample_count: 10 }
            }
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["income"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "income" }
          description: "Float feature has NaN values."
          severity: ERROR
          short_description: "Invalid values"
          reason {
            type: FLOAT_TYPE_HAS_NAN
            short_description: "Invalid values"
            description: "Float feature has NaN values."
          })");
    expected_anomalies["string_encoded_float"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "string_encoded_float" }
          description: "Float feature has NaN values."
          severity: ERROR
          short_description: "Invalid values"
          reason {
            type: FLOAT_TYPE_HAS_NAN
            short_description: "Invalid values"
            description: "Float feature has NaN values."
          })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}
TEST(SchemaAnomalies, NansDisallowedNoNansFound) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "age"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 3 }
      type: FLOAT
      float_domain: { disallow_nan: true }
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "age"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
            histograms: {
              num_nan: 0
              buckets: { high_value: 67 sample_count: 50 }
              buckets: { low_value: 15 high_value: 67 sample_count: 100 }
              type: QUANTILES
            }
            mean: 20
            std_dev: .25
            max: 87
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config,
                    std::map<std::string, testing::ExpectedAnomalyInfo>());
  }
}

TEST(SchemaAnomalies, FindsLowSupportedImageFraction) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "image/encoded"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: BYTES
      image_domain: { minimum_supported_image_fraction: 0.85 }
    }
  )");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'image/encoded'
          type: BYTES
          bytes_stats: {
            common_stats: {
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }}
          custom_stats: {
            name: 'image_format_histogram'
            rank_histogram: {
              buckets: {
                label: 'jpeg'
                sample_count: 5
              }
              buckets: {
                label: 'png'
                sample_count: 3
              }
              buckets: {
                label: 'UNKNOWN'
                sample_count: 2
              }
            }
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["image/encoded"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "image/encoded" }
          description: "Fraction of values containing TensorFlow supported images: 0.800000 is lower than the threshold set in the Schema: 0.850000."
          severity: ERROR
          short_description: "Low supported image fraction"
          reason {
            type: LOW_SUPPORTED_IMAGE_FRACTION
            short_description: "Low supported image fraction"
            description: "Fraction of values containing TensorFlow supported images: 0.800000 is lower than the threshold set in the Schema: 0.850000."
          })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

TEST(SchemaAnomalies, NansInFloatAllowed) {
  // Nans in a float feature will not raises anomalies unless disallow_nan is
  // True.
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "income"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
      float_domain: { min: 0 max: 1 }
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "income"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              num_nan: 5
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: 1.0
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config,
                    std::map<std::string, testing::ExpectedAnomalyInfo>());
  }
}

TEST(SchemaAnomalies, FindInfsInFloatDisallowInfs) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "income"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
      float_domain: { disallow_inf: true }
    }
    feature {
      name: "age"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
      float_domain: { disallow_inf: true }
    }
  )");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "income"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              num_nan: 5
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              buckets: { low_value: 0 high_value: inf sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: inf
          }
        }
        features: {
          name: "age"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            mean: .5
            std_dev: .25
            max: 10
            min: -inf
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["income"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "income" }
          description: "Float feature has Inf values."
          severity: ERROR
          short_description: "Invalid values"
          reason {
            type: FLOAT_TYPE_HAS_INF
            short_description: "Invalid values"
            description: "Float feature has Inf values."
          })");
    expected_anomalies["age"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "age" }
          description: "Float feature has Inf values."
          severity: ERROR
          short_description: "Invalid values"
          reason {
            type: FLOAT_TYPE_HAS_INF
            short_description: "Invalid values"
            description: "Float feature has Inf values."
          })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}
TEST(SchemaAnomalies, InfsDisallowedNoInfsFound) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "age"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 3 }
      type: FLOAT
      float_domain: { disallow_inf: true }
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "age"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 10
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 3
              avg_num_values: 1.5
            }
            histograms: {
              num_nan: 0
              buckets: { high_value: 67 sample_count: 50 }
              buckets: { low_value: 15 high_value: 67 sample_count: 100 }
              type: QUANTILES
            }
            mean: 20
            std_dev: .25
            max: 87
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, std::map<string, testing::ExpectedAnomalyInfo>());
  }
}

TEST(SchemaAnomalies, InfsInFloatAllowed) {
  // Infs in a float feature will not raise anomalies unless disallow_inf is
  // True.
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "income"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: FLOAT
      float_domain: { min: 0 }
    })");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "income"
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            histograms: {
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
            mean: .5
            std_dev: .25
            max: inf
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, std::map<string, testing::ExpectedAnomalyInfo>());
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
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindChangesBooleanFloatFeature) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "a_float_outside_range"
      value_count: { min: 1 max: 1 }
      type: FLOAT
      bool_domain {}
    }
    feature {
      name: "a_float_with_nans"
      value_count: { min: 1 max: 1 }
      type: FLOAT
      bool_domain {}
    }
    feature {
      name: "a_float_between_0_and_1"
      value_count: { min: 1 max: 1 }
      type: FLOAT
      bool_domain {}
    }
    feature {
      name: "an_okay_float"
      value_count: { min: 1 max: 1 }
      type: FLOAT
      bool_domain {}
    })");

  // Note: an_okay_float has histogram buckets between 0 and 1, but this is
  // benign for a non-quantile histogram.
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'a_float_outside_range'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            min: 0
            max: 2.0
            median: .5
            histograms: {
              buckets: { high_value: 0.4 sample_count: 200 }
              buckets: { low_value: 1.8 high_value: 2.0 sample_count: 10 }
              type: QUANTILES
            }
          }
        }
        features: {
          name: 'a_float_with_nans'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            min: 0
            max: 1
            histograms: {
              num_nan: 3
              buckets: { high_value: 1 }
              buckets: { low_value: 0 high_value: 1 sample_count: 100 }
              type: QUANTILES
            }
          }
        }
        features: {
          name: 'a_float_between_0_and_1'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            min: 0
            max: 1
            histograms: {
              buckets: { high_value: 0.4 sample_count: 200 }
              buckets: { low_value: 0.4 high_value: 0.7 sample_count: 10 }
              buckets: { low_value: 0.7 high_value: 1 sample_count: 10 }
              type: QUANTILES
            }
          }
        }
        features: {
          name: 'an_okay_float'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1
            }
            min: 0
            max: 1
            histograms: {
              buckets: { high_value: 0.5 sample_count: 10 }
              buckets: { low_value: 0.5 high_value: 0.7 sample_count: 10 }
              buckets: { low_value: 0.7 high_value: 0.9 sample_count: 10 }
            }
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["a_float_outside_range"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "a_float_outside_range" }
      description: "Floats (such as 2) not in {0, 1}: converting to float_domain."
      severity: ERROR
      short_description: "Non-boolean values"
      reason {
        type: BOOL_TYPE_UNEXPECTED_FLOAT
        short_description: "Non-boolean values"
        description: "Floats (such as 2) not in {0, 1}: converting to float_domain."
      })");
    expected_anomalies["a_float_with_nans"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "a_float_with_nans" }
      description: "Floats (such as NaN) not in {0, 1}: converting to float_domain."
      severity: ERROR
      short_description: "Non-boolean values"
      reason {
        type: BOOL_TYPE_UNEXPECTED_FLOAT
        short_description: "Non-boolean values"
        description: "Floats (such as NaN) not in {0, 1}: converting to float_domain."
      })");
    expected_anomalies["a_float_between_0_and_1"]
        .expected_info_without_diff = ParseTextProtoOrDie<
        tensorflow::metadata::v0::AnomalyInfo>(R"(
      path { step: "a_float_between_0_and_1" }
      description: "Float values falling between 0 and 1: converting to float_domain."
      severity: ERROR
      short_description: "Non-boolean values"
      reason {
        type: BOOL_TYPE_UNEXPECTED_FLOAT
        short_description: "Non-boolean values"
        description: "Float values falling between 0 and 1: converting to float_domain."
      })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindChangesDatasetLevelChanges) {
  const DatasetFeatureStatistics stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 1)");
  const DatasetFeatureStatistics previous_version =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(num_examples: 2)");

  const Schema schema_proto = ParseTextProtoOrDie<Schema>(R"(
    dataset_constraints {
      min_examples_count: 2
      num_examples_version_comparator {
        min_fraction_threshold: 1.0,
        max_fraction_threshold: 1.0
      }
    }
  )");
  std::shared_ptr<DatasetStatsView> previous_version_view =
      std::make_shared<DatasetStatsView>(previous_version);
  DatasetStatsView stats_view =
      DatasetStatsView(stats,
                       /* by_weight= */ false,
                       /* environment= */ absl::nullopt,
                       /* previous_span= */ std::shared_ptr<DatasetStatsView>(),
                       /* serving= */ std::shared_ptr<DatasetStatsView>(),
                       /* previous_version= */ previous_version_view);

  testing::ExpectedAnomalyInfo expected_anomaly_info;
  expected_anomaly_info.expected_info_without_diff = ParseTextProtoOrDie<
      metadata::v0::AnomalyInfo>(R"(
    description: "The ratio of num examples in the current dataset versus the previous version is 0.5 (up to six significant digits), which is below the threshold 1. The dataset has 1 examples, which is fewer than expected."
    severity: ERROR,
    short_description: "Multiple errors"
    reason {
      type: COMPARATOR_LOW_NUM_EXAMPLES,
      short_description: "Low num examples in current dataset versus the previous version.",
      description: "The ratio of num examples in the current dataset versus the previous version is 0.5 (up to six significant digits), which is below the threshold 1."
    }
    reason {
      type: DATASET_LOW_NUM_EXAMPLES,
      short_description: "Low num examples in dataset.",
      description: "The dataset has 1 examples, which is fewer than expected."
    })");

  SchemaAnomalies anomalies(schema_proto);
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    TF_CHECK_OK(anomalies.FindChanges(stats_view, absl::nullopt, config));
    tensorflow::metadata::v0::Anomalies actual_anomalies =
        anomalies.GetSchemaDiff(/*enable_diff_regions=*/false);

    testing::TestAnomalyInfo(actual_anomalies.dataset_anomaly_info(),
                             expected_anomaly_info, "");
  }
}

TEST(SchemaAnomalies, SemanticTypeUpdates) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"pb(
    feature {
      name: "old_nl_feature"
      value_count: { min: 1 max: 1 }
      type: BYTES
    })pb");

  const DatasetFeatureStatistics statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"pb(
    features: {
      name: 'old_nl_feature'
      type: BYTES
      num_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
      custom_stats: { name: "domain_info" str: "natural_language_domain {}" }
    }
    features: {
      name: 'new_nl_feature'
      type: BYTES
      num_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
      custom_stats: { name: "domain_info" str: "natural_language_domain {}" }
    })pb");

  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  // Anomaly for updating an existing feature with semantic type.
  expected_anomalies["old_nl_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path { step: "old_nl_feature" }
        description: "Updated semantic domain for feature: old_nl_feature"
        severity: ERROR
        short_description: "Updated semantic domain"
        reason {
          type: SEMANTIC_DOMAIN_UPDATE
          short_description: "Updated semantic domain"
          description: "Updated semantic domain for feature: old_nl_feature"
        })pb");
  // Anomaly for creating a new feature with semantic type.
  expected_anomalies["new_nl_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"pb(
        path { step: "new_nl_feature" }
        description: "New column (column in data but not in schema)"
        severity: ERROR
        short_description: "New column"
        reason {
          type: SCHEMA_NEW_COLUMN
          short_description: "New column"
          description: "New column (column in data but not in schema)"
        })pb");
  TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
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
    std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

TEST(SchemaAnomalies, FindSkewStringFeature) {
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
          /* previous_span= */ std::shared_ptr<DatasetStatsView>(),
          serving_view,
          /* previous_version= */ std::shared_ptr<DatasetStatsView>());

  SchemaAnomalies skew(schema_proto);
  TF_CHECK_OK(skew.FindSkew(*training_view));
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path { step: "foo" }
    description: "The Linfty distance between training and serving is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
    severity: ERROR
    short_description: "High Linfty distance between training and serving"
    reason {
      type: COMPARATOR_L_INFTY_HIGH
      short_description: "High Linfty distance between training and serving"
      description: "The Linfty distance between training and serving is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
    })");
  const std::vector<tensorflow::metadata::v0::DriftSkewInfo>
      expected_drift_skew_infos = {
          ParseTextProtoOrDie<tensorflow::metadata::v0::DriftSkewInfo>(R"(
            path { step: "foo" }
            skew_measurements {
              type: L_INFTY
              value: 0.19999999999999998
              threshold: 0.1
            }
          )")};
  TestAnomalies(skew.GetSchemaDiff(/*enable_diff_regions=*/false), schema_proto,
                expected_anomalies, expected_drift_skew_infos);
}

TEST(SchemaAnomalies, FindSkewNumericFeature) {
  const DatasetFeatureStatistics training =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
              R"(name: 'foo'
                 type: INT
                 num_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   histograms {
                     buckets {
                       low_value: 1.0
                       high_value: 2.0
                       sample_count: 1.0
                     }
                     buckets {
                       low_value: 2.0
                       high_value: 3.0
                       sample_count: 1.0
                     }
                     type: STANDARD
                   }
                 })"));
  const DatasetFeatureStatistics serving =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
              R"(name: 'foo'
                 type: INT
                 num_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   histograms {
                     buckets {
                       low_value: 5.0
                       high_value: 6.0
                       sample_count: 1.0
                     }
                     buckets {
                       low_value: 6.0
                       high_value: 7.0
                       sample_count: 1.0
                     }
                     type: STANDARD
                   }
                 })"));

  const Schema schema_proto = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: 'foo'
      type: INT
      skew_comparator { jensen_shannon_divergence: { threshold: 0.1 } }
    })");
  const std::shared_ptr<DatasetStatsView> serving_view =
      std::make_shared<DatasetStatsView>(serving);
  const std::shared_ptr<DatasetStatsView> training_view =
      std::make_shared<DatasetStatsView>(
          training,
          /* by_weight= */ false,
          /* environment= */ absl::nullopt,
          /* previous_span= */ std::shared_ptr<DatasetStatsView>(),
          serving_view,
          /* previous_version= */ std::shared_ptr<DatasetStatsView>());

  SchemaAnomalies skew(schema_proto);
  TF_CHECK_OK(skew.FindSkew(*training_view));
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["foo"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path { step: "foo" }
    description: "The approximate Jensen-Shannon divergence between training and serving is 1 (up to six significant digits), above the threshold 0.1."
    severity: ERROR
    short_description: "High approximate Jensen-Shannon divergence between training and serving"
    reason {
      type: COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH
      short_description: "High approximate Jensen-Shannon divergence between training and serving"
      description: "The approximate Jensen-Shannon divergence between training and serving is 1 (up to six significant digits), above the threshold 0.1."
    })");
  const std::vector<tensorflow::metadata::v0::DriftSkewInfo>
      expected_drift_skew_infos = {
          ParseTextProtoOrDie<tensorflow::metadata::v0::DriftSkewInfo>(R"(
            path { step: "foo" }
            skew_measurements {
              type: JENSEN_SHANNON_DIVERGENCE
              value: 1
              threshold: 0.1
            }
          )")};
  TestAnomalies(skew.GetSchemaDiff(/*enable_diff_regions=*/false), schema_proto,
                expected_anomalies, expected_drift_skew_infos);
}

TEST(SchemaAnomalies,
     FindSkewDistributionChangeWithinThresholdDoesNotRaiseAnomaly) {
  // Training and serving statistics have the same distribution of values in the
  // standard histogram.
  const DatasetFeatureStatistics training =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
              R"(name: 'foo'
                 type: INT
                 num_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   histograms {
                     buckets {
                       low_value: 1.0
                       high_value: 2.0
                       sample_count: 1.0
                     }
                     buckets {
                       low_value: 2.0
                       high_value: 3.0
                       sample_count: 1.0
                     }
                     type: STANDARD
                   }
                 })"));
  const DatasetFeatureStatistics serving =
      testing::GetDatasetFeatureStatisticsForTesting(
          ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
              R"(name: 'foo'
                 type: INT
                 num_stats: {
                   common_stats: { num_missing: 0 max_num_values: 1 }
                   histograms {
                     buckets {
                       low_value: 1.0
                       high_value: 2.0
                       sample_count: 1.0
                     }
                     buckets {
                       low_value: 2.0
                       high_value: 3.0
                       sample_count: 1.0
                     }
                     type: STANDARD
                   }
                 })"));

  const Schema schema_proto = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: 'foo'
      type: INT
      skew_comparator { jensen_shannon_divergence: { threshold: 0.5 } }
    })");
  const std::shared_ptr<DatasetStatsView> serving_view =
      std::make_shared<DatasetStatsView>(serving);
  const std::shared_ptr<DatasetStatsView> training_view =
      std::make_shared<DatasetStatsView>(
          training,
          /* by_weight= */ false,
          /* environment= */ absl::nullopt,
          /* previous_span= */ std::shared_ptr<DatasetStatsView>(),
          serving_view,
          /* previous_version= */ std::shared_ptr<DatasetStatsView>());

  SchemaAnomalies skew(schema_proto);
  TF_CHECK_OK(skew.FindSkew(*training_view));
  // No anomalies are expected.
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  const std::vector<tensorflow::metadata::v0::DriftSkewInfo>
      expected_drift_skew_infos = {
          ParseTextProtoOrDie<tensorflow::metadata::v0::DriftSkewInfo>(R"(
            path { step: "foo" }
            skew_measurements {
              type: JENSEN_SHANNON_DIVERGENCE
              value: 0
              threshold: 0.5
            }
          )")};
  TestAnomalies(skew.GetSchemaDiff(/*enable_diff_regions=*/false), schema_proto,
                expected_anomalies, expected_drift_skew_infos);
}

TEST(SchemaAnomalies, UniqueNotInRange) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "categorical_feature"
      type: INT
      int_domain { is_categorical: true }
      unique_constraints { min: 1 max: 1 }
    }
    feature {
      name: "string_feature"
      type: BYTES
      unique_constraints { min: 5 max: 5 }
    }
    feature {
      name: "numeric_feature"
      type: FLOAT
      unique_constraints { min: 1 max: 1 }
    })");

  const DatasetFeatureStatistics statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features: {
      name: 'categorical_feature'
      type: INT
      string_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 5 }
        unique: 5
      }
    }
    features: {
      name: 'string_feature'
      type: BYTES
      string_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 9 }
        unique: 1
      }
    }
    features: {
      name: 'numeric_feature'
      type: FLOAT
      num_stats: {
        common_stats: { min_num_values: 1 max_num_values: 1 num_non_missing: 1 }
      }
    })");

  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["categorical_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path { step: "categorical_feature" }
        description: "Expected no more than 1 unique values but found 5."
        severity: ERROR
        short_description: "High number of unique values"
        reason {
          type: FEATURE_TYPE_HIGH_UNIQUE
          short_description: "High number of unique values"
          description: "Expected no more than 1 unique values but found 5."
        })");
  expected_anomalies["string_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path { step: "string_feature" }
        description: "Expected at least 5 unique values but found only 1."
        severity: ERROR
        short_description: "Low number of unique values"
        reason {
          type: FEATURE_TYPE_LOW_UNIQUE
          short_description: "Low number of unique values"
          description: "Expected at least 5 unique values but found only 1."
        })");
  expected_anomalies["numeric_feature"]
      .expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"(
    path { step: "numeric_feature" }
    description: "UniqueConstraints specified for the feature, but unique values were not counted (i.e., feature is not string or categorical)."
    severity: ERROR
    short_description: "No unique values"
    reason {
      type: FEATURE_TYPE_NO_UNIQUE
      short_description: "No unique values"
      description: "UniqueConstraints specified for the feature, but unique values were not counted (i.e., feature is not string or categorical)."
    })");
  TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

TEST(SchemaAnomalies, FeatureShapeDropped) {
  const Schema schema = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "f1"
      type: INT
      shape { dim { size: 1 } }
      presence { min_fraction: 1 min_count: 1 }
    })");
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        features: {
          name: "f1"
          type: INT
          num_stats: {
            common_stats: {
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 2  # anomaly
            }
          }
        })");
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["f1"].expected_info_without_diff = ParseTextProtoOrDie<
      tensorflow::metadata::v0::AnomalyInfo>(R"pb(
    path { step: "f1" }
    short_description: "Feature shape dropped"
    severity: ERROR
    description: "The feature has a shape, but it\'s not always present (if the feature is nested, then it should always be present at each nested level) or its value lengths vary."
    reason {
      type: INVALID_FEATURE_SHAPE
      short_description: "Feature shape dropped"
      description: "The feature has a shape, but it\'s not always present (if the feature is nested, then it should always be present at each nested level) or its value lengths vary."
    })pb");
  TestFindChanges(schema, DatasetStatsView(statistics, /*by_weight=*/false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

  TestFindChanges(Schema(), DatasetStatsView(statistics, /*by_weight=*/false),
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

  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

    TestFindChanges(schema, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
  expected_anomalies["new_feature"].expected_info_without_diff.set_severity(
      tensorflow::metadata::v0::AnomalyInfo::WARNING);

  // Set to warning severity using legacy new_features_are_warnings
  TestFindChanges(schema, DatasetStatsView(statistics, /*by_weight=*/false),
                  ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
                      "new_features_are_warnings: true"),
                  expected_anomalies);

  // Set to warning severity using severity_overrides
  TestFindChanges(schema, DatasetStatsView(statistics, /*by_weight=*/false),
                  ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
                      R"pb(severity_overrides: {
                             type: SCHEMA_NEW_COLUMN
                             severity: WARNING
                           })pb"),
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

  TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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
  auto result = anomalies.GetSchemaDiff(/*enable_diff_regions=*/false);

  TestAnomalies(result, initial, expected_anomalies);

  // Test that severity overrides affect severity in output anomalies.
  SchemaAnomalies anomalies_with_overrides(initial);
  TF_CHECK_OK(anomalies_with_overrides.FindChanges(
      DatasetStatsView(statistics), features,
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          R"pb(severity_overrides: {
                 type: FEATURE_TYPE_HIGH_NUMBER_VALUES
                 severity: WARNING
               })pb")));
  auto result_with_overrides = anomalies_with_overrides.GetSchemaDiff(
      /*enable_diff_regions=*/false);

  expected_anomalies["bar"].expected_info_without_diff.set_severity(
      tensorflow::metadata::v0::AnomalyInfo::WARNING);
  TestAnomalies(result_with_overrides, initial, expected_anomalies);
}

TEST(GetSchemaDiff, ValidSparseFeature) {
  // Note: This schema is incomplete, as it does not fully define the index
  // and value features and we do not generate feature stats for them.
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
  // Note: This schema is incomplete, as it does not fully define the index
  // and value features and we do not generate feature stats for them.
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["existing_feature"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path: { step: "existing_feature" }
        description: "Sparse feature name collision."
        severity: ERROR
        short_description: "Sparse feature name collision"
        reason {
          type: SPARSE_FEATURE_NAME_COLLISION
          short_description: "Sparse feature name collision"
          description: "Sparse feature name collision."
        })");
  TestFindChanges(schema_proto,
                  DatasetStatsView(no_anomaly_stats, /* by_weight= */ false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

// Test feature missing from stats.
TEST(GetSchemaDiff, SchemaMissingColumn) {
  const Schema schema_proto =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature: {
          name: 'f1'
          type: INT
          presence: { min_fraction: 1 }
        })");

  const DatasetFeatureStatistics empty_stats =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(features: {})");
  Schema schema_deprecated = schema_proto;
  DeprecateFeature(schema_deprecated.mutable_feature(0));
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
  expected_anomalies["f1"].expected_info_without_diff =
      ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
        path: { step: "f1" }
        description: "Column is completely missing"
        severity: WARNING
        short_description: "Column dropped"
        reason {
          type: SCHEMA_MISSING_COLUMN
          short_description: "Column dropped"
          description: "Column is completely missing"
        })");
  TestFindChanges(schema_proto,
                  DatasetStatsView(empty_stats, /*by_weight=*/false),
                  ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
                      R"pb(severity_overrides: {
                             type: SCHEMA_MISSING_COLUMN
                             severity: WARNING
                           })pb"),
                  expected_anomalies);
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

  testing::TestAnomalyInfo(
      anomaly.GetAnomalyInfo(baseline, /*enable_diff_regions=*/false),
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

  testing::TestAnomalyInfo(
      anomaly.GetAnomalyInfo(baseline, /*enable_diff_regions=*/false),
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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
                  std::map<std::string, testing::ExpectedAnomalyInfo>());
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
  // Note: This schema is incomplete, as it does not fully define the index
  // and value features and we do not generate feature stats for them.
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

  // Severity will be ERROR even though one anomaly is overidden to WARNING
  // because the max severity takes precedence.
  TestFindChanges(
      schema_proto,
      DatasetStatsView(missing_features_stats, /* by_weight= */ false),
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          R"pb(severity_overrides: {
                 type: SPARSE_FEATURE_MISSING_VALUE
                 severity: WARNING
               })pb"),
      expected_anomalies);
}

TEST(GetSchemaDiff, LengthMismatchSparseFeature) {
  // Note: This schema is incomplete, as it does not fully define the index
  // and value features and we do not generate feature stats for them.
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

  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

  // Test that severity overrides take effect.
  expected_anomalies["sparse_feature"].expected_info_without_diff.set_severity(
      tensorflow::metadata::v0::AnomalyInfo::WARNING);
  TestFindChanges(
      schema_proto,
      DatasetStatsView(length_mismatch_stats, /* by_weight= */ false),
      ParseTextProtoOrDie<FeatureStatisticsToProtoConfig>(
          R"pb(severity_overrides: {
                 type: SPARSE_FEATURE_LENGTH_MISMATCH
                 severity: WARNING
               })pb"),
      expected_anomalies);
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
  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;

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
  TestAnomalies(anomalies.GetSchemaDiff(/*enable_diff_regions=*/false), initial,
                expected_anomalies);
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

  std::map<std::string, testing::ExpectedAnomalyInfo> expected_anomalies;
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

  TestFindChanges(Schema(), DatasetStatsView(statistics, /*by_weight=*/false),
                  FeatureStatisticsToProtoConfig(), expected_anomalies);
}

TEST(SchemaAnomalies, FindsMaxImageByteSizeExceeded) {
  const Schema initial = ParseTextProtoOrDie<Schema>(R"(
    feature {
      name: "image/encoded"
      presence: { min_count: 1 min_fraction: 1.0 }
      value_count: { min: 1 max: 1 }
      type: BYTES
      image_domain: {
        max_image_byte_size: 100
      }
    }
  )");

  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'image/encoded'
          type: BYTES
          bytes_stats: {
            max_num_bytes_int: 101
            common_stats: {
              num_non_missing: 10
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  for (const auto& config : GetFeatureStatisticsToProtoConfigs()) {
    std::map<string, testing::ExpectedAnomalyInfo> expected_anomalies;
    expected_anomalies["image/encoded"].expected_info_without_diff =
        ParseTextProtoOrDie<tensorflow::metadata::v0::AnomalyInfo>(R"(
          path { step: "image/encoded" }
          description: "The largest image has bytes: 101. The max allowed byte size is: 100."
          severity: ERROR
          short_description: "Num bytes exceeds the max byte size."
          reason {
            type: MAX_IMAGE_BYTE_SIZE_EXCEEDED
            short_description: "Num bytes exceeds the max byte size."
            description: "The largest image has bytes: 101. The max allowed byte size is: 100."
          })");
    TestFindChanges(initial, DatasetStatsView(statistics, /*by_weight=*/false),
                    config, expected_anomalies);
  }
}

}  // namespace

}  // namespace data_validation
}  // namespace tensorflow
