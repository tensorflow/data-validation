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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/bool_domain_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::BoolDomain;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

// Used for friend for BoolType.
class BoolTypeTest : public ::testing::Test {};

struct BoolTypeBoolTypeFeatureNameStatisticsTest {
  const string name;
  const FeatureNameStatistics input;
  const BoolDomain expected;
};

TEST_F(BoolTypeTest, BoolTypeFeatureNameStatistics) {
  const std::vector<BoolTypeBoolTypeFeatureNameStatisticsTest> tests = {
      {"true_false_string", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "true" }
             buckets: { label: "false" }
           }
         })pb"),
       ParseTextProtoOrDie<BoolDomain>(R"pb(
         true_value: "true"
         false_value: "false")pb")},
      {"true_false_int", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           min: 0.0
           max: 1.0
         }
       )pb"),
       BoolDomain()},
  };
  for (const auto& test : tests) {
    const BoolDomain result = BoolDomainFromStats(
        testing::DatasetForTesting(test.input).feature_stats_view());
    EXPECT_THAT(result, EqualsProto(test.expected));
  }
}

struct BoolTypeIsValidTest {
  const string name;
  const BoolDomain original;
  const FeatureNameStatistics input;
  const bool expected;
};

TEST_F(BoolTypeTest, IsValid) {
  const std::vector<BoolTypeIsValidTest> tests = {
      {"string_true_false",
       ParseTextProtoOrDie<BoolDomain>(
           "true_value: 'true' false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true" }
             buckets: { label: "false" }
           }
         })pb"),
       true},
      {"string_true_only",
       ParseTextProtoOrDie<BoolDomain>(
           "true_value: 'true' false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 1
           rank_histogram: { buckets: { label: "true" } }
         })pb"),
       true},
      {"string_false_only",
       ParseTextProtoOrDie<BoolDomain>(
           "true_value: 'true' false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 1
           rank_histogram: { buckets: { label: "false" } }
         })pb"),
       true},
      {"float_valid_stats_and_domain_config", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 1.0
           histograms: {
             buckets: { sample_count: 1.5 low_value: 0.0 high_value: 0.0 }
             buckets: { sample_count: 1.5 low_value: 0.0 high_value: 0.0 }
             buckets: { sample_count: 0 low_value: 0.0 high_value: 1.0 }
             buckets: { sample_count: 2 low_value: 1.0 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       true},
      {"float_valid_stats_only_0s", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 0.0
           histograms: {
             buckets: { sample_count: 2 low_value: 0.0 high_value: 0.0 }
             type: QUANTILES
           }
         }
       )pb"),
       true},
      {"float_valid_stats_only_1s", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 1.0
           max: 1.0
           histograms: {
             buckets: { sample_count: 2  low_value: 1.0 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       true},
      {"int_value", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           min: 0.0
           max: 1.0
         })pb"),
       true},
      {"int_valid_stats_only_0s", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 0.0 max: 0.0 }
       )pb"),
       true},
      {"int_valid_stats_only_1s", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 1.0 max: 1.0 }
       )pb"),
       true},
  };
  for (const auto& test : tests) {
    Feature feature;
    *feature.mutable_bool_domain() = test.original;
    const Feature original_feature = feature;

    std::vector<Description> description = UpdateBoolDomain(
        testing::DatasetForTesting(test.input).feature_stats_view(), &feature);

    // IsValid is the equivalent of an empty description.
    EXPECT_EQ(description.empty(), test.expected);
    if (test.expected) {
      // If it is valid, then the schema shouldn't be updated.
      EXPECT_THAT(feature, EqualsProto(original_feature));
    }
  }
}

struct BoolTypeUpdateTest {
  const string name;
  const Feature original;
  const FeatureNameStatistics input;
  const Feature expected;
  const bool expected_deprecated;
};

TEST_F(BoolTypeTest, Update) {
  const std::vector<BoolTypeUpdateTest> tests = {
      {"string_invalid_missing_true_config", ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
         bool_domain { false_value: "false_val" }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true_val" }
             buckets: { label: "false_val" }
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
       )pb"),
       false},
      {"string_invalid_missing_false_config", ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
         bool_domain { true_value: "true_val" }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true_val" }
             buckets: { label: "false_val" }
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
       )pb"),
       false},
      {"string_invalid_missing_domain_config",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true_val" }
             buckets: { label: "false_val" }
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
       )pb"),
       false},
      {"string_invalid_unexpected_string", ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
         bool_domain { true_value: "true_val" false_value: "false_val" }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true_val" }
             buckets: { label: "dummy_val" }
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: BYTES
       )pb"),
       false},
      {"float_invalid_domain_config_has_true_value",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain { true_value: 'dummy_val' }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 1.0
           histograms: {
             buckets: { low_value: 0.0 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: 0.0 max: 1.0 }
       )pb"),
       false},
      {"float_invalid_domain_config_has_false_value",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain { false_value: 'dummy_val' }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 1.0
           histograms: {
             buckets: { low_value: 0.0 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: 0.0 max: 1.0 }
       )pb"),
       false},
      {"float_invalid_stats_small_min", ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: { min: -1.0 max: 1.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: -1.0 max: 1 }
       )pb"),
       false},
      {"float_invalid_stats_large_max", ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: { min: 0.0 max: 2.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: 0.0 max: 2.0 }
       )pb"),
       false},
      {"float_invalid_stats_has_nan", ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 1.0
           histograms: {
             num_nan: 1
             buckets: { low_value: 0.0 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: 0.0 max: 1.0 }
       )pb"),
       false},
      {"float_invalid_stats_unexpected_value",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: {
           min: 0.0
           max: 1.0
           histograms: {
             buckets: { sample_count: 1 low_value: 0.0 high_value: 0.5 }
             buckets: { sample_count: 1 low_value: 0.5 high_value: 1.0 }
             type: QUANTILES
           }
         }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: FLOAT
         float_domain: { min: 0.0 max: 1.0 }
       )pb"),
       false},
      {"invalid_domain_config_has_true_value",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         bool_domain { true_value: 'dummy_val' }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 0.0 max: 1.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         int_domain: { min: 0 max: 1 }
       )pb"),
       false},
      {"invalid_domain_config_has_false_value",
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         bool_domain { false_value: 'dummy_val' }
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 0.0 max: 1.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         int_domain: { min: 0 max: 1 }
       )pb"),
       false},
      {"invalid_stats_small_min", ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: -1.0 max: 1.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         int_domain: { min: -1 max: 1 }
       )pb"),
       false},
      {"invalid_stats_large_max", ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         bool_domain {}
       )pb"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 0.0 max: 2.0 }
       )pb"),
       ParseTextProtoOrDie<Feature>(R"pb(
         type: INT
         int_domain: { min: 0 max: 2 }
       )pb"),
       false},
  };
  for (const auto& test : tests) {
    Feature to_modify = test.original;
    const std::vector<Description> description = UpdateBoolDomain(
        testing::DatasetForTesting(test.input).feature_stats_view(),
        &to_modify);
    EXPECT_EQ(description.empty(), test.expected_deprecated);
    EXPECT_THAT(to_modify, EqualsProto(test.expected));
  }
}

struct BoolTypeIsCandidateTest {
  const string name;
  const FeatureNameStatistics input;
  bool expected;
};

TEST_F(BoolTypeTest, IsCandidate) {
  const std::vector<BoolTypeIsCandidateTest> tests = {
      {"string valid unique values count",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true" }
             buckets: { label: "false" }
           }
         })pb"),
       true},
      {"true_1 (can't have two positive values, make enum instead)",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 2
           rank_histogram: {
             buckets: { label: "true" }
             buckets: { label: "1" }
           }
         })pb"),
       false},
      {"true_false with wacky value",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "true" }
             buckets: { label: "wacky" }
           }
         })pb"),
       false},
      {"false_only", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 1
           rank_histogram: { buckets: { label: "false" } }
         })pb"),
       true},
      {"true_only", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 1
           rank_histogram: { buckets: { label: "true" } }
         })pb"),
       true},
      {"int_value", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           min: 0.0
           max: 1.0
         })pb"),
       true},
      {"int_value_big_range", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           min: 0.0
           max: 2.0
         })pb"),
       false},
      {"int valid min max equal to 1",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { min: 1.0 max: 1.0 }
       )pb"),
       true},
      {"int invalid min", ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { num_zeros: 2 min: -1.0 max: 1.0 }
       )pb"),
       false},
      {"int invalid max not equal 1",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { num_zeros: 2 min: 0.0 max: 0.0 }
       )pb"),
       false},
      {"int invalid missing max",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: { num_zeros: 2 min: 0.0 }
       )pb"),
       false},
      {"int missing min max without zeros count",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: INT
         num_stats: {}
       )pb"),
       false},
      {"never infer bool domain for stats with FLOAT type",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: FLOAT
         num_stats: { num_zeros: 2 min: 0.0 max: 1.0 }
       )pb"),
       false},
      {"never infer bool domain for stats with BYTES type",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: BYTES
         bytes_stats: {}
       )pb"),
       false},
      {"never infer bool domain for stats with STRUCT type",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
         name: 'bar'
         type: STRUCT
         struct_stats: {}
       )pb"),
       false},
  };
  for (const auto& test : tests) {
    EXPECT_EQ(IsBoolDomainCandidate(
                  testing::DatasetForTesting(test.input).feature_stats_view()),
              test.expected)
        << test.name;
  }
}

struct UpdateBoolDomainSelfTest {
  const string name;
  const BoolDomain original;
  const BoolDomain expected;
  const bool descriptions_empty;
};

// Update a BoolDomain by itself. Namely, if true and false
// both have the same value, clear the false.
TEST(BoolDomainUtil, UpdateBoolDomainSelf) {
  std::vector<UpdateBoolDomainSelfTest> tests = {
      {"correct_present",
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"foo" false_value:"bar")"),
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"foo" false_value:"bar")"),
       true},
      {"correct_false_only",
       ParseTextProtoOrDie<BoolDomain>(R"(false_value:"bar")"),
       ParseTextProtoOrDie<BoolDomain>(R"(false_value:"bar")"), true},
      {"correct_true_only",
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"bar")"),
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"bar")"), true},
      {"correct_empty", ParseTextProtoOrDie<BoolDomain>(R"()"),
       ParseTextProtoOrDie<BoolDomain>(R"()"), true},
      {"broken_identical",
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"foo" false_value:"foo")"),
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"foo")"), false},
      {"broken_empty_strings",
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"" false_value:"")"),
       ParseTextProtoOrDie<BoolDomain>(R"(true_value:"")"), false},
  };
  for (const auto& test : tests) {
    BoolDomain to_modify = test.original;
    const std::vector<Description> descriptions =
        UpdateBoolDomainSelf(&to_modify);
    EXPECT_THAT(to_modify, EqualsProto(test.expected))
        << " test: " << test.name;
    EXPECT_EQ(descriptions.empty(), test.descriptions_empty)
        << " test: " << test.name;
  }
}
}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
