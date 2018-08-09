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

#include "tensorflow_data_validation/anomalies/bool_domain_util.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/types.h"
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
      {"true_false", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: {
               label: "true"
             }
             buckets: {
               label: "false"}}})"),
       ParseTextProtoOrDie<BoolDomain>(R"(
             true_value: "true"
             false_value: "false")")}};
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
      {"true_false",
       ParseTextProtoOrDie<BoolDomain>(
           "true_value: 'true' false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       true},
      {"true_false with wacky value",
       ParseTextProtoOrDie<BoolDomain>(
           "true_value: 'true' false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "wacky"
            }}})"),
       false},
      {"false_only", ParseTextProtoOrDie<BoolDomain>("false_value: 'false'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "false"
            }}})"),
       true},
      {"true_only", ParseTextProtoOrDie<BoolDomain>("true_value: 'true'"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }}})"),
       true},
      {"int_value", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
        num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          min: 0.0
          max: 1.0})"),
       true},
      {"int_value_big_range", BoolDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
        num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          min: 0.0
          max: 2.0})"),
       false}};
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
      {"true_false", ParseTextProtoOrDie<Feature>(R"(
           type: BYTES
           bool_domain {
               true_value: "true"
               false_value: "false"})"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       ParseTextProtoOrDie<Feature>(R"(
           type: BYTES
           bool_domain {
               true_value: "true"
               false_value: "false"})"),
       false},
      {"true_false", ParseTextProtoOrDie<Feature>(R"(
           type: BYTES
           bool_domain {
               true_value: "true"
               false_value: "false"})"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       ParseTextProtoOrDie<Feature>(R"(
           type: BYTES
           bool_domain {
               true_value: "true"
               false_value: "false"})"),
       false}};
  for (const auto& test : tests) {
    Feature to_modify = test.original;
    const std::vector<Description> descriptions = UpdateBoolDomain(
        testing::DatasetForTesting(test.input).feature_stats_view(),
        &to_modify);
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
      {"true_false", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       true},
      {"true_1 (can't have two positive values, make enum instead)",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "1"
            }}})"),
       false},
      {"true_false with wacky value",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "wacky"
            }}})"),
       false},
      {"false_only", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "false"
            }}})"),
       true},
      {"true_only", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }}})"),
       true},
      {"int_value", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
        num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          min: 0.0
          max: 1.0})"),
       true},
      {"int_value_big_range", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
        num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          min: 0.0
          max: 2.0})"),
       false},
      {"query_tokens", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          name: "query_tokens"
          type: STRING
          string_stats {
            common_stats {
              num_non_missing: 90
              num_missing: 10
              min_num_values: 1
              max_num_values: 10
              avg_num_values: 3.4
            }
            unique: 10037})"),
       false}};
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
