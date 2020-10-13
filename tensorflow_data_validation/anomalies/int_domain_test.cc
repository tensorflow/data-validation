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

#include "tensorflow_data_validation/anomalies/int_domain_util.h"

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

using ::tensorflow::metadata::v0::AnomalyInfo;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::IntDomain;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

struct IntTypeIsValidTest {
  const string name;
  const IntDomain original;
  const FeatureNameStatistics input;
  const bool clear_field;
  const IntDomain expected;
  const AnomalyInfo::Type anomaly_type;
};

TEST(IntDomainTest, IsValid) {
  const std::vector<IntTypeIsValidTest> tests = {
      {"No bounds", IntDomain(), ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "14" }
             buckets: { label: "17" }
           }
         })"),
       false, IntDomain()},
      {"Negative strings with uint32",
       ParseTextProtoOrDie<IntDomain>("min: 0 is_categorical: true"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "-14" }
             buckets: { label: "17" }
           }
         })"),
       false, ParseTextProtoOrDie<IntDomain>("min: -14 is_categorical: true"),
       AnomalyInfo::INT_TYPE_SMALL_INT},
      {"Non-int string", ParseTextProtoOrDie<IntDomain>("min: 0"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "hello" }
             buckets: { label: "17" }
           }
         })"),
       true, ParseTextProtoOrDie<IntDomain>("min: 0"),
       AnomalyInfo::INT_TYPE_NOT_INT_STRING},
      {"Too large categorical int",
       ParseTextProtoOrDie<IntDomain>("max: 1000 is_categorical: true"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "50000000000000000000000000000" }
             buckets: { label: "10000000000000000000000000000" }
           }
         })"),
       true, ParseTextProtoOrDie<IntDomain>("max: 1000 is_categorical: true"),
       AnomalyInfo::DOMAIN_INVALID_FOR_TYPE},
      {"Valid large int with bounds",
       ParseTextProtoOrDie<IntDomain>(
           "max: 1528000000000000000 is_categorical: true"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           max: 1527900000000000000
           min: -1527900000000000000
         })"),
       false,
       ParseTextProtoOrDie<IntDomain>(
           "max: 1528000000000000000 is_categorical: true")},
      {"Large int no bounds", IntDomain(),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           max: 10000000000000000000.0
           min: -10000000000000000000.0
         })"),
       false, IntDomain()},
      {"Too large int with bounds", ParseTextProtoOrDie<IntDomain>("max: 1000"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           max: 100000000000000000000.0
           min: -100000000000000000000.0
         })"),
       true, ParseTextProtoOrDie<IntDomain>("max: 1000"),
       AnomalyInfo::DOMAIN_INVALID_FOR_TYPE},
      {"Actual int", IntDomain(), ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           max: 14.0
           min: 13.0
         })"),
       false, IntDomain()},
      {"Negative values for uint32", ParseTextProtoOrDie<IntDomain>("min: 0"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         num_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           max: 14.0
           min: -13.0
         })"),
       false, ParseTextProtoOrDie<IntDomain>("min: -13"),
       AnomalyInfo::INT_TYPE_SMALL_INT},
      {"With bounds", ParseTextProtoOrDie<IntDomain>("min: 14 max: 17"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "14" }
             buckets: { label: "17" }
           }
         })"),
       false, ParseTextProtoOrDie<IntDomain>("min: 14 max: 17")},
      // Same as above but with type: INT which is expected for categorical.
      {"Categorical int With bounds",
       ParseTextProtoOrDie<IntDomain>("min: 14 max: 17"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: INT
         string_stats: {
           common_stats: { num_missing: 3 max_num_values: 2 }
           unique: 3
           rank_histogram: {
             buckets: { label: "14" }
             buckets: { label: "17" }
           }
         })"),
       false, ParseTextProtoOrDie<IntDomain>("min: 14 max: 17")},
      {"Missing min and max values in stats, treated as 0, 0.",
       ParseTextProtoOrDie<IntDomain>("min: 7 max: 10"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar2'
         type: INT
         num_stats: { common_stats: { num_missing: 3 max_num_values: 2 } })"),
       false, ParseTextProtoOrDie<IntDomain>("min:0 max:10"),
       AnomalyInfo::INT_TYPE_SMALL_INT},
      {"Expand range.", ParseTextProtoOrDie<IntDomain>("min: 7 max: 10"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar3'
         type: INT
         num_stats: {
           min: 2
           max: 27
           common_stats: { num_missing: 3 max_num_values: 2 }
         })"),
       false, ParseTextProtoOrDie<IntDomain>("min:2 max:27"),
       AnomalyInfo::INT_TYPE_SMALL_INT}};
  for (const auto& test : tests) {
    const testing::DatasetForTesting dataset(test.input);
    IntDomain to_modify = test.original;
    UpdateSummary summary =
        UpdateIntDomain(dataset.feature_stats_view(), &to_modify);
    if (test.anomaly_type) {
      EXPECT_FALSE(summary.descriptions.empty());
      EXPECT_EQ(summary.descriptions.at(0).type, test.anomaly_type);
    } else {
      EXPECT_TRUE(summary.descriptions.empty());
      // If there are no descriptions, then there should be no changes.
      EXPECT_FALSE(summary.clear_field) << test.name;
      EXPECT_THAT(to_modify, EqualsProto(test.original)) << test.name;
    }

    EXPECT_EQ(summary.clear_field, test.clear_field)
        << "Test name: " << test.name;
    EXPECT_THAT(to_modify, EqualsProto(test.expected))
        << "Test name: " << test.name;
  }
}

struct IsIntDomainCandidateTest {
  const string name;
  const FeatureNameStatistics input;
  bool expected;
};

TEST(IntDomainTest, IsCandidate) {
  const std::vector<IsIntDomainCandidateTest> tests = {
      {"integer as string", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
              label: "1"
            }
            buckets: {
              label: "4"
            }}})"),
       true},
      {"an actual integer", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
              num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          max: 14.0
          min: 13.0})"),
       false},
      {"random string", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
       false},
      {"float values", ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          min: 0.0
          max: 1.0})"),
       false}};
  for (const auto& test : tests) {
    EXPECT_EQ(IsIntDomainCandidate(
                  testing::DatasetForTesting(test.input).feature_stats_view()),
              test.expected)
        << test.name;
  }
}

}  // namespace data_validation
}  // namespace tensorflow
