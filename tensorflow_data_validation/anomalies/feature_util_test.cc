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

#include "tensorflow_data_validation/anomalies/feature_util.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureComparator;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FeaturePresence;
using ::tensorflow::metadata::v0::ValueCount;
using testing::AddWeightedStats;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

string DescriptionsToString(const std::vector<Description>& descriptions) {
  string result;
  for (const Description& description : descriptions) {
    absl::StrAppend(&result, "short:", description.short_description, "\n");
    absl::StrAppend(&result, "long:", description.long_description, "\n");
  }
  return result;
}

Feature GetFeatureProtoOrDie(
    const tensorflow::metadata::v0::Schema& schema_proto,
    const string& field_name) {
  for (const Feature& feature_proto :
       schema_proto.feature()) {
    if (field_name == feature_proto.name()) {
      return feature_proto;
    }
  }
  LOG(FATAL) << "Name " << field_name << " not found in "
             << schema_proto.DebugString();
}

// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct FeatureIsDeprecatedTest {
  Feature feature_proto;
  bool is_deprecated;
};

Feature GetFeatureWithLifecycleStage(
    const tensorflow::metadata::v0::LifecycleStage& lifecycle_stage) {
  Feature feature;
  feature.set_lifecycle_stage(lifecycle_stage);
  return feature;
}

std::vector<FeatureIsDeprecatedTest> GetFeatureIsDeprecatedTests() {
  return {
      // Do not set deprecated and lifecycle_stage.
      {ParseTextProtoOrDie<Feature>(""), false},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::DEPRECATED),
       true},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::ALPHA), true},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::PLANNED), true},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::DEBUG_ONLY),
       true},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::PRODUCTION),
       false},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::BETA), false},
      {GetFeatureWithLifecycleStage(tensorflow::metadata::v0::UNKNOWN_STAGE),
       false}};
}

TEST(FeatureUtilTest, ClearDomain) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
          name: "bytes_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
        )");
  EXPECT_EQ(feature.domain_info_case(), Feature::DOMAIN_INFO_NOT_SET);
  feature.mutable_int_domain();
  EXPECT_EQ(feature.domain_info_case(), Feature::kIntDomain);
  ClearDomain(&feature);
  EXPECT_EQ(feature.domain_info_case(), Feature::DOMAIN_INFO_NOT_SET);
}

TEST(FeatureUtilTest, FeatureIsDeprecated) {
  for (const auto& test : GetFeatureIsDeprecatedTests()) {
    EXPECT_EQ(FeatureIsDeprecated(test.feature_proto), test.is_deprecated)
        << "Failed on  " << test.feature_proto.DebugString() << " expected "
        << test.is_deprecated;
  }
}

TEST(FeatureTypeTest, Deprecate) {
  for (const auto& test : GetFeatureIsDeprecatedTests()) {
    Feature to_modify = test.feature_proto;
    DeprecateFeature(&to_modify);
    EXPECT_TRUE(FeatureIsDeprecated(to_modify))
        << "Failed to deprecate: " << test.feature_proto.DebugString()
        << " produced " << to_modify.DebugString();
  }
}

// Construct a schema from a proto field, and then write it to a
// Feature.
struct FeatureNameStatisticsConstructorTest {
  FeatureNameStatistics statistics;
  Feature feature_proto;
};

// Repurpose for InitValueCountAndPresence.
// Also, break apart a separate test for other util constructors.
TEST(FeatureTypeTest, ConstructFromFeatureNameStatistics) {
  const std::vector<FeatureNameStatisticsConstructorTest> tests = {
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar1'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
              num_non_missing: 10
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       ParseTextProtoOrDie<Feature>(R"(
           value_count {
             min: 1
           }
           presence {
             min_count: 1
           }
           )")},
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          name: 'bar2'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 10
              max_num_values: 1
            }
            unique: 3
            rank_histogram: {
              buckets: {
                label: "foo"
              }
              buckets: {
                label: "bar"
              }
              buckets: {
                label: "baz"}}})"),
       ParseTextProtoOrDie<Feature>(R"(
           value_count {
             min: 1
             max: 1
           }
           presence {
             min_count: 1
           }
           )")},
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          name: 'bar3'
          type: INT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 1
              max_num_values: 2}})"),
       ParseTextProtoOrDie<Feature>(R"(
           value_count {
             min: 1
           }
           presence {
             min_count: 1
           }
           )")},
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 1
              max_num_values: 1
              min_num_values: 1}})"),
       ParseTextProtoOrDie<Feature>(R"(
           value_count {
             min: 1
             max: 1
           }
           presence {
             min_count: 1
             min_fraction: 1.0
           }
           )")},
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          name: 'bar4'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 1
              weighted_common_stats: {
                num_missing: 0
                num_non_missing: 0.5
              }
            }})"),
       ParseTextProtoOrDie<Feature>(R"(
           value_count {
             min: 1
             max: 1
           }
           presence { min_count: 1 }
       )")},
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
          name: 'bar5'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 100
              num_non_missing: 0
            }})"),
       ParseTextProtoOrDie<Feature>(R"(
           presence { min_count: 0 }
       )")}};
  for (const auto& test : tests) {
    Feature feature;

    const testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                             true);
    InitValueCountAndPresence(dataset.feature_stats_view(), &feature);
    EXPECT_THAT(feature, EqualsProto(test.feature_proto));
  }
}

// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct UpdateValueCountTest {
  string name;
  FeatureNameStatistics statistics;
  // Result of IsValid().
  bool expected_description_empty;
  // Initial feature proto.
  ValueCount original;
  // Result of Update().
  ValueCount expected;
};

const std::vector<UpdateValueCountTest> GetUpdateValueCountTests() {
  const std::vector<UpdateValueCountTest> battery_a = {
      {"optional_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)")},
      {"optional_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_string'
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
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       false, ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 2)")},
      {"optional_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_int64'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)")},
      {"optional_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min:1 max: 1)")},
      {"repeated_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min:1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min:1)")},
      {"repeated_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1)")},
      {"repeated_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1)")}};
  const std::vector<UpdateValueCountTest> battery_b = {
      {"repeated_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1)")},
      {"string_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"
               }
             }})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)")},
      {"string_int64_to_repeated",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       false, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 2)")},
      {
          "string_int32_valid",
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
                  buckets: {
                    label: "12"
                  }
                  buckets: {
                    label: "39"
                  }
                  buckets: {
                    label: "256"}}})"),
          true,
          ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
          ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
      },
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
               buckets: {
                 label: "FOO"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       false, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 2)")},
      {"min_max_5",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'string_int32'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 5
               max_num_values: 5
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "FOO"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 5 max: 5)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 5 max: 5)")},
      {"min_max_5_wrong",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'string_int32'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 3
               max_num_values: 8
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "FOO"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       false, ParseTextProtoOrDie<ValueCount>(R"(min: 5 max: 5)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 3 max: 8)")},
      {"string_uint32_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)")}};
  const std::vector<UpdateValueCountTest> battery_c = {
      {"string_uint32_negatives",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(name: 'string_uint32'
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
               buckets: {
                 label: "-12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)")},
      {"few_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 3
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 3)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 3)")},
      {"float_very_common_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 4
               num_non_missing: 6
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)")},
      {"float_very_common_invalid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 7  # 7/10 missing.
               num_non_missing: 3
               min_num_values: 1
               max_num_values: 1
               weighted_common_stats: {
                 num_non_missing: 3.0
                 num_missing: 7.0
               }
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)"),
       ParseTextProtoOrDie<ValueCount>(R"(min: 1 max: 1)")}};
  std::vector<UpdateValueCountTest> result(battery_a);
  result.insert(result.end(), battery_b.begin(), battery_b.end());
  result.insert(result.end(), battery_c.begin(), battery_c.end());
  return result;
}

// TODO(martinz): this is too many test cases that test too little.
// Write a two test cases focusing on min=max=5 (success and failure).
// Remove redundant tests.
TEST(FeatureTypeTest, UpdateValueCountTest) {
  for (const auto& test : GetUpdateValueCountTests()) {
    for (bool by_weight : {false, true}) {
      ValueCount to_modify = test.original;
      DatasetFeatureStatistics statistics;
      statistics.set_num_examples(10);
      *statistics.add_features() = test.statistics;
      testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                         by_weight);
      std::vector<Description> description_b;
      description_b =
          UpdateValueCount(dataset.feature_stats_view(), &to_modify);
      EXPECT_EQ(test.expected_description_empty, description_b.empty());
      EXPECT_THAT(to_modify, EqualsProto(test.expected))
          << "Test:" << test.name << "(by_weight: " << by_weight
          << ") Reason: " << DescriptionsToString(description_b);
    }
  }
}

// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct UpdatePresenceTest {
  string name;
  FeatureNameStatistics statistics;
  bool expected_description_empty;
  // Initial feature proto.
  FeaturePresence original;
  // Result of Update().
  FeaturePresence expected;
};

// TODO(martinz): this is too many test cases.
// Wrap MinCountInvalid and MinCount tests into here.
// Remove redundant tests.
const std::vector<UpdatePresenceTest> GetUpdatePresenceTests() {
  const std::vector<UpdatePresenceTest> battery_a = {
      {"optional_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_string'
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
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_int64'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")}};
  const std::vector<UpdatePresenceTest> battery_b = {
      {"repeated_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"
               }
             }})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_int64_to_repeated",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {
          "string_int32_valid",
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
                  buckets: {
                    label: "12"
                  }
                  buckets: {
                    label: "39"
                  }
                  buckets: {
                    label: "256"}}})"),
          true,
          ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
          ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
      },
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
               buckets: {
                 label: "FOO"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_uint32_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")}};
  const std::vector<UpdatePresenceTest> battery_c = {
      {"string_uint32_negatives",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
               buckets: {
                 label: "-12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"few_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 3
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"float_very_common_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 4
               num_non_missing: 6
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true,
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)")},
      {"float_very_common_invalid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 7  # 7/10 missing.
               num_non_missing: 3
               min_num_values: 1
               max_num_values: 1
               weighted_common_stats: {
                 num_non_missing: 3.0
                 num_missing: 7.0
               }
             }
             min: 0.0
             max: 1.0})"),
       false,
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.3)")}};
  std::vector<UpdatePresenceTest> result(battery_a);
  result.insert(result.end(), battery_b.begin(), battery_b.end());
  result.insert(result.end(), battery_c.begin(), battery_c.end());
  return result;
}

TEST(FeatureTypeTest, UpdatePresenceTest) {
  for (const auto& test : GetUpdatePresenceTests()) {
    for (bool by_weight : {false, true}) {
      FeaturePresence to_modify = test.original;
      DatasetFeatureStatistics statistics;
      statistics.set_num_examples(10);
      *statistics.add_features() = test.statistics;
      testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                         by_weight);
      std::vector<Description> description_b;
      description_b = UpdatePresence(dataset.feature_stats_view(), &to_modify);
      EXPECT_EQ(test.expected_description_empty, description_b.empty());
      EXPECT_THAT(to_modify, EqualsProto(test.expected))
          << "Test:" << test.name << "(by_weight: " << by_weight
          << ") Reason: " << DescriptionsToString(description_b);
    }
  }
}

TEST(FeatureTypeTest, RareMissingFeatureUnweighted) {
  // Rare feature missing for unweighted stats.
  // Notice that the feature is not technically missing given num_non_missing,
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:1000000000000000
          features {
            name: "one_in_a_quadrillion"
            type:INT
            num_stats: {
              common_stats: {
                num_missing: 1
                num_non_missing: 1000000000000000
                min_num_values: 1
                max_num_values: 1
              }
              min: 0.0
              max: 10.0
            }
          })");
  DatasetStatsView stats_view(statistics);
  const absl::optional<FeatureStatsView> feature_stats_view =
      stats_view.GetByPath(Path({"one_in_a_quadrillion"}));
  Feature to_modify = ParseTextProtoOrDie<Feature>(R"(
      name: "one_in_a_quadrillion"
      presence {
        min_fraction: 1.0
      }
      type: INT
      value_count {
        min: 1
        max: 1
      }
  )");
  std::vector<Description> desc =
      UpdatePresence(*feature_stats_view, to_modify.mutable_presence());
  ASSERT_EQ(desc.size(), 1);
  EXPECT_EQ(
      desc.at(0).long_description,
      "The feature was expected everywhere, but was missing in 1 examples.");
}

TEST(FeatureTypeTest, RareMissingFeatureWeighted) {
  // Rare feature missing for unweighted stats.
  // Notice that the feature is not technically missing given num_non_missing,
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:1000000000000000
          weighted_num_examples: 1000000000000000
          features {
            name: "one_in_a_quadrillion"
            type:INT
            num_stats: {
              common_stats: {
                num_missing: 1
                num_non_missing: 1000000000000000
                min_num_values: 1
                max_num_values: 1
                weighted_common_stats {
                  num_non_missing: 1000000000000000
                  num_missing: 1
                }
              }
              min: 0.0
              max: 10.0
            }
          })");
  DatasetStatsView stats_view(statistics, true);
  const absl::optional<FeatureStatsView> feature_stats_view =
      stats_view.GetByPath(Path({"one_in_a_quadrillion"}));
  Feature to_modify = ParseTextProtoOrDie<Feature>(R"(
      name: "one_in_a_quadrillion"
      presence {
        min_fraction: 1.0
      }
      type: INT
      value_count {
        min: 1
        max: 1
      }
  )");
  std::vector<Description> desc =
      UpdatePresence(*feature_stats_view, to_modify.mutable_presence());
  ASSERT_EQ(desc.size(), 1);
  EXPECT_EQ(
      desc.at(0).long_description,
      "The feature was expected everywhere, but was missing in 1 examples.");
}

// Tests IsValid and Update.
TEST(FeatureTypeTest, TestMinCount) {
  // Implicitly cardinality: CUSTOM.
  const Feature feature_proto =
      ParseTextProtoOrDie<Feature>(
          "name: 'foo' presence:{min_count: 3} type:INT");
  Feature to_modify = feature_proto;
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:10
          weighted_num_examples: 10.0
          features {
            name: 'foo'
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 4
                num_non_missing: 6
                min_num_values: 1
                max_num_values: 1
                weighted_common_stats: {
                  num_non_missing: 6.0
                }
              }
              min: 0.0
              max: 1.0
            }})");
  const testing::DatasetForTesting dataset(statistics, false);
  std::vector<Description> description_b;
  if (to_modify.has_presence()) {
    description_b = UpdatePresence(dataset.feature_stats_view(),
                                   to_modify.mutable_presence());
  }
  EXPECT_TRUE(description_b.empty());
  EXPECT_THAT(to_modify, EqualsProto(feature_proto));
}

// Tests IsValid and Update.
TEST(FeatureTypeTest, TestMinCountInvalid) {
  const Feature feature_proto =
      ParseTextProtoOrDie<Feature>(
          "name: 'foo' presence:{min_count: 7} type:INT");
  Feature to_modify = feature_proto;
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:10
          features {
            name: 'foo'
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 4
                num_non_missing: 6
                min_num_values: 1
                max_num_values: 1
              }
              min: 0.0
              max: 1.0}})");
  const testing::DatasetForTesting dataset(statistics, false);
  std::vector<Description> description_b;
  if (to_modify.has_presence()) {
    description_b = UpdatePresence(dataset.feature_stats_view(),
                                   to_modify.mutable_presence());
  }
  EXPECT_FALSE(description_b.empty());
  EXPECT_THAT(to_modify, EqualsProto(R"(
      name: "foo"
      type: INT
      presence {
        min_count: 6})"));
}

struct UpdateMaxCadinalityConstraintTest {
  // Description of the feature.
  Feature feature_proto;
  // If true then no messages should be generated as part of the update.
  bool empty_messages;
  //
  string description;
};

TEST(FeatureTypeTest, ValidateCustomFeatures) {
  const FeatureNameStatistics feature_stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'foo'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            num_non_missing: 7
            min_num_values: 1
            max_num_values: 2
            weighted_common_stats: {
              num_non_missing: 7.0
            }
          }
          weighted_string_stats: {
          }
        })");

  const std::vector<UpdateMaxCadinalityConstraintTest> tests = {
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       true, "Custom feature should have empty messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       true, "Repeated feature should have empty messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                             max: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       false, "Optional feature should have some messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                             max: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                             min_fraction: 1
                           })"),
       false, "Required feature should have some messages"}};
  for (const auto& test : tests) {
    for (bool by_weight : {true, false}) {
      const testing::DatasetForTesting dataset(feature_stats, false);
      Feature to_modify = test.feature_proto;

      std::vector<Description> description_a;
      if (to_modify.has_value_count()) {
        description_a = UpdateValueCount(dataset.feature_stats_view(),
                                         to_modify.mutable_value_count());
      }
      std::vector<Description> description_b;
      if (to_modify.has_presence()) {
        description_b = UpdatePresence(dataset.feature_stats_view(),
                                       to_modify.mutable_presence());
      }

      EXPECT_EQ(test.empty_messages,
                description_a.empty() && description_b.empty())
          << test.description << "(by_weight: " << by_weight << ")";
    }
  }
}

TEST(FeatureTypeTest, HasSkewComparatorFalse) {
  const Feature feature = ParseTextProtoOrDie<Feature>(R"(name: "feature_name")");
  EXPECT_FALSE(FeatureHasComparator(feature, ComparatorType::SKEW));
}

TEST(FeatureTypeTest, HasSkewComparatorTrue) {
  const Feature feature = ParseTextProtoOrDie<Feature>(R"(name: "feature_name"
    skew_comparator {})");
  EXPECT_TRUE(FeatureHasComparator(feature, ComparatorType::SKEW));
}

TEST(FeatureTypeTest, MutableSkewComparator) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
          threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, ComparatorType::SKEW);
  ASSERT_TRUE(comparator != nullptr);
  EXPECT_THAT(*comparator, EqualsProto(R"(
    infinity_norm: { threshold: 0.1 })"));
}

TEST(FeatureTypeTest, MutableComparator2) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
        threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, ComparatorType::SKEW);
  ASSERT_TRUE(comparator != nullptr);
  comparator->mutable_infinity_norm()->set_threshold(0.2);
  EXPECT_THAT(feature.skew_comparator(),
              EqualsProto(R"(infinity_norm: { threshold: 0.2 })"));
}

TEST(FeatureTypeTest, MutableComparatorWithDrift) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      drift_comparator: {
        infinity_norm: {
        threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, ComparatorType::DRIFT);
  ASSERT_TRUE(comparator != nullptr);
  comparator->mutable_infinity_norm()->set_threshold(0.2);
  EXPECT_THAT(feature.drift_comparator(),
              EqualsProto(R"(infinity_norm: { threshold: 0.2 })"));
}

TEST(FeatureTypeTest, GetComparatorNormal) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
          threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, ComparatorType::SKEW);

  ASSERT_TRUE(comparator != nullptr);
  EXPECT_THAT(*comparator, EqualsProto("infinity_norm: { threshold: 0.1 }"));
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
