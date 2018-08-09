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

#include "tensorflow_data_validation/anomalies/float_domain_util.h"

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
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FloatDomain;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

struct UpdateFloatDomainTest {
  const string name;
  FloatDomain float_domain;
  const FeatureNameStatistics input;
  const bool clear_field;
  FloatDomain expected;
};

std::vector<UpdateFloatDomainTest> GetUpdateFloatDomainTests() {
  return {
      {"transform_as_string", FloatDomain(),
           ParseTextProtoOrDie<FeatureNameStatistics>(R"(
             name: "transform_as_string"
             type: STRING
             string_stats: {
             common_stats: {
               num_missing: 3
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "1.5"
               }
               buckets: {
                 label: "0.25"
               }}})"),
       false, FloatDomain()},
      {"float_value_in_range",
       ParseTextProtoOrDie<FloatDomain>("min: 3.0 max: 5.0"),
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'bar'
                type: FLOAT
                num_stats: {
                  common_stats: {
                    num_missing: 3
                    max_num_values: 2
                  }
                  min: 3.5
                  max: 4.5})"),
       false, ParseTextProtoOrDie<FloatDomain>("min: 3.0 max: 5.0")},
      {"low_value", ParseTextProtoOrDie<FloatDomain>("min: 3.0 max: 5.0"),
           ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'bar'
                type: FLOAT
                num_stats: {
                  common_stats: {
                    num_missing: 3
                    max_num_values: 2
                  }
                  min: 2.5
                  max: 4.5})"),
       false, ParseTextProtoOrDie<FloatDomain>("min: 2.5 max: 5")}};
}

TEST(FloatDomainTest, UpdateFloatDomain) {
  for (const auto& test : GetUpdateFloatDomainTests()) {
    const testing::DatasetForTesting dataset(test.input);
    FloatDomain to_modify = test.float_domain;
    UpdateSummary summary =
        UpdateFloatDomain(dataset.feature_stats_view(), &to_modify);
    if (summary.descriptions.empty()) {
      // If there are no descriptions, then there should be no changes.
      EXPECT_FALSE(summary.clear_field) << test.name;
      EXPECT_THAT(to_modify, EqualsProto(test.float_domain)) << test.name;
    }

    EXPECT_EQ(summary.clear_field, test.clear_field) << test.name;
    EXPECT_THAT(to_modify, EqualsProto(test.expected)) << test.name;
  }
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
