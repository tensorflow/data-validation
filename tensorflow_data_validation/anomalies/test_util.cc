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

#include "tensorflow_data_validation/anomalies/test_util.h"

#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {
namespace testing {

ProtoStringMatcher::ProtoStringMatcher(const string& expected)
    : expected_(expected) {}
ProtoStringMatcher::ProtoStringMatcher(
    const ::tensorflow::protobuf::Message& expected)
    : expected_(expected.DebugString()) {}


void TestAnomalies(
    const tensorflow::metadata::v0::Anomalies& actual,
    const tensorflow::metadata::v0::Schema& old_schema,
    const std::map<string, ExpectedAnomalyInfo>& expected_anomalies) {
  EXPECT_THAT(actual.baseline(), EqualsProto(old_schema));
  for (const auto& pair : expected_anomalies) {
    const string& name = pair.first;
    const ExpectedAnomalyInfo& expected = pair.second;
    ASSERT_TRUE(ContainsKey(actual.anomaly_info(), name))
        << "Expected anomaly for feature name: " << name
        << " not found in Anomalies: " << actual.DebugString();
    TestAnomalyInfo(actual.anomaly_info().at(name), old_schema, expected,
                    absl::StrCat(" column: ", name));
  }
  for (const auto& pair : actual.anomaly_info()) {
    const string& name = pair.first;
    EXPECT_TRUE(ContainsKey(expected_anomalies, name))
        << "Unexpected anomaly: " << name << " "
        << pair.second.DebugString();
  }
}

void TestAnomalyInfo(const tensorflow::metadata::v0::AnomalyInfo& actual,
                     const tensorflow::metadata::v0::Schema& baseline,
                     const ExpectedAnomalyInfo& expected,
                     const string& comment) {
  tensorflow::metadata::v0::AnomalyInfo actual_info = actual;
  EXPECT_THAT(actual_info, EqualsProto(expected.expected_info_without_diff))
      << comment;
}

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow
