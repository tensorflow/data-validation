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

tensorflow::Status ConvertSchemaV0ToV1(
    const tensorflow::metadata::v0::Schema& schema_proto,
    tensorflow::metadata::v0::Schema* schema) {
  *schema = schema_proto;
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSchemaV1ToV0(
    const tensorflow::metadata::v0::Schema& schema,
    tensorflow::metadata::v0::Schema* schema_proto) {
  *schema_proto = schema;
  return tensorflow::Status::OK();
}

void TestSchemaToAnomalies(
    const tensorflow::metadata::v0::Schema& old_schema,
    const std::function<tensorflow::metadata::v0::Anomalies(
        const tensorflow::metadata::v0::Schema&)>& get_diff,
    const std::map<string, ExpectedAnomalyInfo>& expected_anomalies) {
  // Test with V0.
  {
    const tensorflow::metadata::v0::Anomalies result = get_diff(old_schema);
    EXPECT_THAT(result.baseline(), EqualsProto(old_schema));
    for (const auto& pair : expected_anomalies) {
      const string& name = pair.first;
      const ExpectedAnomalyInfo& expected = pair.second;
      ASSERT_TRUE(ContainsKey(result.anomaly_info(), name))
          << "Expected anomaly for feature name: " << name
          << " not found in Anomalies: " << result.DebugString();
      tensorflow::metadata::v0::AnomalyInfo actual_info =
          result.anomaly_info().at(name);
      EXPECT_THAT(actual_info, EqualsProto(expected.expected_info_without_diff))
          << " (V0) column: " << name;
    }
    EXPECT_EQ(result.anomaly_info().size(), expected_anomalies.size())
        << " (V0): " << old_schema.DebugString();
  }
}

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow
