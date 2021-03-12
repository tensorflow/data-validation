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

// Provides a variety of tools for evaluating methods that output Anomalies.
// In particular, allows for tests written for schema version 0 to apply to
// schema version 1.
// Also, allows us to have expected schema protos instead of
// expected diff regions.

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_TEST_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_TEST_UTIL_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {
namespace testing {

using tensorflow::protobuf::TextFormat;

// Simple implementation of a proto matcher comparing string representations.
//
// IMPORTANT: Only use this for protos whose textual representation is
// deterministic (that may not be the case for the map collection type).
// This code has been copied from
// https://github.com/tensorflow/serving/blob/master/tensorflow_serving/test_util/test_util.h

class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const string& expected);
  explicit ProtoStringMatcher(const ::tensorflow::protobuf::Message& expected);

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener* /* listener */) const;

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const string expected_;
};

template <typename T>
T CreateProto(const string& textual_proto) {
  T proto;
  CHECK(TextFormat::ParseFromString(textual_proto, &proto));
  return proto;
}

template <typename Message>
bool ProtoStringMatcher::MatchAndExplain(
    const Message& p, ::testing::MatchResultListener* /* listener */) const {
  // Need to CreateProto and then print as string so that the formatting
  // matches exactly.
  return p.SerializeAsString() ==
         CreateProto<Message>(expected_).SerializeAsString();
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const ::tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Parse input string as a protocol buffer.
template <typename T>
T ParseTextProtoOrDie(const string& input) {
  T result;
  CHECK(TextFormat::ParseFromString(input, &result))
      << "Failed to parse: " << input;
  return result;
}

// Store this as a proto, to make it easier to understand and update tests.
struct ExpectedAnomalyInfo {
  tensorflow::metadata::v0::AnomalyInfo expected_info_without_diff;
  tensorflow::metadata::v0::Schema new_schema;
};

// Test if anomalies is as expected.
void TestAnomalies(
    const tensorflow::metadata::v0::Anomalies& actual,
    const tensorflow::metadata::v0::Schema& old_schema,
    const std::map<string, ExpectedAnomalyInfo>& expected_anomalies,
    const std::vector<tensorflow::metadata::v0::DriftSkewInfo>&
        expected_drift_skew_infos = {});

void TestAnomalyInfo(const tensorflow::metadata::v0::AnomalyInfo& actual,
                     const ExpectedAnomalyInfo& expected,
                     const string& comment);

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_TEST_UTIL_H_
