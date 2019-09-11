/* Copyright 2019 Google LLC

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

#include "tensorflow_data_validation/anomalies/features_needed.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/proto/validation_metadata.pb.h"
#include "tensorflow_data_validation/anomalies/test_util.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::testing::ElementsAre;
using testing::EqualsProto;
using ::testing::Pair;
using testing::ParseTextProtoOrDie;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

TEST(FeaturesNeededTest, CppToProtoToCpp) {
  Path path({"a", "b", "c"});

  auto reason1 = ParseTextProtoOrDie<ReasonFeatureNeeded>("comment: 'test1'");
  auto reason2 = ParseTextProtoOrDie<ReasonFeatureNeeded>("comment: 'test2'");
  FeaturesNeeded features_need;
  features_need[path] = {reason1, reason2};

  FeaturesNeededProto proto_format;
  EXPECT_TRUE(ToFeaturesNeededProto(features_need, &proto_format).ok());

  // Verify that proto_format are correctly generated.
  EXPECT_THAT(proto_format, testing::EqualsProto(R"(
                path_and_reason_feature_need {
                  path { step: "a" step: "b" step: "c" }
                  reason_feature_needed { comment: "test1" }
                  reason_feature_needed { comment: "test2" }
                }
              )"));

  // Verify that C++ -> Proto -> C++ is a noop.
  FeaturesNeeded generated_features_need;
  EXPECT_TRUE(
      FromFeaturesNeededProto(proto_format, &generated_features_need).ok());
  EXPECT_THAT(
      generated_features_need,
      UnorderedElementsAre(
          Pair(path, ElementsAre(EqualsProto(reason1), EqualsProto(reason2)))));
}

TEST(FeaturesNeededTest, ProtoToCppToProto) {
  auto original_proto = ParseTextProtoOrDie<FeaturesNeededProto>(R"(
    path_and_reason_feature_need {
      path { step: "a" step: "b" step: "c" }
      reason_feature_needed { comment: "test1" }
      reason_feature_needed { comment: "test2" }
    }
  )");
  FeaturesNeeded features_need;
  EXPECT_TRUE(FromFeaturesNeededProto(original_proto, &features_need).ok());

  // Verify that C++ object are expected.
  Path expeceted_path({"a", "b", "c"});
  auto expected_reason1 =
      ParseTextProtoOrDie<ReasonFeatureNeeded>("comment: 'test1'");
  auto expected_reason2 =
      ParseTextProtoOrDie<ReasonFeatureNeeded>("comment: 'test2'");
  EXPECT_THAT(features_need,
              UnorderedElementsAre(Pair(
                  expeceted_path, ElementsAre(EqualsProto(expected_reason1),
                                              EqualsProto(expected_reason2)))));

  // Verify that Proto -> C++ -> Proto is a noop.
  FeaturesNeededProto generated_proto_format;
  EXPECT_TRUE(
      ToFeaturesNeededProto(features_need, &generated_proto_format).ok());

  EXPECT_THAT(original_proto, EqualsProto(generated_proto_format));
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
