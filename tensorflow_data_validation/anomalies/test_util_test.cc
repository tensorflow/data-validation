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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {
namespace testing {
namespace {


TEST(TestAnomalies, Basic) {
  const tensorflow::metadata::v0::Schema original =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Schema>(R"(
        feature {
          name: "feature_name"
          type: INT
          skew_comparator: { infinity_norm: { threshold: 0.1 } }
        })");

  tensorflow::metadata::v0::Anomalies result;
  *result.mutable_baseline() = original;
  TestAnomalies(result, original, std::map<string, ExpectedAnomalyInfo>());
}

}  // namespace
}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow
