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

#include <vector>

#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/proto/validation_metadata.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

Status ToFeaturesNeededProto(const FeaturesNeeded& feature_needed,
                             FeaturesNeededProto* result) {
  for (const auto& entry : feature_needed) {
    string key = entry.first.AsProto().SerializeAsString();
    ReasonFeatureNeededList value;
    for (const auto& reason_feature_needed : entry.second) {
      *value.add_reason_feature_needed() = reason_feature_needed;
    }
    (*result->mutable_feature_needed())[key] = value;
  }

  return Status::OK();
}

Status FromFeaturesNeededProto(const FeaturesNeededProto& feature_needed_proto,
                               FeaturesNeeded* result) {
  for (const auto& entry : feature_needed_proto.feature_needed()) {
    metadata::v0::Path path;
    if (!path.ParseFromString(entry.first)) {
      return Status(
          error::INVALID_ARGUMENT,
          "FeaturesNeededProto key can not be parsed as metadata::v0::Path");
    }
    Path key(path);
    std::vector<ReasonFeatureNeeded> value = {
        entry.second.reason_feature_needed().begin(),
        entry.second.reason_feature_needed().end()};
    (*result)[key] = value;
  }

  return Status::OK();
}

}  // namespace data_validation
}  // namespace tensorflow
