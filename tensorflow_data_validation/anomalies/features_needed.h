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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_

#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// TODO(martinz): consider making a protobuf.
// TODO(martinz): consider adding an environment.
// TODO(martinz): consider adding a LifecycleStage.
struct ReasonFeatureNeeded {
  // If there is an issue in creating the field, the comment should help
  // explain why.
  // Example: "This is needed for transform XYZ (see /A/B/C)"
  string comment;
};

using FeaturesNeeded = std::map<Path, std::vector<ReasonFeatureNeeded>>;

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_
