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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_DIFF_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_DIFF_UTIL_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

// The schema diff computation functionality is currently not supported.
// Tracked in https://github.com/tensorflow/data-validation/issues/39
std::vector<tensorflow::metadata::v0::DiffRegion> ComputeDiff(
    const std::vector<absl::string_view>& a_lines,
    const std::vector<absl::string_view>& b_lines);

}  // namespace data_validation
}  // namespace tensorflow
#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_DIFF_UTIL_H_
