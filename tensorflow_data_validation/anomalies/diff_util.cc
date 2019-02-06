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

#include "tensorflow_data_validation/anomalies/diff_util.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data_validation {

std::vector<tensorflow::metadata::v0::DiffRegion> ComputeDiff(
    const std::vector<absl::string_view>& a_lines,
    const std::vector<absl::string_view>& b_lines) {
  CHECK(false) << "Schema diff is currently not supported.";
}

}  // namespace data_validation
}  // namespace tensorflow
