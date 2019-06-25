// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Must be defined before init_numpy.h is included.
#define NUMPY_IMPORT_ARRAY
#include "tensorflow_data_validation/arrow/cc/init_numpy.h"

namespace tensorflow {
namespace data_validation {
void ImportNumpy() {
  static const int kUnused = [] {
    import_array1(-1);
    return 0;
  }();
  (void)kUnused;
}

}  // namespace data_validation
}  // namespace tensorflow
