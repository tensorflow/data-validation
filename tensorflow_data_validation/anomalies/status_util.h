/* Copyright 2023 Google LLC

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

#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATUS_UTIL_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATUS_UTIL_H_

#include "absl/base/log_severity.h"
#include "absl/base/optimization.h"
#include "absl/log/log.h"

namespace tensorflow {
namespace data_validation  {

// For propagating errors when calling a function.
#define TFDV_RETURN_IF_ERROR(...)                       \
  do {                                                     \
    const absl::Status _status = (__VA_ARGS__);            \
    if (ABSL_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#ifndef CHECK_NOTNULL
template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LOG(FATAL).AtLocation(file, line)
        << std::string(exprtext);
  }
  return std::forward<T>(t);
}

#define CHECK_NOTNULL(val)                     \
  ::tensorflow::data_validation::CheckNotNull( \
      __FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

#endif  // CHECK_NOTNULL

}  // namespace data_validation
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_STATUS_UTIL_H_
