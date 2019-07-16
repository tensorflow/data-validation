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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_INTERNAL_TYPES_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_INTERNAL_TYPES_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

// Represents the description of an anomaly, in short and long form.
struct Description {
  tensorflow::metadata::v0::AnomalyInfo::Type type;
  string short_description, long_description;

  friend bool operator==(const Description& a, const Description& b) {
    return (a.type == b.type && a.short_description == b.short_description &&
            a.long_description == b.long_description);
  }

  friend std::ostream& operator<<(std::ostream& strm, const Description& a) {
    return (strm << "{" << a.type << ", " << a.short_description << ", " <<
            a.long_description << "}");
  }
};

// UpdateSummary for a field.
struct UpdateSummary {
  // Clear the field in question. If this is a ``shared'' enum,
  // then the field is dropped.
  UpdateSummary() { clear_field = false; }
  bool clear_field;
  std::vector<Description> descriptions;
};

// Enum for comparators used in feature-level comparisons.
enum class FeatureComparatorType {
  SKEW,  // Compares serving and training data.
  DRIFT  // Compares previous and current spans.
};
// Enum for comparators used in dataset-level comparisons.
enum class DatasetComparatorType {
  DRIFT,   // Compares previous and current spans.
  VERSION  // Compares previous and current versions.
};

// The context for a tensorflow::metadata::v0::FeatureComparator.
// In tensorflow::metadata::v0::Feature, there are two comparisons:
// skew_comparator (that compares serving and training) and
// drift_comparator (that compares previous and current). This struct
// allows us to annotate the objects based upon this information.
struct ComparatorContext {
  string control_name;
  string treatment_name;
};

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_INTERNAL_TYPES_H_
