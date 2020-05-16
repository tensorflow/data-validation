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
// Utilities to modify a dataset constraint in the schema.
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_DATASET_CONSTRAINTS_UTIL_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_DATASET_CONSTRAINTS_UTIL_H_

#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ANOMALIES_DATASET_CONSTRAINTS_UTIL_H_

namespace tensorflow {
namespace data_validation {
// Specifies whether the dataset constraints has a comparator of the specified
// type.
bool DatasetConstraintsHasComparator(
    const tensorflow::metadata::v0::DatasetConstraints& dataset_contraints,
    DatasetComparatorType comparator_type);

// Gets the num examples comparator of the specified type, creating it if it
// does not exist.
tensorflow::metadata::v0::NumericValueComparator* GetNumExamplesComparator(
    tensorflow::metadata::v0::DatasetConstraints* dataset_constraints,
    DatasetComparatorType comparator_type);

// Updates the num examples comparator from the dataset constraints.
std::vector<Description> UpdateNumExamplesComparatorDirect(
    const DatasetStatsView& stats, DatasetComparatorType comparator_type,
    tensorflow::metadata::v0::NumericValueComparator* comparator);

// Updates the min and max examples count from the dataset constraints.
std::vector<Description> UpdateExamplesCount(
    const DatasetStatsView& stats,
    tensorflow::metadata::v0::DatasetConstraints* dataset_constraints);

}  // namespace data_validation
}  // namespace tensorflow
