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

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"

namespace tensorflow {
namespace data_validation {

bool DatasetConstraintsHasComparator(
    const tensorflow::metadata::v0::DatasetConstraints& dataset_constraints,
    DatasetComparatorType comparator_type) {
  switch (comparator_type) {
    case DatasetComparatorType::DRIFT:
      return dataset_constraints.has_num_examples_drift_comparator();
    case DatasetComparatorType::VERSION:
      return dataset_constraints.has_num_examples_version_comparator();
  }
}

tensorflow::metadata::v0::NumericValueComparator* GetNumExamplesComparator(
    tensorflow::metadata::v0::DatasetConstraints* dataset_constraints,
    DatasetComparatorType comparator_type) {
  switch (comparator_type) {
    case DatasetComparatorType::DRIFT:
      return dataset_constraints->mutable_num_examples_drift_comparator();
    case DatasetComparatorType::VERSION:
      return dataset_constraints->mutable_num_examples_version_comparator();
  }
}

std::vector<Description> UpdateNumExamplesComparatorDirect(
    const DatasetStatsView& stats, const DatasetComparatorType comparator_type,
    tensorflow::metadata::v0::NumericValueComparator* comparator) {
  if (!comparator->has_min_fraction_threshold() &&
      !comparator->has_max_fraction_threshold()) {
    return {};
  }
  double num_examples = stats.GetNumExamples();
  // ValidateFeatureStatistics does not attempt to detect anomalies in
  // datasets that have num_examples == 0. Check that here.
  CHECK(num_examples > 0.0)
      << "Invalid input. Num examples must be greater than "
         "0.";

  const absl::optional<DatasetStatsView> control_stats =
      ((comparator_type == DatasetComparatorType::DRIFT)
           ? stats.GetPreviousSpan()
           : stats.GetPreviousVersion());
  if (!control_stats) {
    return {};
  }
  const string control_name =
      (comparator_type == DatasetComparatorType::DRIFT ? "previous span"
                                                       : "previous version");

  std::vector<Description> descriptions = {};
  double control_num_examples = control_stats->GetNumExamples();
  CHECK(control_num_examples >= 0.0) << "Invalid input. Control num examples "
                                      "must not be negative";

  double num_examples_ratio;
  if (control_num_examples != 0.0) {
    num_examples_ratio = num_examples / control_num_examples;
  }

  // TODO(b/138589350): Check for possible case of ratio == 1.0 but num_examples
  // != control_num_examples.
  if (comparator->has_max_fraction_threshold()) {
    double max_threshold = comparator->max_fraction_threshold();
    if (control_num_examples == 0.0) {
      comparator->clear_max_fraction_threshold();
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::COMPARATOR_HIGH_NUM_EXAMPLES,
           absl::StrCat("High num examples in current dataset versus the ",
                        control_name, ", which has 0."),
           absl::StrCat("The ", control_name,
                        " has 0 examples, so there is a high number of "
                        "examples in the current dataset versus the ",
                        control_name, ".")});
    } else if (num_examples_ratio > max_threshold) {
      comparator->set_max_fraction_threshold(num_examples_ratio);
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::COMPARATOR_HIGH_NUM_EXAMPLES,
           absl::StrCat("High num examples in current dataset versus the ",
                        control_name, "."),
           absl::StrCat(
               "The ratio of num examples in the current dataset versus the ",
               control_name, " is ", absl::SixDigits(num_examples_ratio),
               " (up to six significant digits), which is above the "
               "threshold ",
               absl::SixDigits(max_threshold), ".")});
    }
  }
  if (comparator->has_min_fraction_threshold()) {
    double min_threshold = comparator->min_fraction_threshold();
    if (control_num_examples != 0.0 &&
        num_examples_ratio < comparator->min_fraction_threshold()) {
      comparator->set_min_fraction_threshold(num_examples_ratio);
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::COMPARATOR_LOW_NUM_EXAMPLES,
           absl::StrCat("Low num examples in current dataset versus the ",
                        control_name, "."),
           absl::StrCat(
               "The ratio of num examples in the current dataset versus the ",
               control_name, " is ", absl::SixDigits(num_examples_ratio),
               " (up to six significant digits), which is below the threshold ",
               absl::SixDigits(min_threshold), ".")});
    }
  }
  return descriptions;
}

std::vector<Description> UpdateExamplesCount(
    const DatasetStatsView& stats,
    tensorflow::metadata::v0::DatasetConstraints* dataset_constraints) {
  std::vector<Description> descriptions;
  if (dataset_constraints->has_min_examples_count()) {
    const double num_present = stats.GetNumExamples();
    if (num_present < dataset_constraints->min_examples_count()) {
      dataset_constraints->set_min_examples_count(num_present);
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::DATASET_LOW_NUM_EXAMPLES,
           "Low num examples in dataset.",
           absl::StrCat("The dataset has ", num_present,
                        " examples, which is fewer than expected.")});
    }
  }
  if (dataset_constraints->has_max_examples_count()) {
    const double num_present = stats.GetNumExamples();
    if (num_present > dataset_constraints->max_examples_count()) {
      dataset_constraints->set_max_examples_count(num_present);
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::DATASET_HIGH_NUM_EXAMPLES,
           "High num examples in dataset.",
           absl::StrCat("The dataset has ", num_present,
                        " examples, which is more than expected.")});
    }
  }
  return descriptions;
}

}  // namespace data_validation
}  // namespace tensorflow
