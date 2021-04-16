/* Copyright 2021 Google LLC

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

#include "tensorflow_data_validation/anomalies/natural_language_domain_util.h"

#include <set>
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::CustomStatistic;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::NaturalLanguageDomain;
using ::tensorflow::metadata::v0::NaturalLanguageStatistics;

const Description kMissingStatsDescription = {
    tensorflow::metadata::v0::AnomalyInfo::STATS_NOT_AVAILABLE,
    "Natural language stats are not computed.",
    "Constraints specified in natural language domain cannot be "
    "verified because natural language stats have not been computed."};

}  // namespace

std::vector<Description> UpdateNaturalLanguageDomain(
    const FeatureStatsView& feature_stats, Feature* feature) {
  std::vector<Description> results;

  const CustomStatistic* nl_custom_stats =
      feature_stats.GetCustomStatByName("nl_statistics");

  NaturalLanguageStatistics nl_stats;
  bool found_nl_stats = false;
  if (nl_custom_stats) {
    if (!nl_custom_stats->any().UnpackTo(&nl_stats)) {
      LOG(WARNING) << "nl_statistics for feature " << feature->name()
                   << "do not have the expected "
                   << "NaturalLanguageStatistics message format.";
      return results;
    }
    found_nl_stats = true;
  }

  const NaturalLanguageDomain& nl_domain = feature->natural_language_domain();
  if (nl_domain.coverage().has_min_coverage()) {
    if (!found_nl_stats) {
      results.push_back(kMissingStatsDescription);
      feature->clear_natural_language_domain();
      return results;

    } else if (nl_domain.coverage().min_coverage() >
               nl_stats.feature_coverage()) {
      results.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_COVERAGE_TOO_LOW,
           "Feature coverage is too low.",
           absl::StrCat("Fraction of tokens in the vocabulary: ",
                        std::to_string(nl_stats.feature_coverage()),
                        " is lower than the threshold set in the Schema: ",
                        std::to_string(nl_domain.coverage().min_coverage()),
                        ".")});
      feature->mutable_natural_language_domain()
          ->mutable_coverage()
          ->set_min_coverage(nl_stats.feature_coverage());
    }
  }
  return results;
}

}  // namespace data_validation
}  // namespace tensorflow
