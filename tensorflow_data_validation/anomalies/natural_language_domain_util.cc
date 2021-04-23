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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
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
using ::tensorflow::metadata::v0::SequenceValueConstraints;

void VerifyCoverageConstraints(const NaturalLanguageStatistics& nl_stats,
                               NaturalLanguageDomain* nl_domain,
                               std::vector<Description>* result) {
  if (nl_domain->coverage().has_min_coverage() &&
      (nl_domain->coverage().min_coverage() > nl_stats.feature_coverage())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo::FEATURE_COVERAGE_TOO_LOW,
         "Feature coverage is too low.",
         absl::StrCat("Fraction of tokens in the vocabulary: ",
                      nl_stats.feature_coverage(),
                      " is lower than the threshold set in the Schema: ",
                      nl_domain->coverage().min_coverage(), ".")});
    nl_domain->mutable_coverage()->set_min_coverage(
        nl_stats.feature_coverage());
  }
  if (nl_domain->coverage().has_min_avg_token_length() &&
      (nl_stats.avg_token_length() <
       nl_domain->coverage().min_avg_token_length())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo ::
             FEATURE_COVERAGE_TOO_SHORT_AVG_TOKEN_LENGTH,
         "Average token length is too short.",
         absl::StrCat("Average token length is: ", nl_stats.avg_token_length(),
                      " which is lower than the threshold set in the Schema:",
                      " ", nl_domain->coverage().min_avg_token_length(), ".")});
    nl_domain->mutable_coverage()->set_min_avg_token_length(
        nl_stats.avg_token_length());
  }
}

void VerifyTokenConstraints(
    const NaturalLanguageStatistics::TokenStatistics& token_stats,
    const string& token_string, SequenceValueConstraints* constraint,
    std::vector<Description>* result) {
  if (constraint->has_min_fraction_of_sequences() &&
      (constraint->min_fraction_of_sequences() >
       token_stats.fraction_of_sequences())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo ::
             SEQUENCE_VALUE_TOO_SMALL_FRACTION,
         "Value occurs in too small a fraction of sequences.",
         absl::StrCat("Fraction of sequences with value: ", token_string,
                      " is: ", token_stats.fraction_of_sequences(),
                      " which is lower than the threshold set in the "
                      "Schema: ",
                      constraint->min_fraction_of_sequences(), ".")});
    constraint->set_min_fraction_of_sequences(
        token_stats.fraction_of_sequences());
  }

  if (constraint->has_max_fraction_of_sequences() &&
      (constraint->max_fraction_of_sequences() <
       token_stats.fraction_of_sequences())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo ::
             SEQUENCE_VALUE_TOO_LARGE_FRACTION,
         "Value occurs in too large a fraction of sequences.",
         absl::StrCat("Fraction of sequences with value: ", token_string,
                      " is: ", token_stats.fraction_of_sequences(),
                      " which is higher than the threshold set in the "
                      "Schema: ",
                      constraint->max_fraction_of_sequences(), ".")});
    constraint->set_max_fraction_of_sequences(
        token_stats.fraction_of_sequences());
  }

  if (constraint->has_min_per_sequence() &&
      (constraint->min_per_sequence() >
       token_stats.per_sequence_min_frequency())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo ::
             SEQUENCE_VALUE_TOO_FEW_OCCURRENCES,
         "Value has too few per-sequence occurrences.",
         absl::StrCat("Value: ", token_string, " occurs at least: ",
                      token_stats.per_sequence_min_frequency(),
                      " times within a sequence, which is lower than the "
                      "threshold set in the Schema: ",
                      constraint->min_per_sequence(), ".")});
    constraint->set_min_per_sequence(token_stats.per_sequence_min_frequency());
  }

  if (constraint->has_max_per_sequence() &&
      (constraint->max_per_sequence() <
       token_stats.per_sequence_max_frequency())) {
    result->push_back(
        {tensorflow::metadata::v0::AnomalyInfo ::
             SEQUENCE_VALUE_TOO_MANY_OCCURRENCES,
         "Value has too many per-sequence occurrences.",
         absl::StrCat("Value: ", token_string, " occurs at most: ",
                      token_stats.per_sequence_max_frequency(),
                      " times within a sequence, which is higher than the "
                      "threshold set in the Schema: ",
                      constraint->max_per_sequence(), ".")});
    constraint->set_max_per_sequence(token_stats.per_sequence_max_frequency());
  }
}

}  // namespace

std::vector<Description> UpdateNaturalLanguageDomain(
    const FeatureStatsView& feature_stats, Feature* feature) {
  std::vector<Description> result;

  const CustomStatistic* nl_custom_stats =
      feature_stats.GetCustomStatByName("nl_statistics");

  NaturalLanguageStatistics nl_stats;
  bool found_nl_stats = false;
  if (nl_custom_stats) {
    if (!nl_custom_stats->any().UnpackTo(&nl_stats)) {
      LOG(WARNING) << "nl_statistics for feature " << feature->name()
                   << "do not have the expected "
                   << "NaturalLanguageStatistics message format.";
      return result;
    }
    found_nl_stats = true;
  }

  static const auto& kMissingStatsDescription = *new Description{
      tensorflow::metadata::v0::AnomalyInfo::STATS_NOT_AVAILABLE,
      "Natural language stats are not computed.",
      "Constraints specified in natural language domain cannot be "
      "verified because natural language stats have not been computed."};

  NaturalLanguageDomain* nl_domain = feature->mutable_natural_language_domain();
  if ((nl_domain->coverage().has_min_coverage() ||
       nl_domain->coverage().has_min_avg_token_length() ||
       nl_domain->token_constraints_size() > 0) &&
      !found_nl_stats) {
    result.push_back(kMissingStatsDescription);
    feature->clear_natural_language_domain();
    return result;
  }

  VerifyCoverageConstraints(nl_stats, nl_domain, &result);

  std::map<absl::variant<std::string, int>,
           const NaturalLanguageStatistics::TokenStatistics&>
      token_stats_map;

  for (auto& token_stats : nl_stats.token_statistics()) {
    if (token_stats.token_case() ==
        NaturalLanguageStatistics::TokenStatistics::TokenCase::kIntToken) {
      token_stats_map.emplace(token_stats.int_token(), token_stats);
    } else if (token_stats.token_case() ==
               NaturalLanguageStatistics::TokenStatistics::TokenCase::
                   kStringToken) {
      token_stats_map.emplace(token_stats.string_token(), token_stats);
    }
  }

  for (auto& constraint : *nl_domain->mutable_token_constraints()) {
    absl::variant<string, int> constraint_name;
    std::string token_string;
    if (constraint.has_int_value()) {
      constraint_name = constraint.int_value();
      token_string = absl::StrCat(constraint.int_value());

    } else if (constraint.has_string_value()) {
      constraint_name = constraint.string_value();
      token_string = constraint.string_value();
    } else {
      continue;
    }

    auto iter = token_stats_map.find(constraint_name);
    if (iter == token_stats_map.end()) {
      result.push_back(kMissingStatsDescription);
      feature->clear_natural_language_domain();
      return result;
    }
    VerifyTokenConstraints(iter->second, token_string, &constraint, &result);
  }
  return result;
}

}  // namespace data_validation
}  // namespace tensorflow
