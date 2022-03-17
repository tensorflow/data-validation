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

#include "tensorflow_data_validation/anomalies/string_domain_util.h"

#include <math.h>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
using ::tensorflow::metadata::v0::StringDomain;
using ::tensorflow::strings::Printf;

std::set<string> GetStringDomainValues(const StringDomain& string_domain) {
  std::set<string> result;
  for (const string& value : string_domain.value()) {
    result.insert(value);
  }
  return result;
}

std::map<string, double> StringDomainGetMissing(
    const FeatureStatsView& stats, const StringDomain& string_domain) {
  // Missing values and their frequencies.
  const std::set<string> valid = GetStringDomainValues(string_domain);
  std::map<string, double> missing;
  // Iterate over values in <stats> and mark those that are missing.
  for (const auto& p : stats.GetStringValuesWithCounts()) {
    const string& value = p.first;
    if (!ContainsKey(valid, value)) {
      missing.insert(p);
    }
  }
  return missing;
}

void StringDomainAddMissing(const std::map<string, double>& missing,
                            StringDomain* string_domain) {
  for (const auto& p : missing) {
    const string& missing_token = p.first;
    *string_domain->add_value() = missing_token;
  }
}

// Returns a string representation of <count>/<total> as a percentage. If
// the ratio is less than 1% then the string "<1%" is returned, otherwise it
// returns the string "~x%" where x is the floor of the ratio.
// If total is undefined, return "?".
string PercentageAsString(double count, absl::optional<double> total) {
  if ((total == absl::nullopt) || (*total == 0.0)) {
    return "?";
  }
  double percent = 100 * count / *total;
  if (percent < 1.0) {
    return "<1%";
  } else {
    return Printf("~%d%%", static_cast<int>(floor(percent)));
  }
}

}  // namespace

bool IsSimilarStringDomain(const StringDomain& a, const StringDomain& b,
                           const EnumsSimilarConfig& config) {
  // Check the overlap between the valid values in the two enums.
  int overlap = 0;

  const std::set<string> set_a = GetStringDomainValues(a);
  const std::set<string> set_b = GetStringDomainValues(b);

  for (const auto& value : set_b) {
    if (ContainsKey(set_a, value)) {
      ++overlap;
    }
  }
  const int count_a = set_a.size();
  const int count_b = set_b.size();
  double jaccard_similarity = static_cast<double>(overlap) /
                              static_cast<double>(count_a + count_b - overlap);
  // For smaller enums, it has to be a perfect match.
  return ((jaccard_similarity > config.min_jaccard_similarity() &&
           count_a > config.min_count() && count_b > config.min_count()) ||
          jaccard_similarity == 1.0);
}

bool IsStringDomainCandidate(const FeatureStatsView& feature_stats,
                             const int enum_threshold) {
  // GetStringValues() creates a map of all the tokens in feature_stats to their
  // frequency. Theoretically, we could do this once, but we sacrifice speed for
  // simplicity.
  if (feature_stats.HasInvalidUTF8Strings()) {
    return false;
  }
  const std::vector<string> values = feature_stats.GetStringValues();
  return values.size() <= enum_threshold && !values.empty();
}

std::vector<Description> UpdateStringDomainSelf(
    tensorflow::metadata::v0::StringDomain* string_domain) {
  std::set<string> seen_so_far;

  std::vector<string> repeats;
  ::tensorflow::protobuf::RepeatedPtrField<string>* values =
      string_domain->mutable_value();
  for (auto iter = values->begin(); iter != values->end();) {
    if (!seen_so_far.insert(*iter).second) {
      repeats.push_back(*iter);
      // This effectively "increments" the iterator.
      iter = values->erase(iter);
    } else {
      ++iter;
    }
  }
  if (repeats.empty()) {
    return {};
  }
  return {{tensorflow::metadata::v0::AnomalyInfo::INVALID_DOMAIN_SPECIFICATION,
           "Malformed StringDomain",
           absl::StrCat("Repeated values in StringDomain:",
                        absl::StrJoin(repeats, ", "))}};
}

UpdateSummary UpdateStringDomain(const Schema::Updater& updater,
                                 const FeatureStatsView& stats,
                                 double max_off_domain,
                                 StringDomain* string_domain) {
  UpdateSummary summary;
  if (stats.HasInvalidUTF8Strings()) {
    summary.descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::ENUM_TYPE_INVALID_UTF8,
         "Invalid UTF8 strings",
         "Found strings that were not valid UTF8 strings."});
    summary.clear_field = true;
    return summary;
  }
  const std::map<string, double> missing =
      StringDomainGetMissing(stats, *string_domain);
  // Total number of values in the dataset that do not appear in the schema.
  const double missing_count = absl::c_accumulate(
      missing, /*init=*/0.0,
      [](double count, const std::pair<const string, double>& p) -> double {
        return count + p.second;
      });
  const double total_value_count = stats.GetTotalValueCountInExamples();
  if ((missing_count / total_value_count) > max_off_domain ||
      (max_off_domain == 0 && !missing.empty())) {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::
            ENUM_TYPE_UNEXPECTED_STRING_VALUES,
        "Unexpected string values",
        absl::StrCat(
            "Examples contain values missing from the schema: ",
            absl::StrJoin(
                missing, ", ",
                [total_value_count](
                    string* out,
                    const std::pair<string, int64>& value_and_freq) {
                  absl::StrAppend(
                      out,
                      Printf(
                          "%s (%s)",
                          absl::Utf8SafeCEscape(value_and_freq.first).c_str(),
                          PercentageAsString(value_and_freq.second,
                                             total_value_count)
                              .c_str()));
                }),
            ". ")};
    summary.descriptions.push_back(description);
    StringDomainAddMissing(missing, string_domain);
  }
  const int domain_size = string_domain->value().size();
  if (updater.string_domain_too_big(domain_size)) {
    summary.clear_field = true;

    summary.descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_DOMAIN_SPECIFICATION,
         "String domain has too many values",
         Printf("String domain has too many values (%d).", domain_size)});
  }
  return summary;
}

}  // namespace data_validation
}  // namespace tensorflow
