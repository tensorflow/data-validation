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

#include "tensorflow_data_validation/anomalies/bool_domain_util.h"

#include <set>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

constexpr char kNonBooleanValues[] = "Non-boolean values";
constexpr char InvalidBoolDomain[] = "Invalid BoolDomain";

using ::tensorflow::metadata::v0::BoolDomain;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FloatDomain;
using ::tensorflow::metadata::v0::IntDomain;
using ::tensorflow::metadata::v0::NumericStatistics;

// NOTE: GetTrueValues() intersect GetFalseValues() must be empty.
std::set<string> GetTrueValues() {
  return {"TRUE", "true", "SET", "set", "1", ""};
}
std::set<string> GetFalseValues() {
  return {"FALSE", "false", "CLEAR", "clear", "0"};
}

// Assumes that the stats type is STRING or BYTES and the
// IsBoolDomainCandidate(stats) is true.
BoolDomain BoolDomainFromStringField(const FeatureStatsView& stats) {
  BoolDomain result;
  const std::set<string> true_values = GetTrueValues();
  for (const string& label : stats.GetStringValues()) {
    if (ContainsKey(true_values, label)) {
      *result.mutable_true_value() = label;
      break;
    }
  }
  const std::set<string> false_values = GetFalseValues();
  for (const string& label : stats.GetStringValues()) {
    if (ContainsKey(false_values, label)) {
      *result.mutable_false_value() = label;
      break;
    }
  }
  return result;
}

}  // namespace

std::vector<Description> UpdateBoolDomainSelf(
    tensorflow::metadata::v0::BoolDomain* bool_domain) {
  if (bool_domain->has_true_value() && bool_domain->has_false_value() &&
      bool_domain->true_value() == bool_domain->false_value()) {
    bool_domain->clear_false_value();
    return {
        {tensorflow::metadata::v0::AnomalyInfo::INVALID_DOMAIN_SPECIFICATION,
         "Malformed BoolDomain",
         absl::StrCat("True and false value equal for BoolDomain:",
                      bool_domain->true_value(),
                      ". The suggested change is to clear the false_value, "
                      "but a domain expert should review this change.")}};
  }
  return {};
}
BoolDomain BoolDomainFromStats(const FeatureStatsView& stats) {
  switch (stats.type()) {
    case FeatureNameStatistics::BYTES:
    case FeatureNameStatistics::STRING:
      return BoolDomainFromStringField(stats);
    case FeatureNameStatistics::INT:
      DCHECK_GE(stats.num_stats().min(), 0.0)
          << "Cannot have integers less than 0.";
      DCHECK_EQ(stats.num_stats().max(), 1.0)
          << "The largest integer must be 1.";
      return BoolDomain();
    case FeatureNameStatistics::FLOAT:
      // Schema::Create(...) will never cause this code to be reached, because
      // BoolType::IsCandidate(...) will return false.
      LOG(ERROR) << "Cannot infer FLOAT as BoolType.";
      DCHECK(false);
      return BoolDomain();
    default:
      LOG(ERROR) << "Unknown type: " << stats.type();
      DCHECK(false);
      return BoolDomain();
  }
}

bool IsBoolDomainCandidate(const FeatureStatsView& feature_stats) {
  if (feature_stats.type() == FeatureNameStatistics::INT) {
    const NumericStatistics& numeric_statistics = feature_stats.num_stats();
    // Note: if the max is not set, it might look like a boolean, but its not.
    // In general, having a value of zero if it is false, and nothing if it
    // is true, seems weird.
    return numeric_statistics.min() >= 0.0 && numeric_statistics.max() == 1.0;
  }
  // This is to make sure that the rest is consistent.
  if (feature_stats.type() != FeatureNameStatistics::STRING) {
    return false;
  }

  // Can only contain one or two unique values.
  if (feature_stats.GetNumUnique() > 2) {
    return false;
  }

  const std::vector<string> tokens = feature_stats.GetStringValues();
  if (tokens.empty()) {
    return false;
  }

  // Can only have one feature that represents true,
  // and one that represents false.
  std::set<string> valid_true = GetTrueValues();
  std::set<string> valid_false = GetFalseValues();
  bool true_seen = false;
  bool false_seen = false;
  for (const string& token : tokens) {
    if (!true_seen && ContainsKey(valid_true, token)) {
      true_seen = true;
      continue;
    }
    if (!false_seen && ContainsKey(valid_false, token)) {
      false_seen = true;
      continue;
    }
    return false;
  }
  // Since there are 1 or 2 strings, and if there are 2, one is false and
  // the other is true, we are OK.
  // Note that it is possible that there is one false string and no true
  // strings.
  return true;
}

std::set<string> BoolDomainValidStrings(const BoolDomain& bool_domain) {
  std::set<string> valid;
  if (bool_domain.has_true_value()) {
    valid.insert(bool_domain.true_value());
  }
  if (bool_domain.has_false_value()) {
    valid.insert(bool_domain.false_value());
  }
  return valid;
}

string BoolDomainValidStringsDescription(const BoolDomain& bool_domain) {
  const std::set<string> valid_strings = BoolDomainValidStrings(bool_domain);
  // Special casing empty makes sure we don't write {""} when we mean an empty
  // set.
  return (valid_strings.empty())
             ? "{}"
             : absl::StrCat(
                   "{\"",
                   absl::StrJoin(BoolDomainValidStrings(bool_domain), "\", \""),
                   "\"}");
}

std::vector<Description> UpdateBoolDomain(const FeatureStatsView& feature_stats,
                                          Feature* feature) {
  std::vector<Description> descriptions;
  switch (feature_stats.type()) {
    case FeatureNameStatistics::BYTES:
      LOG(ERROR) << "Should not call UpdateBoolDomain with BYTES";
      DCHECK(false);
      return {};
    case FeatureNameStatistics::INT: {
      auto set_int_domain = [](const NumericStatistics& numeric_statistics,
                               Feature* feature) -> void {
        IntDomain* int_domain = feature->mutable_int_domain();
        int_domain->set_max(numeric_statistics.max());
        int_domain->set_min(numeric_statistics.min());
      };

      const NumericStatistics& numeric_statistics = feature_stats.num_stats();
      const BoolDomain& bool_domain = feature->bool_domain();
      if (bool_domain.has_true_value() || bool_domain.has_false_value()) {
        set_int_domain(numeric_statistics, feature);
        return {
            {tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_INVALID_CONFIG,
             InvalidBoolDomain,
             absl::StrCat("User-defined boolean values are not permitted for "
                          "features of INT type.")}};
      }

      if (numeric_statistics.min() < 0.0) {
        set_int_domain(numeric_statistics, feature);
        return {{tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_SMALL_INT,
                 kNonBooleanValues,
                 absl::StrCat("Integers (such as ",
                              absl::SixDigits(numeric_statistics.min()),
                              ") not 0 or 1: converting to an integer.")}};
      } else if (numeric_statistics.max() > 1.0) {
        set_int_domain(numeric_statistics, feature);
        return {{tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_BIG_INT,
                 kNonBooleanValues,
                 absl::StrCat("Integers (such as ",
                              absl::SixDigits(numeric_statistics.max()),
                              ") not 0 or 1: converting to an integer.")}};
      }
      return {};
    }
    case FeatureNameStatistics::FLOAT: {
      auto set_float_domain = [](const NumericStatistics& numeric_statistics,
                                 Feature* feature) -> void {
        FloatDomain* float_domain = feature->mutable_float_domain();
        float_domain->set_max(numeric_statistics.max());
        float_domain->set_min(numeric_statistics.min());
      };

      const NumericStatistics& numeric_statistics = feature_stats.num_stats();
      const BoolDomain& bool_domain = feature->bool_domain();
      if (bool_domain.has_true_value() || bool_domain.has_false_value()) {
        set_float_domain(numeric_statistics, feature);
        return {
            {tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_INVALID_CONFIG,
             InvalidBoolDomain,
             absl::StrCat(
                 "User-defined boolean values are not permitted for "
                 "features of FLOAT type: converting to float_domain.")}};
      }

      double invalid_value = 0;
      if (numeric_statistics.min() != 0.0 && numeric_statistics.min() != 1.0) {
        invalid_value = numeric_statistics.min();
      } else if (numeric_statistics.max() != 0.0 &&
                 numeric_statistics.max() != 1.0) {
        invalid_value = numeric_statistics.max();
      }

      if (invalid_value != 0) {
        set_float_domain(numeric_statistics, feature);
        return {
            {tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_UNEXPECTED_FLOAT,
             kNonBooleanValues,
             absl::StrCat("Floats (such as ", absl::SixDigits(invalid_value),
                          ") not in {0, 1}: converting to float_domain.")}};
      }

      for (const auto& histogram : numeric_statistics.histograms()) {
        // With the quantiles histogram, the boundary must be set with the
        // values in the dataset. Therefore, the lower bound should be either 0
        // or 1and so as the upper bound. Note: if the quantile histogram is not
        // computed, or if there is only one bucket in the quantile histogram
        // and the min is 0 and max is 1, invalid bool_domain values will not be
        // detected.
        if (histogram.type() != metadata::v0::Histogram::QUANTILES) {
          continue;
        }
        if (histogram.num_nan() > 0) {
          // Note: num_nam is computed only for FLOAT features.
          set_float_domain(numeric_statistics, feature);
          return {{tensorflow::metadata::v0::AnomalyInfo::
                       BOOL_TYPE_UNEXPECTED_FLOAT,
                   kNonBooleanValues,
                   absl::StrCat("Floats (such as NaN) not in {0, 1}: "
                                "converting to float_domain.")}};
        }
        for (const auto& bucket : histogram.buckets()) {
          if ((bucket.low_value() != 0.0 && bucket.low_value()  != 1.0) ||
              (bucket.high_value() != 0.0 && bucket.high_value()  != 1.0)) {
            set_float_domain(numeric_statistics, feature);
            return {{tensorflow::metadata::v0::AnomalyInfo::
                          BOOL_TYPE_UNEXPECTED_FLOAT,
                      kNonBooleanValues,
                      absl::StrCat(
                          "Float values falling between 0 and 1: converting to "
                          "float_domain.")}};
          }
        }
      }
      return {};
    }
    case FeatureNameStatistics::STRING: {
      const BoolDomain& bool_domain = feature->bool_domain();
      const std::set<string> valid_strings =
          BoolDomainValidStrings(bool_domain);
      const std::vector<string> string_values = feature_stats.GetStringValues();

      if (valid_strings.size() != 2) {
        // Note that this clears the oneof field domain_info.
        feature->clear_bool_domain();
        return {
            {tensorflow::metadata::v0::AnomalyInfo::BOOL_TYPE_INVALID_CONFIG,
             InvalidBoolDomain,
             "BoolDomain missing true and/or false values."}};
      }

      for (const string& str : string_values) {
        if (!ContainsKey(valid_strings, str)) {
          // We might be able to replace this with an enum, but since it is
          // in all likelihood an error, let's just wipe the bool_domain.
          const string valid_strings_desc =
              BoolDomainValidStringsDescription(bool_domain);
          // Note that this clears the oneof field domain_info.
          feature->clear_bool_domain();
          return {{tensorflow::metadata::v0::AnomalyInfo::
                       BOOL_TYPE_UNEXPECTED_STRING,
                   kNonBooleanValues,
                   absl::StrCat("Saw unexpected value \"", str,
                                "\" instead of ", valid_strings_desc, ".")}};
        }
      }
      return {};
    }
    default:
      LOG(ERROR) << "Should not be here with unknown type: "
                 << feature_stats.type();
      DCHECK(false);
      return {};
  }
}

}  // namespace data_validation
}  // namespace tensorflow
