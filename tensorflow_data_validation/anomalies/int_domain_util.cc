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

#include "tensorflow_data_validation/anomalies/int_domain_util.h"

#include <limits.h>

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using ::absl::variant;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::IntDomain;

constexpr char kOutOfRangeValues[] = "Out-of-range values";
constexpr char kInvalidValues[] = "Invalid values";

// A IntIntervalResult where a byte is not a float.
typedef string ExampleStringNotInt;

// An interval of ints.
struct IntInterval {
  // Min and max values of the interval.
  int64 min, max;
};

// See GetIntInterval
using IntIntervalResult =
    absl::optional<variant<IntInterval, ExampleStringNotInt>>;

// Returns an Interval, if the values are all integers (either int64_list or a
// bytes_list where every value is a decimal representation of an integer).
// Returns UNDEFINED if the statistics are FLOAT or BYTES.
// Returns UNDEFINED if the statistics are STRING but do not represent int64
// numbers.
// Returns EMPTY if the statistics are STRING but there are no common values
//   (i.e., the statistics (not the range) are empty).
// Returns NONEMPTY otherwise, with valid min and max set in
// result.
// NOTE: if GetIntInterval returns anything but NONEMPTY, result is always
// [0,0].
IntIntervalResult GetIntInterval(const FeatureStatsView& feature_stats_view) {
  // Extract string values upfront as it can be useful for categorical INT
  // features.
  const std::vector<string> string_values =
      feature_stats_view.GetStringValues();
  switch (feature_stats_view.type()) {
    case FeatureNameStatistics::STRUCT:
      return absl::nullopt;
    case FeatureNameStatistics::FLOAT:
      return absl::nullopt;
    case FeatureNameStatistics::INT: {
      if (string_values.empty()) {
        // IntDomain is interpreted as being castable to Int64, so we validate
        // that this can be done and consider as a non-conformant IntDomain if
        // it cannot. Note: if the IntDomain has no min and max specified, this
        // will not trigger an anomaly.
        if (feature_stats_view.num_stats().min() < LLONG_MIN) {
          return std::to_string(feature_stats_view.num_stats().min());
        }
        if (feature_stats_view.num_stats().max() > LLONG_MAX) {
          return std::to_string(feature_stats_view.num_stats().max());
        }
        return IntInterval{
            static_cast<int64>(feature_stats_view.num_stats().min()),
            static_cast<int64>(feature_stats_view.num_stats().max())};
      }
      // Intentionally fall through BYTES, STRING case for categorical integer
      // features.
      ABSL_FALLTHROUGH_INTENDED;
    }
    case FeatureNameStatistics::BYTES:
    case FeatureNameStatistics::STRING: {
      absl::optional<IntInterval> interval;
      for (const string& str : string_values) {
        int64 value;
        if (!absl::SimpleAtoi(str, &value)) {
          return str;
        }
        if (!interval) {
          interval = IntInterval{value, value};
        }
        if (interval->min > value) {
          interval->min = value;
        }
        if (interval->max < value) {
          interval->max = value;
        }
      }
      if (interval) {
        return *interval;
      }
      return absl::nullopt;
    }
    default:
      LOG(FATAL) << "Unknown type: " << feature_stats_view.type();
  }
}

}  // namespace

bool IsIntDomainCandidate(const FeatureStatsView& feature_stats) {
  // We are not getting bounds here: we are just identifying that it is a string
  // encoded as an int.
  if (feature_stats.type() != FeatureNameStatistics::STRING ||
      feature_stats.HasInvalidUTF8Strings()) {
    return false;
  }

  const IntIntervalResult result = GetIntInterval(feature_stats);
  if (result) {
    return absl::holds_alternative<IntInterval>(*result);
  }
  return false;
}

UpdateSummary UpdateIntDomain(const FeatureStatsView& feature_stats,
                              tensorflow::metadata::v0::IntDomain* int_domain) {
  UpdateSummary update_summary;
  const IntIntervalResult result = GetIntInterval(feature_stats);
  if (result) {
    const variant<IntInterval, ExampleStringNotInt> actual_result = *result;
    if (absl::holds_alternative<ExampleStringNotInt>(actual_result)) {
      if (feature_stats.GetFeatureType() == metadata::v0::BYTES) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::INT_TYPE_NOT_INT_STRING,
             kInvalidValues,
             absl::StrCat(
                 "String values that were not ints were found, such as \"",
                 *absl::get_if<ExampleStringNotInt>(&actual_result), "\".")});
        update_summary.clear_field = true;
      } else if (feature_stats.GetFeatureType() == metadata::v0::INT) {
        if (int_domain->has_max() || int_domain->has_min()) {
          update_summary.descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::DOMAIN_INVALID_FOR_TYPE,
               kInvalidValues,
               absl::StrCat(
                   "Integer had values that were not valid Int64, such as \"",
                   *absl::get_if<ExampleStringNotInt>(&actual_result), "\".")});
          update_summary.clear_field = true;
        }
      } else {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::DOMAIN_INVALID_FOR_TYPE,
             kInvalidValues,
             absl::StrCat("IntDomain incompatible with feature type ",
                          feature_stats.GetFeatureType())});
        update_summary.clear_field = true;
      }
      return update_summary;
    }
    if (absl::holds_alternative<IntInterval>(actual_result)) {
      const IntInterval interval =
          *absl::get_if<IntInterval>(&actual_result);
      if (int_domain->has_min() && int_domain->min() > interval.min) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::INT_TYPE_SMALL_INT,
             kOutOfRangeValues,
             absl::StrCat("Unexpectedly small value: ", interval.min, ".")});
        int_domain->set_min(interval.min);
      }
      if (int_domain->has_max() && int_domain->max() < interval.max) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::INT_TYPE_BIG_INT,
             kOutOfRangeValues,
             absl::StrCat("Unexpectedly large value: ", interval.max, ".")});
        int_domain->set_max(interval.max);
      }

      return update_summary;
    }
  }
  return update_summary;
}

}  // namespace data_validation
}  // namespace tensorflow
