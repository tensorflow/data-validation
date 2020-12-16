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

#include "tensorflow_data_validation/anomalies/float_domain_util.h"

#include <cmath>
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
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

constexpr char kOutOfRangeValues[] = "Out-of-range values";
constexpr char kInvalidValues[] = "Invalid values";

using ::absl::get_if;
using ::absl::holds_alternative;
using ::absl::optional;
using ::absl::variant;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FloatDomain;

// A FloatIntervalResult where a byte is not a float.
typedef string ExampleStringNotFloat;

// An interval of floats.
struct FloatInterval {
  // Min and max values of the interval.
  float min, max;
};

// See GetFloatInterval
using FloatIntervalResult =
    absl::optional<variant<FloatInterval, ExampleStringNotFloat>>;

// Determines the range of floats represented by the feature_stats, whether
// the data is floats or strings.
// Returns nullopt if there is no data in the field or it is INT.
// Returns ExampleStringNotFloat if there is at least one string that does not
// represent a float.
// Otherwise, returns the interval.
FloatIntervalResult GetFloatInterval(const FeatureStatsView& feature_stats) {
  switch (feature_stats.type()) {
    case FeatureNameStatistics::FLOAT:
      return FloatInterval{static_cast<float>(feature_stats.num_stats().min()),
                           static_cast<float>(feature_stats.num_stats().max())};
    case FeatureNameStatistics::BYTES:
    case FeatureNameStatistics::STRING: {
      absl::optional<FloatInterval> interval;
      for (const string& str : feature_stats.GetStringValues()) {
        float value;
        if (!absl::SimpleAtof(str, &value)) {
          return str;
        }
        if (!interval) {
          interval = FloatInterval{value, value};
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
    case FeatureNameStatistics::INT:
      return absl::nullopt;
    default:
      LOG(FATAL) << "Unknown type: " << feature_stats.type();
  }
}

// Check if there are NaNs in a float feature. If the domain indicates that
// NaNs are disallowed, the presence of a NaN raises an anomaly.
// TODO(askerryryan): Consider merging this logic with FloatIntervalResult.
void CheckFloatNans(const FeatureStatsView& stats,
                    UpdateSummary* update_summary,
                    tensorflow::metadata::v0::FloatDomain* float_domain) {
  bool has_nans = false;
  if (!float_domain->disallow_nan()) {
    return;
  }
  switch (stats.type()) {
    case FeatureNameStatistics::FLOAT:
      for (const auto& histogram : stats.num_stats().histograms()) {
        if (histogram.num_nan() > 0) {
          has_nans = true;
          break;
        }
      }
      break;
    case FeatureNameStatistics::STRING:
      for (const string& str : stats.GetStringValues()) {
        float value;
        if (absl::SimpleAtof(str, &value) && std::isnan(value)) {
          has_nans = true;
          break;
        }
      }
      break;
    default:
      break;
  }
  if (has_nans) {
    update_summary->descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_HAS_NAN,
         kInvalidValues, absl::StrCat("Float feature has NaN values.")});
    float_domain->set_disallow_nan(false);
  }
}
}  // namespace

UpdateSummary UpdateFloatDomain(
    const FeatureStatsView& stats,
    tensorflow::metadata::v0::FloatDomain* float_domain) {
  UpdateSummary update_summary;

  CheckFloatNans(stats, &update_summary, float_domain);

  const FloatIntervalResult result = GetFloatInterval(stats);
  if (result) {
    const variant<FloatInterval, ExampleStringNotFloat> actual_result = *result;
    if (holds_alternative<ExampleStringNotFloat>(actual_result)) {
      update_summary.descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_STRING_NOT_FLOAT,
           kInvalidValues,
           absl::StrCat(
               "String values that were not floats were found, such as \"",
               *absl::get_if<ExampleStringNotFloat>(&actual_result), "\".")});
      update_summary.clear_field = true;
      return update_summary;
    }
    if (holds_alternative<FloatInterval>(actual_result)) {
      const FloatInterval range = *absl::get_if<FloatInterval>(&actual_result);
      if (float_domain->has_min() && range.min < float_domain->min()) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_SMALL_FLOAT,
             kOutOfRangeValues,
             absl::StrCat(
                 "Unexpectedly low values: ", absl::SixDigits(range.min), "<",
                 absl::SixDigits(float_domain->min()),
                 "(upto six significant digits)")});
        float_domain->set_min(range.min);
      }

      if (float_domain->has_max() && range.max > float_domain->max()) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_BIG_FLOAT,
             kOutOfRangeValues,
             absl::StrCat(
                 "Unexpectedly high value: ", absl::SixDigits(range.max), ">",
                 absl::SixDigits(float_domain->max()),
                 "(upto six significant digits)")});
        float_domain->set_max(range.max);
      }

      if (float_domain->disallow_inf() &&
          (std::isinf(abs(range.min)) || std::isinf(abs(range.max)))) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_HAS_INF,
             kInvalidValues, absl::StrCat("Float feature has Inf values.")});
        float_domain->set_disallow_inf(false);
      }
    }
  }
  // If no interval is found, then assume everything is OK.
  return update_summary;
}

bool IsFloatDomainCandidate(const FeatureStatsView& feature_stats) {
  // We don't set float_domain by default unless we are trying to indicate
  // that strings are actually floats.
  if (feature_stats.type() != FeatureNameStatistics::STRING ||
      feature_stats.HasInvalidUTF8Strings()) {
    return false;
  }
  const FloatIntervalResult result = GetFloatInterval(feature_stats);
  if (result) {
    // If all the examples are floats, then maybe we can make this a
    // FloatDomain.
    return holds_alternative<FloatInterval>(*result);
  }
  return false;
}

}  // namespace data_validation
}  // namespace tensorflow
