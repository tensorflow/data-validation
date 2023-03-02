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

#include "tensorflow_data_validation/anomalies/metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::Histogram;
using ::tensorflow::metadata::v0::HistogramSelection;

// Gets the L-infty norm of a vector, represented as a map.
// This is the largest absolute value of any value.
// For convenience, the associated key is also returned.
std::pair<string, double> GetLInftyNorm(const std::map<string, double>& vec) {
  std::pair<string, double> best_so_far;
  for (const auto& pair : vec) {
    const string& key = pair.first;
    const double value = std::abs(pair.second);
    if (value >= best_so_far.second) {
      best_so_far = {key, value};
    }
  }
  return best_so_far;
}

// Returns a set of all of the bucket boundaries in the input histogram.
std::set<double> GetHistogramBoundaries(const Histogram& histogram) {
  std::set<double> boundaries;
  for (const auto& bucket : histogram.buckets()) {
    boundaries.insert(bucket.low_value());
    boundaries.insert(bucket.high_value());
  }
  return boundaries;
}

// Adds new buckets to the histogram that are specified by the
// bucket_boundaries.
// To calculate the sample counts for each new bucket, this function assumes
// a uniform distribution of the total_sample_count values across the
// total_range_covered.
void AddBucketsToHistogram(std::vector<double> bucket_boundaries,
                           double total_sample_count,
                           double total_range_covered, Histogram& histogram) {
  int num_new_buckets = bucket_boundaries.size() - 1;
  for (int i = 0; i < num_new_buckets; ++i) {
    Histogram::Bucket* new_bucket = histogram.add_buckets();
    new_bucket->set_low_value(bucket_boundaries[i]);
    new_bucket->set_high_value(bucket_boundaries[i + 1]);
    const double new_bucket_sample_count =
        ((bucket_boundaries[i + 1] - bucket_boundaries[i]) /
         total_range_covered) *
        total_sample_count;
    new_bucket->set_sample_count(new_bucket_sample_count);
  }
}

// Rebuckets `histogram` so that its value counts are redistributed into new
// buckets that are defined by the specified boundaries. This function assumes a
// uniform distribution of values within a given bucket in the original
// histogram, and assumes that each distinct low or high value in histogram
// appears in boundaries.
// TODO(zwestrick): This function works because the bucket boundaries contain
// all of the histogram boundaries, so there is never a partial overlap between
// new and old buckets. Concretely, low_value > boundaries[index] implies that
// low_value >= boundaries[index + 1]. We may want to replace this code to be
// robust to changes in how bucket boundaries are determined, but this is not
// currently a problem.
void RebucketHistogram(
    const std::vector<double>& boundaries, Histogram& histogram) {
  Histogram rebucketed_histogram;
  const int max_boundaries_index = boundaries.size() - 1;
  int index = 0;
  for (const auto& bucket : histogram.buckets()) {
    const double low_value = bucket.low_value();
    const double high_value = bucket.high_value();
    const double sample_count = bucket.sample_count();
    // Fill in empty buckets up to the first bucket in the existing histogram.
    while (low_value > boundaries[index]) {
      CHECK_LE(index + 1, max_boundaries_index);
      Histogram::Bucket* new_bucket = rebucketed_histogram.add_buckets();
      new_bucket->set_low_value(boundaries[index]);
      ++index;
      new_bucket->set_high_value(boundaries[index]);
      new_bucket->set_sample_count(0);
    }
    if ((low_value == high_value) && (low_value == boundaries[index])) {
      Histogram::Bucket* new_bucket = rebucketed_histogram.add_buckets();
      new_bucket->set_low_value(boundaries[index]);
      ++index;
      new_bucket->set_high_value(boundaries[index]);
      new_bucket->set_sample_count(sample_count);
      continue;
    }
    // Once the current position in boundaries is covered by a bucket in the
    // existing histogram, divide that bucket up based on the boundaries that it
    // covers.
    std::vector<double> covered_boundaries;
    while (high_value > boundaries[index]) {
      covered_boundaries.push_back(boundaries[index]);
      ++index;
    }
    // Add the last boundary to covered_boundaries, so it can be used as the
    // high_value for the last bucket.
    covered_boundaries.push_back(boundaries[index]);
    // Divide the current bucket into new buckets defined by covered_boundaries.
    if (covered_boundaries.size() > 1) {
      AddBucketsToHistogram(covered_boundaries, sample_count,
                            high_value - low_value, rebucketed_histogram);
    }
  }
  // Add additional buckets if there are still boundaries for which new
  // buckets have not already been added.
  for (int i = index; i < max_boundaries_index; ++i) {
    Histogram::Bucket* new_bucket = rebucketed_histogram.add_buckets();
    new_bucket->set_low_value(boundaries[i]);
    new_bucket->set_high_value(boundaries[i + 1]);
    new_bucket->set_sample_count(0);
  }
  histogram = std::move(rebucketed_histogram);
}
// Removes point buckets and buckets with infinite bounds from histogram.
// Returns a stripped histogram, a map from point boundary to mass, and the
// total mass across buckets with infinite bounds.
std::tuple<Histogram, std::map<double, double>, double>
StripPointAndInfiniteBuckets(const Histogram& histogram) {
  Histogram result;
  result.set_num_nan(histogram.num_nan());
  std::map<double, double> point_buckets;
  double infinite_mass = 0;
  for (const auto& bucket : histogram.buckets()) {
    if (!std::isfinite(bucket.low_value()) ||
        !std::isfinite(bucket.high_value())) {
      infinite_mass += bucket.sample_count();
    } else if (bucket.low_value() == bucket.high_value()) {
      point_buckets[bucket.low_value()] += bucket.sample_count();
    } else {
      auto* new_bucket = result.add_buckets();
      new_bucket->set_low_value(bucket.low_value());
      new_bucket->set_high_value(bucket.high_value());
      new_bucket->set_sample_count(bucket.sample_count());
    }
  }
  return {result, point_buckets, infinite_mass};
}

// Adds buckets corresponding to zero width point masses from the original
// histograms.
void AddPointMasses(std::map<double, double> histogram_1_point,
                    std::map<double, double> histogram_2_point,
                    Histogram& histogram1, Histogram& histogram2) {
  std::set<double> point_boundaries;
  for (const auto& point_map : {histogram_1_point, histogram_2_point}) {
    for (const auto& kv : point_map) {
      point_boundaries.insert(kv.first);
    }
  }
  auto value_or_zero =
      [](const std::map<double, double>& point_count,
         double boundary) {
        auto val = point_count.find(boundary);
        if (val != point_count.end()) {
          return val->second;
        }
        return 0.0;
      };
  for (const auto& boundary : point_boundaries) {
    histogram1.add_buckets()->set_sample_count(
        value_or_zero(histogram_1_point, boundary));
    histogram2.add_buckets()->set_sample_count(
        value_or_zero(histogram_2_point, boundary));
  }
}

// Aligns histogram_1 and histogram_2 so that they have the same bucket
// boundaries. This function assumes a uniform distribution of values within a
// given bucket in the original histogram.
void AlignHistograms(Histogram& histogram_1, Histogram& histogram_2) {
  // TODO(zwestrick): Figure out why structured bindings breaks windows kokoro
  // tests herre.
  auto result1 =
      StripPointAndInfiniteBuckets(histogram_1);
  auto histogram_1_stripped = std::get<0>(result1);
  auto histogram_1_point = std::get<1>(result1);
  auto histogram_1_inf = std::get<2>(result1);

  auto result2 =
      StripPointAndInfiniteBuckets(histogram_2);
  auto histogram_2_stripped = std::get<0>(result2);
  auto histogram_2_point = std::get<1>(result2);
  auto histogram_2_inf = std::get<2>(result2);
  histogram_1 = std::move(histogram_1_stripped);
  histogram_2 = std::move(histogram_2_stripped);

  const std::set<double> histogram_1_boundaries =
      GetHistogramBoundaries(histogram_1);
  const std::set<double> histogram_2_boundaries =
      GetHistogramBoundaries(histogram_2);
  // If the histograms have the same bucket boundaries, there is no need to
  // rebucket them.
  if (histogram_1_boundaries != histogram_2_boundaries) {
    std::set<double> boundaries_set;
    std::set_union(histogram_1_boundaries.begin(), histogram_1_boundaries.end(),
                   histogram_2_boundaries.begin(), histogram_2_boundaries.end(),
                   std::inserter(boundaries_set, boundaries_set.end()));
    std::vector<double> boundaries(boundaries_set.begin(),
                                   boundaries_set.end());
    RebucketHistogram(boundaries, histogram_1);
    RebucketHistogram(boundaries, histogram_2);
  }
  // Now we've rebucketed the histograms excluding point masses and infinite
  // buckets.
  // First add back in the point masses.
  AddPointMasses(histogram_1_point, histogram_2_point, histogram_1,
                 histogram_2);

  // Now add back infinite values as mismatching additional buckets.
  if (histogram_1_inf != 0) {
    histogram_1.add_buckets()->set_sample_count(histogram_1_inf);
    histogram_2.add_buckets()->set_sample_count(0);
  }
  if (histogram_2_inf != 0) {
    histogram_1.add_buckets()->set_sample_count(0);
    histogram_2.add_buckets()->set_sample_count(histogram_2_inf);
  }
  // If one or more of histograms have NaN values, add a NaN bucket.
  if (histogram_1.num_nan() > 0 || histogram_2.num_nan() > 0) {
    histogram_1.add_buckets()->set_sample_count(0);
    histogram_1.add_buckets()->set_sample_count(histogram_1.num_nan());
    histogram_2.add_buckets()->set_sample_count(histogram_2.num_nan());
    histogram_2.add_buckets()->set_sample_count(0);
  }
}

// Normalizes `histogram` so that the sum of all sample counts in the histogram
// equals 1.
Status NormalizeHistogram(Histogram& histogram) {
  Histogram normalized_histogram;
  double total_sample_count = 0;
  for (const auto& bucket : histogram.buckets()) {
    total_sample_count += bucket.sample_count();
  }
  if (total_sample_count == 0) {
    return tensorflow::errors::InvalidArgument(
        "Unable to normalize an empty histogram");
  }
  for (const auto& bucket : histogram.buckets()) {
    Histogram::Bucket* new_bucket = normalized_histogram.add_buckets();
    new_bucket->set_low_value(bucket.low_value());
    new_bucket->set_high_value(bucket.high_value());
    new_bucket->set_sample_count(bucket.sample_count() / total_sample_count);
  }
  histogram = std::move(normalized_histogram);
  return Status();
}

// Returns an approximate Kullback-Leibler divergence
// (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measuring
// how histogram_2 differs from histogram_1.
double KullbackLeiblerDivergence(const Histogram& histogram_1,
                                 const Histogram& histogram_2) {
  double result = 0;
  CHECK_EQ(histogram_1.buckets_size(), histogram_2.buckets_size());
  for (int i = 0; i < histogram_1.buckets_size(); ++i) {
    double p = histogram_1.buckets().at(i).sample_count();
    double q = histogram_2.buckets().at(i).sample_count();
    if (p > 0 && q > 0) {
      result += p * std::log2(p / q);
    }
  }
  return result;
}

}  // namespace

std::pair<string, double> MaxNormalizedDifference(
    const std::map<string, double>& counts_a,
    const std::map<string, double>& counts_b,
    NormalizationMode normalization_mode) {
  switch (normalization_mode) {
    case NormalizationMode::kSeparateTotal:
      return GetLInftyNorm(
          GetDifference(Normalize(counts_a), Normalize(counts_b)));
    case NormalizationMode::kCombinedTotal:
      double scale = SumValues(counts_a) + SumValues(counts_b);
      return GetLInftyNorm(
          GetDifference(ScaleBy(counts_a, scale), ScaleBy(counts_b, scale)));
      return GetLInftyNorm(GetDifference(counts_a, counts_b));
  }
}

std::pair<string, double> LInftyDistance(const FeatureStatsView& a,
                                         const FeatureStatsView& b) {
  return MaxNormalizedDifference(a.GetStringValuesWithCounts(),
                                 b.GetStringValuesWithCounts(),
                                 NormalizationMode::kSeparateTotal);
}

std::pair<string, double> NormalizedAbsoluteDifference(
    const FeatureStatsView& a, const FeatureStatsView& b)  {
  return MaxNormalizedDifference(a.GetStringValuesWithCounts(),
                                 b.GetStringValuesWithCounts(),
                                 NormalizationMode::kCombinedTotal);
}

Status JensenShannonDivergence(Histogram& histogram_1, Histogram& histogram_2,
                               double& result) {
  if (histogram_1.type() !=histogram_2.type()) {
    return errors::InvalidArgument(
        "Input histograms must have same type.");
  }
  // Generate new histograms with the same bucket boundaries.
  AlignHistograms(histogram_1, histogram_2);
  TF_RETURN_IF_ERROR(NormalizeHistogram(histogram_1));
  TF_RETURN_IF_ERROR(NormalizeHistogram(histogram_2));

  // JSD(P||Q) = (D(P||M) + D(Q||M))/2
  // where D(P||Q) is the Kullback-Leibler divergence, and M = (P + Q)/2.
  Histogram average_distribution_histogram;
  CHECK_EQ(histogram_1.buckets_size(), histogram_2.buckets_size());
  Histogram::Bucket* new_bucket;
  for (int i = 0; i < histogram_1.buckets_size(); ++i) {
    new_bucket = average_distribution_histogram.add_buckets();
    new_bucket->set_low_value(histogram_1.buckets().at(i).low_value());
    new_bucket->set_high_value(histogram_1.buckets().at(i).high_value());
    new_bucket->set_sample_count((histogram_1.buckets().at(i).sample_count() +
                                  histogram_2.buckets().at(i).sample_count()) /
                                 2);
  }
  result =
      ((KullbackLeiblerDivergence(histogram_1, average_distribution_histogram) +
        KullbackLeiblerDivergence(histogram_2,
                                  average_distribution_histogram)) /
       2);
  return Status();
}


Status JensenShannonDivergence(const std::map<string, double>& map_1,
                               const std::map<string, double>& map_2,
                               double& result) {
  if (map_1.empty() || map_2.empty()) {
    return errors::InvalidArgument("Input maps must not be empty.");
  }
  double a_sum = 0, b_sum = 0;
  std::map<string, std::pair<double, double>> js_values;

  // Map string to value counts from each dataset and calculate the total
  // count for each dataset which will be used to calculate the
  // probability/distribution of the values.
  for (const auto& ele : map_1) {
    if (ele.second <= 0){
      return errors::InvalidArgument("Sample count is a non-positive value.");
    }
    js_values[ele.first].first += ele.second;
    a_sum += ele.second;
  }
  for (const auto& ele : map_2) {
    if (ele.second <= 0){
      return errors::InvalidArgument("Sample count is a non-positive value.");
    }
    js_values[ele.first].second += ele.second;
    b_sum += ele.second;
  }
  // Calculate JSD(P||Q) = (D(P||M) + D(Q||M))/2
  double kl_sum = 0;
  double m = 0;
  double a_ele_prob = 0, b_ele_prob = 0;
  for (const auto& ele : js_values) {
    a_ele_prob = ele.second.first / a_sum;
    b_ele_prob = ele.second.second / b_sum;
    // M = (P + Q)/2.
    m = (a_ele_prob + b_ele_prob) / 2;
    if (ele.second.first != 0) {
      // D(P||M)
      kl_sum += a_ele_prob * std::log2(a_ele_prob / m);
    }
    if (ele.second.second != 0) {
      // D(Q||M)
      kl_sum += b_ele_prob * std::log2(b_ele_prob / m);
    }
  }
  result = kl_sum/2;

  return Status();
}

Status JensenShannonDivergence(const FeatureStatsView& a,
                               const FeatureStatsView& b,
                               const HistogramSelection& source,
                               double& result) {
  std::map<string, double> mapping_1 = a.GetStringValuesWithCounts();
  std::map<string, double> mapping_2 = b.GetStringValuesWithCounts();
  if (!mapping_1.empty() && !mapping_2.empty()) {
    return JensenShannonDivergence(mapping_1, mapping_2, result);
  }
  const absl::optional<Histogram> maybe_histogram_1 =
      a.GetHistogramType(source);
  const absl::optional<Histogram> maybe_histogram_2 =
      b.GetHistogramType(source);
  if (maybe_histogram_1 && maybe_histogram_2) {
    Histogram histogram_1 = std::move(maybe_histogram_1.value());
    Histogram histogram_2 = std::move(maybe_histogram_2.value());
    return JensenShannonDivergence(histogram_1, histogram_2, result);
  }

  if ((mapping_1.empty() != mapping_2.empty()) &&
      (!maybe_histogram_1 != !maybe_histogram_2)){
    return tensorflow::errors::InvalidArgument(
      "Input statistics must be either both numeric or both string in order to "
      "calculate the Jensen-Shannon divergence.");
  }
  return tensorflow::errors::InvalidArgument(
    "One or more feature missing data.");
}

}  // namespace data_validation
}  // namespace tensorflow
