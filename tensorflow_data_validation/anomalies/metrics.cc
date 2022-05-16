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
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::Histogram;

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

// Appends a bucket representing nans to `histogram`.
void AddNanBucket(Histogram& histogram) {
  const double nan_count = histogram.num_nan();
  Histogram::Bucket* nan_bucket = histogram.add_buckets();
  nan_bucket->set_low_value(std::numeric_limits<double>::quiet_NaN());
  nan_bucket->set_high_value(std::numeric_limits<double>::quiet_NaN());
  nan_bucket->set_sample_count(nan_count);
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

// When a histogram contains only a single value, adds that value again
// to the boundaries vector.
void AddSingleValueBoundary(const std::set<double>& histogram_boundaries,
                            const Histogram& histogram,
                            std::vector<double>& boundaries) {
  if (histogram_boundaries.size() != 1) {
    return;
  }
  double bucket_value = histogram.buckets().at(0).low_value();
  auto it =
      std::upper_bound(boundaries.begin(), boundaries.end(), bucket_value);
  boundaries.insert(it, bucket_value);
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
// histogram.
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

// Aligns histogram_1 and histogram_2 so that they have the same bucket
// boundaries. This function assumes a uniform distribution of values within a
// given bucket in the original histogram.
void AlignHistograms(Histogram& histogram_1, Histogram& histogram_2) {
  const std::set<double> histogram_1_boundaries =
      GetHistogramBoundaries(histogram_1);
  const std::set<double> histogram_2_boundaries =
      GetHistogramBoundaries(histogram_2);
  // If the histograms have the same bucket boundaries, there is no need to
  // rebucket them. Just return the original histograms.
  if (histogram_1_boundaries == histogram_2_boundaries) {
    return;
  }
  std::set<double> boundaries_set;
  std::set_union(histogram_1_boundaries.begin(), histogram_1_boundaries.end(),
                 histogram_2_boundaries.begin(), histogram_2_boundaries.end(),
                 std::inserter(boundaries_set, boundaries_set.end()));
  std::vector<double> boundaries(boundaries_set.begin(), boundaries_set.end());
  // If a histogram contains only a single value, add that value
  // to the boundaries vector so that the rebucketing will create a bucket
  // with that single value.
  AddSingleValueBoundary(histogram_1_boundaries, histogram_1, boundaries);
  AddSingleValueBoundary(histogram_2_boundaries, histogram_2, boundaries);
  RebucketHistogram(boundaries, histogram_1);
  RebucketHistogram(boundaries, histogram_2);
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
  return Status::OK();
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

std::pair<string, double> LInftyDistance(
    const std::map<string, double>& counts_a,
    const std::map<string, double>& counts_b) {
  return GetLInftyNorm(GetDifference(Normalize(counts_a), Normalize(counts_b)));
}

std::pair<string, double> LInftyDistance(const FeatureStatsView& a,
                                         const FeatureStatsView& b) {
  const std::map<string, double> prob_a =
      Normalize(a.GetStringValuesWithCounts());
  const std::map<string, double> prob_b =
      Normalize(b.GetStringValuesWithCounts());

  return GetLInftyNorm(GetDifference(prob_a, prob_b));
}

Status JensenShannonDivergence(Histogram& histogram_1, Histogram& histogram_2,
                               double& result) {
  if (histogram_1.type() !=
          Histogram::HistogramType::Histogram_HistogramType_STANDARD ||
      histogram_2.type() !=
          Histogram::HistogramType::Histogram_HistogramType_STANDARD) {
    return errors::InvalidArgument(
        "Input histograms must be of STANDARD type.");
  }
  // Generate new histograms with the same bucket boundaries.
  AlignHistograms(histogram_1, histogram_2);
  // If one or more of histograms have NaN values, add a NaN bucket.
  if (histogram_1.num_nan() > 0 || histogram_2.num_nan() > 0) {
    AddNanBucket(histogram_1);
    AddNanBucket(histogram_2);
  }
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
  return Status::OK();
}

Status JensenShannonDivergence(const FeatureStatsView& a,
                               const FeatureStatsView& b, double& result) {
  const absl::optional<Histogram> maybe_histogram_1 = a.GetStandardHistogram();
  const absl::optional<Histogram> maybe_histogram_2 = b.GetStandardHistogram();
  if (!maybe_histogram_1 || !maybe_histogram_2) {
    return tensorflow::errors::InvalidArgument(
        "Both input statistics must have a standard histogram in order to "
        "calculate the Jensen-Shannon divergence.");
  }
  Histogram histogram_1 = std::move(maybe_histogram_1.value());
  Histogram histogram_2 = std::move(maybe_histogram_2.value());
  return JensenShannonDivergence(histogram_1, histogram_2, result);
}

}  // namespace data_validation
}  // namespace tensorflow
