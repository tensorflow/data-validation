# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to compute quantiles."""
from typing import Tuple

import numpy as np
from tensorflow_metadata.proto.v0 import statistics_pb2


def find_median(quantiles: np.ndarray) -> float:
  """Find median from the quantile boundaries.

  Args:
    quantiles: A numpy array containing the quantile boundaries.

  Returns:
    The median.
  """
  num_quantiles = len(quantiles)
  # We assume that we have at least one quantile boundary.
  assert num_quantiles > 0

  median_index = int(num_quantiles / 2)
  if num_quantiles % 2 == 0:
    # If we have an even number of quantile boundaries, take the mean of the
    # middle boundaries to be the median.
    return (quantiles[median_index - 1] + quantiles[median_index])/2.0
  else:
    # If we have an odd number of quantile boundaries, the middle boundary is
    # the median.
    return quantiles[median_index]


def _get_bin_weights(
    boundaries: np.ndarray,
    cum_bin_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns bin weights from cumulative bin weights.

  Args:
    boundaries: A numpy array of bin boundaries. May not be unique.
    cum_bin_weights: A cumulative sum of bin weights aligned with boundaries.

  Returns:
    A tuple of numpy arrays consisting of bin lower bounds, bin upper bounds,
    and the weight falling in each bin. Weight of duplicated bins is spread
    evenly across duplicates.
  """
  cum_bin_weights = cum_bin_weights.astype(np.float64)
  low_bounds = boundaries[:-1]
  high_bounds = boundaries[1:]
  bin_counts = np.diff(cum_bin_weights)
  i = 0
  # First distribute each count across bins with the same upper bound.
  while i < low_bounds.size:
    for j in range(i + 1, low_bounds.size + 1):
      if j == low_bounds.size:
        break
      if high_bounds[i] != high_bounds[j]:
        break
    if j > i + 1:
      distributed_weight = bin_counts[i:j].sum() / (j - i)
      bin_counts[i:j] = distributed_weight
    i = j
  # Now distribute the min element count across all identical bins.
  for i in range(low_bounds.size + 1):
    if i == low_bounds.size:
      break
    if low_bounds[0] != low_bounds[i] or high_bounds[0] != high_bounds[i]:
      break
  if i > 0:
    bin_counts[0:i] += cum_bin_weights[0] / (i)
  return low_bounds, high_bounds, bin_counts


def rebin_quantiles(quantiles: np.ndarray, cumulative_counts: np.ndarray,
                    reduction_factor: int) -> Tuple[np.ndarray, np.ndarray]:
  """Reduces the number of quantiles bins by a factor."""
  x = (cumulative_counts.size - 1) / reduction_factor
  if x != np.floor(x):
    raise ValueError('Reduction factor %d must divide size %d' %
                     (reduction_factor, cumulative_counts.size - 1))

  low_val, low_count = quantiles[0], cumulative_counts[0]
  quantiles = np.concatenate([[low_val],
                              quantiles[reduction_factor::reduction_factor]])
  cumulative_counts = np.concatenate(
      [[low_count], cumulative_counts[reduction_factor::reduction_factor]])
  return quantiles, cumulative_counts


def generate_quantiles_histogram(
    quantiles: np.ndarray,
    cumulative_counts: np.ndarray) -> statistics_pb2.Histogram:
  """Generate quantiles histogram from the quantile boundaries.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    cumulative_counts: A numpy array of the same length as quantiles containing
      the cumulative quantile counts (sum of weights).

  Returns:
    A statistics_pb2.Histogram proto.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.QUANTILES
  low_bounds, high_bounds, bin_weights = _get_bin_weights(
      quantiles, cumulative_counts)
  for i in range(low_bounds.size):
    result.buckets.add(
        low_value=low_bounds[i],
        high_value=high_bounds[i],
        sample_count=bin_weights[i])
  return result


def _strip_infinities(
    quantiles: np.ndarray, cumulative_counts: np.ndarray, finite_max: float,
    num_pos_inf: float) -> Tuple[np.ndarray, np.ndarray, float]:
  """Removes buckets containing infinite bounds.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    cumulative_counts: A numpy array of the same length as quantiles containing
      the cumulative quantile counts (or cumsum of weights).
    finite_max: The maximum finite value.
    num_pos_inf: The total count of positive infinite values. May be non-
      integral if weighted.

  Returns:
    A tuple consisting of new quantiles, new cumulative counts, and the total
    count of removed buckets ending in negative infinity.

  """
  # Find the largest index containing a -inf bucket upper bound.
  neg_inf_idx = np.searchsorted(quantiles, float('-inf'), side='right')
  # First we strip negative infinities. Because quantiles represents bucket
  # right hand bounds, we can just chop off any buckets with a value of -inf.
  if neg_inf_idx:
    # Strip off negative infinities.
    # Note that the quantiles will be off by num_neg_inf, because they only
    # count finite values.
    num_neg_inf = cumulative_counts[neg_inf_idx - 1]
    cumulative_counts = cumulative_counts[neg_inf_idx:]
    quantiles = quantiles[neg_inf_idx:]
    cumulative_counts = cumulative_counts - num_neg_inf
  else:
    num_neg_inf = 0
  # Now we strip positive infinities. A bucket with a right hand bound of +inf
  # may contain some finite values, so we need to use a separately computed
  # number of positive inf values.
  if num_pos_inf:
    pos_inf_index = np.searchsorted(quantiles, float('inf'), side='left')
    # Subtract num_pos_inf from the total count to get the total count of finite
    # elements.
    finite_max_count = cumulative_counts[-1] - num_pos_inf

    # Strip off +inf
    quantiles = quantiles[:pos_inf_index]
    cumulative_counts = cumulative_counts[:pos_inf_index]

    # If a trailing bucket contained the finite max, concatenate a new bucket
    # ending in that value.
    quantiles = np.concatenate([quantiles, np.array([finite_max])])
    cumulative_counts = np.concatenate(
        [cumulative_counts, np.array([finite_max_count])])
  return quantiles, cumulative_counts, num_neg_inf


def _overlap(bucket: statistics_pb2.Histogram.Bucket, low_bound: float,
             high_bound: float, first_bucket: bool) -> Tuple[float, bool, bool]:
  """Computes overlap fraction between a histogram bucket and an interval.

  Args:
    bucket: a histogram bucket. The low_value and high_value may be negative or
      positive inf respectively.
    low_bound: A finite lower bound of a probe interval.
    high_bound: A finite upper bound of a probe interval.
    first_bucket: Indicates if this is the first interval, which may contain a
      point bucket on its left edge.

  Returns:
    A tuple consisting of the following elements:
    1) The fraction of bucket's sample count falling into a probe
    interval. Samples are assumed to be uniformly distributed within a bucket.
    Buckets with infinite bounds are treated as having full overlap with any
    intervals they overlap with.
    2) A boolean indicating whether the bucket completely precedes the probe
    interval.
    3) A boolean indicating whether the bucket completely follows the probe
    interval.
  """
  # Case 0, the bucket is a point, and is equal to an interval edge.
  # If this is the first bucket, we treat it as overlapping if it falls on the
  # left boundary.
  if first_bucket and bucket.high_value == bucket.low_value == low_bound:
    return 1.0, False, False
  # Otherwise we treat it as not overlapping.
  if not first_bucket and bucket.high_value == bucket.low_value == low_bound:
    return 0.0, True, False
  # Case 1, the bucket entirely precedes the interval.
  #            |bucket|
  #                       |   |
  if bucket.high_value < low_bound:
    return 0.0, True, False
  # Case 2, the bucket entirely follows the interval.
  #            |bucket|
  #      |   |
  if bucket.low_value > high_bound:
    return 0.0, False, True
  # Case 3: bucket is contained.
  #            |bucket|
  #        |              |
  if low_bound <= bucket.low_value and high_bound >= bucket.high_value:
    return 1.0, False, False
  # Case 4: interval overlaps bucket on the left.
  #            |bucket|
  #        |     |
  if low_bound <= bucket.low_value:
    return (high_bound - bucket.low_value) / (bucket.high_value -
                                              bucket.low_value), False, False
  # Case 5: interval overlaps bucket on the right.
  #            |bucket|
  #               |     |
  if high_bound >= bucket.high_value:
    return (bucket.high_value - low_bound) / (bucket.high_value -
                                              bucket.low_value), False, False
  # Case 6: interval falls inside of the bucket.
  #            |bucket|
  #               | |
  if low_bound > bucket.low_value and high_bound < bucket.high_value:
    return (high_bound - low_bound) / (bucket.high_value -
                                       bucket.low_value), False, False
  raise ValueError('Unable to compute overlap between (%f, %f) and %s' %
                   (low_bound, high_bound, bucket))


def generate_equi_width_histogram(
    quantiles: np.ndarray, cumulative_counts: np.ndarray, finite_min: float,
    finite_max: float, num_buckets: int,
    num_pos_inf: float) -> statistics_pb2.Histogram:
  """Generates an equal bucket width hist by combining a quantiles histogram.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    cumulative_counts: A numpy array of the same length as quantiles containing
      the cumulative quantile counts (sum of weights).
    finite_min: The mimimum finite value.
    finite_max: The maximum finite value.
    num_buckets: The required number of buckets in the equi-width histogram.
    num_pos_inf: The number of positive infinite values. May be non- integral if
      weighted.

  Returns:
    A standard histogram. Bucket counts are determined via linear interpolation.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.STANDARD
  # If there were no finite values at all, return a single bucket.
  if not np.isfinite(finite_min) and not np.isfinite(finite_max):
    result.buckets.add(
        low_value=finite_min,
        high_value=finite_max,
        sample_count=cumulative_counts[-1])
    return result

  assert np.isfinite(finite_min)
  assert np.isfinite(finite_max)
  # Verify that quantiles are sorted.
  assert np.all(quantiles[:-1] <= quantiles[1:])
  # First, strip off positive and negative infinities.
  quantiles, cumulative_counts, num_neg_inf = _strip_infinities(
      quantiles, cumulative_counts, finite_max, num_pos_inf)

  # TODO(zwestrick): Skip this and operate directly on the arrays?
  quantiles_hist = generate_quantiles_histogram(quantiles, cumulative_counts)
  if finite_min == finite_max:
    new_boundaries = np.array([finite_min, finite_max])
  else:
    new_boundaries = np.linspace(finite_min, finite_max, num_buckets + 1)
    if not np.isfinite(new_boundaries).all():
      # Something has gone wrong, probably overflow. Bail out and return an
      # empty histogram. We can't meaningfully proceed, but this may not be an
      # error.
      return result
  start_index = 0
  # If we stripped off negative infinities, add them back as a single bucket.
  if num_neg_inf:
    result.buckets.add(
        low_value=float('-inf'),
        high_value=float('-inf'),
        sample_count=num_neg_inf)
  # Now build the standard histogram by merging quantiles histogram buckets.
  for i in range(new_boundaries.size - 1):
    low_bound = new_boundaries[i]
    high_bound = new_boundaries[i + 1]
    sample_count = 0
    # Find the first bucket with nonzero overlap with the first hist.
    for current_index in range(start_index, len(quantiles_hist.buckets)):
      overlap, bucket_precedes, bucket_follows = _overlap(
          quantiles_hist.buckets[current_index],
          low_bound=low_bound,
          high_bound=high_bound,
          first_bucket=i == 0)
      if bucket_follows:
        # We're entirely after the current interval.
        # Time to bail.
        break
      if bucket_precedes:
        # The bucket we considered is totally before the current interval, so
        # we can start subsequent searches from here.
        start_index = current_index
      sample_count += overlap * quantiles_hist.buckets[
          current_index].sample_count
      current_index += 1
    result.buckets.add(
        low_value=low_bound, high_value=high_bound, sample_count=sample_count)
  # If we stripped off positive infinites, add them back as a single bucket.
  if num_pos_inf:
    result.buckets.add(
        low_value=float('inf'),
        high_value=float('inf'),
        sample_count=num_pos_inf)
  return result
