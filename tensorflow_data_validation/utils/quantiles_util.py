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

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import bisect
import collections

import numpy as np
import six
import tensorflow_transform as tft
from typing import Iterable, List, Union
from tensorflow_metadata.proto.v0 import statistics_pb2


class QuantilesCombiner(object):
  """Computes quantiles using a combiner function.

  This class wraps tf.transform's QuantilesCombiner.
  """

  def __init__(self, num_quantiles: int, epsilon: float,
               has_weights: bool = False):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._has_weights = has_weights
    self._quantiles_spec = tft.analyzers.QuantilesCombiner(
        num_quantiles=num_quantiles, epsilon=epsilon,
        bucket_numpy_dtype=np.float32, always_return_num_quantiles=True,
        has_weights=has_weights, include_max_and_min=True)
    # TODO(pachristopher): Consider passing an appropriate (runner-dependent)
    # tf_config, similar to TFT.
    self._quantiles_spec.initialize_local_state(tf_config=None)

  def create_accumulator(self) -> List[List[float]]:
    return self._quantiles_spec.create_accumulator()

  def add_input(
      self, summary: List[List[float]],
      input_batch: List[List[Union[int, float]]]) -> List[List[float]]:
    return self._quantiles_spec.add_input(summary, input_batch)

  def merge_accumulators(
      self, summaries: Iterable[List[List[float]]]) -> List[List[float]]:
    return self._quantiles_spec.merge_accumulators(summaries)

  def extract_output(self, summary: List[List[float]]) -> np.ndarray:
    quantiles = self._quantiles_spec.extract_output(summary)
    # The output of the combiner spec is a list containing a
    # single numpy array which contains the quantile boundaries.
    assert len(quantiles) == 1
    return quantiles[0]


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


def generate_quantiles_histogram(quantiles: np.ndarray,
                                 total_count: float,
                                 num_buckets: int
                                ) -> statistics_pb2.Histogram:
  """Generate quantiles histrogram from the quantile boundaries.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    total_count: The total number of values over which the quantiles
        are computed.
    num_buckets: The required number of buckets in the quantiles histogram.

  Returns:
    A statistics_pb2.Histogram proto.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.QUANTILES

  quantiles = list(quantiles)
  # We assume that the number of quantiles is a multiple of the required
  # number of buckets in the quantiles histogram.
  assert (len(quantiles) - 1) % num_buckets == 0

  # Sample count per bucket based on the computed quantiles.
  sample_count = float(total_count / (len(quantiles)-1))
  width = int((len(quantiles) - 1) / num_buckets)
  # Sample count per merged bucket.
  merged_bucket_sample_count = sample_count * width
  i = 0
  while i + width < len(quantiles):
    result.buckets.add(low_value=quantiles[i],
                       high_value=quantiles[i + width],
                       sample_count=merged_bucket_sample_count)
    i += width

  return result


# Named tuple with details for each bucket in a histogram.
Bucket = collections.namedtuple(
    'Bucket', ['low_value', 'high_value', 'sample_count'])


def generate_equi_width_histogram(quantiles: np.ndarray,
                                  finite_min: float,
                                  finite_max: float,
                                  total_count: float,
                                  num_buckets: int
                                 ) -> statistics_pb2.Histogram:
  """Generate equi-width histrogram from the quantile boundaries.

  Currently we construct the equi-width histogram by using the quantiles.
  Specifically, we compute a large number of quantiles and then compute
  the density for each equi-width histogram bucket by aggregating the
  densities of the smaller quantile intervals that fall within the bucket.
  This approach assumes that the number of quantiles is much higher than
  the required number of buckets in the equi-width histogram.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    finite_min: The mimimum finite value.
    finite_max: The maximum finite value.
    total_count: The total number of values over which the quantiles
        are computed.
    num_buckets: The required number of buckets in the equi-width histogram.

  Returns:
    A statistics_pb2.Histogram proto.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.STANDARD
  buckets = generate_equi_width_buckets(
      list(quantiles), finite_min, finite_max, total_count, num_buckets)
  for bucket_info in buckets:
    result.buckets.add(low_value=bucket_info.low_value,
                       high_value=bucket_info.high_value,
                       sample_count=bucket_info.sample_count)

  return result


def generate_equi_width_buckets(quantiles: List[float],
                                finite_min: float,
                                finite_max: float,
                                total_count: float,
                                num_buckets: int) -> List[Bucket]:
  """Generate buckets for equi-width histogram.

  Args:
    quantiles: A list containing the quantile boundaries.
    finite_min: The mimimum finite value.
    finite_max: The maximum finite value.
    total_count: The total number of values over which the quantiles
        are computed.
    num_buckets: The required number of buckets in the equi-width histogram.

  Returns:
    A list containing the buckets.
  """
  # We assume that the number of quantiles is much higher than
  # the required number of buckets in the equi-width histogram.
  assert len(quantiles) > num_buckets

  # If all values of a feature are equal, have only a single bucket.
  if quantiles[0] == quantiles[-1]:
    return [Bucket(quantiles[0], quantiles[-1], total_count)]

  # Find the index of the first and the last finite value. If there are only
  # -inf and +inf values, we generate two buckets (-inf, -inf) and (+inf, +inf).
  finite_min_index = np.searchsorted(quantiles, float('-inf'), side='right')
  finite_max_index = np.searchsorted(quantiles, float('inf'), side='left') - 1

  # Compute sample count associated with a quantile interval.
  sample_count = total_count / (len(quantiles) - 1)

  if finite_max_index < finite_min_index:
    return [
        # Divide the intersecting bucket (-inf, +inf) sample count equally.
        Bucket(float('-inf'), float('-inf'),
               (finite_min_index - 0.5) * sample_count),
        Bucket(float('inf'), float('inf'),
               (len(quantiles) - finite_max_index - 1.5) * sample_count),
    ]

  # Sample count to account for  (-inf, -inf) buckets.
  start_bucket_count = finite_min_index * sample_count
  # Sample count to account for (inf, inf) buckets.
  last_bucket_count = (len(quantiles) - finite_max_index - 1) * sample_count
  finite_values = quantiles[finite_min_index:finite_max_index+1]
  # Insert finite minimum and maximum if first and last finite quantiles are
  # greater or lesser than the finite mimimum and maximum respectively.
  # Note that if all values of a feature are finite, we will always have the
  # finite min and finite max as the first and last boundaries.
  if finite_min_index > 0 and finite_min < finite_values[0]:
    finite_values.insert(0, finite_min)
    # Since we are adding an extra boundary, we borrow the sample count from the
    # (-inf, -inf) buckets as the first bucket will anyhow be merged with all
    # the (-inf, -inf) buckets.
    start_bucket_count -= sample_count
  if finite_max_index < len(quantiles) - 1 and finite_max > finite_values[-1]:
    finite_values.append(finite_max)
    # Since we are adding an extra boundary, we borrow the sample count from the
    # (+inf, +inf) buckets as the last bucket will anyhow be merged with all
    # the (+inf, +inf) buckets.
    last_bucket_count -= sample_count

  # Check if the finite quantile boundaries are sorted.
  assert np.all(np.diff(finite_values) >= 0), (
      'Quantiles output not sorted %r'  % ','.join(map(str, finite_values)))

  # Construct the list of buckets from finite boundaries.
  result = _generate_equi_width_buckets_from_finite_boundaries(
      finite_values, sample_count, num_buckets)

  # If we have -inf values, update first bucket's low value (to be -inf) and
  # sample count to account for remaining (-inf, -inf) buckets.
  if finite_min_index > 0:
    result[0] = Bucket(
        float('-inf'), result[0].high_value,
        result[0].sample_count + start_bucket_count)
  # If we have +inf values, update last bucket's high value (to be +inf) and
  # sample count to account for remaining (+inf, +inf) buckets.
  if finite_max_index < len(quantiles) - 1:
    result[-1] = Bucket(
        result[-1].low_value, float('inf'),
        result[-1].sample_count + last_bucket_count)
  return result


def _generate_equi_width_buckets_from_finite_boundaries(
    quantiles: List[float], sample_count: float, num_buckets: int):
  """Generates equi width buckets from finite quantiles boundaries."""

  def _compute_count(start_index, end_index, start_pos):
    """Computes sample count of the interval."""
    # Add sample count corresponding to the number of entire quantile
    # intervals included in the current bucket.
    result = (end_index - start_index - 1) * sample_count
    if start_pos > quantiles[start_index]:
      result -= sample_count
      result += ((quantiles[start_index + 1] - start_pos) * sample_count /
                 (quantiles[start_index + 1] - quantiles[start_index]))
    return result

  min_val, max_val = quantiles[0], quantiles[-1]

  # Compute the equal-width of the buckets in the standard histogram.
  width = (max_val - min_val) / num_buckets

  # Iterate to create the first num_buckets - 1 buckets.
  bucket_boundaries = [min_val + (ix * width)
                       for ix in six.moves.range(num_buckets)]

  # Index of the current quantile being processed.
  quantile_index = 0
  # Position within the current quantile bucket being processed.
  quantile_pos = quantiles[quantile_index]

  # Assert min/max values of the bucket boundaries.
  assert np.min(bucket_boundaries) == min_val, (
      'Invalid bucket boundaries %r for quantiles output %r' %
      (','.join(map(str, bucket_boundaries)), ','.join(map(str, quantiles))))
  assert np.max(bucket_boundaries) <= max_val, (
      'Invalid bucket boundaries %r for quantiles output %r' %
      (','.join(map(str, bucket_boundaries)), ','.join(map(str, quantiles))))

  # Initialize the list of buckets.
  result = []

  for (bucket_start, bucket_end) in zip(bucket_boundaries[:-1],
                                        bucket_boundaries[1:]):
    # Initialize sample count of the current bucket.
    bucket_count = 0

    # Iterate over the quantiles to find where the current bucket ends.
    curr_index = bisect.bisect_left(quantiles, bucket_end, lo=quantile_index)

    # Handle case where we have at least one full quantile interval in the
    # current bucket.
    if curr_index > quantile_index + 1:
      # Adds count for the initial range
      # (quantile_pos, quantiles[quantile_index+1]) and remaining full quantile
      # intervals for the current bucket.
      bucket_count += _compute_count(quantile_index, curr_index, quantile_pos)
      quantile_pos = quantiles[curr_index - 1]

    # Add sample count corresponding to the partial last quantile interval.
    # We assume the samples are uniformly distributed in an interval.
    delta = ((bucket_end - quantile_pos) * sample_count /
             (quantiles[curr_index] - quantiles[curr_index - 1]))

    bucket_count += delta

    # Add the current bucket to the result.
    result.append(Bucket(bucket_start, bucket_end, bucket_count))

    quantile_pos = bucket_end
    # Update the index of the quantile to be processed for the next bucket.
    quantile_index = (
        (curr_index - 1) if quantile_pos < quantiles[curr_index] else
        curr_index)

  # Add the remaining sample count to the last bucket
  # (bucket_boundaries[-1], max_val). Add sample count for all quantile
  # intervals from quantile_index. We add the last bucket separately because
  # the bucket end boundary is inclusive for the last bucket.
  bucket_count = _compute_count(quantile_index, len(quantiles), quantile_pos)
  result.append(
      Bucket(bucket_boundaries[-1], quantiles[-1], bucket_count))
  return result
