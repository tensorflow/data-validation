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
from tensorflow_transform import analyzers
from tensorflow_data_validation.types_compat import List, Union
from tensorflow_metadata.proto.v0 import statistics_pb2


class QuantilesCombiner(object):
  """Computes quantiles using a combiner function.

  This class wraps tf.transform's QuantilesCombiner.
  """

  def __init__(self, num_quantiles, epsilon,
               has_weights = False):
    self._num_quantiles = num_quantiles
    self._epsilon = epsilon
    self._has_weights = has_weights
    self._quantiles_spec = analyzers.QuantilesCombiner(
        num_quantiles=num_quantiles, epsilon=epsilon,
        bucket_numpy_dtype=np.float32, always_return_num_quantiles=True,
        has_weights=has_weights)
    # Initializes non-pickleable local state of the combiner spec.
    self._quantiles_spec.initialize_local_state()

  def __reduce__(self):
    return QuantilesCombiner, (self._num_quantiles, self._epsilon,
                               self._has_weights)

  def create_accumulator(self):
    return self._quantiles_spec.create_accumulator()

  def add_input(self, summary,
                input_batch):
    return self._quantiles_spec.add_input(summary, input_batch)

  def merge_accumulators(self, summaries):
    return self._quantiles_spec.merge_accumulators(summaries)

  def extract_output(self, summary):
    quantiles = self._quantiles_spec.extract_output(summary)
    # The output of the combiner spec is a list containing a
    # single numpy array which contains the quantile boundaries.
    assert len(quantiles) == 1
    return quantiles[0]


def find_median(quantiles):
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


def generate_quantiles_histogram(quantiles,
                                 min_val,
                                 max_val,
                                 total_count,
                                 num_buckets
                                ):
  """Generate quantiles histrogram from the quantile boundaries.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    min_val: The minimum value among all values over which the quantiles
        are computed.
    max_val: The maximum value among all values over which the quantiles
        are computed.
    total_count: The total number of values over which the quantiles
        are computed.
    num_buckets: The required number of buckets in the quantiles histogram.

  Returns:
    A statistics_pb2.Histogram proto.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.QUANTILES

  quantiles = list(quantiles)
  # We explicitly add the min and max to the quantiles list as the
  # quantiles combiner returns only the internal boundaries.
  quantiles.insert(0, min_val)  # Insert min_val in the beginning.
  quantiles.append(max_val)  # Append max_val to the end.

  # We assume that the number of quantiles is at least the required number of
  # buckets in the quantiles histogram.
  assert len(quantiles) - 1 >= num_buckets

  # Sample count per bucket based on the computed quantiles.
  current_sample_count = float(total_count / (len(quantiles)-1))
  # Sample count per bucket required for the quantiles histogram.
  required_sample_count = float(total_count / num_buckets)

  # Start of the current bucket.
  bucket_start = min_val
  # Sample count of the current bucket.
  running_sample_count = 0
  # Iterate to create the first num_buckets - 1 buckets.
  for i in range(len(quantiles)-1):
    if running_sample_count + current_sample_count >= required_sample_count:
      # Sample count needed for the current bucket.
      needed_sample_count = required_sample_count - running_sample_count
      # Compute width of the current bucket based on the needed sample count.
      # We assume the samples are uniformly distributed in an interval.
      width = ((quantiles[i+1] - quantiles[i]) *
               needed_sample_count / current_sample_count)

      result.buckets.add(low_value=bucket_start,
                         high_value=quantiles[i]+width,
                         sample_count=required_sample_count)

      # Add any carried over sample count for the next bucket.
      running_sample_count = current_sample_count - needed_sample_count
      # Fix the start of the next bucket.
      bucket_start = quantiles[i] + width

      if len(result.buckets) == num_buckets-1:
        break
    else:
      running_sample_count += current_sample_count

  # Add the last bucket.
  result.buckets.add(low_value=bucket_start, high_value=max_val,
                     sample_count=required_sample_count)

  return result


# Named tuple with details for each bucket in a histogram.
Bucket = collections.namedtuple(
    'Bucket', ['low_value', 'high_value', 'sample_count'])


def generate_equi_width_histogram(quantiles,
                                  min_val,
                                  max_val,
                                  total_count,
                                  num_buckets
                                 ):
  """Generate equi-width histrogram from the quantile boundaries.

  Currently we construct the equi-width histogram by using the quantiles.
  Specifically, we compute a large number of quantiles and then compute
  the density for each equi-width histogram bucket by aggregating the
  densities of the smaller quantile intervals that fall within the bucket.
  This approach assumes that the number of quantiles is much higher than
  the required number of buckets in the equi-width histogram.

  Args:
    quantiles: A numpy array containing the quantile boundaries.
    min_val: The minimum value among all values over which the quantiles
        are computed.
    max_val: The maximum value among all values over which the quantiles
        are computed.
    total_count: The total number of values over which the quantiles
        are computed.
    num_buckets: The required number of buckets in the equi-width histogram.

  Returns:
    A statistics_pb2.Histogram proto.
  """
  result = statistics_pb2.Histogram()
  result.type = statistics_pb2.Histogram.STANDARD
  buckets = generate_equi_width_buckets(
      list(quantiles), min_val, max_val, total_count, num_buckets)
  for bucket_info in buckets:
    result.buckets.add(low_value=bucket_info.low_value,
                       high_value=bucket_info.high_value,
                       sample_count=bucket_info.sample_count)

  return result


def generate_equi_width_buckets(quantiles,
                                min_val,
                                max_val,
                                total_count,
                                num_buckets):
  """Generate buckets for equi-width histogram.

  Args:
    quantiles: A list containing the quantile boundaries.
    min_val: The minimum value among all values over which the quantiles
        are computed.
    max_val: The maximum value among all values over which the quantiles
        are computed.
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
  if min_val == max_val:
    return [Bucket(min_val, max_val, total_count)]

  # We explicitly add the min and max to the quantiles list as the
  # quantiles combiner returns only the internal boundaries.
  quantiles.insert(0, min_val)  # Insert min_val in the beginning.
  quantiles.append(max_val)  # Append max_val to the end.

  # Initialize the list of buckets.
  result = []

  # Compute the equal-width of the buckets in the standard histogram.
  width = (max_val - min_val) / num_buckets

  # Compute sample count associated with a quantile interval.
  sample_count = total_count / (len(quantiles) - 1)

  # Index of the current quantile being processed.
  quantile_index = 0
  # Sample count carried over in case a bucket ends between two quantiles.
  carry_over = 0

  # Iterate to create the first num_buckets - 1 buckets.
  bucket_boundaries = [min_val + (ix * width) for ix in range(num_buckets)]
  for (bucket_start, bucket_end) in zip(bucket_boundaries[:-1],
                                        bucket_boundaries[1:]):
    # Add carried over sample count to the current bucket.
    bucket_count, carry_over = carry_over, 0

    # Iterate over the quantiles to find where the current bucket ends.
    curr_index = bisect.bisect_left(quantiles, bucket_end, lo=quantile_index)

    # Add sample count corresponding to the number of entire quantile
    # intervals included in the current bucket.
    bucket_count += (curr_index - quantile_index - 1) * sample_count

    # Add sample count corresponding to the partial last quantile interval.
    # We assume the samples are uniformly distributed in an interval.
    delta = ((bucket_end - quantiles[curr_index - 1]) * sample_count /
             (quantiles[curr_index] - quantiles[curr_index - 1]))
    bucket_count += delta

    # Update carried over sample count for the next bucket.
    carry_over = sample_count - delta

    # Update the index of the quantile to be processed for the next bucket.
    quantile_index = curr_index

    # Add the current bucket to the result.
    result.append(Bucket(bucket_start, bucket_end, bucket_count))

  # Add the remaining sample count to the last bucket
  # (bucket_boundaries[-1], max_val). Add sample count for all quantile
  # intervals from quantile_index. We add the last bucket separately because
  # the bucket end boundary is inclusive for the last bucket.
  bucket_count = (carry_over +
                  (len(quantiles) - quantile_index - 1) * sample_count)
  result.append(Bucket(bucket_boundaries[-1], max_val, bucket_count))

  return result
