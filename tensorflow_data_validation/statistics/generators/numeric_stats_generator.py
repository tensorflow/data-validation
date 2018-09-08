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
"""Module that computes statistics for features of numeric type.

Specifically, we compute the following statistics across all examples
for each numeric feature,
  - Mean of the values for this feature.
  - Standard deviation of the values for this feature.
  - Number of values that equal zero for this feature.
  - Minimum value for this feature.
  - Maximum value for this feature.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import math
import sys

import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Dict, List, Optional, Union

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialNumericStats(object):
  """Holds partial statistics needed to compute the numeric statistics
  for a single feature."""

  def __init__(self):
    # Explicitly make the sum and the sum of squares to be float in order to
    # avoid numpy overflow warnings.
    # The sum of all the values for this feature.
    self.sum = 0.0
    # The sum of squares of all the values for this feature.
    self.sum_of_squares = 0.0
    # The number of values for this feature that equal 0.
    self.num_zeros = 0
    # The number of NaN values for this feature. This is computed only for
    # FLOAT features.
    self.num_nan = 0
    # The minimum value among all the values for this feature.
    self.min = sys.maxsize
    # The maximum value among all the values for this feature.
    self.max = -sys.maxsize
    # The total number of non-missing values for this feature.
    self.total_num_values = 0
    # Type of this feature.
    self.type = None
    # Summary of the equi-width histogram for the values in this feature.
    self.std_hist_summary = ''
    # Summary of the quantiles histogram for the values in this feature.
    self.quantiles_hist_summary = ''


def _update_numeric_stats(
    numeric_stats, value,
    feature_name,
    feature_type,
    current_batch):
  """Update the partial numeric statistics using the input value."""
  if numeric_stats.type is not None and numeric_stats.type != feature_type:
    raise TypeError('Cannot determine the type of feature %s. '
                    'Found values of types %s and %s.' %
                    (feature_name, numeric_stats.type, feature_type))

  # Iterate through the value array and update the partial stats.
  for v in value:
    # If we have a NaN value, increment num_nan and continue to process
    # the next value.
    if (feature_type == statistics_pb2.FeatureNameStatistics.FLOAT and
        np.isnan(v)):
      numeric_stats.num_nan += 1
      continue
    numeric_stats.sum += v
    numeric_stats.sum_of_squares += v * v
    if v == 0:
      numeric_stats.num_zeros += 1
    numeric_stats.min = min(numeric_stats.min, v)
    numeric_stats.max = max(numeric_stats.max, v)
    current_batch.append(v)
    numeric_stats.total_num_values += 1

  # Update the feature type.
  if numeric_stats.type is None:
    numeric_stats.type = feature_type


def _merge_numeric_stats(
    left, right,
    feature_name):
  """Merge two partial numeric statistics and return the merged statistics."""
  # Check if the types from the two partial statistics are not compatible.
  # If so, raise an error.
  if (left.type is not None and right.type is not None and
      left.type != right.type):
    raise TypeError('Cannot determine the type of feature %s. '
                    'Found values of types %s and %s.' %
                    (feature_name, left.type, right.type))

  result = _PartialNumericStats()
  result.sum = left.sum + right.sum
  result.sum_of_squares = left.sum_of_squares + right.sum_of_squares
  result.num_zeros = left.num_zeros + right.num_zeros
  result.num_nan = left.num_nan + right.num_nan
  result.min = min(left.min, right.min)
  result.max = max(left.max, right.max)
  result.total_num_values = left.total_num_values + right.total_num_values
  result.type = left.type if left.type is not None else right.type
  return result


def _make_feature_stats_proto(
    numeric_stats, feature_name,
    std_hist_combiner,
    quantiles_hist_combiner,
    num_histogram_buckets):
  """Convert the partial numeric statistics into FeatureNameStatistics proto."""
  numeric_stats_proto = statistics_pb2.NumericStatistics()

  # Set the stats in the proto only if we have at least one value for the
  # feature.
  if numeric_stats.total_num_values > 0:
    mean = numeric_stats.sum / numeric_stats.total_num_values
    variance = max(
        0, (numeric_stats.sum_of_squares / numeric_stats.total_num_values) -
        mean * mean)
    numeric_stats_proto.mean = float(mean)
    numeric_stats_proto.std_dev = math.sqrt(variance)
    numeric_stats_proto.num_zeros = numeric_stats.num_zeros
    numeric_stats_proto.min = float(numeric_stats.min)
    numeric_stats_proto.max = float(numeric_stats.max)

    # Add median and equi-width histogram to the numeric stats proto.
    quantiles = std_hist_combiner.extract_output(numeric_stats.std_hist_summary)
    # Note that we find the median from the quantiles used for standard
    # histogram as it uses a large number of buckets and hence results in a more
    # accurate median estimate.
    numeric_stats_proto.median = float(quantiles_util.find_median(quantiles))
    std_histogram = quantiles_util.generate_equi_width_histogram(
        quantiles, numeric_stats.min, numeric_stats.max,
        numeric_stats.total_num_values, num_histogram_buckets)
    std_histogram.num_nan = numeric_stats.num_nan
    new_std_histogram = numeric_stats_proto.histograms.add()
    new_std_histogram.CopyFrom(std_histogram)

    # Add quantiles histogram to the numeric stats proto.
    quantiles = quantiles_hist_combiner.extract_output(
        numeric_stats.quantiles_hist_summary)
    q_histogram = quantiles_util.generate_quantiles_histogram(
        quantiles, numeric_stats.min, numeric_stats.max,
        numeric_stats.total_num_values)
    q_histogram.num_nan = numeric_stats.num_nan
    new_q_histogram = numeric_stats_proto.histograms.add()
    new_q_histogram.CopyFrom(q_histogram)

  # Create a new FeatureNameStatistics proto.
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  result.type = numeric_stats.type
  result.num_stats.CopyFrom(numeric_stats_proto)

  return result


# Currently we construct the equi-width histogram by using the
# quantiles. Specifically, we compute a large number of quantiles (say, N),
# and then compute the density for each bucket by aggregating the densities
# of the smaller quantile intervals that fall within the bucket. We set N to
# be _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM * num_histogram_buckets,
# where num_histogram_buckets is the required number of buckets in the
# histogram.
_NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM = 100


class NumericStatsGenerator(stats_generator.CombinerStatsGenerator):
  """A combiner statistics generator that computes the statistics
  for features of numeric type."""

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name = 'NumericStatsGenerator',
      schema = None,
      num_histogram_buckets = 10,
      num_quantiles_histogram_buckets = 10,
      epsilon = 0.01):
    """Initializes a numeric statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      num_histogram_buckets: An optional number of buckets in a standard
          NumericStatistics.histogram with equal-width buckets.
      num_quantiles_histogram_buckets: An optional number of buckets in a
          quantiles NumericStatistics.histogram.
      epsilon: An optional error tolerance for the computation of quantiles,
          typically a small fraction close to zero (e.g. 0.01). Higher values
          of epsilon increase the quantile approximation, and hence result in
          more unequal buckets, but could improve performance, and resource
          consumption.
    """
    super(NumericStatsGenerator, self).__init__(name)
    self._categorical_features = set(
        stats_util.get_categorical_numeric_features(schema) if schema else [])
    # Initialize quantiles combiner for equi-width histogram.
    self._std_hist_combiner = quantiles_util.QuantilesCombiner(
        _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM * num_histogram_buckets,
        epsilon)
    # Initialize quantiles combiner for quantiles histogram.
    self._quantiles_hist_combiner = quantiles_util.QuantilesCombiner(
        num_quantiles_histogram_buckets, epsilon)
    self._num_histogram_buckets = num_histogram_buckets

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self):
    return {}

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are numpy arrays representing a batch of examples) into the
  # accumulator.
  def add_input(self,
                accumulator,
                input_batch
               ):
    # Iterate through each feature and update the partial numeric stats.
    for feature_name, values in six.iteritems(input_batch):
      # If we have a categorical feature, don't generate numeric stats.
      if feature_name in self._categorical_features:
        continue

      # Update the numeric statistics for every example in the batch.
      current_batch = []
      for value in values:
        # Check if we have a numpy array with at least one value.
        if not isinstance(value, np.ndarray) or value.size == 0:
          continue

        # Check if the numpy array is of numeric type.
        feature_type = stats_util.make_feature_type(value.dtype)
        if feature_type not in [
            statistics_pb2.FeatureNameStatistics.INT,
            statistics_pb2.FeatureNameStatistics.FLOAT
        ]:
          continue

        # If we encounter this feature for the first time, create a
        # new partial numeric stats.
        if feature_name not in accumulator:
          partial_stats = _PartialNumericStats()
          # Store empty summary.
          partial_stats.std_hist_summary = (
              self._std_hist_combiner.create_accumulator())
          partial_stats.quantiles_hist_summary = (
              self._quantiles_hist_combiner.create_accumulator())
          accumulator[feature_name] = partial_stats

        # Update the partial numeric stats and append values
        # to the current batch.
        _update_numeric_stats(accumulator[feature_name], value, feature_name,
                              feature_type, current_batch)

      # Update the equi-width histogram and quantiles histogram sequi-widthor
      # the feature based on the current batch.
      if current_batch:
        accumulator[feature_name].std_hist_summary = (
            self._std_hist_combiner.add_input(
                accumulator[feature_name].std_hist_summary, [current_batch]))
        accumulator[feature_name].quantiles_hist_summary = (
            self._quantiles_hist_combiner.add_input(
                accumulator[feature_name].quantiles_hist_summary,
                [current_batch]))

    return accumulator

  # Merge together a list of partial numeric statistics.
  def merge_accumulators(
      self, accumulators
  ):
    result = {}
    std_hist_summary_per_feature = collections.defaultdict(list)
    quantiles_hist_summary_per_feature = collections.defaultdict(list)

    for accumulator in accumulators:
      for feature_name, numeric_stats in accumulator.items():
        if feature_name not in result:
          result[feature_name] = numeric_stats
        else:
          result[feature_name] = _merge_numeric_stats(
              result[feature_name], numeric_stats, feature_name)

        # Keep track of summaries per feature.
        std_hist_summary_per_feature[feature_name].append(
            numeric_stats.std_hist_summary)
        quantiles_hist_summary_per_feature[feature_name].append(
            numeric_stats.quantiles_hist_summary)

    # Merge the equi-width histogram summaries per feature.
    for feature_name, std_hist_summaries in \
        std_hist_summary_per_feature.items():
      result[feature_name].std_hist_summary = (
          self._std_hist_combiner.merge_accumulators(std_hist_summaries))
    # Merge the quantiles histogram summaries per feature.
    for feature_name, quantiles_hist_summaries in \
        quantiles_hist_summary_per_feature.items():
      result[feature_name].quantiles_hist_summary = (
          self._quantiles_hist_combiner.merge_accumulators(
              quantiles_hist_summaries))

    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator
                    ):
    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, numeric_stats in accumulator.items():
      # Construct the FeatureNameStatistics proto from the partial
      # numeric stats.
      feature_stats_proto = _make_feature_stats_proto(
          numeric_stats, feature_name, self._std_hist_combiner,
          self._quantiles_hist_combiner, self._num_histogram_buckets)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
