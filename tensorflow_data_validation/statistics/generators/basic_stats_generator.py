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
"""Module that computes the basic statistics for the features.

We compute the following common statistics for each feature:
  - Number of examples with at least one value for this feature.
  - Number of examples with no values for this feature.
  - Minimum number of values in a single example for this feature.
  - Maximum number of values in a single example for this feature.
  - Average number of values in a single example for this feature.
  - Total number of values across all examples for this feature.
  - Quantiles histogram over the number of values in an example.

We compute the following statistics across all examples
for each numeric feature:
  - Mean of the values for this feature.
  - Standard deviation of the values for this feature.
  - Median of the values for this feature.
  - Number of values that equal zero for this feature.
  - Minimum value for this feature.
  - Maximum value for this feature.
  - Standard histogram over the feature values.
  - Quantiles histogram over the feature values.

We compute the following statistics for each string feature:
  - Average length of the values for this feature.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import math
import sys

import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.types_compat import Dict, List, Optional, Text, Union

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialCommonStats(object):
  """Holds partial common statistics for a single feature."""

  def __init__(self, has_weights):
    # The number of examples with at least one value for this feature.
    self.num_non_missing = 0
    # The number of examples with no values for this feature.
    self.num_missing = 0
    # The minimum number of values in a single example for this feature.
    self.min_num_values = sys.maxsize
    # The maximum number of values in a single example for this feature.
    self.max_num_values = 0
    # The total number of values for this feature.
    self.total_num_values = 0
    # Type of the feature.
    self.type = None
    # Summary of the quantiles histogram for the number of values in this
    # feature.
    self.num_values_summary = ''

    self.has_weights = has_weights
    # Keep track of partial weighted common stats.
    if has_weights:
      # The sum of weights of all the examples with at least one value for this
      # feature.
      self.weighted_num_non_missing = 0
      # The sum of weights of all the examples with no values for this feature.
      self.weighted_num_missing = 0
      # The sum of weights of all the values for this feature.
      self.weighted_total_num_values = 0

  def __iadd__(self, other):
    """Merge two partial common statistics and return the merged statistics."""
    self.num_non_missing += other.num_non_missing
    self.num_missing += other.num_missing
    self.min_num_values = min(self.min_num_values, other.min_num_values)
    self.max_num_values = max(self.max_num_values, other.max_num_values)
    self.total_num_values += other.total_num_values

    assert self.has_weights == other.has_weights
    if self.has_weights:
      self.weighted_num_non_missing += other.weighted_num_non_missing
      self.weighted_num_missing += other.weighted_num_missing
      self.weighted_total_num_values += other.weighted_total_num_values

    # Set the type of the merged common stats.
    # Case 1: Both the types are None. We set the merged type to be None.
    # Case 2: One of two types is None. We set the merged type to be the type
    # which is not None. For example, if left.type=FLOAT and right.type=None,
    # we set the merged type to be FLOAT.
    # Case 3: Both the types are same (and not None), we set the merged type to
    # be the same type.
    if self.type is None:
      self.type = other.type
    return self

  def update(self,
             value,
             feature_name,
             feature_type,
             weight = None):
    """Update the partial common statistics using the input value."""
    if value is not None:
      # Get the number of values for the feature in the example.
      num_values = value.size
      self.num_non_missing += 1
      if num_values < self.min_num_values:
        self.min_num_values = num_values
      if num_values > self.max_num_values:
        self.max_num_values = num_values
      self.total_num_values += num_values

      if weight is not None:
        self.weighted_num_non_missing += weight
        self.weighted_total_num_values += weight * num_values

      if self.type is None:
        self.type = feature_type
      elif self.type != feature_type:
        raise TypeError('Cannot determine the type of feature %s. '
                        'Found values of types %s and %s.' %
                        (feature_name, self.type, feature_type))
    # If the feature is missing, increment num_missing.
    # We represent a missing value by None.
    else:
      self.num_missing += 1
      if weight is not None:
        self.weighted_num_missing += weight


class _PartialNumericStats(object):
  """Holds partial numeric statistics for a single feature."""

  def __init__(self, has_weights):
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
    # Summary of the quantiles for the values in this feature.
    self.quantiles_summary = ''

    self.has_weights = has_weights
    # Keep track of partial weighted numeric stats.
    if has_weights:
      # The weighted sum of all the values for this feature.
      self.weighted_sum = 0.0
      # The weighted sum of squares of all the values for this feature.
      self.weighted_sum_of_squares = 0.0
      # The sum of weights of all the values for this feature.
      self.weighted_total_num_values = 0.0
      # Summary of the weighted quantiles for the values in this feature.
      self.weighted_quantiles_summary = ''

  def __iadd__(self, other):
    """Merge two partial numeric statistics and return the merged statistics."""
    self.sum += other.sum
    self.sum_of_squares += other.sum_of_squares
    self.num_zeros += other.num_zeros
    self.num_nan += other.num_nan
    self.min = min(self.min, other.min)
    self.max = max(self.max, other.max)

    assert self.has_weights == other.has_weights
    if self.has_weights:
      self.weighted_sum += other.weighted_sum
      self.weighted_sum_of_squares += other.weighted_sum_of_squares
      self.weighted_total_num_values += other.weighted_total_num_values
    return self

  def update(
      self,
      value,
      values_and_weights_numeric_feat,
      weight = None):
    """Update the partial numeric statistics using the input value."""
    isnan = math.isnan
    append_value = values_and_weights_numeric_feat[0].append
    # Iterate through the value array and update the partial stats.
    for v in value:
      # If we have a NaN value, increment num_nan and continue to process
      # the next value.
      if isnan(v):
        self.num_nan += 1
        continue
      self.sum += v
      self.sum_of_squares += v * v
      if v == 0:
        self.num_zeros += 1
      if v < self.min:
        self.min = v
      if v > self.max:
        self.max = v
      append_value(v)

      if weight is not None:
        self.weighted_sum += weight * v
        self.weighted_sum_of_squares += weight * v * v
        self.weighted_total_num_values += weight
        values_and_weights_numeric_feat[1].append(weight)


class _PartialStringStats(object):
  """Holds partial string statistics for a single feature."""

  def __init__(self):
    # The total length of all the values for this feature.
    self.total_bytes_length = 0

  def __iadd__(self, other):
    """Merge two partial string statistics and return the merged statistics."""
    self.total_bytes_length += other.total_bytes_length
    return self

  def update(self, value):
    """Update the partial string statistics using the input value."""
    # Iterate through the value array and update the partial stats.
    for v in value:
      self.total_bytes_length += len(v)


class _PartialBasicStats(object):
  """Holds partial statistics for a single feature."""

  def __init__(self, has_weights):
    self.common_stats = _PartialCommonStats(has_weights=has_weights)
    self.numeric_stats = _PartialNumericStats(has_weights=has_weights)
    self.string_stats = _PartialStringStats()


def _make_common_stats_proto(
    common_stats,
    q_combiner,
    num_values_histogram_buckets,
    has_weights
):
  """Convert the partial common stats into a CommonStatistics proto."""
  result = statistics_pb2.CommonStatistics()
  result.num_non_missing = common_stats.num_non_missing
  result.num_missing = common_stats.num_missing
  result.tot_num_values = common_stats.total_num_values

  # TODO(b/79685042): Need to decide on what is the expected values for
  # statistics like min_num_values, max_num_values, avg_num_values, when
  # all the values for the feature are missing.
  if common_stats.num_non_missing > 0:
    result.min_num_values = common_stats.min_num_values
    result.max_num_values = common_stats.max_num_values
    result.avg_num_values = (
        common_stats.total_num_values / common_stats.num_non_missing)

    # Add num_values_histogram to the common stats proto.
    num_values_quantiles = q_combiner.extract_output(
        common_stats.num_values_summary)
    histogram = quantiles_util.generate_quantiles_histogram(
        num_values_quantiles, common_stats.min_num_values,
        common_stats.max_num_values, common_stats.num_non_missing,
        num_values_histogram_buckets)
    result.num_values_histogram.CopyFrom(histogram)

  # Add weighted common stats to the proto.
  if has_weights:
    weighted_common_stats_proto = statistics_pb2.WeightedCommonStatistics(
        num_non_missing=common_stats.weighted_num_non_missing,
        num_missing=common_stats.weighted_num_missing,
        tot_num_values=common_stats.weighted_total_num_values)

    if common_stats.weighted_num_non_missing > 0:
      weighted_common_stats_proto.avg_num_values = (
          common_stats.weighted_total_num_values /
          common_stats.weighted_num_non_missing)

    result.weighted_common_stats.CopyFrom(
        weighted_common_stats_proto)
  return result


def _make_numeric_stats_proto(
    numeric_stats,
    total_num_values,
    quantiles_combiner,
    num_histogram_buckets,
    num_quantiles_histogram_buckets,
    has_weights
    ):
  """Convert the partial numeric statistics into NumericStatistics proto."""
  result = statistics_pb2.NumericStatistics()

  if numeric_stats.num_nan > 0:
    total_num_values -= numeric_stats.num_nan

  # Set the stats in the proto only if we have at least one value for the
  # feature.
  if total_num_values == 0:
    return result

  mean = numeric_stats.sum / total_num_values
  variance = max(
      0, (numeric_stats.sum_of_squares / total_num_values) -
      mean * mean)
  result.mean = float(mean)
  result.std_dev = math.sqrt(variance)
  result.num_zeros = numeric_stats.num_zeros
  result.min = float(numeric_stats.min)
  result.max = float(numeric_stats.max)

  # Extract the quantiles from the summary.
  quantiles = quantiles_combiner.extract_output(
      numeric_stats.quantiles_summary)

  # Find the median from the quantiles and update the numeric stats proto.
  result.median = float(quantiles_util.find_median(quantiles))

  # Construct the equi-width histogram from the quantiles and add it to the
  # numeric stats proto.
  std_histogram = quantiles_util.generate_equi_width_histogram(
      quantiles, numeric_stats.min, numeric_stats.max,
      total_num_values, num_histogram_buckets)
  std_histogram.num_nan = numeric_stats.num_nan
  new_std_histogram = result.histograms.add()
  new_std_histogram.CopyFrom(std_histogram)

  # Construct the quantiles histogram from the quantiles and add it to the
  # numeric stats proto.
  q_histogram = quantiles_util.generate_quantiles_histogram(
      quantiles, numeric_stats.min, numeric_stats.max,
      total_num_values, num_quantiles_histogram_buckets)
  q_histogram.num_nan = numeric_stats.num_nan
  new_q_histogram = result.histograms.add()
  new_q_histogram.CopyFrom(q_histogram)

  # Add weighted numeric stats to the proto.
  if has_weights:
    weighted_numeric_stats_proto = statistics_pb2.WeightedNumericStatistics()

    if numeric_stats.weighted_total_num_values == 0:
      weighted_mean = 0
      weighted_variance = 0
    else:
      weighted_mean = (numeric_stats.weighted_sum /
                       numeric_stats.weighted_total_num_values)
      weighted_variance = max(0, (numeric_stats.weighted_sum_of_squares /
                                  numeric_stats.weighted_total_num_values)
                              - weighted_mean**2)
    weighted_numeric_stats_proto.mean = weighted_mean
    weighted_numeric_stats_proto.std_dev = math.sqrt(weighted_variance)

    # Extract the weighted quantiles from the summary.
    weighted_quantiles = quantiles_combiner.extract_output(
        numeric_stats.weighted_quantiles_summary)

    # Find the weighted median from the quantiles and update the proto.
    weighted_numeric_stats_proto.median = float(
        quantiles_util.find_median(weighted_quantiles))

    # Construct the weighted equi-width histogram from the quantiles and
    # add it to the numeric stats proto.
    weighted_std_histogram = quantiles_util.generate_equi_width_histogram(
        weighted_quantiles, numeric_stats.min, numeric_stats.max,
        numeric_stats.weighted_total_num_values, num_histogram_buckets)
    weighted_std_histogram.num_nan = numeric_stats.num_nan
    weighted_numeric_stats_proto.histograms.extend([weighted_std_histogram])

    # Construct the weighted quantiles histogram from the quantiles and
    # add it to the numeric stats proto.
    weighted_q_histogram = quantiles_util.generate_quantiles_histogram(
        weighted_quantiles, numeric_stats.min, numeric_stats.max,
        numeric_stats.weighted_total_num_values,
        num_quantiles_histogram_buckets)
    weighted_q_histogram.num_nan = numeric_stats.num_nan
    weighted_numeric_stats_proto.histograms.extend([weighted_q_histogram])

    result.weighted_numeric_stats.CopyFrom(
        weighted_numeric_stats_proto)
  return result


def _make_string_stats_proto(string_stats,
                             total_num_values
                            ):
  """Convert the partial string statistics into StringStatistics proto."""
  result = statistics_pb2.StringStatistics()
  if total_num_values > 0:
    result.avg_length = string_stats.total_bytes_length / total_num_values
  return result


def _make_feature_stats_proto(
    basic_stats, feature_name,
    num_values_q_combiner,
    values_q_combiner,
    num_values_histogram_buckets,
    num_histogram_buckets,
    num_quantiles_histogram_buckets,
    is_categorical, has_weights
):
  """Convert the partial basic stats into a FeatureNameStatistics proto.

  Args:
    basic_stats: The partial basic stats associated with a feature.
    feature_name: The name of the feature.
    num_values_q_combiner: The quantiles combiner used to construct the
        quantiles histogram for the number of values in the feature.
    values_q_combiner: The quantiles combiner used to construct the
        histogram for the values in the feature.
    num_values_histogram_buckets: Number of buckets in the quantiles
        histogram for the number of values per feature.
    num_histogram_buckets: Number of buckets in a standard
        NumericStatistics.histogram with equal-width buckets.
    num_quantiles_histogram_buckets: Number of buckets in a
        quantiles NumericStatistics.histogram.
    is_categorical: A boolean indicating whether the feature is categorical.
    has_weights: A boolean indicating whether a weight feature is specified.

  Returns:
    A statistics_pb2.FeatureNameStatistics proto.
  """
  # Create a new FeatureNameStatistics proto.
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  # Set the feature type.
  # If we have a categorical feature, we preserve the type to be the original
  # INT type. Currently we don't set the type if we cannot infer it, which
  # happens when all the values are missing. We need to add an UNKNOWN type
  # to the stats proto to handle this case.
  if is_categorical:
    result.type = statistics_pb2.FeatureNameStatistics.INT
  elif basic_stats.common_stats.type is None:
    # If a feature is completely missing, we assume the type to be STRING.
    result.type = statistics_pb2.FeatureNameStatistics.STRING
  else:
    result.type = basic_stats.common_stats.type

  # Construct common statistics proto.
  common_stats_proto = _make_common_stats_proto(basic_stats.common_stats,
                                                num_values_q_combiner,
                                                num_values_histogram_buckets,
                                                has_weights)

  # Copy the common stats into appropriate numeric/string stats.
  # If the type is not set, we currently wrap the common stats
  # within numeric stats.
  if (is_categorical or
      result.type == statistics_pb2.FeatureNameStatistics.STRING):
    # Construct string statistics proto.
    string_stats_proto = _make_string_stats_proto(
        basic_stats.string_stats, basic_stats.common_stats.total_num_values)
    # Add the common stats into string stats.
    string_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.string_stats.CopyFrom(string_stats_proto)
  else:
    # Construct numeric statistics proto.
    numeric_stats_proto = _make_numeric_stats_proto(
        basic_stats.numeric_stats, basic_stats.common_stats.total_num_values,
        values_q_combiner, num_histogram_buckets,
        num_quantiles_histogram_buckets, has_weights)
    # Add the common stats into numeric stats.
    numeric_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.num_stats.CopyFrom(numeric_stats_proto)

  return result


# Named tuple containing TFDV metrics.
_TFDVMetrics = collections.namedtuple(
    '_TFDVMetrics', ['num_non_missing', 'min_value_count',
                     'max_value_count', 'total_num_values'])
_TFDVMetrics.__new__.__defaults__ = (0, sys.maxsize, 0, 0)


def _update_tfdv_telemetry(
    accumulator):
  """Update TFDV Beam metrics."""
  num_instances, num_missing_feature_values = 0, 0
  # Aggregate type specific metrics.
  metrics = {
      statistics_pb2.FeatureNameStatistics.INT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.FLOAT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.STRING: _TFDVMetrics()
  }

  for basic_stats in accumulator.values():
    common_stats = basic_stats.common_stats
    if num_instances == 0:
      num_instances = common_stats.num_missing + common_stats.num_non_missing
    num_missing_feature_values += common_stats.num_missing
    if common_stats.type is None:
      continue
    # Update type specific metrics.
    type_metrics = metrics[common_stats.type]
    num_non_missing = (type_metrics.num_non_missing +
                       common_stats.num_non_missing)
    min_value_count = min(type_metrics.min_value_count,
                          common_stats.min_num_values)
    max_value_count = max(type_metrics.max_value_count,
                          common_stats.max_num_values)
    total_num_values = (type_metrics.total_num_values +
                        common_stats.total_num_values)
    metrics[common_stats.type] = _TFDVMetrics(num_non_missing, min_value_count,
                                              max_value_count, total_num_values)

  # Update Beam counters.
  counter = beam.metrics.Metrics.counter
  counter(constants.METRICS_NAMESPACE, 'num_instances').inc(num_instances)
  counter(constants.METRICS_NAMESPACE, 'num_missing_feature_values').inc(
      num_missing_feature_values)

  for feature_type in metrics:
    type_str = statistics_pb2.FeatureNameStatistics.Type.Name(
        feature_type).lower()
    type_metrics = metrics[feature_type]
    counter(constants.METRICS_NAMESPACE,
            'num_' + type_str + '_feature_values').inc(
                type_metrics.num_non_missing)
    is_present = type_metrics.num_non_missing > 0
    counter(constants.METRICS_NAMESPACE,
            type_str + '_feature_values_min_count').inc(
                type_metrics.min_value_count if is_present else -1)
    counter(constants.METRICS_NAMESPACE,
            type_str + '_feature_values_max_count').inc(
                type_metrics.max_value_count if is_present else -1)
    counter(
        constants.METRICS_NAMESPACE,
        type_str + '_feature_values_mean_count').inc(
            int(type_metrics.total_num_values/type_metrics.num_non_missing)
            if is_present else -1)


# Currently we construct the equi-width histogram by using the
# quantiles. Specifically, we compute a large number of quantiles (say, N),
# and then compute the density for each bucket by aggregating the densities
# of the smaller quantile intervals that fall within the bucket. We set N to
# be _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM * num_histogram_buckets,
# where num_histogram_buckets is the required number of buckets in the
# histogram.
_NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM = 100


# TODO(b/79685042): Currently the stats generator operates on the
# Dict representation of input (mapping from feature name to a batch of
# values). But we process each feature independently. We should
# consider making the stats generator to operate per feature.
class BasicStatsGenerator(stats_generator.CombinerStatsGenerator):
  """A combiner statistics generator that computes the common statistics
  for all the features, numeric statistics for numeric features and
  string statistics for string features.
  """

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name = 'BasicStatsGenerator',
      schema = None,
      weight_feature = None,
      num_values_histogram_buckets = 10,
      num_histogram_buckets = 10,
      num_quantiles_histogram_buckets = 10,
      epsilon = 0.01):
    """Initializes basic statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: An optional feature name whose numeric value represents
          the weight of an example.
      num_values_histogram_buckets: An optional number of buckets in a quantiles
          histogram for the number of values per Feature, which is stored in
          CommonStatistics.num_values_histogram.
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
    super(BasicStatsGenerator, self).__init__(name, schema)
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_values_histogram_buckets = num_values_histogram_buckets
    # Initialize quantiles combiner for histogram over number of values.
    self._num_values_quantiles_combiner = quantiles_util.QuantilesCombiner(
        self._num_values_histogram_buckets, epsilon)

    self._num_histogram_buckets = num_histogram_buckets
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    num_buckets = max(
        self._num_quantiles_histogram_buckets,
        _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM * self._num_histogram_buckets)
    # Initialize quantiles combiner for histogram over feature values.
    self._values_quantiles_combiner = quantiles_util.QuantilesCombiner(
        num_buckets, epsilon, has_weights=True)

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self):
    return {}

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are lists representing a batch of examples) into the accumulator.
  def add_input(self, accumulator,
                input_batch
               ):
    if self._weight_feature:
      # TODO(b/118489848): This method also validates the weight feature.
      # Consider moving these validation checks outside of the generators.
      weights = stats_util.get_weight_feature(input_batch, self._weight_feature)

    # Iterate through each feature and update the partial basic stats.
    for feature_name, values in six.iteritems(input_batch):
      # Skip the weight feature.
      if feature_name == self._weight_feature:
        continue
      is_categorical_feature = feature_name in self._categorical_features

      # If we encounter this feature for the first time, create a
      # new partial basic stats.
      if feature_name not in accumulator:
        partial_stats = _PartialBasicStats(self._weight_feature is not None)
        # Store empty summary.
        partial_stats.common_stats.num_values_summary = (
            self._num_values_quantiles_combiner.create_accumulator())
        partial_stats.numeric_stats.quantiles_summary = (
            self._values_quantiles_combiner.create_accumulator())
        accumulator[feature_name] = partial_stats

      # Keep track of the number of values in each example in order to update
      # the common statistics.
      num_values = []

      # Keep track of the values and the weights in the current batch for
      # numeric feature. Note that we store the values in the current batch
      # so that we invoke the quantiles combiner only once per feature for
      # the input batch.
      values_and_weights_numeric_feat = [[], []]

      for i, value in enumerate(values):
        # TODO(b/79685042): Currently we infer the type for each example, which
        # is expensive. Consider doing the type inference only once per batch.
        if isinstance(value, np.ndarray):
          feature_type = get_feature_type(value.dtype)
          if feature_type is None:
            raise TypeError('Feature {} has value {} which is a numpy array '
                            'of type {}, should be int, float or str '
                            'types.'.format(feature_name, value,
                                            value.dtype.name))
        elif value is None:
          # We have a missing value.
          feature_type = None
        else:
          raise TypeError('Feature %s has value of type %s, '
                          'should be numpy.ndarray or None' %
                          (feature_name, type(value).__name__))

        accumulator[feature_name].common_stats.update(
            value, feature_name, feature_type,
            weights[i][0] if self._weight_feature else None)
        if value is None:
          continue
        # Keep track of the number of values in non-missing examples.
        num_values.append(value.size)
        if value.size == 0:
          continue

        if (is_categorical_feature or
            feature_type == statistics_pb2.FeatureNameStatistics.STRING):
          # If we have a categorical feature, convert the value to string type.
          if is_categorical_feature:
            value = value.astype(str)

          # Update the partial string stats.
          accumulator[feature_name].string_stats.update(value)
        else:
          # Update the partial numeric stats and append values
          # to the current batch of values and weights.
          accumulator[feature_name].numeric_stats.update(
              value, values_and_weights_numeric_feat,
              weights[i][0] if self._weight_feature else None)

      # Update the num_vals_histogram summary for the feature based on the
      # current batch.
      if num_values:
        accumulator[feature_name].common_stats.num_values_summary = (
            self._num_values_quantiles_combiner.add_input(
                accumulator[feature_name].common_stats.num_values_summary,
                [num_values]))

      # Update the quantiles summary of the numeric feature values based on the
      # current batch of values and weights.
      if values_and_weights_numeric_feat[0]:
        numeric_stats = accumulator[feature_name].numeric_stats
        # For the unweighted case, explicitly set the weights to be 1. We do
        # this so that we can use the same weighted quantiles combiner for both
        # scenarios.
        numeric_stats.quantiles_summary = (
            self._values_quantiles_combiner.add_input(
                numeric_stats.quantiles_summary,
                [values_and_weights_numeric_feat[0],
                 # Set weights to be 1.
                 [1] * len(values_and_weights_numeric_feat[0])]))

        if self._weight_feature:
          numeric_stats.weighted_quantiles_summary = (
              self._values_quantiles_combiner.add_input(
                  numeric_stats.weighted_quantiles_summary,
                  values_and_weights_numeric_feat))

    return accumulator

  # Merge together a list of basic common statistics.
  def merge_accumulators(
      self, accumulators
  ):
    result = {}
    num_values_summary_per_feature = collections.defaultdict(list)
    values_summary_per_feature = collections.defaultdict(list)
    if self._weight_feature:
      weighted_values_summary_per_feature = collections.defaultdict(list)

    for accumulator in accumulators:
      for feature_name, basic_stats in accumulator.items():
        is_categorical = feature_name in self._categorical_features
        current_type = basic_stats.common_stats.type
        if feature_name not in result:
          result[feature_name] = basic_stats
        else:
          # Check if the types from the two partial statistics are not
          # compatible. If so, raise an error. We consider types to be
          # compatible if both types are same or one of them is None.
          left_type = result[feature_name].common_stats.type
          right_type = basic_stats.common_stats.type
          if (left_type is not None and right_type is not None and
              left_type != right_type):
            raise TypeError('Cannot determine the type of feature %s. '
                            'Found values of types %s and %s.' %
                            (feature_name, left_type, right_type))

          result[feature_name].common_stats += basic_stats.common_stats

          if current_type is not None:
            if (is_categorical or
                current_type == statistics_pb2.FeatureNameStatistics.STRING):
              result[feature_name].string_stats += basic_stats.string_stats
            else:
              result[feature_name].numeric_stats += basic_stats.numeric_stats

        # Keep track of num values quantiles summaries per feature.
        num_values_summary_per_feature[feature_name].append(
            basic_stats.common_stats.num_values_summary)

        if (current_type is not None and
            not is_categorical and
            current_type != statistics_pb2.FeatureNameStatistics.STRING):
          # Keep track of values quantile summaries per feature.
          values_summary_per_feature[feature_name].append(
              basic_stats.numeric_stats.quantiles_summary)

          # Keep track of values weighted quantile summaries per feature.
          if self._weight_feature:
            weighted_values_summary_per_feature[feature_name].append(
                basic_stats.numeric_stats.weighted_quantiles_summary)

    # Merge the summaries per feature.
    for feature_name, num_values_summaries in\
        num_values_summary_per_feature.items():
      result[feature_name].common_stats.num_values_summary = (
          self._num_values_quantiles_combiner.merge_accumulators(
              num_values_summaries))

    # Merge the values quantiles summaries per feature.
    for feature_name, quantiles_summaries in (
        values_summary_per_feature.items()):
      result[feature_name].numeric_stats.quantiles_summary = (
          self._values_quantiles_combiner.merge_accumulators(
              quantiles_summaries))

    # Merge the values weighted quantiles summaries per feature.
    if self._weight_feature:
      for feature_name, weighted_quantiles_summaries in (
          weighted_values_summary_per_feature.items()):
        result[feature_name].numeric_stats.weighted_quantiles_summary = (
            self._values_quantiles_combiner.merge_accumulators(
                weighted_quantiles_summaries))

    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator
                    ):
    # Update TFDV telemetry.
    _update_tfdv_telemetry(accumulator)

    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, basic_stats in accumulator.items():
      # Construct the FeatureNameStatistics proto from the partial
      # basic stats.
      feature_stats_proto = _make_feature_stats_proto(
          basic_stats, feature_name,
          self._num_values_quantiles_combiner, self._values_quantiles_combiner,
          self._num_values_histogram_buckets, self._num_histogram_buckets,
          self._num_quantiles_histogram_buckets,
          feature_name in self._categorical_features,
          self._weight_feature is not None)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
