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
"""Module that computes the common statistics for all feature types.

Specifically, we compute the following common statistics for each feature,
  - Number of examples with at least one value for this feature.
  - Number of examples with no values for this feature.
  - Minimum number of values in a single example for this feature.
  - Maximum number of values in a single example for this feature.
  - Average number of values in a single example for this feature.
  - Total number of values across all examples for this feature.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import sys

import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.types_compat import Dict, List, Optional

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# Namespace for all TFDV metrics.
METRICS_NAMESPACE = 'tfx.DataValidation'


class _PartialCommonStats(object):
  """Holds partial statistics needed to compute the common statistics
  for a single feature."""

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

    # Keep track of partial weighted common stats.
    if has_weights:
      # The sum of weights of all the examples with at least one value for this
      # feature.
      self.weighted_num_non_missing = 0
      # The sum of weights of all the examples with no values for this feature.
      self.weighted_num_missing = 0
      # The sum of weights of all the values for this feature.
      self.weighted_total_num_values = 0


def _update_common_stats(common_stats,
                         value,
                         feature_name,
                         weight = None):
  """Update the partial common statistics using the input value."""
  # Check if the input value is a numpy array. If so, we have a non-missing
  # value to process.
  if isinstance(value, np.ndarray):
    # Get the number of values for the feature in the example.
    num_values = value.size
    common_stats.num_non_missing += 1
    common_stats.min_num_values = min(common_stats.min_num_values, num_values)
    common_stats.max_num_values = max(common_stats.max_num_values, num_values)
    common_stats.total_num_values += num_values

    if weight is not None:
      common_stats.weighted_num_non_missing += weight
      common_stats.weighted_total_num_values += weight * num_values

    feature_type = get_feature_type(value.dtype)
    if feature_type is None:
      raise TypeError('Feature %s has value which is a numpy array of type %s, '
                      'should be int, float or str types.' % (feature_name,
                                                              value.dtype.name))

    if common_stats.type is None:
      common_stats.type = feature_type
    elif common_stats.type != feature_type:
      raise TypeError('Cannot determine the type of feature %s. '
                      'Found values of types %s and %s.' %
                      (feature_name, common_stats.type, feature_type))
  # If the feature is missing, increment num_missing.
  # We represent a missing value by None.
  elif value is None:
    common_stats.num_missing += 1
    if weight is not None:
      common_stats.weighted_num_missing += weight
  else:
    raise TypeError('Feature %s has value of type %s, '
                    'should be numpy.ndarray or None' %
                    (feature_name, type(value).__name__))


def _merge_common_stats(left, right,
                        feature_name, has_weights
                       ):
  """Merge two partial common statistics and return the merged statistics."""
  # Check if the types from the two partial statistics are not compatible.
  # If so, raise an error. We consider types to be compatible if both types
  # are same or one of them is None.
  if (left.type is not None and right.type is not None and
      left.type != right.type):
    raise TypeError('Cannot determine the type of feature %s. '
                    'Found values of types %s and %s.' %
                    (feature_name, left.type, right.type))

  result = _PartialCommonStats(has_weights)
  result.num_non_missing = left.num_non_missing + right.num_non_missing
  result.num_missing = left.num_missing + right.num_missing
  result.min_num_values = min(left.min_num_values, right.min_num_values)
  result.max_num_values = max(left.max_num_values, right.max_num_values)
  result.total_num_values = left.total_num_values + right.total_num_values

  if has_weights:
    result.weighted_num_non_missing = (left.weighted_num_non_missing +
                                       right.weighted_num_non_missing)
    result.weighted_num_missing = (left.weighted_num_missing +
                                   right.weighted_num_missing)
    result.weighted_total_num_values = (left.weighted_total_num_values +
                                        right.weighted_total_num_values)

  # Set the type of the merged common stats.
  # Case 1: Both the types are None. We set the merged type to be None.
  # Case 2: One of two types is None. We set the merged type to be the type
  # which is not None. For example, if left.type=FLOAT and right.type=None,
  # we set the merged type to be FLOAT.
  # Case 3: Both the types are same (and not None), we set the merged type to be
  # the same type.
  result.type = left.type if left.type is not None else right.type
  return result


def _make_feature_stats_proto(
    common_stats, feature_name,
    q_combiner,
    num_values_histogram_buckets,
    is_categorical, has_weights
):
  """Convert the partial common stats into a FeatureNameStatistics proto.

  Args:
    common_stats: The partial common stats associated with a feature.
    feature_name: The name of the feature.
    q_combiner: The quantiles combiner used to construct the quantiles
        histogram for the number of values in the feature.
    num_values_histogram_buckets: Number of buckets in the quantiles
        histogram for the number of values per feature.
    is_categorical: A boolean indicating whether the feature is categorical.
    has_weights: A boolean indicating whether a weight feature is specified.

  Returns:
    A statistics_pb2.FeatureNameStatistics proto.
  """
  common_stats_proto = statistics_pb2.CommonStatistics()
  common_stats_proto.num_non_missing = common_stats.num_non_missing
  common_stats_proto.num_missing = common_stats.num_missing
  common_stats_proto.tot_num_values = common_stats.total_num_values

  if common_stats.num_non_missing > 0:
    common_stats_proto.min_num_values = common_stats.min_num_values
    common_stats_proto.max_num_values = common_stats.max_num_values
    common_stats_proto.avg_num_values = (
        common_stats.total_num_values / common_stats.num_non_missing)

    # Add num_values_histogram to the common stats proto.
    num_values_quantiles = q_combiner.extract_output(
        common_stats.num_values_summary)
    histogram = quantiles_util.generate_quantiles_histogram(
        num_values_quantiles, common_stats.min_num_values,
        common_stats.max_num_values, common_stats.num_non_missing,
        num_values_histogram_buckets)
    common_stats_proto.num_values_histogram.CopyFrom(histogram)

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

    common_stats_proto.weighted_common_stats.CopyFrom(
        weighted_common_stats_proto)

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
  elif common_stats.type is None:
    # If a feature is completely missing, we assume the type to be STRING.
    result.type = statistics_pb2.FeatureNameStatistics.STRING
  else:
    result.type = common_stats.type

  # Copy the common stats into appropriate numeric/string stats.
  # If the type is not set, we currently wrap the common stats
  # within numeric stats.
  if (result.type == statistics_pb2.FeatureNameStatistics.STRING or
      is_categorical):
    # Add the common stats into string stats.
    string_stats_proto = statistics_pb2.StringStatistics()
    string_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.string_stats.CopyFrom(string_stats_proto)
  else:
    # Add the common stats into numeric stats.
    numeric_stats_proto = statistics_pb2.NumericStatistics()
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

  for common_stats in accumulator.values():
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
  beam_metrics = beam.metrics.Metrics.counter
  beam_metrics(METRICS_NAMESPACE, 'num_instances').inc(
      num_instances)
  beam_metrics(METRICS_NAMESPACE, 'num_missing_feature_values').inc(
      num_missing_feature_values)

  for feature_type in metrics:
    type_str = statistics_pb2.FeatureNameStatistics.Type.Name(
        feature_type).lower()
    type_metrics = metrics[feature_type]
    beam_metrics(METRICS_NAMESPACE, 'num_' + type_str + '_feature_values').inc(
        type_metrics.num_non_missing)
    beam_metrics(METRICS_NAMESPACE, type_str + '_feature_values_min_count').inc(
        type_metrics.min_value_count if type_metrics.num_non_missing > 0 else -1
    )
    beam_metrics(METRICS_NAMESPACE, type_str + '_feature_values_max_count').inc(
        type_metrics.max_value_count if type_metrics.num_non_missing > 0 else -1
    )
    beam_metrics(
        METRICS_NAMESPACE, type_str + '_feature_values_mean_count').inc(
            int(type_metrics.total_num_values/type_metrics.num_non_missing)
            if type_metrics.num_non_missing > 0 else -1)


class CommonStatsGenerator(stats_generator.CombinerStatsGenerator):
  """A combiner statistics generator that computes the common statistics
  for all the features."""

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name = 'CommonStatsGenerator',
      schema = None,
      weight_feature = None,
      num_values_histogram_buckets = 10,
      epsilon = 0.01):
    """Initializes a common statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: An optional feature name whose numeric value represents
          the weight of an example.
      num_values_histogram_buckets: An optional number of buckets in a quantiles
          histogram for the number of values per Feature, which is stored in
          CommonStatistics.num_values_histogram.
      epsilon: An optional error tolerance for the computation of quantiles,
          typically a small fraction close to zero (e.g. 0.01). Higher values
          of epsilon increase the quantile approximation, and hence result in
          more unequal buckets, but could improve performance, and resource
          consumption.
    """
    super(CommonStatsGenerator, self).__init__(name, schema)
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_values_histogram_buckets = num_values_histogram_buckets
    # Initialize quantiles combiner.
    self._quantiles_combiner = quantiles_util.QuantilesCombiner(
        self._num_values_histogram_buckets, epsilon)

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self):
    return {}

  # Incorporates the input (a Python dict whose keys are
  # feature names and values are numpy arrays representing a
  # batch of examples) into the accumulator.
  def add_input(self, accumulator,
                input_batch
               ):
    if self._weight_feature:
      weights = stats_util.get_weight_feature(input_batch, self._weight_feature)

    # Iterate through each feature and update the partial common stats.
    for feature_name, values in six.iteritems(input_batch):
      # Skip the weight feature.
      if feature_name == self._weight_feature:
        continue
      # If we encounter this feature for the first time, create a
      # new partial common stats.
      if feature_name not in accumulator:
        partial_stats = _PartialCommonStats(self._weight_feature is not None)
        # Store empty summary.
        partial_stats.num_values_summary = (
            self._quantiles_combiner.create_accumulator())
        accumulator[feature_name] = partial_stats

      # Update the common statistics for every example in the batch.
      num_values = []

      for i, value in enumerate(values):
        _update_common_stats(accumulator[feature_name], value, feature_name,
                             weights[i][0] if self._weight_feature else None)
        # Keep track of the number of values in non-missing examples.
        if isinstance(value, np.ndarray):
          num_values.append(value.size)

      # Update the num_vals_histogram summary for the feature based on the
      # current batch.
      if num_values:
        accumulator[feature_name].num_values_summary = (
            self._quantiles_combiner.add_input(
                accumulator[feature_name].num_values_summary, [num_values]))

    return accumulator

  # Merge together a list of partial common statistics.
  def merge_accumulators(
      self, accumulators
  ):
    result = {}
    num_values_summary_per_feature = collections.defaultdict(list)

    for accumulator in accumulators:
      for feature_name, common_stats in accumulator.items():
        if feature_name not in result:
          result[feature_name] = common_stats
        else:
          result[feature_name] = _merge_common_stats(
              result[feature_name], common_stats, feature_name,
              self._weight_feature is not None)

        # Keep track of summaries per feature.
        num_values_summary_per_feature[feature_name].append(
            common_stats.num_values_summary)

    # Merge the summaries per feature.
    for feature_name, num_values_summaries in\
        num_values_summary_per_feature.items():
      result[feature_name].num_values_summary = (
          self._quantiles_combiner.merge_accumulators(num_values_summaries))

    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator
                    ):
    # Update TFDV telemetry.
    _update_tfdv_telemetry(accumulator)

    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, common_stats in accumulator.items():
      # Construct the FeatureNameStatistics proto from the partial
      # common stats.
      feature_stats_proto = _make_feature_stats_proto(
          common_stats, feature_name, self._quantiles_combiner,
          self._num_values_histogram_buckets,
          feature_name in self._categorical_features,
          self._weight_feature is not None)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
