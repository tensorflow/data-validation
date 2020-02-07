# Copyright 2019 Google LLC
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
import pyarrow as pa
import six
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util
from typing import Any, Dict, Iterable, Optional, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialCommonStats(object):
  """Holds partial common statistics for a single feature."""

  __slots__ = ['num_non_missing', 'min_num_values', 'max_num_values',
               'total_num_values', 'type', 'num_values_summary', 'has_weights',
               'weighted_num_non_missing', 'weighted_total_num_values']

  def __init__(self, has_weights: bool):
    # The number of examples with at least one value for this feature.
    self.num_non_missing = 0
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
    self.num_values_summary = None

    self.has_weights = has_weights
    # Keep track of partial weighted common stats.
    if has_weights:
      # The sum of weights of all the examples with at least one value for this
      # feature.
      self.weighted_num_non_missing = 0
      # The sum of weights of all the values for this feature.
      self.weighted_total_num_values = 0

  def __iadd__(self, other: '_PartialCommonStats') -> '_PartialCommonStats':
    """Merge two partial common statistics and return the merged statistics."""
    self.num_non_missing += other.num_non_missing
    self.min_num_values = min(self.min_num_values, other.min_num_values)
    self.max_num_values = max(self.max_num_values, other.max_num_values)
    self.total_num_values += other.total_num_values

    assert self.has_weights == other.has_weights
    if self.has_weights:
      self.weighted_num_non_missing += other.weighted_num_non_missing
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
             feature_path: types.FeaturePath,
             feature_array: pa.Array,
             feature_type: types.FeatureNameStatisticsType,
             num_values_quantiles_combiner: Any,
             weights: Optional[np.ndarray] = None) -> None:
    """Update the partial common statistics using the input value."""
    # All the values in this column is null and we cannot deduce the type of
    # the feature. This is not an error as this feature might have some values
    # in other batches.
    if feature_type is None:
      return

    if self.type is None:
      self.type = feature_type
    elif self.type != feature_type:
      raise TypeError('Cannot determine the type of feature %s. '
                      'Found values of types %s and %s.' %
                      (feature_path, self.type, feature_type))

    # np.max / np.min below cannot handle empty arrays. And there's nothing
    # we can collect in this case.
    if not feature_array:
      return

    num_values = np.asarray(array_util.ListLengthsFromListArray(feature_array))
    none_mask = np.asarray(
        array_util.GetArrayNullBitmapAsByteArray(feature_array)).view(np.bool)

    self.num_non_missing += len(feature_array) - feature_array.null_count
    num_values_not_none = num_values[~none_mask]
    # We do this check to avoid failing in np.min/max with empty array.
    if num_values_not_none.size == 0:
      return
    # Use np.maximum.reduce(num_values_not_none, initial=self.max_num_values)
    # once we upgrade to numpy 1.16
    self.max_num_values = max(
        np.max(num_values_not_none), self.max_num_values)
    self.min_num_values = min(
        np.min(num_values_not_none), self.min_num_values)
    self.total_num_values += np.sum(num_values_not_none)
    self.num_values_summary = num_values_quantiles_combiner.add_input(
        self.num_values_summary, [num_values_not_none])

    if weights is not None:
      if weights.size != num_values.size:
        raise ValueError('Weight feature must not be missing.')
      self.weighted_total_num_values += np.sum(num_values * weights)
      self.weighted_num_non_missing += np.sum(weights[~none_mask])


class _PartialNumericStats(object):
  """Holds partial numeric statistics for a single feature."""

  __slots__ = ['sum', 'sum_of_squares', 'num_zeros', 'num_nan', 'min', 'max',
               'finite_min', 'finite_max', 'quantiles_summary', 'has_weights',
               'weighted_sum', 'weighted_sum_of_squares',
               'weighted_total_num_values', 'weighted_quantiles_summary']

  def __init__(self, has_weights: bool):
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
    self.min = float('inf')
    # The maximum value among all the values for this feature.
    self.max = float('-inf')
    # The minimum value among all the finite values for this feature.
    self.finite_min = float('inf')
    # The maximum value among all the finite values for this feature.
    self.finite_max = float('-inf')
    # Summary of the quantiles for the values in this feature.
    self.quantiles_summary = None

    self.has_weights = has_weights
    # Keep track of partial weighted numeric stats.
    if has_weights:
      # The weighted sum of all the values for this feature.
      self.weighted_sum = 0.0
      # The weighted sum of squares of all the values for this feature.
      self.weighted_sum_of_squares = 0.0
      # The sum of weights of all the values for this feature (
      # excluding the weights for NaN values)
      self.weighted_total_num_values = 0.0
      # Summary of the weighted quantiles for the values in this feature.
      self.weighted_quantiles_summary = None

  def __iadd__(self, other: '_PartialNumericStats') -> '_PartialNumericStats':
    """Merge two partial numeric statistics and return the merged statistics."""
    self.sum += other.sum
    self.sum_of_squares += other.sum_of_squares
    self.num_zeros += other.num_zeros
    self.num_nan += other.num_nan
    self.min = min(self.min, other.min)
    self.max = max(self.max, other.max)
    self.finite_min = min(self.finite_min, other.finite_min)
    self.finite_max = max(self.finite_max, other.finite_max)

    assert self.has_weights == other.has_weights
    if self.has_weights:
      self.weighted_sum += other.weighted_sum
      self.weighted_sum_of_squares += other.weighted_sum_of_squares
      self.weighted_total_num_values += other.weighted_total_num_values
    return self

  def update(
      self,
      feature_array: pa.Array,
      values_quantiles_combiner: Any,
      weights: Optional[np.ndarray] = None) -> None:
    """Update the partial numeric statistics using the input value."""

    # np.max / np.min below cannot handle empty arrays. And there's nothing
    # we can collect in this case.
    if not feature_array:
      return

    flattened_value_array = feature_array.flatten()
    # Note: to_numpy will fail if flattened_value_array is empty.
    if not flattened_value_array:
      return
    values = np.asarray(flattened_value_array)
    nan_mask = np.isnan(values)
    self.num_nan += np.sum(nan_mask)
    non_nan_mask = ~nan_mask
    values_no_nan = values[non_nan_mask]

    # We do this check to avoid failing in np.min/max with empty array.
    if values_no_nan.size == 0:
      return
    # This is to avoid integer overflow when computing sum or sum of squares.
    values_no_nan_as_double = values_no_nan.astype(np.float64)
    self.sum += np.sum(values_no_nan_as_double)
    self.sum_of_squares += np.sum(
        values_no_nan_as_double* values_no_nan_as_double)
    # Use np.minimum.reduce(values_no_nan, initial=self.min) once we upgrade
    # to numpy 1.16
    curr_min = np.min(values_no_nan)
    curr_max = np.max(values_no_nan)
    self.min = min(self.min, curr_min)
    self.max = max(self.max, curr_max)
    if curr_min == float('-inf') or curr_max == float('inf'):
      finite_values = values_no_nan[np.isfinite(values_no_nan)]
      if finite_values.size > 0:
        self.finite_min = min(self.finite_min, np.min(finite_values))
        self.finite_max = max(self.finite_max, np.max(finite_values))

    self.num_zeros += values_no_nan.size - np.count_nonzero(values_no_nan)
    self.quantiles_summary = values_quantiles_combiner.add_input(
        self.quantiles_summary, [values_no_nan, np.ones_like(values_no_nan)])
    if weights is not None:
      value_parent_indices = np.asarray(
          array_util.GetFlattenedArrayParentIndices(feature_array))
      flat_weights = weights[value_parent_indices]
      flat_weights_no_nan = flat_weights[non_nan_mask]
      weighted_values = flat_weights_no_nan * values_no_nan
      self.weighted_sum += np.sum(weighted_values)
      self.weighted_sum_of_squares += np.sum(weighted_values * values_no_nan)
      self.weighted_quantiles_summary = values_quantiles_combiner.add_input(
          self.weighted_quantiles_summary, [values_no_nan, flat_weights_no_nan])
      self.weighted_total_num_values += np.sum(flat_weights_no_nan)


class _PartialStringStats(object):
  """Holds partial string statistics for a single feature."""

  __slots__ = ['total_bytes_length']

  def __init__(self):
    # The total length of all the values for this feature.
    self.total_bytes_length = 0

  def __iadd__(self, other: '_PartialStringStats') -> '_PartialStringStats':
    """Merge two partial string statistics and return the merged statistics."""
    self.total_bytes_length += other.total_bytes_length
    return self

  def update(self, feature_array: pa.Array) -> None:
    """Update the partial string statistics using the input value."""
    if pa.types.is_null(feature_array.type):
      return
    # Iterate through the value array and update the partial stats.
    flattened_values_array = feature_array.flatten()
    if arrow_util.is_binary_like(flattened_values_array.type):
      # GetBinaryArrayTotalByteSize returns a Python long (to be compatible
      # with Python3). To make sure we do cheaper integer arithemetics in
      # Python2, we first convert it to int.
      self.total_bytes_length += int(array_util.GetBinaryArrayTotalByteSize(
          flattened_values_array))
    elif flattened_values_array:
      # We can only do flattened_values_array.to_numpy() when it's not empty.
      # This could be computed faster by taking log10 of the integer.
      def _len_after_conv(s):
        return len(str(s))
      self.total_bytes_length += np.sum(
          np.vectorize(_len_after_conv,
                       otypes=[np.int32])(np.asarray(flattened_values_array)))


class _PartialBytesStats(object):
  """Holds partial bytes statistics for a single feature."""

  __slots__ = ['total_num_bytes', 'min_num_bytes', 'max_num_bytes']

  def __init__(self):
    # The total number of bytes of all the values for this feature.
    self.total_num_bytes = 0
    # The minimum number of bytes among all the values for this feature.
    self.min_num_bytes = sys.maxsize
    # The maximum number of bytes among all the values for this feature.
    self.max_num_bytes = -sys.maxsize

  def __iadd__(self, other: '_PartialBytesStats') -> '_PartialBytesStats':
    """Merge two partial bytes statistics and return the merged statistics."""
    self.total_num_bytes += other.total_num_bytes
    self.min_num_bytes = min(self.min_num_bytes, other.min_num_bytes)
    self.max_num_bytes = max(self.max_num_bytes, other.max_num_bytes)
    return self

  def update(self, feature_array: pa.Array) -> None:
    """Update the partial bytes statistics using the input value."""
    if pa.types.is_null(feature_array.type):
      return
    # Iterate through the value array and update the partial stats.'
    flattened_values_array = feature_array.flatten()
    if (pa.types.is_floating(flattened_values_array.type) or
        pa.types.is_integer(flattened_values_array.type)):
      raise ValueError('Bytes stats cannot be computed on INT/FLOAT features.')
    if flattened_values_array:
      num_bytes = array_util.GetElementLengths(
          flattened_values_array).to_numpy()
      self.min_num_bytes = min(self.min_num_bytes, np.min(num_bytes))
      self.max_num_bytes = max(self.max_num_bytes, np.max(num_bytes))
      self.total_num_bytes += np.sum(num_bytes)


class _PartialBasicStats(object):
  """Holds partial statistics for a single feature."""

  __slots__ = ['common_stats', 'numeric_stats', 'string_stats', 'bytes_stats']

  def __init__(self, has_weights: bool):
    self.common_stats = _PartialCommonStats(has_weights=has_weights)
    self.numeric_stats = _PartialNumericStats(has_weights=has_weights)
    self.string_stats = _PartialStringStats()
    self.bytes_stats = _PartialBytesStats()


def _make_common_stats_proto(
    common_stats: _PartialCommonStats,
    parent_common_stats: Optional[_PartialCommonStats],
    q_combiner: quantiles_util.QuantilesCombiner,
    num_values_histogram_buckets: int,
    has_weights: bool
) -> statistics_pb2.CommonStatistics:
  """Convert the partial common stats into a CommonStatistics proto."""
  result = statistics_pb2.CommonStatistics()
  result.num_non_missing = common_stats.num_non_missing
  if parent_common_stats is not None:
    result.num_missing = (
        parent_common_stats.total_num_values - common_stats.num_non_missing)
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
        num_values_quantiles, common_stats.num_non_missing,
        num_values_histogram_buckets)
    result.num_values_histogram.CopyFrom(histogram)

  # Add weighted common stats to the proto.
  if has_weights:
    weighted_common_stats_proto = statistics_pb2.WeightedCommonStatistics(
        num_non_missing=common_stats.weighted_num_non_missing,
        tot_num_values=common_stats.weighted_total_num_values)
    if parent_common_stats is not None:
      weighted_common_stats_proto.num_missing = (
          parent_common_stats.weighted_total_num_values -
          common_stats.weighted_num_non_missing)

    if common_stats.weighted_num_non_missing > 0:
      weighted_common_stats_proto.avg_num_values = (
          common_stats.weighted_total_num_values /
          common_stats.weighted_num_non_missing)

    result.weighted_common_stats.CopyFrom(
        weighted_common_stats_proto)
  return result


def _make_numeric_stats_proto(
    numeric_stats: _PartialNumericStats,
    total_num_values: int,
    quantiles_combiner: quantiles_util.QuantilesCombiner,
    num_histogram_buckets: int,
    num_quantiles_histogram_buckets: int,
    has_weights: bool
    ) -> statistics_pb2.NumericStatistics:
  """Convert the partial numeric statistics into NumericStatistics proto."""
  result = statistics_pb2.NumericStatistics()

  if numeric_stats.num_nan > 0:
    total_num_values -= numeric_stats.num_nan

  if total_num_values == 0:
    # If we only have nan values, we only set num_nan.
    if numeric_stats.num_nan > 0:
      result.histograms.add(type=statistics_pb2.Histogram.STANDARD).num_nan = (
          numeric_stats.num_nan)
      result.histograms.add(type=statistics_pb2.Histogram.QUANTILES).num_nan = (
          numeric_stats.num_nan)
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
      quantiles, numeric_stats.finite_min, numeric_stats.finite_max,
      total_num_values, num_histogram_buckets)
  std_histogram.num_nan = numeric_stats.num_nan
  new_std_histogram = result.histograms.add()
  new_std_histogram.CopyFrom(std_histogram)

  # Construct the quantiles histogram from the quantiles and add it to the
  # numeric stats proto.
  q_histogram = quantiles_util.generate_quantiles_histogram(
      quantiles, total_num_values, num_quantiles_histogram_buckets)
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
        weighted_quantiles, numeric_stats.finite_min, numeric_stats.finite_max,
        numeric_stats.weighted_total_num_values, num_histogram_buckets)
    weighted_std_histogram.num_nan = numeric_stats.num_nan
    weighted_numeric_stats_proto.histograms.extend([weighted_std_histogram])

    # Construct the weighted quantiles histogram from the quantiles and
    # add it to the numeric stats proto.
    weighted_q_histogram = quantiles_util.generate_quantiles_histogram(
        weighted_quantiles, numeric_stats.weighted_total_num_values,
        num_quantiles_histogram_buckets)
    weighted_q_histogram.num_nan = numeric_stats.num_nan
    weighted_numeric_stats_proto.histograms.extend([weighted_q_histogram])

    result.weighted_numeric_stats.CopyFrom(
        weighted_numeric_stats_proto)
  return result


def _make_string_stats_proto(string_stats: _PartialStringStats,
                             total_num_values: int
                            ) -> statistics_pb2.StringStatistics:
  """Convert the partial string statistics into StringStatistics proto."""
  result = statistics_pb2.StringStatistics()
  if total_num_values > 0:
    result.avg_length = string_stats.total_bytes_length / total_num_values
  return result


def _make_bytes_stats_proto(bytes_stats: _PartialBytesStats,
                            total_num_values: int
                           ) -> statistics_pb2.BytesStatistics:
  """Convert the partial bytes statistics into BytesStatistics proto."""
  result = statistics_pb2.BytesStatistics()
  if total_num_values > 0:
    result.avg_num_bytes = bytes_stats.total_num_bytes / total_num_values
    result.min_num_bytes = bytes_stats.min_num_bytes
    result.max_num_bytes = bytes_stats.max_num_bytes
  return result


def _make_feature_stats_proto(
    feature_path: types.FeaturePath,
    basic_stats: _PartialBasicStats,
    parent_basic_stats: Optional[_PartialBasicStats],
    num_values_q_combiner: quantiles_util.QuantilesCombiner,
    values_q_combiner: quantiles_util.QuantilesCombiner,
    num_values_histogram_buckets: int,
    num_histogram_buckets: int,
    num_quantiles_histogram_buckets: int,
    is_bytes: bool, is_categorical: bool, has_weights: bool
) -> statistics_pb2.FeatureNameStatistics:
  """Convert the partial basic stats into a FeatureNameStatistics proto.

  Args:
    feature_path: The path of the feature.
    basic_stats: The partial basic stats associated with the feature.
    parent_basic_stats: The partial basic stats of the parent of the feature.
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
    is_bytes: A boolean indicating whether the feature is bytes.
    is_categorical: A boolean indicating whether the feature is categorical.
    has_weights: A boolean indicating whether a weight feature is specified.

  Returns:
    A statistics_pb2.FeatureNameStatistics proto.
  """
  # Create a new FeatureNameStatistics proto.
  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())
  # Set the feature type.
  # If we have a categorical feature, we preserve the type to be the original
  # INT type. Currently we don't set the type if we cannot infer it, which
  # happens when all the values are missing. We need to add an UNKNOWN type
  # to the stats proto to handle this case.
  if is_categorical:
    result.type = statistics_pb2.FeatureNameStatistics.INT
  elif is_bytes:
    result.type = statistics_pb2.FeatureNameStatistics.BYTES
  elif basic_stats.common_stats.type is None:
    # If a feature is completely missing, we assume the type to be STRING.
    result.type = statistics_pb2.FeatureNameStatistics.STRING
  else:
    result.type = basic_stats.common_stats.type

  # Construct common statistics proto.
  common_stats_proto = _make_common_stats_proto(
      basic_stats.common_stats,
      parent_basic_stats.common_stats
      if parent_basic_stats is not None else None,
      num_values_q_combiner,
      num_values_histogram_buckets, has_weights)

  # Copy the common stats into appropriate numeric/string stats.
  # If the type is not set, we currently wrap the common stats
  # within numeric stats.
  if is_bytes:
    # Construct bytes statistics proto.
    bytes_stats_proto = _make_bytes_stats_proto(
        basic_stats.bytes_stats, basic_stats.common_stats.total_num_values)
    # Add the common stats into bytes stats.
    bytes_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.bytes_stats.CopyFrom(bytes_stats_proto)
  if (is_categorical or
      result.type == statistics_pb2.FeatureNameStatistics.STRING):
    # Construct string statistics proto.
    string_stats_proto = _make_string_stats_proto(
        basic_stats.string_stats, basic_stats.common_stats.total_num_values)
    # Add the common stats into string stats.
    string_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.string_stats.CopyFrom(string_stats_proto)
  elif result.type == statistics_pb2.FeatureNameStatistics.STRUCT:
    result.struct_stats.common_stats.CopyFrom(common_stats_proto)
  elif result.type in (statistics_pb2.FeatureNameStatistics.INT,
                       statistics_pb2.FeatureNameStatistics.FLOAT):
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
    accumulator: Dict[types.FeaturePath, _PartialBasicStats]) -> None:
  """Update TFDV Beam metrics."""
  # Aggregate type specific metrics.
  metrics = {
      statistics_pb2.FeatureNameStatistics.INT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.FLOAT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.STRING: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.STRUCT: _TFDVMetrics(),
  }

  for basic_stats in six.itervalues(accumulator):
    common_stats = basic_stats.common_stats
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
  for feature_type in metrics:
    type_str = statistics_pb2.FeatureNameStatistics.Type.Name(
        feature_type).lower()
    type_metrics = metrics[feature_type]
    counter(
        constants.METRICS_NAMESPACE,
        'num_' + type_str + '_feature_values').inc(
            int(type_metrics.num_non_missing))
    if type_metrics.num_non_missing > 0:
      counter(
          constants.METRICS_NAMESPACE,
          type_str + '_feature_values_min_count').inc(
              int(type_metrics.min_value_count))
      counter(
          constants.METRICS_NAMESPACE,
          type_str + '_feature_values_max_count').inc(
              int(type_metrics.max_value_count))
      counter(
          constants.METRICS_NAMESPACE,
          type_str + '_feature_values_mean_count').inc(
              int(type_metrics.total_num_values / type_metrics.num_non_missing))


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
  """A combiner statistics generator that computes basic statistics.

  It computes common statistics for all the features, numeric statistics for
  numeric features and string statistics for string/categorical features.
  """

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name: Text = 'BasicStatsGenerator',
      schema: Optional[schema_pb2.Schema] = None,
      weight_feature: Optional[types.FeatureName] = None,
      num_values_histogram_buckets: Optional[int] = 10,
      num_histogram_buckets: Optional[int] = 10,
      num_quantiles_histogram_buckets: Optional[int] = 10,
      epsilon: Optional[float] = 0.01) -> None:
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

    self._bytes_features = set(
        schema_util.get_bytes_features(schema) if schema else [])
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
    assert num_buckets % self._num_quantiles_histogram_buckets == 0
    # Initialize quantiles combiner for histogram over feature values.
    self._values_quantiles_combiner = quantiles_util.QuantilesCombiner(
        num_buckets, epsilon, has_weights=True)

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self) -> Dict[types.FeaturePath, _PartialBasicStats]:
    return {}

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are lists representing a batch of examples) into the accumulator.
  def add_input(
      self, accumulator: Dict[types.FeaturePath, _PartialBasicStats],
      examples_table: pa.Table) -> Dict[types.FeaturePath, _PartialBasicStats]:
    for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
        examples_table,
        weight_column=self._weight_feature,
        enumerate_leaves_only=False):
      stats_for_feature = accumulator.get(feature_path)
      if stats_for_feature is None:
        stats_for_feature = _PartialBasicStats(self._weight_feature is not None)
        # Store empty summary for each of the quantiles computations.
        stats_for_feature.common_stats.num_values_summary = (
            self._num_values_quantiles_combiner.create_accumulator())
        stats_for_feature.numeric_stats.quantiles_summary = (
            self._values_quantiles_combiner.create_accumulator())
        stats_for_feature.numeric_stats.weighted_quantiles_summary = (
            self._values_quantiles_combiner.create_accumulator())
        accumulator[feature_path] = stats_for_feature

      feature_type = stats_util.get_feature_type_from_arrow_type(
          feature_path, feature_array.type)
      stats_for_feature.common_stats.update(feature_path,
                                            feature_array, feature_type,
                                            self._num_values_quantiles_combiner,
                                            weights)
      if feature_path in self._bytes_features:
        stats_for_feature.bytes_stats.update(feature_array)
      elif (feature_path in self._categorical_features or
            feature_type == statistics_pb2.FeatureNameStatistics.STRING):
        stats_for_feature.string_stats.update(feature_array)
      elif feature_type in (statistics_pb2.FeatureNameStatistics.INT,
                            statistics_pb2.FeatureNameStatistics.FLOAT):
        stats_for_feature.numeric_stats.update(feature_array,
                                               self._values_quantiles_combiner,
                                               weights)

    return accumulator

  # Merge together a list of basic common statistics.
  def merge_accumulators(
      self, accumulators: Iterable[Dict[types.FeaturePath, _PartialBasicStats]]
  ) -> Dict[types.FeaturePath, _PartialBasicStats]:
    result = {}
    num_values_summary_per_feature = collections.defaultdict(list)
    values_summary_per_feature = collections.defaultdict(list)
    if self._weight_feature:
      weighted_values_summary_per_feature = collections.defaultdict(list)

    for accumulator in accumulators:
      for feature_path, basic_stats in six.iteritems(accumulator):
        is_categorical = feature_path in self._categorical_features
        current_type = basic_stats.common_stats.type
        existing_stats = result.get(feature_path)
        if existing_stats is None:
          existing_stats = basic_stats
          result[feature_path] = basic_stats
        else:
          # Check if the types from the two partial statistics are not
          # compatible. If so, raise an error. We consider types to be
          # compatible if both types are same or one of them is None.
          left_type = existing_stats.common_stats.type
          right_type = current_type
          if (left_type is not None and right_type is not None and
              left_type != right_type):
            raise TypeError('Cannot determine the type of feature %s. '
                            'Found values of types %s and %s.' %
                            (feature_path, left_type, right_type))

          existing_stats.common_stats += basic_stats.common_stats

          if current_type is not None:
            if feature_path in self._bytes_features:
              existing_stats.bytes_stats += basic_stats.bytes_stats
            elif (is_categorical or
                  current_type == statistics_pb2.FeatureNameStatistics.STRING):
              existing_stats.string_stats += basic_stats.string_stats
            elif current_type in (statistics_pb2.FeatureNameStatistics.INT,
                                  statistics_pb2.FeatureNameStatistics.FLOAT):
              existing_stats.numeric_stats += basic_stats.numeric_stats

        # Keep track of num values quantiles summaries per feature.
        num_values_summary_per_feature[feature_path].append(
            basic_stats.common_stats.num_values_summary)

        if (current_type is not None and
            not is_categorical and
            current_type != statistics_pb2.FeatureNameStatistics.STRING):
          # Keep track of values quantile summaries per feature.
          values_summary_per_feature[feature_path].append(
              basic_stats.numeric_stats.quantiles_summary)

          # Keep track of values weighted quantile summaries per feature.
          if self._weight_feature:
            weighted_values_summary_per_feature[feature_path].append(
                basic_stats.numeric_stats.weighted_quantiles_summary)

    # Merge the summaries per feature.
    for feature_path, num_values_summaries in (
        six.iteritems(num_values_summary_per_feature)):
      result[feature_path].common_stats.num_values_summary = (
          self._num_values_quantiles_combiner.merge_accumulators(
              num_values_summaries))

    # Merge the values quantiles summaries per feature.
    for feature_path, quantiles_summaries in (
        six.iteritems(values_summary_per_feature)):
      result[feature_path].numeric_stats.quantiles_summary = (
          self._values_quantiles_combiner.merge_accumulators(
              quantiles_summaries))

    # Merge the values weighted quantiles summaries per feature.
    if self._weight_feature:
      for feature_path, weighted_quantiles_summaries in (
          six.iteritems(weighted_values_summary_per_feature)):
        result[feature_path].numeric_stats.weighted_quantiles_summary = (
            self._values_quantiles_combiner.merge_accumulators(
                weighted_quantiles_summaries))

    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator: Dict[types.FeaturePath, _PartialBasicStats]
                    ) -> statistics_pb2.DatasetFeatureStatistics:
    # Update TFDV telemetry.
    _update_tfdv_telemetry(accumulator)

    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_path, basic_stats in six.iteritems(accumulator):
      # Construct the FeatureNameStatistics proto from the partial
      # basic stats.
      feature_stats_proto = _make_feature_stats_proto(
          feature_path,
          basic_stats,
          accumulator.get(feature_path.parent()),
          self._num_values_quantiles_combiner, self._values_quantiles_combiner,
          self._num_values_histogram_buckets, self._num_histogram_buckets,
          self._num_quantiles_histogram_buckets,
          feature_path in self._bytes_features,
          feature_path in self._categorical_features,
          self._weight_feature is not None)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
