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
"""Module that computes the basic statistics for a collection of RecordBatches.

It computes common statistics for each column in the
RecordBatches. And if a column is of struct type (i.e. it consists of
child arrays), it recursively computes common statistics for each child array.
Common statistics are about the presence and valency of the column. The column
is assumed to be at least 1-nested (i.e. a list<primitive>, which means the
value of the column at each row of a RecordBatch is a list of primitives) and
could be more deeply nested (i.e. a of list<list<...>> type). We compute
presence and valency stats for each nest level, relative to its outer level.
Note that the presence and valency of the outermost nest level is relative to a
RecordBatch row. The following presence and valency stats are computed:
  * Number of missing (value == null) elements.
    Note:
      - For the out-most level, this number means number of rows that does not
        have values at this column. And this number is actually not computed
        here because we need num_rows (or num_examples) to compute it and that
        is not tracked here. See stats_impl.py.
      - An empty list is distinguished from a null and is not counted as
        missing.
  * Number of present elements.
  * Maximum valency of elements.
  * Minimum valency of elements. Note that the valency of an empty list is 0
    but a null element has no valency (does not contribute to the result).
  * Total number of values (sum of valency).
  * Quantiles histogram over the valency.

It computes the following statistics for each numeric column (or leaf numeric
array contained in some struct column):
  - Mean of the values.
  - Standard deviation of the values.
  - Median of the values.
  - Number of values that equal zero.
  - Minimum value.
  - Maximum value.
  - Standard histogram over the values.
  - Quantiles histogram over the values.

We compute the following statistics for each string column (or leaf numeric
array contained in some struct column):
  - Average length of the values for this feature.
  - Number of non-UTF8 values (only for string or binary typed columns)
"""

import collections
import itertools
import math
import sys
from typing import Any, Callable, Iterable, List, Mapping, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import feature_partition_util
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import top_k_uniques_stats_util
from tensorflow_data_validation.utils import variance_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl import sketches
from tfx_bsl.arrow import array_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PresenceAndValencyStats(object):
  """Contains stats on presence and valency of a feature."""
  __slots__ = [
      'num_non_missing', 'min_num_values', 'max_num_values', 'total_num_values',
      'weighted_total_num_values', 'weighted_num_non_missing',
      'num_values_summary']

  def __init__(
      self,
      make_quantiles_sketch_fn: Callable[
          [], Optional[sketches.QuantilesSketch]
      ],
  ):
    # The number of examples with at least one value for this feature.
    self.num_non_missing = 0
    # The minimum number of values in a single example for this feature.
    self.min_num_values = sys.maxsize
    # The maximum number of values in a single example for this feature.
    self.max_num_values = 0
    # The total number of values for this feature.
    self.total_num_values = 0
    # The sum of weights of all the values for this feature.
    self.weighted_total_num_values = 0
    # The sum of weights of all the examples with at least one value for this
    # feature.
    self.weighted_num_non_missing = 0
    self.num_values_summary = make_quantiles_sketch_fn()

  def merge_with(self, other: '_PresenceAndValencyStats') -> None:
    self.num_non_missing += other.num_non_missing
    self.min_num_values = min(self.min_num_values, other.min_num_values)
    self.max_num_values = max(self.max_num_values, other.max_num_values)
    self.total_num_values += other.total_num_values
    self.weighted_num_non_missing += other.weighted_num_non_missing
    self.weighted_total_num_values += other.weighted_total_num_values
    if self.num_values_summary is not None:
      self.num_values_summary.Merge(other.num_values_summary)

  def update(self, feature_array: pa.Array, presence_mask: np.ndarray,
             num_values: np.ndarray, num_values_not_none: np.ndarray,
             weights: Optional[np.ndarray]) -> None:
    """Updates the stats with a feature array."""
    self.num_non_missing += len(feature_array) - feature_array.null_count

    self.max_num_values = np.maximum.reduce(
        num_values_not_none, initial=self.max_num_values)
    self.min_num_values = np.minimum.reduce(num_values_not_none,
                                            initial=self.min_num_values)
    self.total_num_values += np.sum(num_values_not_none)
    # num values tends to vary little. pre-aggregate them by values would help
    # reduce the cost in AddValues().
    num_values_grouped = pa.array(num_values_not_none).value_counts()

    if self.num_values_summary is not None:
      self.num_values_summary.AddValues(
          num_values_grouped.field(0), num_values_grouped.field(1)
      )

    if weights is not None:
      if weights.size != num_values.size:
        raise ValueError('Weight feature must not be missing.')
      self.weighted_total_num_values += np.sum(num_values * weights)
      self.weighted_num_non_missing += np.sum(weights[presence_mask])


class _PartialCommonStats(object):
  """Holds partial common statistics for a single feature."""

  __slots__ = ['type', 'has_weights', 'presence_and_valency_stats']

  def __init__(self, has_weights: bool):
    # Type of the feature.
    self.type = None  # type: Optional[types.FeatureNameStatisticsType]
    # This will be a List[_PresenceAndValencyStats] once `update()` is called.
    # presence_and_valency_stats[i] contains the stats at nest level i.
    # for example: a feature of type list<list<int>> will have
    # presence_and_valency_stats of length 2. presence_and_valency_stats[0]
    # contains the stats about the outer list.
    self.presence_and_valency_stats = None  # type: Optional[List[Any]]
    self.has_weights = has_weights

  def merge_with(
      self, feature_path: types.FeaturePath, other: '_PartialCommonStats'
      ) -> None:
    """Merges two partial common statistics and return the merged statistics.

    Note that this DOES NOT merge self.num_values_summaries. See
    `merge_num_values_summaries()`.

    Args:
      feature_path: path of the feature that `self` is associated with.
      other: a _PartialCommonStats to merge with.
    """

    assert self.has_weights == other.has_weights
    if self.presence_and_valency_stats is None:
      self.presence_and_valency_stats = other.presence_and_valency_stats
    elif other.presence_and_valency_stats is not None:
      this_nest_level = len(self.presence_and_valency_stats)
      other_nest_level = len(other.presence_and_valency_stats)
      if this_nest_level != other_nest_level:
        raise ValueError(
            'Unable to merge common stats with different nest levels for '
            'feature {}: {} vs {}'.format(
                feature_path, this_nest_level, other_nest_level))
      for self_stats, other_stats in zip(self.presence_and_valency_stats,
                                         other.presence_and_valency_stats):
        self_stats.merge_with(other_stats)

    # Set the type of the merged common stats.
    # Case 1: Both the types are None. We set the merged type to be None.
    # Case 2: One of two types is None. We set the merged type to be the type
    # which is not None. For example, if left.type=FLOAT and right.type=None,
    # we set the merged type to be FLOAT.
    # Case 3: Both the types are same (and not None), we set the merged type to
    # be the same type.
    if self.type is None:
      self.type = other.type

  def update(
      self,
      feature_path: types.FeaturePath,
      feature_array: pa.Array,
      feature_type: types.FeatureNameStatisticsType,
      make_quantiles_sketch_fn: Callable[
          [], Optional[sketches.QuantilesSketch]
      ],
      weights: Optional[np.ndarray] = None,
  ) -> None:
    """Update the partial common statistics using the input value."""
    if self.type is None:
      self.type = feature_type  # pytype: disable=annotation-type-mismatch
    elif feature_type is not None and self.type != feature_type:
      raise TypeError('Cannot determine the type of feature %s. '
                      'Found values of types %s and %s.' %
                      (feature_path, self.type, feature_type))

    nest_level = arrow_util.get_nest_level(feature_array.type)
    if self.presence_and_valency_stats is None:
      self.presence_and_valency_stats = [
          _PresenceAndValencyStats(make_quantiles_sketch_fn)
          for _ in range(nest_level)
      ]
    elif nest_level != len(self.presence_and_valency_stats):
      raise ValueError('Inconsistent nestedness in feature {}: {} vs {}'.format(
          feature_path, nest_level, len(self.presence_and_valency_stats)))

    # And there's nothing we can collect in this case.
    if not feature_array:
      return

    level = 0
    while array_util.is_list_like(feature_array.type):
      presence_mask = ~np.asarray(
          array_util.GetArrayNullBitmapAsByteArray(feature_array)).view(bool)
      num_values = np.asarray(
          array_util.ListLengthsFromListArray(feature_array))
      num_values_not_none = num_values[presence_mask]
      self.presence_and_valency_stats[level].update(feature_array,
                                                    presence_mask, num_values,
                                                    num_values_not_none,
                                                    weights)
      flattened = feature_array.flatten()
      if weights is not None:
        parent_indices = array_util.GetFlattenedArrayParentIndices(
            feature_array).to_numpy()
        weights = weights[parent_indices]
      feature_array = flattened
      level += 1


class _PartialNumericStats(object):
  """Holds partial numeric statistics for a single feature."""

  __slots__ = [
      'num_zeros', 'num_nan', 'min', 'max', 'finite_min', 'finite_max',
      'pos_inf_count', 'pos_inf_weighted_count', 'quantiles_summary',
      'has_weights', 'weighted_quantiles_summary', 'mean_var_accumulator',
      'weighted_mean_var_accumulator'
  ]

  def __init__(
      self,
      has_weights: bool,
      make_quantiles_sketch_fn: Callable[
          [], Optional[sketches.QuantilesSketch]
      ],
  ):
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
    # The total count of positive inf values.
    self.pos_inf_count = 0.0
    # The total weight sum of positive inf values, if weights are used.
    self.pos_inf_weighted_count = 0.0
    # Summary of the quantiles for the values in this feature.
    self.quantiles_summary = make_quantiles_sketch_fn()

    self.has_weights = has_weights

    # Accumulator for mean and variance.
    self.mean_var_accumulator = variance_util.MeanVarAccumulator()
    # Keep track of partial weighted numeric stats.
    if has_weights:
      # Summary of the weighted quantiles for the values in this feature.
      self.weighted_quantiles_summary = make_quantiles_sketch_fn()
      # Accumulator for weighted mean and weighted variance.
      self.weighted_mean_var_accumulator = (
          variance_util.WeightedMeanVarAccumulator())
    else:
      self.weighted_mean_var_accumulator = None

  def __iadd__(self, other: '_PartialNumericStats') -> '_PartialNumericStats':
    """Merge two partial numeric statistics and return the merged statistics."""
    self.num_zeros += other.num_zeros
    self.num_nan += other.num_nan
    self.min = min(self.min, other.min)
    self.max = max(self.max, other.max)
    self.finite_min = min(self.finite_min, other.finite_min)
    self.finite_max = max(self.finite_max, other.finite_max)
    self.pos_inf_count += other.pos_inf_count
    self.pos_inf_weighted_count += other.pos_inf_weighted_count
    if self.quantiles_summary is not None:
      self.quantiles_summary.Merge(other.quantiles_summary)
    self.mean_var_accumulator.merge(other.mean_var_accumulator)
    assert self.has_weights == other.has_weights
    if self.has_weights:
      if self.weighted_quantiles_summary is not None:
        self.weighted_quantiles_summary.Merge(other.weighted_quantiles_summary)
      assert self.weighted_mean_var_accumulator is not None
      self.weighted_mean_var_accumulator.merge(
          other.weighted_mean_var_accumulator)
    return self

  def update(
      self,
      feature_array: pa.Array,
      weights: Optional[np.ndarray] = None) -> None:
    """Update the partial numeric statistics using the input value."""

    # np.max / np.min below cannot handle empty arrays. And there's nothing
    # we can collect in this case.
    if not feature_array:
      return

    flattened_value_array, value_parent_indices = array_util.flatten_nested(
        feature_array, weights is not None)
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
    self.mean_var_accumulator.update(values_no_nan_as_double)
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
    else:
      self.finite_min = min(self.finite_min, curr_min)
      self.finite_max = max(self.finite_max, curr_max)
    self.pos_inf_count += np.isposinf(values_no_nan).sum()
    self.num_zeros += values_no_nan.size - np.count_nonzero(values_no_nan)
    if self.quantiles_summary is not None:
      self.quantiles_summary.AddValues(pa.array(values_no_nan))
    if weights is not None:
      flat_weights = weights[value_parent_indices]
      flat_weights_no_nan = flat_weights[non_nan_mask]
      assert self.weighted_mean_var_accumulator is not None
      self.weighted_mean_var_accumulator.update(values_no_nan_as_double,
                                                flat_weights_no_nan)
      if self.weighted_quantiles_summary is not None:
        self.weighted_quantiles_summary.AddValues(
            pa.array(values_no_nan), pa.array(flat_weights_no_nan)
        )
      self.pos_inf_weighted_count += flat_weights_no_nan[np.isposinf(
          values_no_nan)].sum()


class _PartialStringStats(object):
  """Holds partial string statistics for a single feature."""

  __slots__ = ['total_bytes_length', 'invalid_utf8_count']

  def __init__(self):
    # The total length of all the values for this feature.
    self.total_bytes_length = 0
    # The count of invalid utf-8 strings (in flattened arrays).
    self.invalid_utf8_count = 0

  def __iadd__(self, other: '_PartialStringStats') -> '_PartialStringStats':
    """Merge two partial string statistics and return the merged statistics."""
    self.total_bytes_length += other.total_bytes_length
    self.invalid_utf8_count += other.invalid_utf8_count
    return self

  def update(self, feature_array: pa.Array) -> None:
    """Update the partial string statistics using the input value."""
    if pa.types.is_null(feature_array.type):
      return
    # Iterate through the value array and update the partial stats.
    flattened_values_array, _ = array_util.flatten_nested(feature_array)
    if arrow_util.is_binary_like(flattened_values_array.type):
      # GetBinaryArrayTotalByteSize returns a Python long (to be compatible
      # with Python3). To make sure we do cheaper integer arithemetics in
      # Python2, we first convert it to int.
      self.total_bytes_length += int(array_util.GetBinaryArrayTotalByteSize(
          flattened_values_array))
      self.invalid_utf8_count += array_util.CountInvalidUTF8(
          flattened_values_array)
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
    flattened_values_array, _ = array_util.flatten_nested(feature_array)
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

  def __init__(
      self,
      has_weights: bool,
      make_quantiles_sketch_fn: Callable[
          [], Optional[sketches.QuantilesSketch]
      ],
  ):
    self.common_stats = _PartialCommonStats(has_weights=has_weights)
    self.numeric_stats = _PartialNumericStats(
        has_weights=has_weights,
        make_quantiles_sketch_fn=make_quantiles_sketch_fn)
    self.string_stats = _PartialStringStats()
    self.bytes_stats = _PartialBytesStats()


def _make_presence_and_valency_stats_protos(
    parent_presence_and_valency: Optional[_PresenceAndValencyStats],
    presence_and_valency: List[_PresenceAndValencyStats],
    num_examples: int,
) -> List[statistics_pb2.PresenceAndValencyStatistics]:
  """Converts presence and valency stats to corresponding protos."""
  result = []
  # The top-level non-missing is computed by
  # num_examples - top_level.num_non_missing (outside BasicStatsGenerator as
  # num_examples cannot be computed here). For all other levels,
  # it's previous_level.total_num_values - this_level.num_non_missing.
  for prev_s, s in zip(
      itertools.chain([parent_presence_and_valency], presence_and_valency),
      presence_and_valency):
    proto = statistics_pb2.PresenceAndValencyStatistics()
    if prev_s is not None:
      proto.num_missing = (prev_s.total_num_values - s.num_non_missing)
    else:
      proto.num_missing = num_examples - s.num_non_missing
    proto.num_non_missing = s.num_non_missing
    if s.num_non_missing > 0:
      proto.min_num_values = s.min_num_values
      proto.max_num_values = s.max_num_values
      proto.tot_num_values = s.total_num_values
    result.append(proto)
  return result


def _make_weighted_presence_and_valency_stats_protos(
    parent_presence_and_valency: Optional[_PresenceAndValencyStats],
    presence_and_valency: List[_PresenceAndValencyStats],
    weighted_num_examples: int,
) -> List[statistics_pb2.WeightedCommonStatistics]:
  """Converts weighted presence and valency stats to corresponding protos."""
  result = []
  # The top-level non-missing is computed by
  # weighted_num_examples - top_level.weighted_num_non_missing (outside
  # BasicStatsGenerator as num_examples cannot be computed here).
  # For all other levels,
  # it's (previous_level.weighted_total_num_values -
  # this_level.weighted_num_non_missing).
  for prev_s, s in zip(
      itertools.chain([parent_presence_and_valency], presence_and_valency),
      presence_and_valency):
    proto = statistics_pb2.WeightedCommonStatistics()
    if prev_s is not None:
      proto.num_missing = (
          prev_s.weighted_total_num_values - s.weighted_num_non_missing)
    else:
      proto.num_missing = weighted_num_examples - s.weighted_num_non_missing
    proto.num_non_missing = s.weighted_num_non_missing
    proto.tot_num_values = s.weighted_total_num_values
    if s.weighted_num_non_missing > 0:
      proto.avg_num_values = (
          s.weighted_total_num_values / s.weighted_num_non_missing)
    result.append(proto)
  return result


def _make_common_stats_proto(
    common_stats: _PartialCommonStats,
    parent_common_stats: Optional[_PartialCommonStats],
    make_quantiles_sketch_fn: Callable[[], sketches.QuantilesSketch],
    num_values_histogram_buckets: int,
    has_weights: bool,
    num_examples: int,
    weighted_num_examples: int,
) -> statistics_pb2.CommonStatistics:
  """Convert the partial common stats into a CommonStatistics proto."""
  result = statistics_pb2.CommonStatistics()
  parent_presence_and_valency = None
  if parent_common_stats is not None:
    parent_presence_and_valency = (
        _PresenceAndValencyStats(make_quantiles_sketch_fn)
        if parent_common_stats.presence_and_valency_stats is None else
        parent_common_stats.presence_and_valency_stats[-1])

  presence_and_valency_stats = common_stats.presence_and_valency_stats
  # the CommonStatistics already contains the presence and valency
  # for a 1-nested feature.
  if (presence_and_valency_stats is not None and
      len(presence_and_valency_stats) > 1):
    result.presence_and_valency_stats.extend(
        _make_presence_and_valency_stats_protos(parent_presence_and_valency,
                                                presence_and_valency_stats,
                                                num_examples))
    if has_weights:
      result.weighted_presence_and_valency_stats.extend(
          _make_weighted_presence_and_valency_stats_protos(
              parent_presence_and_valency,
              common_stats.presence_and_valency_stats, weighted_num_examples))

  top_level_presence_and_valency = (
      _PresenceAndValencyStats(make_quantiles_sketch_fn)
      if common_stats.presence_and_valency_stats is None else
      common_stats.presence_and_valency_stats[0])
  result.num_non_missing = top_level_presence_and_valency.num_non_missing
  if parent_presence_and_valency is not None:
    result.num_missing = (
        parent_presence_and_valency.total_num_values -
        top_level_presence_and_valency.num_non_missing)
  else:
    result.num_missing = (
        num_examples - top_level_presence_and_valency.num_non_missing)
  result.tot_num_values = top_level_presence_and_valency.total_num_values

  # TODO(b/79685042): Need to decide on what is the expected values for
  # statistics like min_num_values, max_num_values, avg_num_values, when
  # all the values for the feature are missing.
  if top_level_presence_and_valency.num_non_missing > 0:
    result.min_num_values = top_level_presence_and_valency.min_num_values
    result.max_num_values = top_level_presence_and_valency.max_num_values
    result.avg_num_values = (
        top_level_presence_and_valency.total_num_values /
        top_level_presence_and_valency.num_non_missing)

    if top_level_presence_and_valency.num_values_summary is not None:

      # Add num_values_histogram to the common stats proto.
      num_values_quantiles, num_values_counts = (
          _get_quantiles_counts(
              top_level_presence_and_valency.num_values_summary,
              num_values_histogram_buckets))
      histogram = quantiles_util.generate_quantiles_histogram(
          num_values_quantiles, num_values_counts)
      result.num_values_histogram.CopyFrom(histogram)

  # Add weighted common stats to the proto.
  if has_weights:
    weighted_common_stats_proto = statistics_pb2.WeightedCommonStatistics(
        num_non_missing=top_level_presence_and_valency.weighted_num_non_missing,
        tot_num_values=top_level_presence_and_valency.weighted_total_num_values)
    if parent_presence_and_valency is not None:
      weighted_common_stats_proto.num_missing = (
          parent_presence_and_valency.weighted_total_num_values -
          top_level_presence_and_valency.weighted_num_non_missing)
    else:
      weighted_common_stats_proto.num_missing = (
          weighted_num_examples -
          top_level_presence_and_valency.weighted_num_non_missing)

    if top_level_presence_and_valency.weighted_num_non_missing > 0:
      weighted_common_stats_proto.avg_num_values = (
          top_level_presence_and_valency.weighted_total_num_values /
          top_level_presence_and_valency.weighted_num_non_missing)

    result.weighted_common_stats.CopyFrom(
        weighted_common_stats_proto)
  return result


def _get_quantiles_counts(qs: sketches.QuantilesSketch,
                          num_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
  quantiles, counts = qs.GetQuantilesAndCumulativeWeights(num_buckets)
  quantiles = quantiles.flatten().to_numpy(zero_copy_only=False)
  counts = counts.flatten().to_numpy(zero_copy_only=False)
  return quantiles, counts


def _make_numeric_stats_proto(
    numeric_stats: _PartialNumericStats,
    total_num_values: int,
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

  result.mean = float(numeric_stats.mean_var_accumulator.mean)
  result.std_dev = math.sqrt(
      max(0, numeric_stats.mean_var_accumulator.variance))
  result.num_zeros = numeric_stats.num_zeros
  result.min = float(numeric_stats.min)
  result.max = float(numeric_stats.max)

  # Extract the quantiles from the summary.
  if numeric_stats.quantiles_summary is not None:
    # Construct the equi-width histogram from the quantiles and add it to the
    # numeric stats proto.
    quantiles, counts = _get_quantiles_counts(
        numeric_stats.quantiles_summary,
        _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM
        * num_quantiles_histogram_buckets,
    )

    # Find the median from the quantiles and update the numeric stats proto.
    result.median = float(quantiles_util.find_median(quantiles))

    std_histogram = quantiles_util.generate_equi_width_histogram(
        quantiles=quantiles,
        cumulative_counts=counts,
        finite_min=numeric_stats.finite_min,
        finite_max=numeric_stats.finite_max,
        num_buckets=num_histogram_buckets,
        num_pos_inf=numeric_stats.pos_inf_count,
    )
    std_histogram.num_nan = numeric_stats.num_nan
    new_std_histogram = result.histograms.add()
    new_std_histogram.CopyFrom(std_histogram)

    # Construct the quantiles histogram from the quantiles and add it to the
    # numeric stats proto.
    q_histogram = quantiles_util.generate_quantiles_histogram(
        *quantiles_util.rebin_quantiles(
            quantiles, counts, _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM
        )
    )
    q_histogram.num_nan = numeric_stats.num_nan
    new_q_histogram = result.histograms.add()
    new_q_histogram.CopyFrom(q_histogram)
  else:
    result.median = np.nan

  # Add weighted numeric stats to the proto.
  if has_weights:
    assert numeric_stats.weighted_mean_var_accumulator is not None
    weighted_numeric_stats_proto = statistics_pb2.WeightedNumericStatistics()
    weighted_mean = numeric_stats.weighted_mean_var_accumulator.mean
    weighted_variance = max(
        0, numeric_stats.weighted_mean_var_accumulator.variance)
    weighted_numeric_stats_proto.mean = weighted_mean
    weighted_numeric_stats_proto.std_dev = math.sqrt(weighted_variance)

    # Extract the weighted quantiles from the summary.
    if numeric_stats.weighted_quantiles_summary is not None:

      weighted_quantiles, weighted_counts = _get_quantiles_counts(
          numeric_stats.weighted_quantiles_summary,
          _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM
          * num_quantiles_histogram_buckets,
      )

      # Find the weighted median from the quantiles and update the proto.
      weighted_numeric_stats_proto.median = float(
          quantiles_util.find_median(weighted_quantiles)
      )

      # Construct the weighted equi-width histogram from the quantiles and
      # add it to the numeric stats proto.
      weighted_std_histogram = quantiles_util.generate_equi_width_histogram(
          quantiles=weighted_quantiles,
          cumulative_counts=weighted_counts,
          finite_min=numeric_stats.finite_min,
          finite_max=numeric_stats.finite_max,
          num_buckets=num_histogram_buckets,
          num_pos_inf=numeric_stats.pos_inf_weighted_count,
      )
      weighted_std_histogram.num_nan = numeric_stats.num_nan
      weighted_numeric_stats_proto.histograms.extend([weighted_std_histogram])

      # Construct the weighted quantiles histogram from the quantiles and
      # add it to the numeric stats proto.

      weighted_q_histogram = quantiles_util.generate_quantiles_histogram(
          *quantiles_util.rebin_quantiles(
              weighted_quantiles,
              weighted_counts,
              _NUM_QUANTILES_FACTOR_FOR_STD_HISTOGRAM,
          )
      )
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
  result.invalid_utf8_count = string_stats.invalid_utf8_count
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
    result.max_num_bytes_int = bytes_stats.max_num_bytes
  return result


def _make_num_values_custom_stats_proto(
    common_stats: _PartialCommonStats,
    num_histogram_buckets: int,
    ) -> List[statistics_pb2.CustomStatistic]:
  """Returns a list of CustomStatistic protos that contains histograms.

  Those histograms captures the distribution of number of values at each
  nest level.

  It will only create histograms for nest levels greater than 1. Because
  the histogram of nest level 1 is already in
  CommonStatistics.num_values_histogram.

  Args:
    common_stats: a _PartialCommonStats.
    num_histogram_buckets: number of buckets in the histogram.
  Returns:
    a (potentially empty) list of statistics_pb2.CustomStatistic.
  """
  result = []
  if common_stats.type is None:
    return result
  presence_and_valency_stats = common_stats.presence_and_valency_stats
  if presence_and_valency_stats is None:
    return result

  # The top level histogram is included in CommonStats -- skip.
  for level, presence_and_valency in zip(
      itertools.count(2), presence_and_valency_stats[1:]):
    if presence_and_valency.num_values_summary is None:
      continue
    num_values_quantiles, num_values_counts = (
        _get_quantiles_counts(presence_and_valency.num_values_summary,
                              num_histogram_buckets))

    histogram = quantiles_util.generate_quantiles_histogram(
        num_values_quantiles, num_values_counts)
    proto = statistics_pb2.CustomStatistic()
    proto.name = 'level_{}_value_list_length_quantiles'.format(level)
    proto.histogram.CopyFrom(histogram)
    result.append(proto)

    standard_histogram = quantiles_util.generate_equi_width_histogram(
        quantiles=num_values_quantiles,
        cumulative_counts=num_values_counts,
        finite_min=presence_and_valency.min_num_values,
        finite_max=presence_and_valency.max_num_values,
        num_buckets=num_histogram_buckets,
        num_pos_inf=0,
    )
    proto = statistics_pb2.CustomStatistic()
    proto.name = 'level_{}_value_list_length_standard'.format(level)
    proto.histogram.CopyFrom(standard_histogram)
    result.append(proto)
  return result


def _make_feature_stats_proto(
    feature_path: types.FeaturePath, basic_stats: _PartialBasicStats,
    parent_basic_stats: Optional[_PartialBasicStats],
    make_quantiles_sketch_fn: Callable[[], sketches.QuantilesSketch],
    num_values_histogram_buckets: int, num_histogram_buckets: int,
    num_quantiles_histogram_buckets: int, is_bytes: bool,
    categorical_numeric_types: Mapping[types.FeaturePath,
                                       'schema_pb2.FeatureType'],
    has_weights: bool, num_examples: int,
    weighted_num_examples: int) -> statistics_pb2.FeatureNameStatistics:
  """Convert the partial basic stats into a FeatureNameStatistics proto.

  Args:
    feature_path: The path of the feature.
    basic_stats: The partial basic stats associated with the feature.
    parent_basic_stats: The partial basic stats of the parent of the feature.
    make_quantiles_sketch_fn: A callable to create a quantiles sketch.
    num_values_histogram_buckets: Number of buckets in the quantiles
        histogram for the number of values per feature.
    num_histogram_buckets: Number of buckets in a standard
        NumericStatistics.histogram with equal-width buckets.
    num_quantiles_histogram_buckets: Number of buckets in a
        quantiles NumericStatistics.histogram.
    is_bytes: A boolean indicating whether the feature is bytes.
    categorical_numeric_types: A mapping from feature path to type derived from
        the schema.
    has_weights: A boolean indicating whether a weight feature is specified.
    num_examples: The global (across feature) number of examples.
    weighted_num_examples: The global (across feature) weighted number of
      examples.

  Returns:
    A statistics_pb2.FeatureNameStatistics proto.
  """
  # Create a new FeatureNameStatistics proto.
  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())
  # Set the feature type.
  inferred_type = basic_stats.common_stats.type
  if inferred_type is not None:
    # The user claims the feature to be BYTES. Only trust them if the inferred
    # type is STRING (which means the actual data is in strings/bytes). We
    # never infer BYTES.
    if (is_bytes and
        inferred_type == statistics_pb2.FeatureNameStatistics.STRING):
      result.type = statistics_pb2.FeatureNameStatistics.BYTES
    else:
      result.type = inferred_type
  # The inferred type being None means we don't see any value for this feature.
  # We trust user's claim.
  elif is_bytes:
    result.type = statistics_pb2.FeatureNameStatistics.BYTES
  else:
    # We don't have an "unknown" type, so default to STRING here.
    result.type = statistics_pb2.FeatureNameStatistics.STRING

  # Construct common statistics proto.
  common_stats_proto = _make_common_stats_proto(
      basic_stats.common_stats, parent_basic_stats.common_stats
      if parent_basic_stats is not None else None, make_quantiles_sketch_fn,
      num_values_histogram_buckets, has_weights, num_examples,
      weighted_num_examples)

  # this is the total number of values at the leaf level.
  total_num_values = (
      0 if basic_stats.common_stats.presence_and_valency_stats is None else
      basic_stats.common_stats.presence_and_valency_stats[-1].total_num_values)

  # Copy the common stats into appropriate numeric/string stats.
  # If the type is not set, we currently wrap the common stats
  # within numeric stats.
  if result.type == statistics_pb2.FeatureNameStatistics.BYTES:
    # Construct bytes statistics proto.
    bytes_stats_proto = _make_bytes_stats_proto(
        basic_stats.bytes_stats, common_stats_proto.tot_num_values)
    # Add the common stats into bytes stats.
    bytes_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.bytes_stats.CopyFrom(bytes_stats_proto)
  # TODO(b/187054148): Update to allow FLOAT
  if (result.type == statistics_pb2.FeatureNameStatistics.STRING or
      top_k_uniques_stats_util.output_categorical_numeric(
          categorical_numeric_types, feature_path, result.type)):
    # Construct string statistics proto.
    string_stats_proto = _make_string_stats_proto(basic_stats.string_stats,
                                                  total_num_values)
    # Add the common stats into string stats.
    string_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.string_stats.CopyFrom(string_stats_proto)
  elif result.type == statistics_pb2.FeatureNameStatistics.STRUCT:
    result.struct_stats.common_stats.CopyFrom(common_stats_proto)
  elif result.type in (statistics_pb2.FeatureNameStatistics.INT,
                       statistics_pb2.FeatureNameStatistics.FLOAT):
    # Construct numeric statistics proto.
    numeric_stats_proto = _make_numeric_stats_proto(
        basic_stats.numeric_stats, total_num_values,
        num_histogram_buckets, num_quantiles_histogram_buckets, has_weights)
    # Add the common stats into numeric stats.
    numeric_stats_proto.common_stats.CopyFrom(common_stats_proto)
    result.num_stats.CopyFrom(numeric_stats_proto)

  result.custom_stats.extend(_make_num_values_custom_stats_proto(
      basic_stats.common_stats,
      num_values_histogram_buckets))
  return result


# Named tuple containing TFDV metrics.
_TFDVMetrics = collections.namedtuple(
    '_TFDVMetrics', ['num_non_missing', 'min_value_count',
                     'max_value_count', 'total_num_values'])
_TFDVMetrics.__new__.__defaults__ = (0, sys.maxsize, 0, 0)


def _update_tfdv_telemetry(accumulator: '_BasicAcctype') -> None:
  """Update TFDV Beam metrics."""
  # Aggregate type specific metrics.
  metrics = {
      statistics_pb2.FeatureNameStatistics.INT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.FLOAT: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.STRING: _TFDVMetrics(),
      statistics_pb2.FeatureNameStatistics.STRUCT: _TFDVMetrics(),
  }

  for basic_stats in accumulator.values():
    common_stats = basic_stats.common_stats
    if common_stats.type is None:
      continue
    # Take the leaf level stats.
    presence_and_valency = (
        _PresenceAndValencyStats(lambda: None)
        if common_stats.presence_and_valency_stats is None else
        common_stats.presence_and_valency_stats[-1])
    # Update type specific metrics.
    type_metrics = metrics[common_stats.type]
    num_non_missing = (type_metrics.num_non_missing +
                       presence_and_valency.num_non_missing)
    min_value_count = min(type_metrics.min_value_count,
                          presence_and_valency.min_num_values)
    max_value_count = max(type_metrics.max_value_count,
                          presence_and_valency.max_num_values)
    total_num_values = (type_metrics.total_num_values +
                        presence_and_valency.total_num_values)
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


class _BasicAcctype(collections.abc.MutableMapping):
  """Maintains per-feature state and example counts.

  This class may be accessed as a dict.
  """

  def __init__(self):
    self._dict = {}
    self.num_examples = 0
    self.weighted_num_examples = 0

  def __getitem__(self, key: types.FeaturePath) -> _PartialBasicStats:
    return self._dict[key]

  def __setitem__(self, key: types.FeaturePath, value: _PartialBasicStats):
    self._dict[key] = value

  def __delitem__(self, key: types.FeaturePath):
    del self._dict[key]

  def __iter__(self):
    return iter(self._dict)

  def __len__(self):
    return len(self._dict)


# TODO(b/79685042): Currently the stats generator operates on all features as
# a batch (or a subset of features with optional partitioning enabled).
# Because each feature is actually processed independently, we should
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
      example_weight_map: ExampleWeightMap = ExampleWeightMap(),
      num_values_histogram_buckets: Optional[int] = 10,
      num_histogram_buckets: Optional[int] = 10,
      num_quantiles_histogram_buckets: Optional[int] = 10,
      epsilon: Optional[float] = 0.01,
      feature_config: Optional[types.PerFeatureStatsConfig] = None,
  ) -> None:
    """Initializes basic statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
        corresponding weight column.
      num_values_histogram_buckets: An optional number of buckets in a quantiles
        histogram for the number of values per Feature, which is stored in
        CommonStatistics.num_values_histogram.
      num_histogram_buckets: An optional number of buckets in a standard
        NumericStatistics.histogram with equal-width buckets.
      num_quantiles_histogram_buckets: An optional number of buckets in a
        quantiles NumericStatistics.histogram.
      epsilon: An optional error tolerance for the computation of quantiles,
        typically a small fraction close to zero (e.g. 0.01). Higher values of
        epsilon increase the quantile approximation, and hence result in more
        unequal buckets, but could improve performance, and resource
        consumption.
      feature_config: Provides granular control of what stats are computed per-
        feature. Experimental.
    """
    super(BasicStatsGenerator, self).__init__(name, schema)

    self._bytes_features = set(
        schema_util.get_bytes_features(schema) if schema else [])
    self._categorical_numeric_types = schema_util.get_categorical_numeric_feature_types(
        schema) if schema else {}
    self._example_weight_map = example_weight_map
    self._num_values_histogram_buckets = num_values_histogram_buckets
    self._num_histogram_buckets = num_histogram_buckets
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    self._epsilon = epsilon
    self._make_quantiles_sketch_fn = lambda: sketches.QuantilesSketch(  # pylint: disable=g-long-lambda
        eps=epsilon,
        max_num_elements=1 << 32,
        num_streams=1)

    # Local state to support feature partitioning.
    self._partition_index = -1
    self._column_hasher = None
    self._feature_config = (
        feature_config or types.PerFeatureStatsConfig.default()
    )

  def _copy_for_partition_index(
      self, index: int, num_partitions: int) -> stats_generator.StatsGenerator:
    if index < 0 or num_partitions <= 1 or index >= num_partitions:
      raise ValueError('Index or num_partitions out of range: %d, %d' %
                       (index, num_partitions))
    copy = BasicStatsGenerator(
        name=self.name,
        schema=self.schema,
        example_weight_map=self._example_weight_map,
        num_values_histogram_buckets=self._num_values_histogram_buckets,
        num_histogram_buckets=self._num_histogram_buckets,
        num_quantiles_histogram_buckets=self._num_quantiles_histogram_buckets)
    copy._partition_index = index  # pylint: disable=protected-access
    copy._column_hasher = feature_partition_util.ColumnHasher(num_partitions)  # pylint: disable=protected-access
    return copy

  def _column_select_fn(self) -> Optional[Callable[[types.FeatureName], bool]]:
    if self._column_hasher is None:
      return None
    return lambda f: self._column_hasher.assign(f) == self._partition_index

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self) -> _BasicAcctype:
    return _BasicAcctype()

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are lists representing a batch of examples) into the accumulator.
  def add_input(self, accumulator: _BasicAcctype,
                examples: pa.RecordBatch) -> _BasicAcctype:
    accumulator.num_examples += examples.num_rows
    # Get the default weight, if it exists. This is always the weight we use
    # for weighted num examples.
    maybe_weight_feature = self._example_weight_map.get(types.FeaturePath([]))
    if maybe_weight_feature:
      weights_column = arrow_util.get_column(examples, maybe_weight_feature)
      accumulator.weighted_num_examples += np.sum(
          np.asarray(weights_column.flatten()))

    for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
        examples,
        example_weight_map=self._example_weight_map,
        enumerate_leaves_only=False,
        column_select_fn=self._column_select_fn()):
      if not self._feature_config.should_compute_histograms(feature_path):
        quantiles_sketch_fn = lambda: None
      else:
        quantiles_sketch_fn = self._make_quantiles_sketch_fn
      stats_for_feature = accumulator.get(feature_path)
      if stats_for_feature is None:
        stats_for_feature = _PartialBasicStats(
            weights is not None, quantiles_sketch_fn
        )
        accumulator[feature_path] = stats_for_feature
      feature_type = stats_util.get_feature_type_from_arrow_type(
          feature_path, feature_array.type)
      stats_for_feature.common_stats.update(
          feature_path,
          feature_array,
          feature_type,
          quantiles_sketch_fn,
          weights,
      )
      # The user may make certain claims about a feature's data type
      # (e.g. _bytes_features imply string data type). However we should not
      # trust those claims because TFDV is also responsible for detecting
      # mismatching types. We collect stats according to the actual type, and
      # only when the actual type matches the claim do we collect the
      # type-specific stats (like for categorical int and bytes features).
      if feature_type == statistics_pb2.FeatureNameStatistics.STRING:
        if feature_path in self._bytes_features:
          stats_for_feature.bytes_stats.update(feature_array)
        else:
          stats_for_feature.string_stats.update(feature_array)
      # We want to compute string stats for a numeric only if a top-k stats
      # generator is running, hence the dependency on this library function.
      elif top_k_uniques_stats_util.output_categorical_numeric(
          self._categorical_numeric_types, feature_path, feature_type):
        stats_for_feature.string_stats.update(feature_array)
      elif feature_type in (statistics_pb2.FeatureNameStatistics.FLOAT,
                            statistics_pb2.FeatureNameStatistics.INT):
        stats_for_feature.numeric_stats.update(feature_array, weights)
    return accumulator

  # Merge together a list of basic common statistics.
  def merge_accumulators(
      self, accumulators: Iterable[_BasicAcctype]) -> _BasicAcctype:
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      result.num_examples += accumulator.num_examples
      result.weighted_num_examples += accumulator.weighted_num_examples
      for feature_path, basic_stats in accumulator.items():
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

          existing_stats.common_stats.merge_with(feature_path,
                                                 basic_stats.common_stats)

          if current_type is not None:
            if feature_path in self._bytes_features:
              existing_stats.bytes_stats += basic_stats.bytes_stats
            elif (top_k_uniques_stats_util.output_categorical_numeric(
                self._categorical_numeric_types, feature_path, current_type) or
                  current_type == statistics_pb2.FeatureNameStatistics.STRING):
              existing_stats.string_stats += basic_stats.string_stats
            elif current_type in (statistics_pb2.FeatureNameStatistics.INT,
                                  statistics_pb2.FeatureNameStatistics.FLOAT):
              existing_stats.numeric_stats += basic_stats.numeric_stats

    return result

  def compact(self, accumulator: _BasicAcctype) -> _BasicAcctype:
    for stats in accumulator.values():
      if stats.numeric_stats.quantiles_summary is not None:
        stats.numeric_stats.quantiles_summary.Compact()
      if (
          stats.numeric_stats.has_weights
          and stats.numeric_stats.weighted_quantiles_summary is not None
      ):
        stats.numeric_stats.weighted_quantiles_summary.Compact()
      if stats.common_stats.presence_and_valency_stats is not None:
        for p_and_v_stat in stats.common_stats.presence_and_valency_stats:
          if p_and_v_stat.num_values_summary is not None:
            p_and_v_stat.num_values_summary.Compact()
    return accumulator

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(
      self,
      accumulator: _BasicAcctype) -> statistics_pb2.DatasetFeatureStatistics:
    # Update TFDV telemetry.
    _update_tfdv_telemetry(accumulator)

    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()
    result.num_examples = accumulator.num_examples
    result.weighted_num_examples = accumulator.weighted_num_examples
    for feature_path, basic_stats in accumulator.items():
      # Construct the FeatureNameStatistics proto from the partial
      # basic stats.
      feature_stats_proto = _make_feature_stats_proto(
          feature_path, basic_stats, accumulator.get(feature_path.parent()),
          self._make_quantiles_sketch_fn, self._num_values_histogram_buckets,
          self._num_histogram_buckets, self._num_quantiles_histogram_buckets,
          feature_path in self._bytes_features, self._categorical_numeric_types,
          self._example_weight_map.get(feature_path) is not None,
          accumulator.num_examples, accumulator.weighted_num_examples)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
