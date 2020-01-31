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
r"""Computes lifts between one feature and a set of categorical features.

We define the feature value lift(x_i, y_i) for features X and Y as:

  P(Y=y_i|X=x_i) / P(Y=y_i)

This quantitatively captures the notion of probabilistic independence, such that
when X and Y are independent, the lift will be 1. It also indicates the degree
to which the presence of x_i increases or decreases the probablity of the
presence of y_i. When X or Y is multivalent, the expressions `X=x_i` and `Y=y_i`
are intepreted as the set membership checks, `x_i \in X` and `y_i \in Y`.

When Y is a label and Xs are the set of categorical features, lift can be used
to assess feature importance. However, in the presence of correlated features,
because lift is computed independently for each feature, it will not be a
reliable indicator of the expected impact on model quality from adding or
removing that feature.

This TransformStatsGenerator computes feature value lift for all pairs of X and
Y, where Y is a single, user-configured feature and X is either a manually
specified list of features, or all categorical features in the provided schema.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import operator

import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
import six

from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import bin_util
from tensorflow_data_validation.utils import schema_util
from tfx_bsl.arrow import array_util
import typing
from typing import Any, Dict, Iterator, Iterable, List, Optional, Text, Tuple, Union
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

_XType = Union[Text, bytes]
_YType = Union[Text, bytes, int]

_SlicedYKey = typing.NamedTuple('_SlicedYKey', [('slice_key', types.SliceKey),
                                                ('y', _YType)])

_SlicedXKey = typing.NamedTuple('_SlicedXKey', [('slice_key', types.SliceKey),
                                                ('x_path', types.FeaturePath),
                                                ('x', _XType)])

_SlicedXYKey = typing.NamedTuple('_SlicedXYKey', [('slice_key', types.SliceKey),
                                                  ('x_path', types.FeaturePath),
                                                  ('x', _XType), ('y', _YType)])

_LiftSeriesKey = typing.NamedTuple('_LiftSeriesKey',
                                   [('slice_key', types.SliceKey),
                                    ('x_path', types.FeaturePath),
                                    ('y', _YType), ('y_count', int)])

_SlicedFeatureKey = typing.NamedTuple('_SlicedFeatureKey',
                                      [('slice_key', types.SliceKey),
                                       ('x_path', types.FeaturePath)])

_ConditionalYRate = typing.NamedTuple('_ConditionalYRate',
                                      [('x_path', types.FeaturePath),
                                       ('x', _XType), ('xy_count', int),
                                       ('x_count', int)])

_YRate = typing.NamedTuple('_YRate', [('y_count', int), ('example_count', int)])

_LiftInfo = typing.NamedTuple('_LiftInfo', [('x', _XType), ('y', _YType),
                                            ('lift', float), ('xy_count', int),
                                            ('x_count', int), ('y_count', int)])

_LiftValue = typing.NamedTuple('_LiftValue', [('x', _XType), ('lift', float),
                                              ('xy_count', int),
                                              ('x_count', int)])

_LiftSeries = typing.NamedTuple('_LiftSeries',
                                [('y', _YType), ('y_count', int),
                                 ('lift_values', Iterable[_LiftValue])])


def _get_example_value_presence(
    table: pa.Table, path: types.FeaturePath,
    boundaries: Optional[Iterable[float]]) -> Optional[pd.Series]:
  """Returns information about which examples contained which values.

  This function treats all values for a given path within a single example
  as a set and and returns a mapping between each example index and the distinct
  values which are present in that example.

  The result of calling this function for path 'p' on an arrow table with the
  two records [{'p': ['a', 'a', 'b']}, {'p': [a]}] will be
  pd.Series(['a', 'b', 'a'], index=[0, 0, 1]).

  If the array retrieved from get_array is null, this function returns None.

  Args:
    table: The table in which to look up the path.
    path: The FeaturePath for which to fetch values.
    boundaries: Optionally, a set of bin boundaries to use for binning the array
      values.

  Returns:
    A Pandas Series containing distinct pairs of array values and example
    indices. The series values will be the array values, and the series index
    values will be the example indices.
  """
  arr, example_indices = arrow_util.get_array(
      table, path, return_example_indices=True)
  if pa.types.is_null(arr.type):
    return None

  arr_flat = arr.flatten()
  example_indices_flat = example_indices[
      array_util.GetFlattenedArrayParentIndices(arr).to_numpy()]
  if boundaries is not None:
    element_indices, bins = bin_util.bin_array(arr_flat, boundaries)
    df = pd.DataFrame({
        'example_indices': example_indices_flat[element_indices],
        'values': bins
    })
  else:
    df = pd.DataFrame({
        'example_indices': example_indices_flat,
        'values': np.asarray(arr_flat)
    })
  df_unique = df.drop_duplicates()
  return df_unique.set_index('example_indices')['values']


def _to_partial_copresence_counts(
    sliced_table: types.SlicedTable, y_path: types.FeaturePath,
    x_paths: Iterable[types.FeaturePath],
    y_boundaries: Optional[Iterable[float]]
) -> Iterator[Tuple[_SlicedXYKey, int]]:
  """Yields per-(slice, path_x, x, y) counts of examples with x and y.

  This method generates the number of times a given pair of y- and x-values
  appear in the same record, for a slice_key and x_path. Records in which either
  x or y is absent will be skipped.

  Args:
    sliced_table: A tuple of (slice_key, table) representing a slice of examples
    y_path: The path to use as Y in the lift expression: lift = P(Y=y|X=x) /
      P(Y=y).
    x_paths: A set of x_paths for which to compute lift.
    y_boundaries: Optionally, a set of bin boundaries to use for binning y_path
      values.

  Yields:
    Tuples of the form (_SlicedXYKey(slice_key, x_path, x, y), count) for each
    combination of  x_path, x, and y  in the input table.
  """
  slice_key, table = sliced_table
  y_series = _get_example_value_presence(table, y_path, y_boundaries)
  if y_series is None:
    return
  x_column, y_column = 'x', 'y'
  for x_path in x_paths:
    x_series = _get_example_value_presence(table, x_path, boundaries=None)
    if x_series is None:
      continue
    # merge using inner join implicitly drops null entries.
    copresence_df = pd.merge(
        x_series.rename(x_column),
        y_series.rename(y_column),
        how='inner',
        left_index=True,
        right_index=True)
    copresence_counts = copresence_df.groupby([x_column, y_column]).size()
    for (x, y), count in copresence_counts.items():
      yield _SlicedXYKey(slice_key=slice_key, x_path=x_path, x=x, y=y), count


def _to_partial_y_counts(
    sliced_table: types.SlicedTable, y_path: types.FeaturePath,
    y_boundaries: Optional[Iterable[float]]
) -> Iterator[Tuple[_SlicedYKey, int]]:
  """Yields per-(slice, y) counts of the examples with y in y_path."""
  slice_key, table = sliced_table
  series = _get_example_value_presence(table, y_path, y_boundaries)
  if series is None:
    return
  for y, y_count in series.value_counts().items():
    yield _SlicedYKey(slice_key, y), y_count


def _to_partial_x_counts(
    sliced_table: types.SlicedTable,
    x_paths: Iterable[types.FeaturePath]) -> Iterator[Tuple[_SlicedXKey, int]]:
  """Yields per-(slice, x_path, x) counts of the examples with x in x_path."""
  slice_key, table = sliced_table
  for x_path in x_paths:
    series = _get_example_value_presence(table, x_path, boundaries=None)
    if series is None:
      continue
    for x, x_count in series.value_counts().items():
      yield _SlicedXKey(slice_key, x_path, x), x_count


def _make_dataset_feature_stats_proto(
    lifts: Tuple[_SlicedFeatureKey, _LiftSeries],
    y_path: types.FeaturePath, y_boundaries: Optional[np.ndarray]
) -> Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]:
  """Generates DatasetFeatureStatistics proto for a given x_path, y_path pair.

  Args:
    lifts: The result of two successive group bys of lift values. The innermost
      grouping collects all the lift values for a given (slice, x_path and
      y_value) tuple (corresponding to a single LiftSeries message). The
      outermost grouping collects all the lift values for the same (slice,
      x_path) tuple (corresponding to the set of the LiftSeries which share the
      same value of y_path). The full structure of lifts is described by:
        (slice, x_path), [(y, y_count, [(x, lift, xy_count, x_count)])]
    y_path: The path used as Y in the lift expression: lift = P(Y=y|X=x) /
      P(Y=y).
    y_boundaries: Optionally, a set of bin boundaries used for binning y_path
      values.

  Returns:
    The populated DatasetFeatureStatistics proto.
  """
  key, lift_series_list = lifts
  stats = statistics_pb2.DatasetFeatureStatistics()
  cross_stats = stats.cross_features.add(
      path_x=key.x_path.to_proto(), path_y=y_path.to_proto())
  for lift_series in sorted(lift_series_list):
    lift_series_proto = (
        cross_stats.categorical_cross_stats.lift.lift_series.add(
            y_count=lift_series.y_count))
    y = lift_series.y
    if y_boundaries is not None:
      low_value, high_value = bin_util.get_boundaries(y, y_boundaries)
      lift_series_proto.y_bucket.low_value = low_value
      lift_series_proto.y_bucket.high_value = high_value
    elif isinstance(y, six.string_types):
      lift_series_proto.y_string = y
    else:
      lift_series_proto.y_int = y

    # dedupe possibly overlapping top_k and bottom_k x values.
    lift_values_deduped = {v.x: v for v in lift_series.lift_values}
    # sort by lift DESC, x ASC
    lift_values_sorted = sorted(lift_values_deduped.values(),
                                key=lambda v: (-v.lift, v.x))
    for lift_value in lift_values_sorted:
      lift_value_proto = lift_series_proto.lift_values.add(
          lift=lift_value.lift, x_count=lift_value.x_count,
          x_and_y_count=lift_value.xy_count)
      x = lift_value.x
      if isinstance(x, six.string_types):
        lift_value_proto.x_string = x
      else:
        lift_value_proto.x_int = x
  return key.slice_key, stats


def _cross_join_y_keys(
    join_info: Tuple[types.SliceKey, Dict[Text, List[Any]]]
    # TODO(b/147153346) fix annotation to include specific Dict value type:
    # Union[_YKey, Tuple[_YType, Tuple[types.FeaturePath, _XType, int]]]
) -> Iterator[Tuple[_SlicedXYKey, int]]:
  slice_key, join_args = join_info
  for x_path, x, _ in join_args['x_counts']:
    for y in join_args['y_keys']:
      yield _SlicedXYKey(slice_key=slice_key, x_path=x_path, x=x, y=y), 0


def _join_x_counts(
    join_info: Tuple[_SlicedXKey, Dict[Text, List[Any]]]
    # TODO(b/147153346) fix annotation to include specific Dict value type:
    # Union[int, Tuple[_YType, int]]
) -> Iterator[Tuple[_SlicedYKey, _ConditionalYRate]]:
  """Joins x_count with all xy_counts for that x.

  This function expects the result of a CoGroupByKey, in which the key is a
  tuple of the form (slice_key, x_path, x), and one of the grouped streams has
  just one element, the number of examples in a given slice for which x is
  present in x_path, and the other grouped stream is the set of all (x, y) pairs
  for that x along with the number of examples in which  both x and y are
  present in their respective paths. Schematically, join_info looks like:

  (slice, x_path, x), {'x_count': [x_count],
                       'xy_counts': [(y_1, xy_1_count), ..., (y_k, xy_k_count)]}

  If the value of x_count is less than min_x_count, no rates will be yielded.

  Args:
    join_info: A CoGroupByKey result

  Yields:
    Per-(slice, x_path, y, x) tuples of the form (_SlicedYKey(slice, y),
    _ConditionalYRate(x_path, x, xy_count, x_count)).
  """
  # (slice_key, x_path, x), join_inputs = join_info
  key, join_inputs = join_info
  if not join_inputs['x_count']:
    return
  x_count = join_inputs['x_count'][0]
  for y, xy_count in join_inputs['xy_counts']:
    yield _SlicedYKey(key.slice_key, y), _ConditionalYRate(
        x_path=key.x_path, x=key.x, xy_count=xy_count, x_count=x_count)


def _join_example_counts(
    join_info: Tuple[types.SliceKey, Dict[Text, List[Any]]]
    # TODO(b/147153346) fix annotation to include specific Dict value type:
    # Union[int, Tuple[_YType, int]]
) -> Iterator[Tuple[_SlicedYKey, _YRate]]:
  """Joins slice example count with all values of y within that slice.

  This function expects the result of a CoGroupByKey, in which the key is the
  slice_key, one of the grouped streams has just one element, the total number
  of examples within the slice, and the other grouped stream is the set of all
  y values and number of times that y value appears in this slice.
  Schematically, join_info looks like:

  slice_key, {'example_count': [example_count],
              'y_counts': [(y_1, y_1_count), ..., (y_k, y_k_count)]}

  Args:
    join_info: A CoGroupByKey result.

  Yields:
    Per-(slice, y) tuples (_SlicedYKey(slice, y),
                           _YRate(y_count, example_count)).
  """
  slice_key, join_inputs = join_info
  example_count = join_inputs['example_count'][0]
  for y, y_count in join_inputs['y_counts']:
    yield _SlicedYKey(slice_key, y), _YRate(y_count, example_count)


def _compute_lifts(
    join_info: Tuple[_SlicedYKey, Dict[Text, List[Any]]]
    # TODO(b/147153346) fix annotation to include specific Dict value type:
    # List[Union[_YRate, _ConditionalYRate]]
) -> Iterator[Tuple[_SlicedFeatureKey, _LiftInfo]]:
  """Joins y_counts with all x-y pairs for that y and computes lift.

  This function expects the result of a CoGroupByKey, in which the key is a
  tuple of the form (slice_key, y), one of the grouped streams has just one
  element, the y_rate for that value of y, and the other grouped stream is the
  set of all conditional_y_rate values for that same value of y. Schematically,
  join_info looks like:

  (slice_key, y), {'y_rate': [y_count, example_count], 'conditional_y_rate': [
      (x_path_1, x_1, x_1_y_count, x_1_count), ...,
      (x_path_1, x_k, x_k_y_count, x_k_count)
      ...
      (x_path_m, x_1, x_1_y_count, x_1_count), ...,
      (x_path_m, x_k, x_k_y_count, x_k_count)]}

  Args:
    join_info: A CoGroupByKey result.

  Yields:
    Per-(slice, x_path) tuples of the form ((slice_key, x_path),
    _LiftInfo(x, y, lift, xy_count, x_count, y_count)).
  """
  (slice_key, y), join_inputs = join_info
  y_rate = join_inputs['y_rate'][0]
  for conditional_y_rate in join_inputs['conditional_y_rate']:
    lift = ((float(conditional_y_rate.xy_count) / conditional_y_rate.x_count) /
            (float(y_rate.y_count) / y_rate.example_count))
    yield (_SlicedFeatureKey(slice_key, conditional_y_rate.x_path),
           _LiftInfo(
               x=conditional_y_rate.x,
               y=y,
               lift=lift,
               xy_count=conditional_y_rate.xy_count,
               x_count=conditional_y_rate.x_count,
               y_count=y_rate.y_count))


@beam.typehints.with_input_types(Tuple[_SlicedFeatureKey, _LiftInfo])
@beam.typehints.with_output_types(Tuple[_SlicedFeatureKey, _LiftSeries])
class _FilterLifts(beam.PTransform):
  """A PTransform for filtering and truncating lift values."""

  def __init__(self, top_k_per_y: Optional[int], bottom_k_per_y: Optional[int]):
    self._top_k_per_y = top_k_per_y
    self._bottom_k_per_y = bottom_k_per_y

  def expand(self, lifts: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    """Takes top k and bottom k x values (sorted by lift) per slice and y value.

    Args:
      lifts: A PCollection of tuples of the form: (
        _SlicedFeatureKey(slice_key, x_path),
        _LiftInfo(x, y, lift, xy_count, x_count, y_count)).

    Returns:
      A PCollection resulting from a group by with the keys of the form
      (slice_key, x_path) and a stream of values of the form
      (y, y_count, [(x, lift, xy_count, x_count)], in which the stream of values
      has been limited to the top k and bottom k elements per key.
    """

    def move_y_info_to_key(key, value):
      slice_key, x_path = key
      return (_LiftSeriesKey(
          slice_key=slice_key, x_path=x_path, y=value.y, y_count=value.y_count),
              _LiftValue(
                  x=value.x,
                  lift=value.lift,
                  xy_count=value.xy_count,
                  x_count=value.x_count))

    # Push y_* into key so that we get per-slice, per-x-path, per-y top and
    # bottom k when calling {Largest,Smallest}PerKey.
    # (_LiftSequenceKey(slice, x_path, y, y_count),
    #      _LiftValue(x, lift, xy_count, x_count))
    lifts = lifts | 'MoveYToKey' >> beam.MapTuple(move_y_info_to_key)

    top_key = operator.attrgetter('lift', 'x')
    if self._top_k_per_y:
      # (_LiftSequenceKey(slice, x_path, y, y_count),
      #      [_LiftValue(x, lift, xy_count, x_count)])
      top_k = (
          lifts
          | 'TopK' >> beam.transforms.combiners.Top.PerKey(
              n=self._top_k_per_y, key=top_key))
    if self._bottom_k_per_y:
      # (_LiftSequenceKey(slice, x_path, y, y_count),
      #      [_LiftValue(x, lift, xy_count, x_count)])
      bottom_k = (
          lifts
          | 'BottomK' >> beam.transforms.combiners.Top.PerKey(
              n=self._bottom_k_per_y, reverse=True, key=top_key))

    if self._top_k_per_y and self._bottom_k_per_y:
      # (_LiftSeriesKey(slice, x_path, y, y_count),
      #      [_LiftValue(x, lift, xy_count, x_count)])
      grouped_lifts = ((top_k, bottom_k)
                       | 'MergeTopAndBottom' >> beam.Flatten()
                       | 'FlattenTopAndBottomLifts' >>
                       beam.FlatMapTuple(lambda k, vs: ((k, v) for v in vs))
                       | 'ReGroupTopAndBottom' >> beam.GroupByKey())
    elif self._top_k_per_y:
      grouped_lifts = top_k
    elif self._bottom_k_per_y:
      grouped_lifts = bottom_k
    else:
      grouped_lifts = lifts | 'GroupByYs' >> beam.GroupByKey()

    def move_y_info_to_value(key, lift_values):
      return (_SlicedFeatureKey(key.slice_key, key.x_path),
              _LiftSeries(
                  y=key.y, y_count=key.y_count, lift_values=lift_values))

    # (_SlicedFeatureKey(slice, x_path),
    #      _LiftSeries(y, y_count, [_LiftValue(x, lift, xy_count, x_count)]))
    return (grouped_lifts
            | 'MoveYInfoToValue' >> beam.MapTuple(move_y_info_to_value))


# No typehint for input, since it's a multi-input PTransform for which Beam
# doesn't yet support typehints (BEAM-3280).
@beam.typehints.with_output_types(Tuple[_SlicedXYKey, int])
class _GetPlaceholderCopresenceCounts(beam.PTransform):
  """A PTransform for computing all possible x-y pairs, to support 0 lifts."""

  def __init__(self, x_paths, min_x_count):
    self._x_paths = x_paths
    self._min_x_count = min_x_count

  def expand(self,
             x_counts_and_ys: Tuple[Tuple[_SlicedXKey, int], _SlicedYKey]):
    x_counts, y_keys = x_counts_and_ys

    # slice, y
    y_keys_by_slice = (
        y_keys
        | 'MoveYToValue_YKey' >> beam.Map(lambda k: (k.slice_key, k.y)))
    # slice, (x_path, x, x_count)
    x_counts_by_slice = (
        x_counts
        | 'MoveXToValue_XCountsKey' >> beam.MapTuple(
            lambda k, v: (k.slice_key, (k.x_path, k.x, v))))

    # _SlicedXYKey(slice, x_path, x, y), 0
    return (
        {
            'y_keys': y_keys_by_slice,
            'x_counts': x_counts_by_slice
        }
        | 'CoGroupByForPlaceholderYRates' >> beam.CoGroupByKey()
        | 'CrossXYValues' >> beam.FlatMap(_cross_join_y_keys))


# No typehint for input, since it's a multi-input PTransform for which Beam
# doesn't yet support typehints (BEAM-3280).
@beam.typehints.with_output_types(Tuple[_SlicedYKey, _ConditionalYRate])
class _GetConditionalYRates(beam.PTransform):
  """A PTransform for computing the rate of each y value, given an x value."""

  def __init__(self, y_path, y_boundaries, x_paths, min_x_count):
    self._y_path = y_path
    self._y_boundaries = y_boundaries
    self._x_paths = x_paths
    self._min_x_count = min_x_count

  def expand(self, sliced_tables_and_ys: Tuple[types.SlicedTable, _SlicedYKey]):
    sliced_tables, y_keys = sliced_tables_and_ys

    # _SlicedXYKey(slice, x_path, x, y), xy_count
    partial_copresence_counts = (
        sliced_tables
        | 'ToPartialCopresenceCounts' >> beam.FlatMap(
            _to_partial_copresence_counts, self._y_path, self._x_paths,
            self._y_boundaries))

    # Compute placerholder copresence counts.
    # partial_copresence_counts will only include x-y pairs that are present,
    # but we would also like to keep track of x-y pairs that never appear, as
    # long as x and y independently occur in the slice.

    # _SlicedXKey(slice, x_path, x), x_count
    x_counts = (
        sliced_tables
        | 'ToPartialXCounts' >> beam.FlatMap(
            _to_partial_x_counts, self._x_paths)
        | 'SumXCounts' >> beam.CombinePerKey(sum))
    if self._min_x_count:
      x_counts = x_counts | 'FilterXCounts' >> beam.Filter(
          lambda kv: kv[1] > self._min_x_count)

    # _SlicedXYKey(slice, x_path, x, y), 0
    placeholder_copresence_counts = (
        (x_counts, y_keys)
        | 'GetPlaceholderCopresenceCounts' >> _GetPlaceholderCopresenceCounts(
            self._x_paths, self._min_x_count))

    def move_y_to_value(key, xy_count):
      return _SlicedXKey(key.slice_key, key.x_path, key.x), (key.y, xy_count)

    # _SlicedXKey(slice, x_path, x), (y, xy_count)
    copresence_counts = (
        (placeholder_copresence_counts, partial_copresence_counts)
        | 'FlattenCopresenceCounts' >> beam.Flatten()
        | 'SumCopresencePairs' >> beam.CombinePerKey(sum)
        | 'MoveYToValue' >> beam.MapTuple(move_y_to_value))

    # _SlicedYKey(slice, y), _ConditionalYRate(x_path, x, xy_count, x_count)
    return ({
        'x_count': x_counts,
        'xy_counts': copresence_counts
    }
            | 'CoGroupByForConditionalYRates' >> beam.CoGroupByKey()
            | 'JoinXCounts' >> beam.FlatMap(_join_x_counts))


@beam.typehints.with_input_types(types.SlicedTable)
@beam.typehints.with_output_types(Tuple[_SlicedYKey, _YRate])
class _GetYRates(beam.PTransform):
  """A PTransform for computing the rate of each y value within each slice."""

  def __init__(self, y_path, y_boundaries):
    self._y_path = y_path
    self._y_boundaries = y_boundaries

  def expand(self, sliced_tables):
    # slice, example_count
    example_counts = (
        sliced_tables
        | 'ToExampleCounts' >> beam.MapTuple(lambda k, v: (k, v.num_rows))
        | 'SumExampleCounts' >> beam.CombinePerKey(sum))

    def move_y_to_value(slice_and_y, y_count):
      slice_key, y = slice_and_y
      return slice_key, (y, y_count)

    # slice, (y, y_count)
    y_counts = (
        sliced_tables
        | 'ToPartialYCounts' >> beam.FlatMap(_to_partial_y_counts, self._y_path,
                                             self._y_boundaries)
        | 'SumYCounts' >> beam.CombinePerKey(sum)
        | 'MoveYToValue' >> beam.MapTuple(move_y_to_value))

    # _SlicedYKey(slice, y), _YRate(y_count, example_count)
    return ({
        'y_counts': y_counts,
        'example_count': example_counts
    }
            | 'CoGroupByForYRates' >> beam.CoGroupByKey()
            | 'JoinExampleCounts' >> beam.FlatMap(_join_example_counts))


@beam.typehints.with_input_types(types.SlicedTable)
@beam.typehints.with_output_types(Tuple[types.SliceKey,
                                        statistics_pb2.DatasetFeatureStatistics]
                                 )
class _LiftStatsGenerator(beam.PTransform):
  """A PTransform implementing a TransformStatsGenerator to compute lift.

  This transform computes lift for a set of feature pairs (y, x_1), ... (y, x_k)
  for a collection of x_paths, and a single y_path. The y_path must be either
  a categorical feature, or numeric feature (in which case binning boundaries
  are also required). The x_paths can be manually provided or will be
  automatically inferred as the set of categorical features in the schema
  (excluding y_path).
  """

  def __init__(self, y_path: types.FeaturePath,
               schema: Optional[schema_pb2.Schema],
               x_paths: Optional[Iterable[types.FeaturePath]],
               y_boundaries: Optional[Iterable[float]], min_x_count: int,
               top_k_per_y: Optional[int], bottom_k_per_y: Optional[int],
               name: Text) -> None:
    """Initializes a lift statistics generator.

    Args:
      y_path: The path to use as Y in the lift expression:
        lift = P(Y=y|X=x) / P(Y=y).
     schema: An optional schema for the dataset. If not provided, x_paths must
        be specified. If x_paths are not specified, the schema is used to
        identify all categorical columns for which Lift should be computed.
      x_paths: An optional list of path to use as X in the lift expression:
        lift = P(Y=y|X=x) / P(Y=y). If None (default), all categorical features,
        exluding the feature passed as y_path, will be used.
      y_boundaries: An optional list of boundaries to be used for binning
        y_path. If provided with b boundaries, the binned values will be treated
        as a categorical feature with b+1 different values. For example, the
        y_boundaries value [0.1, 0.8] would lead to three buckets: [-inf, 0.1),
        [0.1, 0.8) and [0.8, inf].
      min_x_count: The minimum number of examples in which a specific x value
        must appear, in order for its lift to be output.
      top_k_per_y:  Optionally, the number of top x values per y value, ordered
        by descending lift, for which to output lift. If both top_k_per_y and
        bottom_k_per_y are unset, all values will be output.
      bottom_k_per_y:  Optionally, the number of bottom x values per y value,
        ordered by descending lift, for which to output lift. If both
        top_k_per_y and bottom_k_per_y are unset, all values will be output.
      name: An optional unique name associated with the statistics generator.
    """
    self._name = name
    self._schema = schema
    self._y_path = y_path
    self._min_x_count = min_x_count
    self._top_k_per_y = top_k_per_y
    self._bottom_k_per_y = bottom_k_per_y
    self._y_boundaries = (
        np.array(sorted(set(y_boundaries))) if y_boundaries else None)

    # If a schema is provided, we can do some additional validation of the
    # provided y_feature and boundaries.
    if self._schema is not None:
      y_feature = schema_util.get_feature(self._schema, y_path)
      y_is_categorical = schema_util.is_categorical_feature(y_feature)
      if self._y_boundaries is not None:
        if y_is_categorical:
          raise ValueError(
              'Boundaries cannot be applied to a categorical y_path')
      else:
        if not y_is_categorical:
          raise ValueError('Boundaries must be provided with a non-categorical '
                           'y_path.')
    if x_paths is not None:
      self._x_paths = x_paths
    elif self._schema is not None:
      self._x_paths = (
          set(schema_util.get_categorical_features(schema)) - set([y_path]))
    else:
      raise ValueError('Either a schema or x_paths must be provided.')

  def expand(self,
             sliced_tables: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    # Compute P(Y=y)
    # _SlicedYKey(slice, y), _YRate(y_count, example_count)
    y_rates = sliced_tables | 'GetYRates' >> _GetYRates(self._y_path,
                                                        self._y_boundaries)
    y_keys = y_rates | 'ExtractYKeys' >> beam.Keys()

    # Compute P(Y=y | X=x)
    # _SlicedYKey(slice, y), _ConditionalYRate(x_path, x, xy_count, x_count)
    conditional_y_rates = (
        (sliced_tables, y_keys)
        | 'GetConditionalYRates' >> _GetConditionalYRates(
            self._y_path, self._y_boundaries, self._x_paths, self._min_x_count))

    return (
        {
            'conditional_y_rate': conditional_y_rates,
            'y_rate': y_rates
        }
        | 'CoGroupByForLift' >> beam.CoGroupByKey()
        | 'ComputeLifts' >> beam.FlatMap(_compute_lifts)
        | 'FilterLifts' >> _FilterLifts(self._top_k_per_y, self._bottom_k_per_y)
        | 'GroupLiftsForOutput' >> beam.GroupByKey()
        | 'MakeProtos' >> beam.Map(_make_dataset_feature_stats_proto,
                                   self._y_path, self._y_boundaries))


class LiftStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform stats generator for computing lift between two features."""

  def __init__(self,
               y_path: types.FeaturePath,
               schema: Optional[schema_pb2.Schema] = None,
               x_paths: Optional[Iterable[types.FeaturePath]] = None,
               y_boundaries: Optional[Iterable[float]] = None,
               min_x_count: int = 0,
               top_k_per_y: Optional[int] = None,
               bottom_k_per_y: Optional[int] = None,
               name: Text = 'LiftStatsGenerator') -> None:
    super(LiftStatsGenerator, self).__init__(
        name,
        ptransform=_LiftStatsGenerator(y_path, schema, x_paths, y_boundaries,
                                       min_x_count, top_k_per_y, bottom_k_per_y,
                                       name),
        schema=schema)
