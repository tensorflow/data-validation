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

import logging
import operator

import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
import six

from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import bin_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
import typing
from typing import Any, Dict, Iterator, Iterable, Optional, Sequence, Text, Tuple, Union
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

_XType = Union[Text, bytes]
_YType = Union[Text, bytes, int]
_CountType = Union[int, float]

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
                                    ('y', _YType), ('y_count', _CountType)])

_SlicedFeatureKey = typing.NamedTuple('_SlicedFeatureKey',
                                      [('slice_key', types.SliceKey),
                                       ('x_path', types.FeaturePath)])

_ConditionalYRate = typing.NamedTuple('_ConditionalYRate',
                                      [('x_path', types.FeaturePath),
                                       ('x', _XType), ('xy_count', _CountType),
                                       ('x_count', _CountType)])

_YRate = typing.NamedTuple('_YRate', [('y_count', _CountType),
                                      ('example_count', _CountType)])

_LiftInfo = typing.NamedTuple('_LiftInfo', [('x', _XType), ('y', _YType),
                                            ('lift', float),
                                            ('xy_count', _CountType),
                                            ('x_count', _CountType),
                                            ('y_count', _CountType)])

_LiftValue = typing.NamedTuple('_LiftValue', [('x', _XType), ('lift', float),
                                              ('xy_count', _CountType),
                                              ('x_count', _CountType)])

_LiftSeries = typing.NamedTuple('_LiftSeries',
                                [('y', _YType), ('y_count', _CountType),
                                 ('lift_values', Iterable[_LiftValue])])


def _get_example_value_presence(
    record_batch: pa.RecordBatch, path: types.FeaturePath,
    boundaries: Optional[Sequence[float]],
    weight_column_name: Optional[Text]) -> Optional[pd.DataFrame]:

  """Returns information about which examples contained which values.

  This function treats all values for a given path within a single example
  as a set and and returns a mapping between each example index and the distinct
  values which are present in that example.

  The result of calling this function for path 'p' on an arrow record batch with
  the two records [{'p': ['a', 'a', 'b']}, {'p': [a]}] will be
  pd.Series(['a', 'b', 'a'], index=[0, 0, 1]).

  If the array retrieved from get_array is null, this function returns None.

  Args:
    record_batch: The RecordBatch in which to look up the path.
    path: The FeaturePath for which to fetch values.
    boundaries: Optionally, a set of bin boundaries to use for binning the array
      values.
    weight_column_name: Optionally, a weight column to return in addition to the
      value and example index.

  Returns:
    A Pandas DataFrame containing distinct pairs of array values and example
    indices, along with the corresponding flattened example weights. The index
    will be the example indices and the values will be stored in a column named
    'values'. If weight_column_name is provided, a second column will be
    returned containing the array values, and 'weights' containing the weights
    for the example from which each value came.
  """
  arr, example_indices = arrow_util.get_array(
      record_batch, path, return_example_indices=True)
  if stats_util.get_feature_type_from_arrow_type(path, arr.type) is None:
    return None

  arr_flat, parent_indices = arrow_util.flatten_nested(
      arr, return_parent_indices=True)
  is_binary_like = arrow_util.is_binary_like(arr_flat.type)
  assert boundaries is None or not is_binary_like, (
      'Boundaries can only be applied to numeric columns')
  if is_binary_like:
    # use dictionary_encode so we can use np.unique on object arrays
    dict_array = arr_flat.dictionary_encode()
    arr_flat = dict_array.indices
    arr_flat_dict = np.asarray(dict_array.dictionary)
  example_indices_flat = example_indices[parent_indices]
  if boundaries is not None:
    element_indices, bins = bin_util.bin_array(arr_flat, boundaries)
    rows = np.vstack([example_indices_flat[element_indices], bins])
  else:
    rows = np.vstack([example_indices_flat, np.asarray(arr_flat)])
  if not rows.size:
    return None
  # Deduplicate values which show up more than once in the same example. This
  # makes P(X=x|Y=y) in the standard lift definition behave as
  # P(x \in Xs | y \in Ys) if examples contain more than one value of X and Y.
  unique_rows = np.unique(rows, axis=1)
  example_indices = unique_rows[0, :]
  values = unique_rows[1, :]
  if is_binary_like:
    # return binary like values a pd.Categorical wrapped in a Series. This makes
    # subsqeuent operations like pd.Merge cheaper.
    values = pd.Categorical.from_codes(values, categories=arr_flat_dict)
  columns = {'example_indices': example_indices, 'values': values}
  if weight_column_name:
    weights = arrow_util.get_weight_feature(record_batch, weight_column_name)
    columns['weights'] = np.asarray(weights)[example_indices]
  df = pd.DataFrame(columns)
  return df.set_index('example_indices')


def _to_partial_copresence_counts(
    sliced_record_batch: types.SlicedRecordBatch, y_path: types.FeaturePath,
    x_paths: Iterable[types.FeaturePath], y_boundaries: Optional[np.ndarray],
    weight_column_name: Optional[Text]
) -> Iterator[Tuple[_SlicedXYKey, _CountType]]:
  """Yields per-(slice, path_x, x, y) counts of examples with x and y.

  This method generates the number of times a given pair of y- and x-values
  appear in the same record, for a slice_key and x_path. Records in which either
  x or y is absent will be skipped.

  Args:
    sliced_record_batch: A tuple of (slice_key, record_batch) representing a
      slice of examples
    y_path: The path to use as Y in the lift expression: lift = P(Y=y|X=x) /
      P(Y=y).
    x_paths: A set of x_paths for which to compute lift.
    y_boundaries: Optionally, a set of bin boundaries to use for binning y_path
      values.
    weight_column_name: Optionally, a weight column to use for weighting
      copresence counts by the example weight in which an X and Y value were
      copresent.

  Yields:
    Tuples of the form (_SlicedXYKey(slice_key, x_path, x, y), count) for each
    combination of  x_path, x, and y  in the input record batch.
  """
  slice_key, record_batch = sliced_record_batch
  y_df = _get_example_value_presence(record_batch, y_path, y_boundaries,
                                     weight_column_name)
  if y_df is None:
    return
  for x_path in x_paths:
    x_df = _get_example_value_presence(
        record_batch,
        x_path,
        boundaries=None,
        weight_column_name=weight_column_name)
    if x_df is None:
      continue
    # merge using inner join implicitly drops null entries.
    copresence_df = pd.merge(
        x_df, y_df, how='inner', left_index=True, right_index=True)
    # pd.merge automatically appends '_x' and '_y' to the first and second join
    # args respectively.
    grouped = copresence_df.groupby(['values_x', 'values_y'], observed=True)
    if weight_column_name:
      copresence_counts = grouped['weights_x'].sum()
    else:
      copresence_counts = grouped.size()
    for (x, y), count in copresence_counts.items():
      yield _SlicedXYKey(slice_key=slice_key, x_path=x_path, x=x, y=y), count


def _to_partial_counts(
    sliced_record_batch: types.SlicedRecordBatch, path: types.FeaturePath,
    boundaries: Optional[np.ndarray], weight_column_name: Optional[Text]
) -> Iterator[Tuple[Tuple[types.SliceKey, Union[_XType, _YType]], _CountType]]:
  """Yields per-(slice, value) counts of the examples with value in path."""
  slice_key, record_batch = sliced_record_batch
  df = _get_example_value_presence(record_batch, path, boundaries,
                                   weight_column_name)
  if df is None:
    return
  for value, group in df.groupby('values'):
    if weight_column_name:
      count = group['weights'].sum()
    else:
      count = group['values'].size
    yield (slice_key, value), count


def _to_partial_x_counts(
    sliced_record_batch: types.SlicedRecordBatch,
    x_paths: Iterable[types.FeaturePath], weight_column_name: Optional[Text]
) -> Iterator[Tuple[_SlicedXKey, _CountType]]:
  """Yields per-(slice, x_path, x) counts of the examples with x in x_path."""
  for x_path in x_paths:
    for (slice_key, x), x_count in _to_partial_counts(
        sliced_record_batch,
        x_path,
        boundaries=None,
        weight_column_name=weight_column_name):
      yield _SlicedXKey(slice_key, x_path, x), x_count


def _get_unicode_value(value: Union[Text, bytes], path: types.FeaturePath):
  value = stats_util.maybe_get_utf8(value)
  # Check if we have a valid utf-8 string. If not, assign a placeholder.
  if value is None:
    logging.warning('Feature "%s" has bytes value "%s" which cannot be '
                    'decoded as a UTF-8 string.', path, value)
    value = constants.NON_UTF8_PLACEHOLDER
  return value


def _make_dataset_feature_stats_proto(
    lifts: Tuple[_SlicedFeatureKey, Iterable[_LiftSeries]],
    y_path: types.FeaturePath, y_boundaries: Optional[np.ndarray],
    weighted_examples: bool, output_custom_stats: bool
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
    weighted_examples: Whether lift is computed over weighted examples, in which
      case the proto will output weighted counts (as floats) rather than simple
      counts (as ints).
    output_custom_stats: Whether to output custom stats for use with Facets.

  Returns:
    The populated DatasetFeatureStatistics proto.
  """
  key, lift_series_list = lifts
  stats = statistics_pb2.DatasetFeatureStatistics()
  cross_stats = stats.cross_features.add(
      path_x=key.x_path.to_proto(), path_y=y_path.to_proto())
  if output_custom_stats:
    feature_stats = stats.features.add(path=key.x_path.to_proto())
  for lift_series in sorted(lift_series_list):
    lift_series_proto = (
        cross_stats.categorical_cross_stats.lift.lift_series.add())
    if weighted_examples:
      lift_series_proto.weighted_y_count = lift_series.y_count
    else:
      lift_series_proto.y_count = lift_series.y_count
    y = lift_series.y
    if y_boundaries is not None and isinstance(y, int):
      low_value, high_value = bin_util.get_boundaries(y, y_boundaries)
      lift_series_proto.y_bucket.low_value = low_value
      lift_series_proto.y_bucket.high_value = high_value
      y_display_fmt = '[{},{}]' if high_value == float('inf') else '[{},{})'
      y_display_val = y_display_fmt.format(low_value, high_value)
    elif isinstance(y, six.text_type):
      lift_series_proto.y_string = y
      y_display_val = y
    elif isinstance(y, six.binary_type):
      y_string = _get_unicode_value(y, y_path)
      lift_series_proto.y_string = y_string
      y_display_val = y_string
    else:
      lift_series_proto.y_int = y
      y_display_val = str(y)

    if output_custom_stats:
      hist = feature_stats.custom_stats.add(
          name='Lift (Y={})'.format(y_display_val)).rank_histogram

    # dedupe possibly overlapping top_k and bottom_k x values.
    lift_values_deduped = {v.x: v for v in lift_series.lift_values}
    # sort by lift DESC, x ASC
    lift_values_sorted = sorted(lift_values_deduped.values(),
                                key=lambda v: (-v.lift, v.x))
    for lift_value in lift_values_sorted:
      lift_value_proto = lift_series_proto.lift_values.add(lift=lift_value.lift)
      if weighted_examples:
        lift_value_proto.weighted_x_count = lift_value.x_count
        lift_value_proto.weighted_x_and_y_count = lift_value.xy_count
      else:
        lift_value_proto.x_count = lift_value.x_count
        lift_value_proto.x_and_y_count = lift_value.xy_count
      x = lift_value.x
      if isinstance(x, six.text_type):
        lift_value_proto.x_string = x
        x_display_val = x
      elif isinstance(x, six.binary_type):
        x_string = _get_unicode_value(x, key.x_path)
        lift_value_proto.x_string = x_string
        x_display_val = x_string
      else:
        lift_value_proto.x_int = x
        x_display_val = str(x)

      if output_custom_stats:
        hist.buckets.add(label=x_display_val, sample_count=lift_value.lift)

  return key.slice_key, stats


def _cross_join_y_keys(
    join_info: Tuple[types.SliceKey, Dict[Text, Sequence[Any]]]
    # TODO(b/147153346) update dict value list element type annotation to:
    # Union[_YKey, Tuple[_YType, Tuple[types.FeaturePath, _XType, _CountType]]]
) -> Iterator[Tuple[_SlicedXYKey, _CountType]]:
  slice_key, join_args = join_info
  for x_path, x, _ in join_args['x_counts']:
    for y in join_args['y_keys']:
      yield _SlicedXYKey(slice_key=slice_key, x_path=x_path, x=x, y=y), 0


def _join_x_counts(
    join_info: Tuple[_SlicedXKey, Dict[Text, Sequence[Any]]]
    # TODO(b/147153346) update dict value list element type annotation to:
    # Union[_CountType, Tuple[_YType, _CountType]]
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
    join_info: Tuple[types.SliceKey, Dict[Text, Sequence[Any]]]
    # TODO(b/147153346) update dict value list element type annotation to:
    # Union[_CountType, Tuple[_YType, _CountType]]
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
    join_info: Tuple[_SlicedYKey, Dict[Text, Sequence[Any]]]
    # TODO(b/147153346) update dict value list element type annotation to:
    # Sequence[Union[_YRate, _ConditionalYRate]]
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
@beam.typehints.with_output_types(Tuple[_SlicedXYKey, _CountType])
class _GetPlaceholderCopresenceCounts(beam.PTransform):
  """A PTransform for computing all possible x-y pairs, to support 0 lifts."""

  def __init__(self, x_paths: Iterable[types.FeaturePath], min_x_count: int):
    self._x_paths = x_paths
    self._min_x_count = min_x_count

  def expand(self, x_counts_and_ys: Tuple[Tuple[_SlicedXKey, _CountType],
                                          _SlicedYKey]):
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

  def __init__(self, y_path: types.FeaturePath,
               y_boundaries: Optional[np.ndarray],
               x_paths: Iterable[types.FeaturePath], min_x_count: int,
               weight_column_name: Optional[Text]):
    self._y_path = y_path
    self._y_boundaries = y_boundaries
    self._x_paths = x_paths
    self._min_x_count = min_x_count
    self._weight_column_name = weight_column_name

  def expand(self, sliced_record_batchs_and_ys: Tuple[types.SlicedRecordBatch,
                                                      _SlicedYKey]):
    sliced_record_batchs, y_keys = sliced_record_batchs_and_ys

    # _SlicedXYKey(slice, x_path, x, y), xy_count
    partial_copresence_counts = (
        sliced_record_batchs
        | 'ToPartialCopresenceCounts' >> beam.FlatMap(
            _to_partial_copresence_counts, self._y_path, self._x_paths,
            self._y_boundaries, self._weight_column_name))

    # Compute placerholder copresence counts.
    # partial_copresence_counts will only include x-y pairs that are present,
    # but we would also like to keep track of x-y pairs that never appear, as
    # long as x and y independently occur in the slice.

    # _SlicedXKey(slice, x_path, x), x_count
    x_counts = (
        sliced_record_batchs
        | 'ToPartialXCounts' >> beam.FlatMap(
            _to_partial_x_counts, self._x_paths, self._weight_column_name)
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


@beam.typehints.with_input_types(types.SlicedRecordBatch)
@beam.typehints.with_output_types(Tuple[_SlicedYKey, _YRate])
class _GetYRates(beam.PTransform):
  """A PTransform for computing the rate of each y value within each slice."""

  def __init__(self, y_path: types.FeaturePath,
               y_boundaries: Optional[np.ndarray],
               weight_column_name: Optional[Text]):
    self._y_path = y_path
    self._y_boundaries = y_boundaries
    self._weight_column_name = weight_column_name

  def expand(self, sliced_record_batchs):
    # slice, example_count
    example_counts = (
        sliced_record_batchs
        | 'ToExampleCounts' >> beam.MapTuple(lambda k, v: (k, v.num_rows))
        | 'SumExampleCounts' >> beam.CombinePerKey(sum))

    def move_y_to_value(slice_and_y, y_count):
      slice_key, y = slice_and_y
      return slice_key, (y, y_count)

    # slice, (y, y_count)
    y_counts = (
        sliced_record_batchs
        | 'ToPartialYCounts' >>
        beam.FlatMap(_to_partial_counts, self._y_path, self._y_boundaries,
                     self._weight_column_name)
        | 'SumYCounts' >> beam.CombinePerKey(sum)
        | 'MoveYToValue' >> beam.MapTuple(move_y_to_value))

    # _SlicedYKey(slice, y), _YRate(y_count, example_count)
    return ({
        'y_counts': y_counts,
        'example_count': example_counts
    }
            | 'CoGroupByForYRates' >> beam.CoGroupByKey()
            | 'JoinExampleCounts' >> beam.FlatMap(_join_example_counts))


@beam.typehints.with_input_types(types.SlicedRecordBatch)
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
               y_boundaries: Optional[Sequence[float]], min_x_count: int,
               top_k_per_y: Optional[int], bottom_k_per_y: Optional[int],
               weight_column_name: Optional[Text],
               output_custom_stats: bool, name: Text) -> None:
    """Initializes a lift statistics generator.

    Args:
      y_path: The path to use as Y in the lift expression: lift = P(Y=y|X=x) /
        P(Y=y).
     schema: An optional schema for the dataset. If not provided, x_paths must
       be specified. If x_paths are not specified, the schema is used to
       identify all categorical columns for which Lift should be computed.
      x_paths: An optional list of path to use as X in the lift expression: lift
        = P(Y=y|X=x) / P(Y=y). If None (default), all categorical features,
        exluding the feature passed as y_path, will be used.
      y_boundaries: An optional list of boundaries to be used for binning
        y_path. If provided with b boundaries, the binned values will be treated
        as a categorical feature with b+1 different values. For example, the
        y_boundaries value [0.1, 0.8] would lead to three buckets: [-inf, 0.1),
          [0.1, 0.8) and [0.8, inf].
      min_x_count: The minimum number of examples in which a specific x value
        must appear, in order for its lift to be output.
      top_k_per_y: Optionally, the number of top x values per y value, ordered
        by descending lift, for which to output lift. If both top_k_per_y and
        bottom_k_per_y are unset, all values will be output.
      bottom_k_per_y: Optionally, the number of bottom x values per y value,
        ordered by descending lift, for which to output lift. If both
        top_k_per_y and bottom_k_per_y are unset, all values will be output.
      weight_column_name: Optionally, a weight column to use for converting
        counts of x or y into weighted counts.
      output_custom_stats: Whether to output custom stats for use with Facets.
      name: An optional unique name associated with the statistics generator.
    """
    self._name = name
    self._schema = schema
    self._y_path = y_path
    self._min_x_count = min_x_count
    self._top_k_per_y = top_k_per_y
    self._bottom_k_per_y = bottom_k_per_y
    self._output_custom_stats = output_custom_stats
    self._y_boundaries = (
        np.array(sorted(set(y_boundaries))) if y_boundaries else None)
    self._weight_column_name = weight_column_name

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

  def expand(
      self,
      sliced_record_batchs: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    # Compute P(Y=y)
    # _SlicedYKey(slice, y), _YRate(y_count, example_count)
    y_rates = sliced_record_batchs | 'GetYRates' >> _GetYRates(
        self._y_path, self._y_boundaries, self._weight_column_name)
    y_keys = y_rates | 'ExtractYKeys' >> beam.Keys()

    # Compute P(Y=y | X=x)
    # _SlicedYKey(slice, y), _ConditionalYRate(x_path, x, xy_count, x_count)
    conditional_y_rates = ((sliced_record_batchs, y_keys)
                           | 'GetConditionalYRates' >> _GetConditionalYRates(
                               self._y_path, self._y_boundaries, self._x_paths,
                               self._min_x_count, self._weight_column_name))

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
                                   self._y_path, self._y_boundaries,
                                   self._weight_column_name is not None,
                                   self._output_custom_stats))


@beam.typehints.with_input_types(types.SlicedRecordBatch)
@beam.typehints.with_output_types(Tuple[types.SliceKey,
                                        statistics_pb2.DatasetFeatureStatistics]
                                 )
class _UnweightedAndWeightedLiftStatsGenerator(beam.PTransform):
  """A PTransform to compute both unweighted and weighted lift.

  This simply wraps the logic in _LiftStatsGenerator and, depending on the value
  of weight_column_name, either calls it once to compute unweighted lift, or
  twice to compute both the unweighted and weighted lift. The result will be a
  PCollection of stats per slice, with possibly two stats protos for the same
  slice: one for the unweighted lift and one for the weighted lift.
  """

  def __init__(self, weight_column_name: Optional[Text], **kwargs):
    """Initializes a weighted lift statistics generator.

    Args:
      weight_column_name: Optionally, a weight column to use for converting
        counts of x or y into weighted counts.
      **kwargs: The set of args to be passed to _LiftStatsGenerator.
    """
    self._unweighted_generator = _LiftStatsGenerator(
        weight_column_name=None, **kwargs)
    self._weight_column_name = weight_column_name
    if weight_column_name:
      self._weighted_generator = _LiftStatsGenerator(
          weight_column_name=weight_column_name, **kwargs)

  def expand(
      self,
      sliced_record_batchs: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    unweighted_protos = (
        sliced_record_batchs
        | 'ComputeUnweightedLift' >> self._unweighted_generator)
    if not self._weight_column_name:
      # If no weight column name is given, only compute unweighted lift.
      return unweighted_protos

    weighted_protos = (
        sliced_record_batchs
        | 'ComputeWeightedLift' >> self._weighted_generator)

    return ((unweighted_protos, weighted_protos)
            | 'MergeUnweightedAndWeightedProtos' >> beam.Flatten())


class LiftStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform stats generator for computing lift between two features."""

  def __init__(self,
               y_path: types.FeaturePath,
               schema: Optional[schema_pb2.Schema] = None,
               x_paths: Optional[Iterable[types.FeaturePath]] = None,
               y_boundaries: Optional[Sequence[float]] = None,
               min_x_count: int = 0,
               top_k_per_y: Optional[int] = None,
               bottom_k_per_y: Optional[int] = None,
               weight_column_name: Optional[Text] = None,
               output_custom_stats: Optional[bool] = False,
               name: Text = 'LiftStatsGenerator') -> None:
    super(LiftStatsGenerator, self).__init__(
        name,
        ptransform=_UnweightedAndWeightedLiftStatsGenerator(
            weight_column_name=weight_column_name,
            schema=schema,
            y_path=y_path,
            x_paths=x_paths,
            y_boundaries=y_boundaries,
            min_x_count=min_x_count,
            top_k_per_y=top_k_per_y,
            bottom_k_per_y=bottom_k_per_y,
            output_custom_stats=output_custom_stats,
            name=name),
        schema=schema)
