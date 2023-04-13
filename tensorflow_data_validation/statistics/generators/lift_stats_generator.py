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
"""Provides LiftStatsGenerator for quantifying feature-label correlations."""

import collections
import datetime
import operator
from typing import Any, Dict, Hashable, Iterator, Iterable, List, Optional, Sequence, Text, Tuple, TypeVar, Union

import apache_beam as beam
from apache_beam.utils import shared
import numpy as np
import pyarrow as pa
import six

from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import bin_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl.arrow import array_util
from tfx_bsl.arrow import path as tfx_bsl_path
from tfx_bsl.arrow import table_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(b/170996403): Switch to`collections.namedtuple` or `typing.NamedTuple`
# once the Spark issue is resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order

_XType = Union[Text, bytes]
_YType = Union[Text, bytes, int]
_CountType = Union[int, float]

_JoinKeyType = TypeVar('_JoinKeyType')

_LeftJoinValueType = TypeVar('_LeftJoinValueType')

_RightJoinValueType = TypeVar('_RightJoinValueType')

_SlicedYKey = tfx_namedtuple.TypedNamedTuple('_SlicedYKey',
                                             [('slice_key', types.SliceKey),
                                              ('y', _YType)])

# TODO(embr,zhuo): FeaturePathTuple is used instead of FeaturePath because:
#  - FeaturePath does not have a deterministic coder
#  - Even if it does, beam does not automatically derive a coder for a
#    NamedTuple.
#  Once the latter is supported we can change all FEaturePathTuples back to
#  FeaturePaths.
_SlicedXKey = tfx_namedtuple.TypedNamedTuple(
    '_SlicedXKey', [('slice_key', types.SliceKey),
                    ('x_path', types.FeaturePathTuple), ('x', _XType)])

_SlicedXYKey = tfx_namedtuple.TypedNamedTuple(
    '_SlicedXYKey', [('slice_key', types.SliceKey),
                     ('x_path', types.FeaturePathTuple), ('x', _XType),
                     ('y', _YType)])

_LiftSeriesKey = tfx_namedtuple.TypedNamedTuple(
    '_LiftSeriesKey', [('slice_key', types.SliceKey),
                       ('x_path', types.FeaturePathTuple), ('y', _YType),
                       ('y_count', _CountType)])

_SlicedFeatureKey = tfx_namedtuple.TypedNamedTuple(
    '_SlicedFeatureKey', [('slice_key', types.SliceKey),
                          ('x_path', types.FeaturePathTuple)])

_ConditionalYRate = tfx_namedtuple.TypedNamedTuple(
    '_ConditionalYRate', [('x_path', types.FeaturePathTuple), ('x', _XType),
                          ('xy_count', _CountType), ('x_count', _CountType)])

_YRate = tfx_namedtuple.TypedNamedTuple('_YRate',
                                        [('y_count', _CountType),
                                         ('example_count', _CountType)])

_LiftInfo = tfx_namedtuple.TypedNamedTuple('_LiftInfo',
                                           [('x', _XType), ('y', _YType),
                                            ('lift', float),
                                            ('xy_count', _CountType),
                                            ('x_count', _CountType),
                                            ('y_count', _CountType)])

_LiftValue = tfx_namedtuple.TypedNamedTuple('_LiftValue',
                                            [('x', _XType), ('lift', float),
                                             ('xy_count', _CountType),
                                             ('x_count', _CountType)])

_LiftSeries = tfx_namedtuple.TypedNamedTuple(
    '_LiftSeries', [('y', _YType), ('y_count', _CountType),
                    ('lift_values', Iterable[_LiftValue])])
_ValuePresence = tfx_namedtuple.TypedNamedTuple(
    '_ValuePresence', [('example_indices', np.ndarray), ('values', np.ndarray),
                       ('weights', np.ndarray)])

# Beam counter to track the number of non-utf8 values.
_NON_UTF8_VALUES_COUNTER = beam.metrics.Metrics.counter(
    constants.METRICS_NAMESPACE, 'num_non_utf8_values_lift_generator')


def _get_example_value_presence(
    record_batch: pa.RecordBatch, path: types.FeaturePath,
    boundaries: Optional[Sequence[float]],
    weight_column_name: Optional[Text]) -> Optional[_ValuePresence]:
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
    A _ValuePresence tuple which contains three numpy arrays: example indices,
    values, and weights.
  """
  arr, example_indices = table_util.get_array(
      record_batch,
      tfx_bsl_path.ColumnPath(path.steps()),
      return_example_indices=True,
  )
  if stats_util.get_feature_type_from_arrow_type(path, arr.type) is None:
    return

  arr_flat, parent_indices = array_util.flatten_nested(
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
    return
  # Deduplicate values which show up more than once in the same example. This
  # makes P(X=x|Y=y) in the standard lift definition behave as
  # P(x \in Xs | y \in Ys) if examples contain more than one value of X and Y.
  unique_rows = np.unique(rows, axis=1)
  example_indices = unique_rows[0, :]
  values = unique_rows[1, :]
  if is_binary_like:
    # return binary like values a pd.Categorical wrapped in a Series. This makes
    # subsqeuent operations like pd.Merge cheaper.
    values = arr_flat_dict[values].tolist()
  else:
    values = values.tolist()  # converts values to python native types.
  if weight_column_name:
    weights = arrow_util.get_weight_feature(record_batch, weight_column_name)
    weights = np.asarray(weights)[example_indices].tolist()
  else:
    weights = np.ones(len(example_indices), dtype=int).tolist()
  return _ValuePresence(example_indices.tolist(), values, weights)


def _to_partial_copresence_counts(
    sliced_record_batch: types.SlicedRecordBatch,
    y_path: types.FeaturePath,
    x_paths: Iterable[types.FeaturePath],
    y_boundaries: Optional[np.ndarray],
    example_weight_map: ExampleWeightMap,
    num_xy_pairs_batch_copresent: Optional[
        beam.metrics.metric.Metrics.DelegatingDistribution] = None
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
    example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
        corresponding weight column.
    num_xy_pairs_batch_copresent: A counter tracking the number of different xy
      pairs that are copresent within each batch. If the same pair of xy values
      are copresent in more than one batch, this counter will be incremented
      once for each batch in which they are copresent.

  Yields:
    Tuples of the form (_SlicedXYKey(slice_key, x_path, x, y), count) for each
    combination of  x_path, x, and y  in the input record batch.
  """
  slice_key, record_batch = sliced_record_batch
  y_presence = _get_example_value_presence(
      record_batch, y_path, y_boundaries, weight_column_name=None)
  if y_presence is None:
    return
  ys_by_example = collections.defaultdict(list)
  for example_index, y in zip(y_presence.example_indices, y_presence.values):
    ys_by_example[example_index].append(y)
  for x_path in x_paths:
    weight_column_name = example_weight_map.get(x_path)
    x_presence = _get_example_value_presence(
        record_batch,
        x_path,
        boundaries=None,
        weight_column_name=weight_column_name)
    if x_presence is None:
      continue
    if weight_column_name is not None:
      copresence_counts = collections.defaultdict(float)
    else:
      copresence_counts = collections.defaultdict(int)

    for example_index, x, weight in zip(x_presence.example_indices,
                                        x_presence.values, x_presence.weights):
      for y in ys_by_example[example_index]:
        copresence_counts[(x, y)] += weight

    if num_xy_pairs_batch_copresent:
      num_xy_pairs_batch_copresent.update(len(copresence_counts))
    for (x, y), count in copresence_counts.items():
      sliced_xy_key = _SlicedXYKey(
          slice_key=slice_key, x_path=x_path.steps(), x=x, y=y)
      yield sliced_xy_key, count


def _to_partial_counts(
    sliced_record_batch: types.SlicedRecordBatch, path: types.FeaturePath,
    boundaries: Optional[np.ndarray], weight_column_name: Optional[Text]
) -> Iterator[Tuple[Tuple[types.SliceKey, Union[_XType, _YType]], _CountType]]:
  """Yields per-(slice, value) counts of the examples with value in path."""
  slice_key, record_batch = sliced_record_batch
  value_presence = _get_example_value_presence(record_batch, path, boundaries,
                                               weight_column_name)
  if value_presence is None:
    return value_presence

  if weight_column_name is not None:
    grouped_values = collections.defaultdict(float)
  else:
    grouped_values = collections.defaultdict(int)

  for value, weight in zip(value_presence.values, value_presence.weights):
    grouped_values[value] += weight

  for value, count in grouped_values.items():
    yield (slice_key, value), count


def _to_partial_x_counts(
    sliced_record_batch: types.SlicedRecordBatch,
    x_paths: Iterable[types.FeaturePath], example_weight_map: ExampleWeightMap
) -> Iterator[Tuple[_SlicedXKey, _CountType]]:
  """Yields per-(slice, x_path, x) counts of the examples with x in x_path."""
  for x_path in x_paths:
    for (slice_key, x), x_count in _to_partial_counts(
        sliced_record_batch,
        x_path,
        boundaries=None,
        weight_column_name=example_weight_map.get(x_path)):
      yield _SlicedXKey(slice_key, x_path.steps(), x), x_count


def _get_unicode_value(value: Union[Text, bytes]) -> Text:
  """Get feature value decoded as utf-8."""
  decoded_value = stats_util.maybe_get_utf8(value)
  # Check if we have a valid utf-8 string. If not, assign a placeholder.
  if decoded_value is None:
    _NON_UTF8_VALUES_COUNTER.inc()
    decoded_value = constants.NON_UTF8_PLACEHOLDER
  return decoded_value


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
  x_path = types.FeaturePath(key.x_path)
  stats = statistics_pb2.DatasetFeatureStatistics()
  cross_stats = stats.cross_features.add(
      path_x=x_path.to_proto(), path_y=y_path.to_proto())
  if output_custom_stats:
    feature_stats = stats.features.add(path=x_path.to_proto())
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
      y_string = _get_unicode_value(y)
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
        x_string = _get_unicode_value(x)
        lift_value_proto.x_string = x_string
        x_display_val = x_string
      else:
        lift_value_proto.x_int = x
        x_display_val = str(x)

      if output_custom_stats:
        hist.buckets.add(label=x_display_val, sample_count=lift_value.lift)

  return key.slice_key, stats


def _make_placeholder_counts(
    join_result: Tuple[types.SliceKey, Tuple[types.FeaturePathTuple, _XType,
                                             _CountType], _YType]
) -> Tuple[_SlicedXYKey, _CountType]:
  slice_key, x_path_value_and_count, y = join_result
  x_path, x, _ = x_path_value_and_count
  return _SlicedXYKey(slice_key=slice_key, x_path=x_path, x=x, y=y), 0


def _make_conditional_y_rates(
    join_result: Tuple[_SlicedXKey, Tuple[_YType, _CountType], _CountType],
    num_xy_pairs_distinct: beam.metrics.metric.Metrics.DelegatingCounter
) -> Tuple[_SlicedYKey, _ConditionalYRate]:
  """Creates conditional y rates from slice y rates and the per-x y rates."""
  sliced_x_key, y_and_xy_count, x_count = join_result
  y, xy_count = y_and_xy_count
  num_xy_pairs_distinct.inc(1)
  sliced_y_key = _SlicedYKey(sliced_x_key.slice_key, y)
  conditional_y_rate = _ConditionalYRate(
      x_path=sliced_x_key.x_path,
      x=sliced_x_key.x,
      xy_count=xy_count,
      x_count=x_count)
  return sliced_y_key, conditional_y_rate


def _make_y_rates(
    join_result: Tuple[types.SliceKey, Tuple[_YType, _CountType], _CountType]
) -> Tuple[_SlicedYKey, _YRate]:
  slice_key, y_and_count, example_count = join_result
  y, y_count = y_and_count
  sliced_y_key = _SlicedYKey(slice_key, y)
  y_rate = _YRate(y_count=y_count, example_count=example_count)
  return sliced_y_key, y_rate


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


class _WeakRefFrozenMapping(collections.abc.Mapping, object):
  """A weakly-referencable dict, necessary to allow use with shared.Shared.

  Note that the mapping will not be frozen until freeze() is called.
  """

  def __init__(self):
    self._dict = {}
    self._is_frozen = False

  def __setitem__(self, key: Hashable, value: Any):
    assert not self._is_frozen
    self._dict[key] = value

  def freeze(self):
    self._is_frozen = True

  def __getitem__(self, key: Hashable) -> Any:
    return self._dict[key]

  def __iter__(self) -> Iterator[Hashable]:
    return iter(self._dict)

  def __len__(self) -> int:
    return len(self._dict)


class _LookupInnerJoinDoFn(beam.DoFn):
  """A DoFn which performs a lookup inner join using a side input."""

  def __init__(self):
    self._shared_handle = shared.Shared()
    self._right_lookup_contruction_seconds_distribution = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'right_lookup_construction_seconds'))
    # These should be gauges, but not all runners support gauges so they are
    # made distributions, which are equivalent.
    # TODO(b/130840752): support gauges in the internal runner.
    self._right_lookup_num_keys = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'right_lookup_num_keys'))
    self._right_lookup_num_values = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'right_lookup_num_values'))

  def process(
      self, left_element: Tuple[_JoinKeyType, _LeftJoinValueType],
      right_iterable: Iterable[Tuple[_JoinKeyType, _RightJoinValueType]]
  ) -> Iterator[Tuple[_JoinKeyType, _LeftJoinValueType, _RightJoinValueType]]:

    def construct_lookup():
      start = datetime.datetime.now()
      result = _WeakRefFrozenMapping()
      num_values = 0
      for key, value in right_iterable:
        lst = result.get(key, None)
        if lst is None:
          lst = []
          result[key] = lst
        lst.append(value)
        num_values += 1
      result.freeze()
      self._right_lookup_contruction_seconds_distribution.update(
          int((datetime.datetime.now() - start).total_seconds()))
      self._right_lookup_num_keys.update(len(result))
      self._right_lookup_num_values.update(num_values)
      return result

    right_lookup = self._shared_handle.acquire(construct_lookup)
    key, left_value = left_element
    right_values = right_lookup.get(key)
    if right_values is None:
      return
    for right_value in right_values:
      yield key, left_value, right_value


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
      lift_series_key = _LiftSeriesKey(
          slice_key=slice_key, x_path=x_path, y=value.y, y_count=value.y_count)
      lift_value = _LiftValue(
          x=value.x,
          lift=value.lift,
          xy_count=value.xy_count,
          x_count=value.x_count)
      return lift_series_key, lift_value

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
                       | 'ReGroupTopAndBottom' >> beam.CombinePerKey(
                           beam.combiners.ToListCombineFn()))
    elif self._top_k_per_y:
      grouped_lifts = top_k
    elif self._bottom_k_per_y:
      grouped_lifts = bottom_k
    else:
      grouped_lifts = lifts | 'CombinePerY' >> beam.CombinePerKey(
          beam.combiners.ToListCombineFn())

    def move_y_info_to_value(
        key: _LiftSeriesKey,
        lift_values: List[_LiftValue]) -> Tuple[_SlicedFeatureKey, _LiftSeries]:
      return (_SlicedFeatureKey(key.slice_key, key.x_path),
              _LiftSeries(
                  y=key.y, y_count=key.y_count, lift_values=lift_values))

    # (_SlicedFeatureKey(slice, x_path),
    #      _LiftSeries(y, y_count, [_LiftValue(x, lift, xy_count, x_count)]))
    return (grouped_lifts
            | 'MoveYInfoToValue' >> beam.MapTuple(move_y_info_to_value))


class _GetPlaceholderCopresenceCounts(beam.PTransform):
  """A PTransform for computing all possible x-y pairs, to support 0 lifts."""

  def __init__(self, x_paths: Iterable[types.FeaturePath], min_x_count: int):
    self._x_paths = x_paths
    self._min_x_count = min_x_count

  def expand(
      self, x_counts_and_ys: Tuple[beam.PCollection[Tuple[_SlicedXKey,
                                                          _CountType]],
                                   beam.PCollection[_SlicedYKey]]
  ) -> beam.PCollection[Tuple[_SlicedXYKey, _CountType]]:
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

    # TODO(b/201480787): consider creating the cross product of all distinct
    # x-values and y-values in the entire dataset (rather than per slice)
    # _SlicedXYKey(slice, x_path, x, y), 0
    return (x_counts_by_slice
            | 'JoinWithPlaceholderYRates' >> beam.ParDo(
                _LookupInnerJoinDoFn(),
                right_iterable=beam.pvalue.AsIter(y_keys_by_slice))
            | 'MakePlaceholderCounts' >> beam.Map(_make_placeholder_counts))


class _GetConditionalYRates(beam.PTransform):
  """A PTransform for computing the rate of each y value, given an x value."""

  def __init__(self, y_path: types.FeaturePath,
               y_boundaries: Optional[np.ndarray],
               x_paths: Iterable[types.FeaturePath], min_x_count: int,
               example_weight_map: Optional[ExampleWeightMap]):
    self._y_path = y_path
    self._y_boundaries = y_boundaries
    self._x_paths = x_paths
    self._min_x_count = min_x_count
    self._example_weight_map = example_weight_map
    self._num_xy_pairs_distinct = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_xy_pairs_distinct')
    self._num_xy_pairs_batch_copresent = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'num_xy_pairs_batch_copresent')

  def expand(
      self, sliced_record_batchs_and_ys: Tuple[
          beam.PCollection[types.SlicedRecordBatch],
          beam.PCollection[_SlicedYKey]]
  ) -> beam.PCollection[Tuple[_SlicedYKey, _ConditionalYRate]]:
    sliced_record_batchs, y_keys = sliced_record_batchs_and_ys

    # _SlicedXYKey(slice, x_path, x, y), xy_count
    partial_copresence_counts = (
        sliced_record_batchs
        | 'ToPartialCopresenceCounts' >> beam.FlatMap(
            _to_partial_copresence_counts, self._y_path, self._x_paths,
            self._y_boundaries, self._example_weight_map,
            self._num_xy_pairs_batch_copresent))

    # Compute placeholder copresence counts.
    # partial_copresence_counts will only include x-y pairs that are present,
    # but we would also like to keep track of x-y pairs that never appear, as
    # long as x and y independently occur in the slice.

    # _SlicedXKey(slice, x_path, x), x_count
    x_counts = (
        sliced_record_batchs
        | 'ToPartialXCounts' >> beam.FlatMap(
            _to_partial_x_counts, self._x_paths, self._example_weight_map)
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
    return (
        copresence_counts
        | 'JoinXCounts' >> beam.ParDo(
            _LookupInnerJoinDoFn(), right_iterable=beam.pvalue.AsIter(x_counts))
        | 'MakeConditionalYRates' >> beam.Map(
            _make_conditional_y_rates,
            num_xy_pairs_distinct=self._num_xy_pairs_distinct))


class _GetYRates(beam.PTransform):
  """A PTransform for computing the rate of each y value within each slice."""

  def __init__(self, y_path: types.FeaturePath,
               y_boundaries: Optional[np.ndarray],
               weight_column_name: Optional[Text]):
    self._y_path = y_path
    self._y_boundaries = y_boundaries
    self._weight_column_name = weight_column_name

  def expand(
      self, sliced_record_batchs: beam.PCollection[types.SlicedRecordBatch]
  ) -> beam.PCollection[Tuple[_SlicedYKey, _YRate]]:
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
    return (y_counts
            | 'JoinExampleCounts' >> beam.ParDo(
                _LookupInnerJoinDoFn(),
                right_iterable=beam.pvalue.AsIter(example_counts))
            | 'MakeYRates' >> beam.Map(_make_y_rates))


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
               y_boundaries: Optional[Iterable[float]], min_x_count: int,
               top_k_per_y: Optional[int], bottom_k_per_y: Optional[int],
               example_weight_map: ExampleWeightMap,
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
      example_weight_map: Optionally, an ExampleWeightMap that maps a
        FeaturePath to its corresponding weight column. If provided and if
        it's not an empty map (i.e. no feature has a corresponding weight column
        ), unweighted lift stats will be populated, otherwise weighted lift
        stats will be populated.
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
    self._example_weight_map = example_weight_map

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
        self._y_path, self._y_boundaries,
        self._example_weight_map.get(self._y_path))
    y_keys = y_rates | 'ExtractYKeys' >> beam.Keys()

    # Compute P(Y=y | X=x)
    # _SlicedYKey(slice, y), _ConditionalYRate(x_path, x, xy_count, x_count)
    conditional_y_rates = ((sliced_record_batchs, y_keys)
                           | 'GetConditionalYRates' >> _GetConditionalYRates(
                               self._y_path, self._y_boundaries, self._x_paths,
                               self._min_x_count, self._example_weight_map))

    return (
        {
            'conditional_y_rate': conditional_y_rates,
            'y_rate': y_rates
        }
        | 'CoGroupByForLift' >> beam.CoGroupByKey()
        | 'ComputeLifts' >> beam.FlatMap(_compute_lifts)
        | 'FilterLifts' >> _FilterLifts(self._top_k_per_y, self._bottom_k_per_y)
        | 'GroupLiftsForOutput' >> beam.GroupByKey()
        | 'MakeProtos' >> beam.Map(
            _make_dataset_feature_stats_proto, self._y_path, self._y_boundaries,
            bool(self._example_weight_map.all_weight_features()),
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

  def __init__(self, example_weight_map: ExampleWeightMap, **kwargs):
    """Initializes a weighted lift statistics generator.

    Args:
      example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
        corresponding weight column.
      **kwargs: The set of args to be passed to _LiftStatsGenerator.
    """
    self._unweighted_generator = _LiftStatsGenerator(
        example_weight_map=ExampleWeightMap(), **kwargs)
    self._has_any_weight = bool(example_weight_map.all_weight_features())
    if self._has_any_weight:
      self._weighted_generator = _LiftStatsGenerator(
          example_weight_map=example_weight_map, **kwargs)

  def expand(
      self,
      sliced_record_batchs: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    unweighted_protos = (
        sliced_record_batchs
        | 'ComputeUnweightedLift' >> self._unweighted_generator)
    if not self._has_any_weight:
      # If no weight column name is given, only compute unweighted lift.
      return unweighted_protos

    weighted_protos = (
        sliced_record_batchs
        | 'ComputeWeightedLift' >> self._weighted_generator)

    return ((unweighted_protos, weighted_protos)
            | 'MergeUnweightedAndWeightedProtos' >> beam.Flatten())


class LiftStatsGenerator(stats_generator.TransformStatsGenerator):
  r"""A transform stats generator for computing lift between two features.

  We define the feature value lift(x_i, y_i) for features X and Y as:

    P(Y=y_i|X=x_i) / P(Y=y_i)

  This quantitatively captures the notion of probabilistic independence, such
  that when X and Y are independent, the lift will be 1. It also indicates the
  degree to which the presence of x_i increases or decreases the probablity of
  the presence of y_i. When X or Y is multivalent, the expressions `X=x_i` and
  `Y=y_i` are intepreted as the set membership checks, `x_i \in X` and
  `y_i \in Y`.

  When Y is a label and Xs are the set of categorical features, lift can be used
  to assess feature importance. However, in the presence of correlated features,
  because lift is computed independently for each feature, it will not be a
  reliable indicator of the expected impact on model quality from adding or
  removing that feature.

  This generator computes lift for a set of feature pairs (y, x_1), ... (y, x_k)
  for a collection of x_paths, and a single y_path. The y_path must be either
  a categorical feature, or numeric feature (in which case binning boundaries
  are also required). The x_paths can be manually provided or will be
  automatically inferred as the set of categorical features in the schema
  (excluding y_path).

  This calculation can also be done using per-example weights. If no
  ExampleWeightMap is provided, or there is no weight for y_path, only
  unweighted lift will be computed. In the case where the ExampleWeightMap
  contains a weight_path or a per-feature override for y_path (y_weight), a
  weighted version of lift will be computed in which each example is treated as
  if it occured y_weight times.
  """

  def __init__(self,
               y_path: types.FeaturePath,
               schema: Optional[schema_pb2.Schema] = None,
               x_paths: Optional[Iterable[types.FeaturePath]] = None,
               y_boundaries: Optional[Iterable[float]] = None,
               min_x_count: int = 0,
               top_k_per_y: Optional[int] = None,
               bottom_k_per_y: Optional[int] = None,
               example_weight_map: ExampleWeightMap = ExampleWeightMap(),
               output_custom_stats: Optional[bool] = False,
               name: Text = 'LiftStatsGenerator') -> None:
    """Initializes a LiftStatsGenerator.

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
      example_weight_map: Optionally, an ExampleWeightMap that maps a
        FeaturePath to its corresponding weight column. If provided and if
        it's not an empty map (i.e. no feature has a corresponding weight column
        ), unweighted lift stats will be populated, otherwise both unweighted
        and weighted lift stats will be populated.
      output_custom_stats: Whether to output custom stats for use with Facets.
      name: An optional unique name associated with the statistics generator.
    """
    super(LiftStatsGenerator, self).__init__(
        name,
        ptransform=_UnweightedAndWeightedLiftStatsGenerator(
            example_weight_map=example_weight_map,
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
