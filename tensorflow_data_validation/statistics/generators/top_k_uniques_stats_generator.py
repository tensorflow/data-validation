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

"""Computes top-k most frequent values and number of unique values.

This generator computes these values for string and categorical features.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import logging
import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util
from typing import Any, Iterable, Iterator, FrozenSet, List, Optional, Set, Text, Tuple, Union

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

FeatureValueCount = collections.namedtuple('FeatureValueCount',
                                           ['feature_value', 'count'])

# Pickling types.FeaturePath is slow, so we use tuples directly where pickling
# happens frequently.
FeaturePathTuple = Tuple[types.FeatureName]

_INVALID_STRING = '__BYTES_VALUE__'


def _make_feature_stats_proto_with_uniques_stats(
    feature_path: types.FeaturePath, count: int,
    is_categorical: bool) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing the uniques stats."""
  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (
      statistics_pb2.FeatureNameStatistics.INT
      if is_categorical else statistics_pb2.FeatureNameStatistics.STRING)
  result.string_stats.unique = count
  return result


def _make_dataset_feature_stats_proto_with_uniques_for_single_feature(
    feature_path_to_value_count: Tuple[Tuple[types.SliceKey,
                                             FeaturePathTuple], int],
    categorical_features: Set[types.FeaturePath]
) -> Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]:
  """Makes a DatasetFeatureStatistics proto with uniques stats for a feature."""
  (slice_key, feature_path_tuple), count = feature_path_to_value_count
  feature_path = types.FeaturePath(feature_path_tuple)
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto_with_uniques_stats(
          feature_path, count, feature_path in categorical_features))
  return slice_key, result


def make_feature_stats_proto_with_topk_stats(
    feature_path: types.FeaturePath,
    top_k_value_count_list: List[FeatureValueCount], is_categorical: bool,
    is_weighted_stats: bool, num_top_values: int,
    frequency_threshold: Union[float, int],
    num_rank_histogram_buckets: int) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing the top-k stats.

  Args:
    feature_path: The path of the feature.
    top_k_value_count_list: A list of FeatureValueCount tuples.
    is_categorical: Whether the feature is categorical.
    is_weighted_stats: Whether top_k_value_count_list incorporates weights.
    num_top_values: The number of most frequent feature values to keep for
      string features.
    frequency_threshold: The minimum number of examples (possibly weighted) the
      most frequent values must be present in.
    num_rank_histogram_buckets: The number of buckets in the rank histogram for
      string features.

  Returns:
    A FeatureNameStatistics proto containing the top-k stats.
  """
  # Sort (a copy of) the top_k_value_count_list in descending order by count.
  # Where multiple feature values have the same count, consider the feature with
  # the 'larger' feature value to be larger for purposes of breaking the tie.
  top_k_value_count_list = sorted(
      top_k_value_count_list,
      key=lambda counts: (counts[1], counts[0]),
      reverse=True)

  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (statistics_pb2.FeatureNameStatistics.INT if is_categorical
                 else statistics_pb2.FeatureNameStatistics.STRING)

  if is_weighted_stats:
    string_stats = result.string_stats.weighted_string_stats
  else:
    string_stats = result.string_stats

  for i in range(len(top_k_value_count_list)):
    value, count = top_k_value_count_list[i]
    if count < frequency_threshold:
      break
    # Check if we have a valid utf-8 string. If not, assign a default invalid
    # string value.
    if isinstance(value, six.binary_type):
      value = stats_util.maybe_get_utf8(value)
      if value is None:
        logging.warning('Feature "%s" has bytes value "%s" which cannot be '
                        'decoded as a UTF-8 string.', feature_path, value)
        value = _INVALID_STRING
    elif not isinstance(value, six.text_type):
      value = str(value)

    if i < num_top_values:
      freq_and_value = string_stats.top_values.add()
      freq_and_value.value = value
      freq_and_value.frequency = count
    if i < num_rank_histogram_buckets:
      bucket = string_stats.rank_histogram.buckets.add()
      bucket.low_rank = i
      bucket.high_rank = i
      bucket.sample_count = count
      bucket.label = value
  return result


def _make_dataset_feature_stats_proto_with_topk_for_single_feature(
    feature_path_to_value_count_list: Tuple[Tuple[types.SliceKey,
                                                  FeaturePathTuple],
                                            List[FeatureValueCount]],
    categorical_features: Set[types.FeaturePath], is_weighted_stats: bool,
    num_top_values: int, frequency_threshold: float,
    num_rank_histogram_buckets: int
) -> Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]:
  """Makes a DatasetFeatureStatistics proto with top-k stats for a feature."""
  (slice_key, feature_path_tuple), value_count_list = (
      feature_path_to_value_count_list)
  feature_path = types.FeaturePath(feature_path_tuple)
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      make_feature_stats_proto_with_topk_stats(
          feature_path, value_count_list, feature_path in categorical_features,
          is_weighted_stats, num_top_values, frequency_threshold,
          num_rank_histogram_buckets))
  return slice_key, result


def _weighted_unique(values: np.ndarray, weights: np.ndarray
                    ) -> Iterator[Tuple[Any, int, Union[int, float]]]:
  """Computes weighted uniques.

  Args:
    values: 1-D array.
    weights: 1-D numeric array. Should have the same size as `values`.
  Returns:
    An iterator of tuples (unique_value, count, sum_weight).

  Implementation note: we use Pandas and pay the cost of copying the
  input numpy arrays into a DataFrame because Pandas can perform group-by
  without sorting. A numpy-only implementation with sorting is possible but
  slower because of the calls to the string comparator.
  """
  df = pd.DataFrame({
      'value': values,
      'count': np.ones_like(values, dtype=np.int32),
      'weight': weights,
  })
  gb = df.groupby(
      'value', as_index=False, sort=False)['count', 'weight'].sum()
  return six.moves.zip(
      gb['value'].tolist(), gb['count'].tolist(), gb['weight'].tolist())


def _to_topk_tuples(
    sliced_table: Tuple[Text, pa.Table],
    bytes_features: FrozenSet[types.FeaturePath],
    categorical_features: FrozenSet[types.FeaturePath],
    weight_feature: Optional[Text]
) -> Iterable[
    Tuple[Tuple[Text, FeaturePathTuple, Any],
          Union[int, Tuple[int, Union[int, float]]]]]:
  """Generates tuples for computing top-k and uniques from input tables."""
  slice_key, table = sliced_table

  for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
      table,
      weight_column=weight_feature,
      enumerate_leaves_only=True):
    feature_array_type = feature_array.type
    if pa.types.is_null(feature_array_type):
      continue
    if feature_path in bytes_features:
      continue
    if (feature_path in categorical_features or
        stats_util.get_feature_type_from_arrow_type(
            feature_path,
            feature_array_type) == statistics_pb2.FeatureNameStatistics.STRING):
      flattened_values = feature_array.flatten()
      if weights is not None and flattened_values:
        # Slow path: weighted uniques.
        flattened_values_np = np.asarray(flattened_values)
        parent_indices = (
            np.asarray(
                array_util.GetFlattenedArrayParentIndices(feature_array)))
        weights_ndarray = weights[parent_indices]
        for value, count, weight in _weighted_unique(
            flattened_values_np, weights_ndarray):
          yield (slice_key, feature_path.steps(), value), (count, weight)
      else:
        value_counts = array_util.ValueCounts(flattened_values)
        values = value_counts.field('values').to_pylist()
        counts = value_counts.field('counts').to_pylist()
        for value, count in six.moves.zip(values, counts):
          yield ((slice_key, feature_path.steps(), value), count)


class _ComputeTopKUniquesStats(beam.PTransform):
  """A ptransform that computes top-k and uniques for string features."""

  def __init__(self, schema: schema_pb2.Schema,
               weight_feature: types.FeatureName, num_top_values: int,
               frequency_threshold: int, weighted_frequency_threshold: float,
               num_rank_histogram_buckets: int):
    """Initializes _ComputeTopKUniquesStats.

    Args:
      schema: An schema for the dataset. None if no schema is available.
      weight_feature: Feature name whose numeric value represents the weight
          of an example. None if there is no weight feature.
      num_top_values: The number of most frequent feature values to keep for
          string features.
      frequency_threshold: The minimum number of examples the most frequent
          values must be present in.
      weighted_frequency_threshold: The minimum weighted number of examples the
          most frequent weighted values must be present in.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
          for string features.
    """
    self._bytes_features = set(
        schema_util.get_bytes_features(schema) if schema else [])
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_top_values = num_top_values
    self._frequency_threshold = frequency_threshold
    self._weighted_frequency_threshold = weighted_frequency_threshold
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:

    def _sum_pairwise(
        iter_of_pairs: Iterator[Tuple[Union[int, float], Union[int, float]]]
    ) -> Tuple[Union[int, float], Union[int, float]]:
      """Computes sum of counts and weights."""
      # We take advantage of the fact that constructing a np array from a list
      # is much faster as the length is known beforehand.
      if isinstance(iter_of_pairs, list):
        arr = np.array(
            iter_of_pairs, dtype=[('c', np.int64), ('w', np.float)])
      else:
        arr = np.fromiter(
            iter_of_pairs, dtype=[('c', np.int64), ('w', np.float)])
      return arr['c'].sum(), arr['w'].sum()

    if self._weight_feature is not None:
      sum_fn = _sum_pairwise
    else:
      # For non-weighted case, use sum combine fn over integers to allow Beam
      # to use Cython combiner.
      sum_fn = sum
    top_k_tuples_combined = (
        pcoll
        | 'ToTopKTuples' >> beam.FlatMap(
            _to_topk_tuples,
            bytes_features=self._bytes_features,
            categorical_features=self._categorical_features,
            weight_feature=self._weight_feature)
        | 'CombineCountsAndWeights' >> beam.CombinePerKey(sum_fn)
        | 'Rearrange' >> beam.MapTuple(lambda k, v: ((k[0], k[1]), (v, k[2]))))
    # (slice_key, feature), (count_and_maybe_weight, value)

    top_k = top_k_tuples_combined
    if self._weight_feature is not None:
      top_k |= 'Unweighted_DropWeightsAndRearrange' >> beam.MapTuple(
          lambda k, v: (k, (v[0][0], v[1])))
      # (slice_key, feature), (count, value)
    top_k = (
        top_k
        | 'Unweighted_TopK' >> beam.combiners.Top().PerKey(
            max(self._num_top_values, self._num_rank_histogram_buckets))
        | 'Unweighted_ToFeatureValueCount' >> beam.MapTuple(
            lambda k, v: (k, [FeatureValueCount(t[1], t[0]) for t in v]))
        | 'Unweighted_ToProto' >> beam.Map(
            _make_dataset_feature_stats_proto_with_topk_for_single_feature,
            categorical_features=self._categorical_features,
            is_weighted_stats=False,
            num_top_values=self._num_top_values,
            frequency_threshold=self._frequency_threshold,
            num_rank_histogram_buckets=self._num_rank_histogram_buckets))
    uniques = (
        top_k_tuples_combined
        | 'Uniques_Keys' >> beam.Keys()
        | 'Uniques_CountPerFeatureName' >> beam.combiners.Count().PerElement()
        | 'Uniques_ConvertToSingleFeatureStats' >> beam.Map(
            _make_dataset_feature_stats_proto_with_uniques_for_single_feature,
            categorical_features=self._categorical_features))
    result_protos = [top_k, uniques]

    if self._weight_feature is not None:
      weighted_top_k = (
          top_k_tuples_combined
          | 'Weighted_DropCountsAndRearrange'
          >> beam.MapTuple(lambda k, v: (k, (v[0][1], v[1])))
          # (slice_key, feature), (weight, value)
          | 'Weighted_TopK' >> beam.combiners.Top().PerKey(
              max(self._num_top_values, self._num_rank_histogram_buckets))
          | 'Weighted_ToFeatureValueCount' >> beam.MapTuple(
              lambda k, v: (k, [FeatureValueCount(t[1], t[0]) for t in v]))
          | 'Weighted_ToProto' >> beam.Map(
              _make_dataset_feature_stats_proto_with_topk_for_single_feature,
              categorical_features=self._categorical_features,
              is_weighted_stats=True,
              num_top_values=self._num_top_values,
              frequency_threshold=self._weighted_frequency_threshold,
              num_rank_histogram_buckets=self._num_rank_histogram_buckets))
      result_protos.append(weighted_top_k)

    return (result_protos
            | 'FlattenTopKUniquesFeatureStatsProtos' >> beam.Flatten())


class TopKUniquesStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes top-k and uniques."""

  def __init__(self,
               name: Text = 'TopKUniquesStatsGenerator',
               schema: Optional[schema_pb2.Schema] = None,
               weight_feature: Optional[types.FeatureName] = None,
               num_top_values: int = 2,
               frequency_threshold: int = 1,
               weighted_frequency_threshold: float = 1.0,
               num_rank_histogram_buckets: int = 1000) -> None:
    """Initializes top-k and uniques stats generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: An optional feature name whose numeric value
          (must be of type INT or FLOAT) represents the weight of an example.
      num_top_values: An optional number of most frequent feature values to keep
          for string features (defaults to 2).
      frequency_threshold: An optional minimum number of examples
        the most frequent values must be present in (defaults to 1).
      weighted_frequency_threshold: An optional minimum weighted
        number of examples the most frequent weighted values must be
        present in (defaults to 1.0).
      num_rank_histogram_buckets: An optional number of buckets in the rank
          histogram for string features (defaults to 1000).
    """
    super(TopKUniquesStatsGenerator, self).__init__(
        name,
        schema=schema,
        ptransform=_ComputeTopKUniquesStats(
            schema=schema,
            weight_feature=weight_feature,
            num_top_values=num_top_values,
            frequency_threshold=frequency_threshold,
            weighted_frequency_threshold=weighted_frequency_threshold,
            num_rank_histogram_buckets=num_rank_histogram_buckets))
