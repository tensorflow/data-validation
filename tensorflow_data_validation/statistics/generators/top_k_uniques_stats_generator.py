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

import logging
from typing import Any, FrozenSet, Iterable, Iterator, Mapping, Optional, Text, Tuple, Union
import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import top_k_uniques_stats_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl.arrow import array_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


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
      'value', as_index=False, sort=False)[['count', 'weight']].sum()
  return zip(gb['value'].tolist(), gb['count'].tolist(), gb['weight'].tolist())


def _should_run(categorical_numeric_types: Mapping[types.FeaturePath,
                                                   'schema_pb2.FeatureType'],
                feature_path: types.FeaturePath,
                feature_type: Optional[int]) -> bool:
  """Check if top-k analysis should run on a feature."""
  # if it's not a categorical int feature nor a string feature, we don't
  # bother with topk stats.
  if feature_type == statistics_pb2.FeatureNameStatistics.STRING:
    return True
  if top_k_uniques_stats_util.output_categorical_numeric(
      categorical_numeric_types, feature_path, feature_type):
    # This top-k uniques generator implementation only supports categorical
    # INT.
    if feature_type == statistics_pb2.FeatureNameStatistics.INT:
      return True
    else:
      logging.error(
          'Categorical float feature %s not supported for TopKUniquesStatsGenerator',
          feature_path)
    return feature_type == statistics_pb2.FeatureNameStatistics.INT
  return False


def _to_topk_tuples(
    sliced_record_batch: Tuple[types.SliceKey, pa.RecordBatch],
    bytes_features: FrozenSet[types.FeaturePath],
    categorical_numeric_types: Mapping[types.FeaturePath,
                                       'schema_pb2.FeatureType'],
    example_weight_map: ExampleWeightMap,
) -> Iterable[Tuple[Tuple[types.SliceKey, types.FeaturePathTuple, Any], Tuple[
    int, Union[int, float]]]]:
  """Generates tuples for computing top-k and uniques from the input."""
  slice_key, record_batch = sliced_record_batch

  for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
      record_batch,
      example_weight_map=example_weight_map,
      enumerate_leaves_only=True):
    feature_array_type = feature_array.type
    feature_type = stats_util.get_feature_type_from_arrow_type(
        feature_path, feature_array_type)
    if feature_path in bytes_features:
      continue
    if not _should_run(categorical_numeric_types, feature_path, feature_type):
      continue
    flattened_values, parent_indices = array_util.flatten_nested(
        feature_array, weights is not None)
    if weights is not None and flattened_values:
      # Slow path: weighted uniques.
      flattened_values_np = np.asarray(flattened_values)
      weights_ndarray = weights[parent_indices]
      for value, count, weight in _weighted_unique(flattened_values_np,
                                                   weights_ndarray):
        yield (slice_key, feature_path.steps(), value), (count, weight)
    else:
      value_counts = flattened_values.value_counts()
      values = value_counts.field('values').to_pylist()
      counts = value_counts.field('counts').to_pylist()
      for value, count in zip(values, counts):
        yield ((slice_key, feature_path.steps(), value), (count, 1))


class _ComputeTopKUniquesStats(beam.PTransform):
  """A ptransform that computes top-k and uniques for string features."""

  def __init__(self, schema: schema_pb2.Schema,
               example_weight_map: ExampleWeightMap, num_top_values: int,
               frequency_threshold: int, weighted_frequency_threshold: float,
               num_rank_histogram_buckets: int):
    """Initializes _ComputeTopKUniquesStats.

    Args:
      schema: An schema for the dataset. None if no schema is available.
      example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
          corresponding weight column.
      num_top_values: The number of most frequent feature values to keep for
          string features.
      frequency_threshold: The minimum number of examples the most frequent
          values must be present in.
      weighted_frequency_threshold: The minimum weighted number of examples the
          most frequent weighted values must be present in.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
          for string features.
    """
    self._bytes_features = frozenset(
        schema_util.get_bytes_features(schema) if schema else [])
    self._categorical_numeric_types = (
        schema_util.get_categorical_numeric_feature_types(schema)
        if schema else {})
    self._example_weight_map = example_weight_map
    self._num_top_values = num_top_values
    self._frequency_threshold = frequency_threshold
    self._weighted_frequency_threshold = weighted_frequency_threshold
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:

    def _sum_pairwise(
        iter_of_pairs: Iterable[Tuple[Union[int, float], Union[int, float]]]
    ) -> Tuple[Union[int, float], Union[int, float]]:
      """Computes sum of counts and weights."""
      # We take advantage of the fact that constructing a np array from a list
      # is much faster as the length is known beforehand.
      if isinstance(iter_of_pairs, list):
        arr = np.array(
            iter_of_pairs, dtype=[('c', np.int64), ('w', float)])
      else:
        arr = np.fromiter(
            iter_of_pairs, dtype=[('c', np.int64), ('w', float)])
      return int(arr['c'].sum()), float(arr['w'].sum())

    has_any_weight = bool(self._example_weight_map.all_weight_features())

    class CombineCountsAndWeights(beam.PTransform):

      def expand(self, pcoll):
        if has_any_weight:
          return pcoll | beam.CombinePerKey(_sum_pairwise)
        else:
          # For non-weighted case, use sum combine fn over integers to allow
          # Beam to use Cython combiner.
          return (pcoll
                  | 'RemoveWeights' >> beam.MapTuple(lambda k, v: (k, v[0]))
                  | beam.CombinePerKey(sum))

    top_k_tuples_combined = (
        pcoll
        | 'ToTopKTuples' >> beam.FlatMap(
            _to_topk_tuples,
            bytes_features=self._bytes_features,
            categorical_numeric_types=self._categorical_numeric_types,
            example_weight_map=self._example_weight_map)
        | 'CombineCountsAndWeights' >> CombineCountsAndWeights()
        | 'Rearrange' >> beam.MapTuple(lambda k, v: ((k[0], k[1]), (v, k[2]))))
    # (slice_key, feature_path_steps), (count_and_maybe_weight, value)

    top_k = top_k_tuples_combined
    if has_any_weight:
      top_k |= 'Unweighted_DropWeightsAndRearrange' >> beam.MapTuple(
          lambda k, v: (k, (v[0][0], v[1])))
      # (slice_key, feature_path_steps), (count, value)

    top_k = (
        top_k
        | 'Unweighted_TopK' >> beam.combiners.Top().PerKey(
            max(self._num_top_values, self._num_rank_histogram_buckets))
        | 'Unweighted_ToFeatureValueCount' >> beam.MapTuple(
            # pylint: disable=g-long-lambda
            lambda k, v: (k, [
                top_k_uniques_stats_util.FeatureValueCount(t[1], t[0])
                for t in v
            ])
            # pylint: enable=g-long-lambda
        )
        | 'Unweighted_ToProto' >> beam.MapTuple(
            # pylint: disable=g-long-lambda
            lambda k, v:
            (k[0],
             top_k_uniques_stats_util.
             make_dataset_feature_stats_proto_topk_single(
                 feature_path_tuple=k[1],
                 value_count_list=v,
                 is_weighted_stats=False,
                 num_top_values=self._num_top_values,
                 frequency_threshold=self._frequency_threshold,
                 num_rank_histogram_buckets=self._num_rank_histogram_buckets))
            # pylint: enable=g-long-lambda
        ))
    # (slice_key, DatasetFeatureStatistics)

    uniques = (
        top_k_tuples_combined
        | 'Uniques_Keys' >> beam.Keys()
        | 'Uniques_CountPerFeatureName' >> beam.combiners.Count().PerElement()
        | 'Uniques_ConvertToSingleFeatureStats' >> beam.MapTuple(
            # pylint: disable=g-long-lambda
            lambda k, v:
            (k[0],
             top_k_uniques_stats_util.
             make_dataset_feature_stats_proto_unique_single(
                 feature_path_tuple=k[1],
                 num_uniques=v))
            # pylint: enable=g-long-lambda
        ))
    # (slice_key, DatasetFeatureStatistics)

    result_protos = [top_k, uniques]

    if has_any_weight:
      weighted_top_k = (
          top_k_tuples_combined
          | 'Weighted_DropCountsAndRearrange' >>
          beam.MapTuple(lambda k, v: (k, (v[0][1], v[1])))
          # (slice_key, feature), (weight, value)
          | 'Weighted_TopK' >> beam.combiners.Top().PerKey(
              max(self._num_top_values, self._num_rank_histogram_buckets))
          | 'Weighted_ToFeatureValueCount' >> beam.MapTuple(
              # pylint: disable=g-long-lambda
              lambda k, v: (k, [
                  top_k_uniques_stats_util.FeatureValueCount(t[1], t[0])
                  for t in v
              ])
              # pylint: enable=g-long-lambda
          )
          | 'Weighted_ToProto' >> beam.MapTuple(
              # pylint: disable=g-long-lambda
              lambda k, v:
              (k[0],
               top_k_uniques_stats_util.
               make_dataset_feature_stats_proto_topk_single(
                   feature_path_tuple=k[1],
                   value_count_list=v,
                   is_weighted_stats=True,
                   num_top_values=self._num_top_values,
                   frequency_threshold=self._weighted_frequency_threshold,
                   num_rank_histogram_buckets=self._num_rank_histogram_buckets))
              # pylint: enable=g-long-lambda
          ))
      # (slice_key, DatasetFeatureStatistics)

      result_protos.append(weighted_top_k)

    return (result_protos
            | 'FlattenTopKUniquesFeatureStatsProtos' >> beam.Flatten())


class TopKUniquesStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes top-k and uniques."""

  def __init__(self,
               name: Text = 'TopKUniquesStatsGenerator',
               schema: Optional[schema_pb2.Schema] = None,
               example_weight_map: ExampleWeightMap = ExampleWeightMap(),
               num_top_values: int = 2,
               frequency_threshold: int = 1,
               weighted_frequency_threshold: float = 1.0,
               num_rank_histogram_buckets: int = 1000) -> None:
    """Initializes top-k and uniques stats generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      example_weight_map: An optional feature name whose numeric value
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
            example_weight_map=example_weight_map,
            num_top_values=num_top_values,
            frequency_threshold=frequency_threshold,
            weighted_frequency_threshold=weighted_frequency_threshold,
            num_rank_histogram_buckets=num_rank_histogram_buckets))
