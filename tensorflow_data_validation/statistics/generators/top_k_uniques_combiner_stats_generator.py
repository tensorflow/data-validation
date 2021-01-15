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
"""Module that computes the top-k and uniques statistics for string features."""

import collections
from typing import Any, Dict, Iterable, Optional, Text

import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import top_k_uniques_stats_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# _ValueCounts holds two collections.Counter objects: one that holds the
# unweighted counts of each unique value for a given feature, and one that holds
# the weighted counts of each unique value for a given feature.
_ValueCounts = collections.namedtuple('_ValueCounts',
                                      ['unweighted_counts', 'weighted_counts'])


class _WeightedCounter(collections.defaultdict):
  """A counter that support weighted counts."""

  def __init__(self):
    super(_WeightedCounter, self).__init__(int)

  def weighted_update(self, values: Iterable[Any],
                      weights: Iterable[Any]) -> None:
    """Updates the count of elements in `values` with weights."""

    for v, w in zip(values, weights):
      self[v] += w

  def update(self, other: '_WeightedCounter') -> None:
    for k, v in other.items():
      self[k] += v

  def __reduce__(self):
    return type(self), (), None, None, iter(self.items())


class TopKUniquesCombinerStatsGenerator(
    stats_generator.CombinerStatsGenerator):
  """Combiner statistics generator that computes top-k and uniques stats.

  This generator is for in-memory data only. The TopKStatsGenerator and
  UniquesStatsGenerator are for distributed data.
  """

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name: Text = 'TopKUniquesCombinerStatsGenerator',
      schema: Optional[schema_pb2.Schema] = None,
      example_weight_map: ExampleWeightMap = ExampleWeightMap(),
      num_top_values: int = 2,
      frequency_threshold: int = 1,
      weighted_frequency_threshold: float = 1.0,
      num_rank_histogram_buckets: int = 1000) -> None:
    """Initializes a top-k and uniques combiner statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
          corresponding weight column.
      num_top_values: The number of most frequent feature values to keep for
        string features.
      frequency_threshold: An optional minimum number of examples
        the most frequent values must be present in (defaults to 1).
      weighted_frequency_threshold: An optional minimum weighted
        number of examples the most frequent weighted values must be
        present in (defaults to 1.0).
      num_rank_histogram_buckets: The number of buckets in the rank histogram
        for string features.
    """
    super(TopKUniquesCombinerStatsGenerator, self).__init__(name, schema)
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._example_weight_map = example_weight_map
    self._num_top_values = num_top_values
    self._frequency_threshold = frequency_threshold
    self._weighted_frequency_threshold = weighted_frequency_threshold
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def create_accumulator(self) -> Dict[types.FeatureName, _ValueCounts]:
    return {}

  def add_input(
      self, accumulator: Dict[types.FeaturePath,
                              _ValueCounts], input_record_batch: pa.RecordBatch
  ) -> Dict[types.FeaturePath, _ValueCounts]:
    for feature_path, leaf_array, weights in arrow_util.enumerate_arrays(
        input_record_batch,
        example_weight_map=self._example_weight_map,
        enumerate_leaves_only=True):
      feature_type = stats_util.get_feature_type_from_arrow_type(
          feature_path, leaf_array.type)
      # if it's not a categorical int feature nor a string feature, we don't
      # bother with topk stats.
      if ((feature_type == statistics_pb2.FeatureNameStatistics.INT and
           feature_path in self._categorical_features) or
          feature_type == statistics_pb2.FeatureNameStatistics.STRING):
        flattened_values, parent_indices = arrow_util.flatten_nested(
            leaf_array, weights is not None)
        unweighted_counts = collections.Counter()
        # Compute unweighted counts.
        value_counts = flattened_values.value_counts()
        values = value_counts.field('values').to_pylist()
        counts = value_counts.field('counts').to_pylist()
        for value, count in zip(values, counts):
          unweighted_counts[value] = count

        # Compute weighted counts if a weight feature is specified.
        weighted_counts = _WeightedCounter()
        if weights is not None:
          flattened_values_np = np.asarray(flattened_values)
          weighted_counts.weighted_update(
              flattened_values_np, weights[parent_indices])

        if feature_path not in accumulator:
          accumulator[feature_path] = _ValueCounts(
              unweighted_counts=unweighted_counts,
              weighted_counts=weighted_counts)
        else:
          accumulator[feature_path].unweighted_counts.update(unweighted_counts)
          accumulator[feature_path].weighted_counts.update(weighted_counts)

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[Dict[types.FeaturePath, _ValueCounts]]
  ) -> Dict[types.FeaturePath, _ValueCounts]:
    result = {}
    for accumulator in accumulators:
      for feature_path, value_counts in accumulator.items():
        if feature_path not in result:
          result[feature_path] = value_counts
        else:
          result[feature_path].unweighted_counts.update(
              value_counts.unweighted_counts)
          if result[feature_path].weighted_counts:
            result[feature_path].weighted_counts.update(
                value_counts.weighted_counts)
    return result

  def extract_output(self, accumulator: Dict[types.FeaturePath, _ValueCounts]
                    ) -> statistics_pb2.DatasetFeatureStatistics:

    result = statistics_pb2.DatasetFeatureStatistics()
    for feature_path, value_counts in accumulator.items():
      if not value_counts.unweighted_counts:
        assert not value_counts.weighted_counts
        continue
      feature_value_counts = [
          top_k_uniques_stats_util.FeatureValueCount(key, value)
          for key, value in value_counts.unweighted_counts.items()
      ]
      weighted_feature_value_counts = None
      if value_counts.weighted_counts:
        weighted_feature_value_counts = [
            top_k_uniques_stats_util.FeatureValueCount(key, value)
            for key, value in value_counts.weighted_counts.items()
        ]

      feature_stats_proto = (
          top_k_uniques_stats_util.make_feature_stats_proto_topk_uniques(
              feature_path=feature_path,
              is_categorical=feature_path in self._categorical_features,
              frequency_threshold=self._frequency_threshold,
              weighted_frequency_threshold=self._weighted_frequency_threshold,
              num_top_values=self._num_top_values,
              num_rank_histogram_buckets=self._num_rank_histogram_buckets,
              num_unique=len(feature_value_counts),
              value_count_list=feature_value_counts,
              weighted_value_count_list=weighted_feature_value_counts))

      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
