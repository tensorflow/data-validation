# Copyright 2020 Google LLC
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
"""Module that estimates top-k and unique statistics for categorical features.

Uses the Misra-Gries sketch to estimate top unweighted item counts and weighted
(if provided) item counts and the K-Minimum Values sketch to estimate number of
unique items.
"""

import collections
from typing import Dict, Iterable, Optional, Text

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types as tfdv_types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import top_k_uniques_stats_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap

from tfx_bsl.arrow import array_util
from tfx_bsl.sketches import KmvSketch
from tfx_bsl.sketches import MisraGriesSketch

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# Tuple for containing estimates from querying a _CombinedSketch.
_CombinedEstimate = collections.namedtuple(
    "_CombinedEstimate", ["distinct", "topk_unweighted", "topk_weighted"])


# Strings longer than this will be attributed to a single "large string" token
# (constants.LARGE_BYTES_PLACEHOLDER) for top-k computations.
_LARGE_STRING_THRESHOLD = 1024


class _CombinedSketch(object):
  """Wrapper for the three sketches for a single feature."""
  __slots__ = ["_distinct", "_topk_unweighted", "_topk_weighted"]

  def __init__(self, distinct, topk_unweighted, topk_weighted=None):
    self._distinct = distinct
    self._topk_unweighted = topk_unweighted
    self._topk_weighted = topk_weighted

  def add(self, values, weights=None):
    self._distinct.AddValues(values)
    self._topk_unweighted.AddValues(values)
    if weights is not None:
      self._topk_weighted.AddValues(values, weights)

  def merge(self, other_sketch):
    # pylint: disable=protected-access
    self._distinct.Merge(other_sketch._distinct)
    self._topk_unweighted.Merge(other_sketch._topk_unweighted)
    self._topk_weighted.Merge(other_sketch._topk_weighted)
    # pylint: enable=protected-access

  def estimate(self):
    # Converts the result struct array into list of FeatureValueCounts.
    topk_unweighted = self._topk_unweighted.Estimate().to_pylist()
    topk_unweighted_counts = [top_k_uniques_stats_util.FeatureValueCount(
        pair["values"], pair["counts"]) for pair in topk_unweighted]
    topk_weighted = self._topk_weighted.Estimate().to_pylist()
    topk_weighted_counts = [top_k_uniques_stats_util.FeatureValueCount(
        pair["values"], pair["counts"]) for pair in topk_weighted]
    return _CombinedEstimate(
        self._distinct.Estimate(), topk_unweighted_counts, topk_weighted_counts)


class TopKUniquesSketchStatsGenerator(stats_generator.CombinerStatsGenerator):
  """Generates statistics for number unique and top-k item counts.

  Uses mergeable K-Minimum Values (KMV) and Misra-Gries sketches to estimate
  statistics.
  """

  def __init__(
      self,
      name: Text = "TopKUniquesSketchStatsGenerator",
      schema: Optional[schema_pb2.Schema] = None,
      example_weight_map: ExampleWeightMap = ExampleWeightMap(),
      num_top_values: int = 2,
      num_rank_histogram_buckets: int = 128,
      frequency_threshold: int = 1,
      weighted_frequency_threshold: float = 1.0,
      num_misragries_buckets: int = 128,
      num_kmv_buckets: int = 128,
      store_output_in_custom_stats: bool = False,
      length_counter_sampling_rate: float = 0.01
  ):
    """Initializes a top-k and uniques sketch combiner statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
        corresponding weight column.
      num_top_values: The number of most frequent feature values to keep for
        string features.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
        for string features.
      frequency_threshold: An optional minimum number of examples the most
        frequent values must be present in (defaults to 1).
      weighted_frequency_threshold: An optional minimum weighted number of
        examples the most frequent weighted values must be present in (defaults
        to 1.0).
      num_misragries_buckets: Number of buckets to use for MisraGries sketch.
      num_kmv_buckets: Number of buckets to use for KMV sketch.
      store_output_in_custom_stats: Boolean to indicate if the output stats need
        to be stored in custom stats. If False, the output is stored in
        `uniques` and `rank_histogram` fields.
      length_counter_sampling_rate: The sampling rate to update the byte length
        counter.
    """
    super(
        TopKUniquesSketchStatsGenerator,
        self,
    ).__init__(name, schema)
    self._num_misragries_buckets = num_misragries_buckets
    self._num_kmv_buckets = num_kmv_buckets
    self._num_top_values = num_top_values
    self._example_weight_map = example_weight_map
    self._num_rank_histogram_buckets = num_rank_histogram_buckets
    self._categorical_numeric_types = (
        schema_util.get_categorical_numeric_feature_types(schema)
        if schema else {})
    self._bytes_features = frozenset(
        schema_util.get_bytes_features(schema) if schema else [])
    self._byte_feature_is_categorical_values = (
        schema_util.get_bytes_features_categorical_value(schema))
    self._frequency_threshold = frequency_threshold
    self._weighted_frequency_threshold = weighted_frequency_threshold
    self._store_output_in_custom_stats = store_output_in_custom_stats
    self._length_counter_sampling_rate = length_counter_sampling_rate
    # They should be gauges, but not all runners support gauges so they are
    # made distributions.
    # TODO(b/130840752): support gauges in the internal runner.
    self._num_top_values_gauge = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, "num_top_values")
    self._num_rank_histogram_buckets_gauge = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, "num_rank_histogram_buckets")
    self._num_mg_buckets_gauge = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, "num_mg_buckets")
    self._num_kmv_buckets_gauge = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, "num_kmv_buckets")

  def _update_combined_sketch_for_feature(
      self, feature_name: tfdv_types.FeaturePath, values: pa.Array,
      weights: Optional[np.ndarray],
      accumulator: Dict[tfdv_types.FeaturePath, _CombinedSketch]):
    """Updates combined sketch with values (and weights if provided)."""
    flattened_values, parent_indices = array_util.flatten_nested(
        values, weights is not None)

    combined_sketch = accumulator.get(feature_name, None)
    if combined_sketch is None:
      self._num_kmv_buckets_gauge.update(self._num_kmv_buckets)

      def make_mg_sketch():
        num_buckets = max(self._num_misragries_buckets, self._num_top_values,
                          self._num_rank_histogram_buckets)
        self._num_mg_buckets_gauge.update(num_buckets)
        self._num_top_values_gauge.update(self._num_top_values)
        self._num_rank_histogram_buckets_gauge.update(
            self._num_rank_histogram_buckets)
        categorical = self._byte_feature_is_categorical_values.get(
            feature_name,
            schema_pb2.StringDomain.Categorical.CATEGORICAL_UNSPECIFIED
        ) == schema_pb2.StringDomain.Categorical.CATEGORICAL_YES
        return MisraGriesSketch(
            num_buckets=num_buckets,
            invalid_utf8_placeholder=constants.NON_UTF8_PLACEHOLDER,
            # Maximum sketch size:
            # _LARGE_STRING_THRESHOLD * num_buckets * constant_factor.
            large_string_threshold=_LARGE_STRING_THRESHOLD
            if not categorical else None,
            large_string_placeholder=constants.LARGE_BYTES_PLACEHOLDER
            if not categorical else None)

      self._num_top_values_gauge.update(self._num_top_values)
      combined_sketch = _CombinedSketch(
          distinct=KmvSketch(self._num_kmv_buckets),
          topk_unweighted=make_mg_sketch(),
          topk_weighted=make_mg_sketch())
    weight_array = None
    if weights is not None:
      flattened_weights = weights[parent_indices]
      weight_array = pa.array(flattened_weights, type=pa.float32())
    combined_sketch.add(flattened_values, weight_array)
    accumulator[feature_name] = combined_sketch

  def create_accumulator(self) -> Dict[tfdv_types.FeaturePath, _CombinedSketch]:
    return {}

  def _should_run(self, feature_path: tfdv_types.FeaturePath,
                  feature_type: Optional[int]) -> bool:
    # Only compute top-k and unique stats for categorical numeric and string
    # features (excluding string features declared as bytes and features that
    # indicates as non categorical under StringDomain).
    if feature_type == statistics_pb2.FeatureNameStatistics.STRING:
      return (feature_path not in self._bytes_features and
              self._byte_feature_is_categorical_values.get(feature_path, 0) !=
              schema_pb2.StringDomain.Categorical.CATEGORICAL_NO)
    return top_k_uniques_stats_util.output_categorical_numeric(
        self._categorical_numeric_types, feature_path, feature_type)

  def add_input(
      self, accumulator: Dict[tfdv_types.FeaturePath, _CombinedSketch],
      input_record_batch: pa.RecordBatch
  ) -> Dict[tfdv_types.FeaturePath, _CombinedSketch]:

    def update_length_counters(
        feature_type: tfdv_types.FeatureNameStatisticsType,
        leaf_array: pa.Array):
      if np.random.random() > self._length_counter_sampling_rate: return
      if feature_type == statistics_pb2.FeatureNameStatistics.STRING:
        distinct_count = collections.defaultdict(int)
        values, _ = array_util.flatten_nested(leaf_array)
        for value in values:
          binary_scalar_len = int(np.log2(max(value.as_buffer().size, 1)))
          distinct_count[binary_scalar_len] += 1
        for k, v in distinct_count.items():
          beam.metrics.Metrics.counter(constants.METRICS_NAMESPACE,
                                       "binary_scalar_len_" + str(k)).inc(v)

    for feature_path, leaf_array, weights in arrow_util.enumerate_arrays(
        input_record_batch,
        example_weight_map=self._example_weight_map,
        enumerate_leaves_only=True):
      feature_type = stats_util.get_feature_type_from_arrow_type(
          feature_path, leaf_array.type)
      if self._should_run(feature_path, feature_type):
        self._update_combined_sketch_for_feature(feature_path, leaf_array,
                                                 weights, accumulator)
        update_length_counters(feature_type, leaf_array)
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[Dict[tfdv_types.FeaturePath, _CombinedSketch]]
      ) -> Dict[tfdv_types.FeaturePath, _CombinedSketch]:
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      for feature_name, combined_sketch in accumulator.items():
        existing_sketch = result.get(feature_name, None)
        if existing_sketch is None:
          result[feature_name] = combined_sketch
        else:
          existing_sketch.merge(combined_sketch)
          result[feature_name] = existing_sketch
    return result

  def extract_output(
      self, accumulator: Dict[tfdv_types.FeaturePath, _CombinedSketch]
  ) -> statistics_pb2.DatasetFeatureStatistics:
    result = statistics_pb2.DatasetFeatureStatistics()
    for feature_path, combined_sketch in accumulator.items():
      combined_estimate = combined_sketch.estimate()
      if not combined_estimate.topk_unweighted:
        assert not combined_estimate.topk_weighted
        continue
      make_feature_stats_proto = (
          top_k_uniques_stats_util.make_feature_stats_proto_topk_uniques)
      if self._store_output_in_custom_stats:
        make_feature_stats_proto = (
            top_k_uniques_stats_util.
            make_feature_stats_proto_topk_uniques_custom_stats)

      feature_stats_proto = (
          make_feature_stats_proto(
              feature_path=feature_path,
              frequency_threshold=self._frequency_threshold,
              weighted_frequency_threshold=self._weighted_frequency_threshold,
              num_top_values=self._num_top_values,
              num_rank_histogram_buckets=self._num_rank_histogram_buckets,
              num_unique=combined_estimate.distinct,
              value_count_list=combined_estimate.topk_unweighted,
              weighted_value_count_list=combined_estimate.topk_weighted))

      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
