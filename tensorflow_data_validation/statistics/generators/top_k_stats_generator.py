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

"""Module computing top-k most frequent values for string features."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Generator, List, Optional, Set, Tuple
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

_FeatureValueCount = collections.namedtuple('_FeatureValueCount',
                                            ['feature_value', 'count'])


def _make_feature_stats_proto(
    feature_name,
    top_k_value_count_list, is_categorical,
    num_top_values,
    num_rank_histogram_buckets):
  """Makes a FeatureNameStatistics proto containing the top k stats."""
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (statistics_pb2.FeatureNameStatistics.INT if is_categorical
                 else statistics_pb2.FeatureNameStatistics.STRING)

  for i in range(len(top_k_value_count_list)):
    value, count = top_k_value_count_list[i]
    if i < num_top_values:
      freq_and_value = result.string_stats.top_values.add()
      freq_and_value.value = value
      freq_and_value.frequency = count
    if i < num_rank_histogram_buckets:
      bucket = result.string_stats.rank_histogram.buckets.add()
      bucket.low_rank = i
      bucket.high_rank = i
      bucket.sample_count = count
      bucket.label = value
  return result


def _make_dataset_feature_stats_proto_with_single_feature(
    feature_name_to_value_count_list, categorical_features,
    num_top_values,
    num_rank_histogram_buckets):
  """Makes a DatasetFeatureStatistics containing one single feature."""
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto(
          feature_name_to_value_count_list[0],
          feature_name_to_value_count_list[1],
          feature_name_to_value_count_list[0] in categorical_features,
          num_top_values, num_rank_histogram_buckets))
  return result


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatistics)
class _ComputeTopKStats(beam.PTransform):
  """A ptransform that computes the top-k most frequent feature values for
  string features.
  """

  def __init__(self, num_top_values, num_rank_histogram_buckets,
               schema):
    """Initializes _ComputeTopKStats.

    Args:
      num_top_values: The number of most frequent feature values to keep for
          string features.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
          for string features.
      schema: An schema for the dataset. None if no schema is available.
    """
    self._categorical_features = set(
        stats_util.get_categorical_numeric_features(schema) if schema else [])
    self._num_top_values = num_top_values
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def _filter_irrelevant_features(
      self, input_batch
  ):
    """Filters out non-string features."""
    for feature_name, values_batch in six.iteritems(input_batch):
      is_categorical = feature_name in self._categorical_features
      for values in values_batch:
        # Check if we have a numpy array with at least one value.
        if not isinstance(values, np.ndarray) or values.size == 0:
          continue
        # If the feature is neither categorical nor of string type, then
        # skip the feature.
        if not (is_categorical or
                stats_util.make_feature_type(values.dtype) ==
                statistics_pb2.FeatureNameStatistics.STRING):
          continue

        yield (feature_name, values.astype(str) if is_categorical else values)

  def expand(self, pcoll):
    """Computes top-k most frequent values for string features."""
    # Count the number of appearance of each feature_value. Output is a
    # pcollection of (feature_name, (feature_value, count))
    counts = (
        pcoll
        | 'TopK_FilterIrrelevantFeatures' >>
        (beam.FlatMap(self._filter_irrelevant_features).with_output_types(
            beam.typehints.KV[types.BeamFeatureName, np.ndarray]))
        | 'TopK_FlattenToFeatureNameValueTuples' >>
        beam.FlatMap(
            lambda (name, value_list): [(name, value) for value in value_list])
        | 'TopK_CountFeatureNameValueTuple' >>
        beam.combiners.Count().PerElement()
        # Convert from ((feature_name, feature_value), count) to
        # (feature_name, (feature_value, count))
        | 'TopK_ModifyKeyToFeatureName' >>
        beam.Map(lambda x: (x[0][0], _FeatureValueCount(x[0][1], x[1]))))

    def _feature_value_count_less(a, b):
      """Compares two _FeatureValueCount tuples."""
      # To keep the result deterministic, if two feature values have the same
      # number of appearances, the one with the 'larger' feature value will be
      # larger.
      return (a.count < b.count or
              (a.count == b.count and a.feature_value < b.feature_value))

    return (counts
            # Obtain the top-k most frequent feature value for each feature.
            | 'TopK_GetTopK' >> beam.combiners.Top().PerKey(
                max(self._num_top_values, self._num_rank_histogram_buckets),
                _feature_value_count_less)
            | 'TopK_ConvertToSingleFeatureStats' >>
            beam.Map(
                _make_dataset_feature_stats_proto_with_single_feature,
                categorical_features=self._categorical_features,
                num_top_values=self._num_top_values,
                num_rank_histogram_buckets=self._num_rank_histogram_buckets))


class TopKStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes the top-k most frequent
  feature values for string features."""

  def __init__(self,
               name = 'TopKStatsGenerator',
               schema = None,
               num_top_values = 2,
               num_rank_histogram_buckets = 1000):
    """Initializes top-k stats generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      num_top_values: An optional number of most frequent feature values to keep
          for string features (defaults to 2).
      num_rank_histogram_buckets: An optional number of buckets in the rank
          histogram for string features (defaults to 1000).
    """
    super(TopKStatsGenerator, self).__init__(
        name,
        schema=schema,
        ptransform=_ComputeTopKStats(
            schema=schema,
            num_top_values=num_top_values,
            num_rank_histogram_buckets=num_rank_histogram_buckets),
    )
