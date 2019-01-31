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
import logging
import apache_beam as beam
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.types_compat import Iterator, List, Optional, Set, Text, Tuple, Union
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

FeatureValueCount = collections.namedtuple('FeatureValueCount',
                                           ['feature_value', 'count'])

_FeatureNameAndValueListWithWeight = collections.namedtuple(
    '_FeatureNameAndValueListWithWeight',
    ['feature_name', 'value_list', 'weight'])

_INVALID_STRING = '__BYTES_VALUE__'


def _is_valid_utf8(value):
  try:
    value.decode('utf-8')
  except UnicodeError:
    return False
  return True


def make_feature_stats_proto_with_topk_stats(
    feature_name,
    top_k_value_count_list, is_categorical,
    is_weighted_stats, num_top_values,
    num_rank_histogram_buckets):
  """Makes a FeatureNameStatistics proto containing the top-k stats.

  Args:
    feature_name: The feature name.
    top_k_value_count_list: A list of FeatureValueCount tuples.
    is_categorical: Whether the feature is categorical.
    is_weighted_stats: Whether top_k_value_count_list incorporates weights.
    num_top_values: The number of most frequent feature values to keep for
      string features.
    num_rank_histogram_buckets: The number of buckets in the rank histogram for
      string features.
  Returns:
    A FeatureNameStatistics proto containing the top-k stats.
  """
  # Sort the top_k_value_count_list in descending order by count. Where
  # multiple feature values have the same count, consider the feature with the
  # 'larger' feature value to be larger for purposes of breaking the tie.
  top_k_value_count_list.sort(
      key=lambda counts: (counts[1], counts[0]),
      reverse=True)

  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
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
    # Check if we have a valid utf-8 string. If not, assign a default invalid
    # string value.
    if isinstance(value, bytes) and not _is_valid_utf8(value):
      logging.warning('Feature "%s" has bytes value "%s" which cannot be '
                      'decoded as a UTF-8 string.', feature_name, value)
      value = _INVALID_STRING

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


def _make_dataset_feature_stats_proto_with_single_feature(
    feature_name_to_value_count_list,
    categorical_features, is_weighted_stats,
    num_top_values, num_rank_histogram_buckets):
  """Makes a DatasetFeatureStatistics containing one single feature."""
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      make_feature_stats_proto_with_topk_stats(
          feature_name_to_value_count_list[0],
          feature_name_to_value_count_list[1],
          feature_name_to_value_count_list[0] in categorical_features,
          is_weighted_stats, num_top_values, num_rank_histogram_buckets))
  return result.SerializeToString()


def _convert_input_to_feature_values_with_weights(
    example,
    categorical_features,
    weight_feature = None
):
  """Converts input example to tuples containing feature values and weights.

  Specifically, iterates over all the STRING features in the input example and
  outputs tuples containing feature name, feature value and the weight
  associated with the value (if a weight feature is provided).

  Args:
    example: Input example.
    categorical_features: Set of names of categorical features.
    weight_feature: Name of the weight feature. None if there is no
        weight feature.

  Yields:
    A tuple (feature_name, feature_value_list, optional weight).
  """
  if weight_feature is not None:
    weight = example[weight_feature][0]

  for feature_name, values in example.items():
    if feature_name == weight_feature:
      continue

    is_categorical = feature_name in categorical_features
    # Check if we have a numpy array with at least one value.
    if not isinstance(values, np.ndarray) or values.size == 0:
      continue
    # If the feature is neither categorical nor of string type, then
    # skip the feature.
    if not (is_categorical or
            get_feature_type(values.dtype) ==
            statistics_pb2.FeatureNameStatistics.STRING):
      continue

    yield _FeatureNameAndValueListWithWeight(
        feature_name,
        values.astype(str) if is_categorical else values,
        weight if weight_feature else None)


_StringFeatureValue = Union[Text, bytes]


def _flatten_value_list(
    entry
):
  """Flatten list of feature values.

  Example: ('x', ['a', 'b'], ?) -> ('x', 'a'), ('x', 'b')

  Args:
    entry: A tuple (feature_name, feature_value_list, optional weight)

  Yields:
    A tuple (feature_name, feature_value)
  """
  for value in entry.value_list:
    yield entry.feature_name, value


def _flatten_weighted_value_list(
    entry
):
  """Flatten list of weighted feature values.

  Example: ('x', ['a', 'b'], w) -> (('x', 'a'), w), (('x', 'b'), w)

  Args:
    entry: A tuple (feature_name, feature_value_list, weight)

  Yields:
    A tuple ((feature_name, feature_value), weight)
  """
  for value in entry.value_list:
    yield (entry.feature_name, value), entry.weight


def _feature_value_count_comparator(a,
                                    b):
  """Compares two FeatureValueCount tuples."""
  # To keep the result deterministic, if two feature values have the same
  # number of appearances, the one with the 'larger' feature value will be
  # larger.
  return (a.count < b.count or
          (a.count == b.count and a.feature_value < b.feature_value))


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.Example)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatistics)
class _ComputeTopKStats(beam.PTransform):
  """A ptransform that computes the top-k most frequent feature values for
  string features.
  """

  def __init__(self, schema,
               weight_feature,
               num_top_values, num_rank_histogram_buckets):
    """Initializes _ComputeTopKStats.

    Args:
      schema: An schema for the dataset. None if no schema is available.
      weight_feature: Feature name whose numeric value represents the weight
          of an example. None if there is no weight feature.
      num_top_values: The number of most frequent feature values to keep for
          string features.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
          for string features.
    """
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_top_values = num_top_values
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def expand(self, pcoll):
    """Computes top-k most frequent values for string features."""
    # Convert input example to tuples of form
    # (feature_name, feature_value_list, optional weight)
    # corresponding to each example.
    feature_values_with_weights = (
        pcoll
        | 'TopK_ConvertInputToFeatureValuesWithWeights' >>
        beam.FlatMap(
            _convert_input_to_feature_values_with_weights,
            categorical_features=self._categorical_features,
            weight_feature=self._weight_feature).with_output_types(
                beam.typehints.KV[types.BeamFeatureName, np.ndarray]))

    result_protos = []
    # Find topk values for each feature.
    topk = (
        feature_values_with_weights
        # Flatten (feature_name, feature_value_list, optional weight) to
        # (feature_name, feature_value)
        | 'TopK_FlattenToFeatureNameValueTuples' >>
        beam.FlatMap(_flatten_value_list)
        # Compute the frequency of each feature_value. Output is a
        # PCollection of ((feature_name, feature_value), count)
        | 'TopK_CountFeatureNameValueTuple' >>
        beam.combiners.Count().PerElement()
        # Convert from ((feature_name, feature_value), count) to
        # (feature_name, (feature_value, count))
        | 'TopK_ModifyKeyToFeatureName' >>
        beam.Map(lambda x: (x[0][0], FeatureValueCount(x[0][1], x[1])))
        # Obtain the top-k most frequent feature value for each feature.
        | 'TopK_GetTopK' >> beam.combiners.Top().PerKey(
            max(self._num_top_values, self._num_rank_histogram_buckets),
            _feature_value_count_comparator)
        | 'TopK_ConvertToSingleFeatureStats' >> beam.Map(
            _make_dataset_feature_stats_proto_with_single_feature,
            categorical_features=self._categorical_features,
            is_weighted_stats=False,
            num_top_values=self._num_top_values,
            num_rank_histogram_buckets=self._num_rank_histogram_buckets))

    result_protos.append(topk)

    # If a weight feature is provided, find the weighted topk values for each
    # feature.
    if self._weight_feature is not None:
      weighted_topk = (
          feature_values_with_weights
          # Flatten (feature_name, feature_value_list, weight) to
          # ((feature_name, feature_value), weight)
          | 'TopKWeighted_FlattenToFeatureNameValueTuples' >>
          beam.FlatMap(_flatten_weighted_value_list)
          # Sum the weights of each feature_value. Output is a
          # PCollection of ((feature_name, feature_value), weighted_count)
          | 'TopKWeighted_CountFeatureNameValueTuple' >> beam.CombinePerKey(sum)
          # Convert from ((feature_name, feature_value), weighted_count) to
          # (feature_name, (feature_value, weighted_count))
          | 'TopKWeighted_ModifyKeyToFeatureName' >>
          beam.Map(lambda x: (x[0][0], FeatureValueCount(x[0][1], x[1])))
          # Obtain the top-k most frequent feature value for each feature.
          | 'TopKWeighted_GetTopK' >> beam.combiners.Top().PerKey(
              max(self._num_top_values, self._num_rank_histogram_buckets),
              _feature_value_count_comparator)
          | 'TopKWeighted_ConvertToSingleFeatureStats' >> beam.Map(
              _make_dataset_feature_stats_proto_with_single_feature,
              categorical_features=self._categorical_features,
              is_weighted_stats=True,
              num_top_values=self._num_top_values,
              num_rank_histogram_buckets=self._num_rank_histogram_buckets))
      result_protos.append(weighted_topk)

    def _deserialize_feature_stats_proto(serialized_feature_stats_proto):
      result = statistics_pb2.DatasetFeatureStatistics()
      result.ParseFromString(serialized_feature_stats_proto)
      return result

    return (result_protos
            | 'FlattenTopKResults' >> beam.Flatten()
            # TODO(b/121152126): This deserialization stage is a workaround.
            # Remove this once it is no longer needed.
            | 'DeserializeTopKFeatureStatsProto' >> beam.Map(
                _deserialize_feature_stats_proto))


class TopKStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes the top-k most frequent
  feature values for string features."""

  def __init__(self,
               name = 'TopKStatsGenerator',
               schema = None,
               weight_feature = None,
               num_top_values = 2,
               num_rank_histogram_buckets = 1000):
    """Initializes top-k stats generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: An optional feature name whose numeric value
          (must be of type INT or FLOAT) represents the weight of an example.
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
            weight_feature=weight_feature,
            num_top_values=num_top_values,
            num_rank_histogram_buckets=num_rank_histogram_buckets)
    )
