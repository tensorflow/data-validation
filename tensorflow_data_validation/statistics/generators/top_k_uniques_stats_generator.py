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
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.utils.stats_util import is_valid_utf8
from tensorflow_data_validation.types_compat import Iterator, List, Optional, Set, Text, Tuple, Union
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

FeatureValueCount = collections.namedtuple('FeatureValueCount',
                                           ['feature_value', 'count'])

_SlicedFeatureNameAndValueListWithWeight = collections.namedtuple(
    '_SlicedFeatureNameAndValueListWithWeight',
    ['slice_key', 'feature_name', 'value_list', 'weight'])

_INVALID_STRING = '__BYTES_VALUE__'


def _make_feature_stats_proto_with_uniques_stats(
    feature_name, count,
    is_categorical):
  """Makes a FeatureNameStatistics proto containing the uniques stats."""
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (
      statistics_pb2.FeatureNameStatistics.INT
      if is_categorical else statistics_pb2.FeatureNameStatistics.STRING)
  result.string_stats.unique = count
  return result


def _make_dataset_feature_stats_proto_with_uniques_for_single_feature(
    feature_name_to_value_count,
    categorical_features
):
  """Makes a DatasetFeatureStatistics proto with uniques stats for a feature."""
  (slice_key, feature_name), count = feature_name_to_value_count
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto_with_uniques_stats(
          feature_name, count, feature_name in categorical_features))
  return slice_key, result.SerializeToString()


def make_feature_stats_proto_with_topk_stats(
    feature_name,
    top_k_value_count_list, is_categorical,
    is_weighted_stats, num_top_values,
    frequency_threshold,
    num_rank_histogram_buckets):
  """Makes a FeatureNameStatistics proto containing the top-k stats.

  Args:
    feature_name: The feature name.
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
    if count < frequency_threshold:
      break
    # Check if we have a valid utf-8 string. If not, assign a default invalid
    # string value.
    if isinstance(value, bytes) and not is_valid_utf8(value):
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


def _make_dataset_feature_stats_proto_with_topk_for_single_feature(
    feature_name_to_value_count_list,
    categorical_features, is_weighted_stats,
    num_top_values, frequency_threshold,
    num_rank_histogram_buckets):
  """Makes a DatasetFeatureStatistics proto with top-k stats for a feature."""
  (slice_key, feature_name), value_count_list = feature_name_to_value_count_list
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      make_feature_stats_proto_with_topk_stats(
          feature_name, value_count_list, feature_name in categorical_features,
          is_weighted_stats, num_top_values, frequency_threshold,
          num_rank_histogram_buckets))
  return slice_key, result.SerializeToString()


def _convert_input_to_feature_values_with_weights(
    sliced_example,
    categorical_features,
    weight_feature = None
):
  """Converts input example to tuples containing feature values and weights.

  Specifically, iterates over all the STRING features in the input example and
  outputs tuples containing slice key, feature name, feature value and the
  weight associated with the value (if a weight feature is provided).

  Args:
    sliced_example: Tuple (slice_key, example).
    categorical_features: Set of names of categorical features.
    weight_feature: Name of the weight feature. None if there is no
        weight feature.

  Yields:
    A tuple (slice_key, feature_name, feature_value_list, optional weight).
  """
  slice_key, example = sliced_example
  if weight_feature is not None:
    weight = example[weight_feature][0]

  for feature_name, values in example.items():
    if feature_name == weight_feature:
      continue

    is_categorical = feature_name in categorical_features
    # Check if we have a non-missing feature with at least one value.
    if values is None or values.size == 0:
      continue
    # If the feature is neither categorical nor of string type, then
    # skip the feature.
    if not (is_categorical or
            get_feature_type(values.dtype) ==
            statistics_pb2.FeatureNameStatistics.STRING):
      continue

    yield _SlicedFeatureNameAndValueListWithWeight(
        slice_key,
        feature_name,
        values.astype(str) if is_categorical else values,
        weight if weight_feature else None)


_StringFeatureValue = Union[Text, bytes]


def _flatten_value_list(
    entry
):
  """Flatten list of feature values.

  Example: ('key', 'x', ['a', 'b'], ?) -> ('key', 'x', 'a'), ('key', 'x', 'b')

  Args:
    entry: Tuple (slice_key, feature_name, feature_value_list, optional weight)

  Yields:
    Tuple (slice_key, feature_name, feature_value)
  """
  (slice_key, feature_name, value_list, _) = entry
  for value in value_list:
    yield slice_key, feature_name, value


def _flatten_weighted_value_list(
    entry
):
  """Flatten list of weighted feature values.

  Example: ('key', 'x', ['a', 'b'], w) ->
                    (('key', 'x', 'a'), w), (('key', 'x', 'b'), w)

  Args:
    entry: Tuple (slice_key, feature_name, feature_value_list, weight)

  Yields:
    Tuple ((slice_key, feature_name, feature_value), weight)
  """
  (slice_key, feature_name, value_list, weight) = entry
  for value in value_list:
    yield (slice_key, feature_name, value), weight


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.SlicedExample)
@beam.typehints.with_output_types(
    Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics])
class _ComputeTopKUniquesStats(beam.PTransform):
  """A ptransform that computes top-k and uniques for string features."""

  def __init__(self, schema,
               weight_feature, num_top_values,
               frequency_threshold, weighted_frequency_threshold,
               num_rank_histogram_buckets):
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
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_top_values = num_top_values
    self._frequency_threshold = frequency_threshold
    self._weighted_frequency_threshold = weighted_frequency_threshold
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def expand(self, pcoll):
    """Computes top-k most frequent values and number of uniques."""
    # Convert input example to tuples of form
    # (slice_key, feature_name, feature_value_list, optional weight)
    # corresponding to each example.
    feature_values_with_weights = (
        pcoll
        | 'TopKUniques_ConvertInputToFeatureValuesWithWeights' >> beam.FlatMap(
            _convert_input_to_feature_values_with_weights,
            categorical_features=self._categorical_features,
            weight_feature=self._weight_feature))

    # Lambda to convert from ((slice_key, feature_name, feature_value), count)
    # to ((slice_key, feature_name), (feature_value, count))
    modify_key = (
        lambda x: ((x[0][0], x[0][1]), FeatureValueCount(x[0][2], x[1])))

    # Key to order values.
    key_fn = lambda x: (x.count, x.feature_value)

    sliced_feature_name_value_count = (
        feature_values_with_weights
        # Flatten (slice_key, feature_name, feature_value_list, optional weight)
        # to (slice_key, feature_name, feature_value)
        | 'TopKUniques_FlattenToSlicedFeatureNameValueTuples' >>
        beam.FlatMap(_flatten_value_list)
        # Compute the frequency of each feature_value per slice. Output is a
        # PCollection of ((slice_key, feature_name, feature_value), count)
        | 'TopKUniques_CountSlicedFeatureNameValueTuple' >>
        beam.combiners.Count().PerElement()
        # Convert from ((slice_key, feature_name, feature_value), count) to
        # ((slice_key, feature_name), (feature_value, count))
        | 'TopKUniques_ModifyKeyToSlicedFeatureName' >> beam.Map(modify_key)
    )

    result_protos = []
    # Find topk values for each feature.
    topk = (
        sliced_feature_name_value_count
        # Obtain the top-k most frequent feature value for each feature in a
        # slice.
        | 'TopK_GetTopK' >> beam.combiners.Top.PerKey(
            max(self._num_top_values, self._num_rank_histogram_buckets),
            key=key_fn)
        | 'TopK_ConvertToSingleFeatureStats' >> beam.Map(
            _make_dataset_feature_stats_proto_with_topk_for_single_feature,
            categorical_features=self._categorical_features,
            is_weighted_stats=False,
            num_top_values=self._num_top_values,
            frequency_threshold=self._frequency_threshold,
            num_rank_histogram_buckets=self._num_rank_histogram_buckets))

    result_protos.append(topk)

    # If a weight feature is provided, find the weighted topk values for each
    # feature.
    if self._weight_feature is not None:
      weighted_topk = (
          # Flatten (slice_key, feature_name, feature_value_list, weight) to
          # ((slice_key, feature_name, feature_value), weight)
          feature_values_with_weights
          | 'TopKWeighted_FlattenToSlicedFeatureNameValueTuples' >>
          beam.FlatMap(_flatten_weighted_value_list)
          # Sum the weights of each feature_value per slice. Output is a
          # PCollection of
          # ((slice_key, feature_name, feature_value), weighted_count)
          | 'TopKWeighted_CountSlicedFeatureNameValueTuple' >>
          beam.CombinePerKey(sum)
          # Convert from
          # ((slice_key, feature_name, feature_value), weighted_count) to
          # ((slice_key, feature_name), (feature_value, weighted_count))
          | 'TopKWeighted_ModifyKeyToSlicedFeatureName' >> beam.Map(modify_key)
          # Obtain the top-k most frequent feature value for each feature in a
          # slice.
          | 'TopKWeighted_GetTopK' >> beam.combiners.Top().PerKey(
              max(self._num_top_values, self._num_rank_histogram_buckets),
              key=key_fn)
          | 'TopKWeighted_ConvertToSingleFeatureStats' >> beam.Map(
              _make_dataset_feature_stats_proto_with_topk_for_single_feature,
              categorical_features=self._categorical_features,
              is_weighted_stats=True,
              num_top_values=self._num_top_values,
              frequency_threshold=self._weighted_frequency_threshold,
              num_rank_histogram_buckets=self._num_rank_histogram_buckets))
      result_protos.append(weighted_topk)

    uniques = (
        sliced_feature_name_value_count
        # Drop the values to only have the slice_key and feature_name with
        # each repeated the number of unique values times.
        | 'Uniques_DropValues' >> beam.Keys()
        | 'Uniques_CountPerFeatureName' >> beam.combiners.Count().PerElement()
        | 'Uniques_ConvertToSingleFeatureStats' >> beam.Map(
            _make_dataset_feature_stats_proto_with_uniques_for_single_feature,
            categorical_features=self._categorical_features))
    result_protos.append(uniques)

    def _deserialize_sliced_feature_stats_proto(entry):
      feature_stats_proto = statistics_pb2.DatasetFeatureStatistics()
      feature_stats_proto.ParseFromString(entry[1])
      return entry[0], feature_stats_proto

    return (
        result_protos
        | 'FlattenTopKUniquesResults' >> beam.Flatten()
        # TODO(b/121152126): This deserialization stage is a workaround.
        # Remove this once it is no longer needed.
        | 'DeserializeTopKUniquesFeatureStatsProto' >>
        beam.Map(_deserialize_sliced_feature_stats_proto))


class TopKUniquesStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes top-k and uniques."""

  def __init__(self,
               name = 'TopKUniquesStatsGenerator',
               schema = None,
               weight_feature = None,
               num_top_values = 2,
               frequency_threshold = 1,
               weighted_frequency_threshold = 1.0,
               num_rank_histogram_buckets = 1000):
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
