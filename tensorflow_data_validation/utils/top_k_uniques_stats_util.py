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
"""Utilities for Top-K Uniques stats generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Mapping, Optional, Union

import apache_beam as beam
import six
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import stats_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order

# Tuple to hold feature value and count for an item.
FeatureValueCount = tfx_namedtuple.namedtuple('FeatureValueCount',
                                              ['feature_value', 'count'])

# Custom stats names.
_TOPK_SKETCH_CUSTOM_STATS_NAME = 'topk_sketch_rank_histogram'
_WEIGHTED_TOPK_SKETCH_CUSTOM_STATS_NAME = 'weighted_topk_sketch_rank_histogram'
_UNIQUES_SKETCH_CUSTOM_STATS_NAME = 'uniques_sketch_num_uniques'

# Beam counter to track the number of non-utf8 values.
_NON_UTF8_VALUES_COUNTER = beam.metrics.Metrics.counter(
    constants.METRICS_NAMESPACE, 'num_non_utf8_values_topk_uniques_generator')


def make_feature_stats_proto_topk_uniques(
    feature_path: types.FeaturePath,
    num_top_values: int,
    num_rank_histogram_buckets: int,
    num_unique: int,
    value_count_list: List[FeatureValueCount],
    weighted_value_count_list: Optional[List[FeatureValueCount]] = None,
    frequency_threshold: int = 1,
    weighted_frequency_threshold: Optional[float] = None
) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing top-k and uniques stats.

  Args:
    feature_path: The path of the feature.
    num_top_values: The number of most frequent feature values to keep for
      string features.
    num_rank_histogram_buckets: The number of buckets in the rank histogram for
      string features.
    num_unique: The number of unique values in the feature.
    value_count_list: A list of FeatureValueCount tuples.
    weighted_value_count_list: An optional list of FeatureValueCount tuples for
      weighted features.
    frequency_threshold: The minimum number of examples the most frequent values
      must be present in.
    weighted_frequency_threshold: The minimum weighted number of examples the
      most frequent weighted values must be present in. Optional.

  Returns:
    A FeatureNameStatistics proto containing the top-k and uniques stats.
  """

  # Create a FeatureNameStatistics proto that includes the unweighted top-k
  # stats.
  result = _make_feature_stats_proto_topk(feature_path, value_count_list, False,
                                          num_top_values, frequency_threshold,
                                          num_rank_histogram_buckets)

  # If weights were provided, create another FeatureNameStatistics proto that
  # includes the weighted top-k stats, and then copy those weighted top-k stats
  # into the result proto.
  if weighted_value_count_list:
    assert weighted_frequency_threshold is not None
    weighted_result = _make_feature_stats_proto_topk(
        feature_path, weighted_value_count_list, True, num_top_values,
        weighted_frequency_threshold, num_rank_histogram_buckets)

    result.string_stats.weighted_string_stats.CopyFrom(
        weighted_result.string_stats.weighted_string_stats)

  # Add the number of uniques to the FeatureNameStatistics proto.
  result.string_stats.unique = num_unique
  return result


def make_feature_stats_proto_topk_uniques_custom_stats(
    feature_path: types.FeaturePath,
    num_top_values: int,
    num_rank_histogram_buckets: int,
    num_unique: int,
    value_count_list: List[FeatureValueCount],
    weighted_value_count_list: Optional[List[FeatureValueCount]] = None,
    frequency_threshold: int = 1,
    weighted_frequency_threshold: Optional[float] = None
) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing top-k and uniques stats.

  Args:
    feature_path: The path of the feature.
    num_top_values: The number of most frequent feature values to keep for
      string features.
    num_rank_histogram_buckets: The number of buckets in the rank histogram for
      string features.
    num_unique: The number of unique values in the feature.
    value_count_list: A list of FeatureValueCount tuples.
    weighted_value_count_list: An optional list of FeatureValueCount tuples for
      weighted features.
    frequency_threshold: The minimum number of examples the most frequent values
      must be present in.
    weighted_frequency_threshold: The minimum weighted number of examples the
      most frequent weighted values must be present in. Optional.

  Returns:
    A FeatureNameStatistics proto containing the top-k and uniques stats.
  """

  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())

  # Create a FeatureNameStatistics proto that includes the unweighted top-k
  # stats.
  topk_stats = _make_feature_stats_proto_topk(feature_path, value_count_list,
                                              False, num_top_values,
                                              frequency_threshold,
                                              num_rank_histogram_buckets)

  # Topk rank histogram.
  topk_custom_stats = result.custom_stats.add(
      name=_TOPK_SKETCH_CUSTOM_STATS_NAME)
  topk_custom_stats.rank_histogram.CopyFrom(
      topk_stats.string_stats.rank_histogram)

  # If weights were provided, create another FeatureNameStatistics proto that
  # includes the weighted top-k stats, and then copy those weighted top-k stats
  # into the result proto.
  if weighted_value_count_list:
    assert weighted_frequency_threshold is not None
    weighted_topk_stats = _make_feature_stats_proto_topk(
        feature_path, weighted_value_count_list, True, num_top_values,
        weighted_frequency_threshold, num_rank_histogram_buckets)

    # Weighted Topk rank histogram.
    weighted_topk_custom_stats = result.custom_stats.add(
        name=_WEIGHTED_TOPK_SKETCH_CUSTOM_STATS_NAME)
    weighted_topk_custom_stats.rank_histogram.CopyFrom(
        weighted_topk_stats.string_stats.weighted_string_stats.rank_histogram)

  # Add the number of uniques to the FeatureNameStatistics proto.
  result.custom_stats.add(
      name=_UNIQUES_SKETCH_CUSTOM_STATS_NAME, num=num_unique)
  return result


def make_dataset_feature_stats_proto_unique_single(
    feature_path_tuple: types.FeaturePathTuple,
    num_uniques: int,
) -> statistics_pb2.DatasetFeatureStatistics:
  """Makes a DatasetFeatureStatistics proto with uniques stats for a feature."""
  feature_path = types.FeaturePath(feature_path_tuple)
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto_uniques(feature_path, num_uniques))
  return result


def make_dataset_feature_stats_proto_topk_single(
    feature_path_tuple: types.FeaturePathTuple,
    value_count_list: List[FeatureValueCount],
    is_weighted_stats: bool, num_top_values: int,
    frequency_threshold: Union[int, float],
    num_rank_histogram_buckets: int) -> statistics_pb2.DatasetFeatureStatistics:
  """Makes a DatasetFeatureStatistics proto with top-k stats for a feature."""
  feature_path = types.FeaturePath(feature_path_tuple)
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto_topk(feature_path, value_count_list,
                                     is_weighted_stats, num_top_values,
                                     frequency_threshold,
                                     num_rank_histogram_buckets))
  return result


def _make_feature_stats_proto_uniques(
    feature_path: types.FeaturePath, num_uniques: int,
) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing the uniques stats."""
  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())
  result.string_stats.unique = num_uniques
  return result


def _make_feature_stats_proto_topk(
    feature_path: types.FeaturePath,
    top_k_values_pairs: List[FeatureValueCount],
    is_weighted_stats: bool, num_top_values: int,
    frequency_threshold: Union[float, int],
    num_rank_histogram_buckets: int) -> statistics_pb2.FeatureNameStatistics:
  """Makes a FeatureNameStatistics proto containing the top-k stats."""
  # Sort (a copy of) the top_k_value_pairs in descending order by count.
  # Where multiple feature values have the same count, consider the feature with
  # the 'larger' feature value to be larger for purposes of breaking the tie.

  top_k_values_pairs = sorted(
      top_k_values_pairs,
      key=lambda pair: (pair.count, pair.feature_value),
      reverse=True)

  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())

  if is_weighted_stats:
    string_stats = result.string_stats.weighted_string_stats
  else:
    string_stats = result.string_stats

  for i in range(len(top_k_values_pairs)):
    value, count = top_k_values_pairs[i]
    if count < frequency_threshold:
      break
    # Check if we have a valid utf-8 string. If not, assign a default invalid
    # string value.
    if isinstance(value, six.binary_type):
      decoded_value = stats_util.maybe_get_utf8(value)
      if decoded_value is None:
        _NON_UTF8_VALUES_COUNTER.inc()
        value = constants.NON_UTF8_PLACEHOLDER
      else:
        value = decoded_value
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


def output_categorical_numeric(categorical_numeric_types: Mapping[
    types.FeaturePath, 'schema_pb2.FeatureType'],
                               feature_path: types.FeaturePath,
                               feature_type: Optional[int]) -> bool:
  """Check if a feature path should be treated as a numeric categorical.

  Args:
    categorical_numeric_types: A mapping from feature path to schema feature
      type.
    feature_path: The path of a feature.
    feature_type: Either a statistics_pb2.FeatureNameStatistics.Type or None.

  Returns:
    True feature_type is INT and feature_path was expressed in the schema as an
    INT.
  """
  if feature_path not in categorical_numeric_types:
    return False
  schema_type = categorical_numeric_types[feature_path]

  # Only output categorical numeric if the feature was declared categorical
  # numeric and the Arrow type (INT/FLOAT) matches the schema type (INT/FLOAT).
  if (feature_type == statistics_pb2.FeatureNameStatistics.INT and
      schema_type == schema_pb2.INT):
    return True
  if (feature_type == statistics_pb2.FeatureNameStatistics.FLOAT and
      schema_type == schema_pb2.FLOAT):
    return True

  return False
