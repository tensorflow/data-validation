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

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators import top_k_stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.types_compat import Dict, List, Optional, Set, Text
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# _ValueCounts holds two collections.Counter objects: one that holds the
# unweighted counts of each unique value for a given feature, and one that holds
# the weighted counts of each unique value for a given feature.
_ValueCounts = collections.namedtuple('_ValueCounts',
                                      ['unweighted_counts', 'weighted_counts'])


def _make_feature_stats_proto(
    feature_name,
    value_count_list,
    weighted_value_count_list,
    is_categorical, num_top_values,
    num_rank_histogram_buckets):
  """Makes a FeatureNameStatistics proto containing top-k and uniques stats."""
  # Create a FeatureNameStatistics proto that includes the unweighted top-k
  # stats.
  result = top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
      feature_name, value_count_list, is_categorical, False, num_top_values,
      num_rank_histogram_buckets)

  # If weights were provided, create another FeatureNameStatistics proto that
  # includes the weighted top-k stats, and then copy those weighted top-k stats
  # into the result proto.
  if weighted_value_count_list:
    weighted_result = (
        top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
            feature_name, weighted_value_count_list, is_categorical, True,
            num_top_values, num_rank_histogram_buckets))

    result.string_stats.weighted_string_stats.CopyFrom(
        weighted_result.string_stats.weighted_string_stats)

  # Add the number of uniques to the FeatureNameStatistics proto.
  result.string_stats.unique = len(value_count_list)

  return result


def _make_dataset_feature_stats_proto_with_multiple_features(
    feature_names_to_value_counts,
    weighted_feature_names_to_value_counts,
    categorical_features, num_top_values,
    num_rank_histogram_buckets):
  """Makes a DatasetFeatureStatistics proto containing multiple features."""
  result = statistics_pb2.DatasetFeatureStatistics()
  for feature_name, value_count in feature_names_to_value_counts.items():
    if weighted_feature_names_to_value_counts:
      weighted_value_count = weighted_feature_names_to_value_counts[
          feature_name]
    else:
      weighted_value_count = None
    result.features.add().CopyFrom(
        _make_feature_stats_proto(feature_name, value_count,
                                  weighted_value_count,
                                  feature_name in categorical_features,
                                  num_top_values, num_rank_histogram_buckets))
  return result


class _WeightedCounter(collections.Counter):
  """An extension of collections.Counter that supports weights."""

  def weighted_update(self, iterable=None, weight=1):
    """Like Counter.update(), except weights the counts when adding them."""

    if iterable is not None:
      self_get = self.get
      for elem in iterable:
        self[elem] = self_get(elem, 0) + weight


class TopKUniquesCombinerStatsGenerator(stats_generator.CombinerStatsGenerator):
  """Combiner statistics generator that computes top-k and uniques stats.

  This generator is for in-memory data only. The TopKStatsGenerator and
  UniquesStatsGenerator are for distributed data.
  """

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name = 'TopKUniquesCombinerStatsGenerator',
      schema = None,
      weight_feature = None,
      num_top_values = 2,
      num_rank_histogram_buckets = 1000):
    """Initializes a top-k and uniques combiner statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: Feature name whose numeric value represents the weight of
        an example. None if there is no weight feature.
      num_top_values: The number of most frequent feature values to keep for
        string features.
      num_rank_histogram_buckets: The number of buckets in the rank histogram
        for string features.
    """
    super(TopKUniquesCombinerStatsGenerator, self).__init__(name, schema)
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])
    self._weight_feature = weight_feature
    self._num_top_values = num_top_values
    self._num_rank_histogram_buckets = num_rank_histogram_buckets

  def create_accumulator(self):
    return {}

  def add_input(
      self, accumulator,
      input_batch):
    if self._weight_feature is not None:
      weights = stats_util.get_weight_feature(input_batch, self._weight_feature)

    for feature_name, values in six.iteritems(input_batch):
      # Skip the weight feature.
      if feature_name == self._weight_feature:
        continue
      unweighted_counts = collections.Counter()
      weighted_counts = _WeightedCounter()

      for i, value in enumerate(values):
        # Check if we have a numpy array with at least one value.
        if not isinstance(value, np.ndarray) or value.size == 0:
          continue
        # Check that the feature is either categorical or of string type.
        if not (feature_name in self._categorical_features or get_feature_type(
            value.dtype) == statistics_pb2.FeatureNameStatistics.STRING):
          continue
        if feature_name in self._categorical_features:
          value = value.astype(str)

        unweighted_counts.update(value)
        if self._weight_feature is not None:
          weighted_counts.weighted_update(value, weights[i][0])

      if feature_name not in accumulator:
        accumulator[feature_name] = _ValueCounts(
            unweighted_counts=unweighted_counts,
            weighted_counts=weighted_counts)
      else:
        accumulator[feature_name].unweighted_counts.update(unweighted_counts)
        accumulator[feature_name].weighted_counts.update(weighted_counts)
    return accumulator

  def merge_accumulators(
      self, accumulators
  ):
    result = {}
    for accumulator in accumulators:
      for feature_name, value_counts in accumulator.items():
        if feature_name not in result:
          result[feature_name] = value_counts
        else:
          result[feature_name].unweighted_counts.update(
              value_counts.unweighted_counts)
          if result[feature_name].weighted_counts:
            result[feature_name].weighted_counts.update(
                value_counts.weighted_counts)
    return result

  def extract_output(self, accumulator
                    ):
    feature_names_to_value_counts = dict()
    feature_names_to_weighted_value_counts = dict()

    for feature_name, value_counts in accumulator.items():
      if value_counts.unweighted_counts:
        feature_value_counts = [
            top_k_stats_generator.FeatureValueCount(key, value)
            for key, value in value_counts.unweighted_counts.items()
        ]
        feature_names_to_value_counts[feature_name] = feature_value_counts
      if value_counts.weighted_counts:
        weighted_feature_value_counts = [
            top_k_stats_generator.FeatureValueCount(key, value)
            for key, value in value_counts.weighted_counts.items()
        ]
        feature_names_to_weighted_value_counts[
            feature_name] = weighted_feature_value_counts

    return _make_dataset_feature_stats_proto_with_multiple_features(
        feature_names_to_value_counts, feature_names_to_weighted_value_counts,
        self._categorical_features, self._num_top_values,
        self._num_rank_histogram_buckets)
