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

"""Module that computes statistics for features of string type.

Specifically, we compute the following statistics for each string feature:
  - Average length of the values for this feature.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Dict, List, Optional
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialStringStats(object):
  """Holds partial statistics needed to compute the string statistics
  for a single feature."""

  def __init__(self):
    # The total length of all the values for this feature.
    self.total_bytes_length = 0
    # The total number of values for this feature.
    self.total_num_values = 0


def _merge_string_stats(left,
                        right):
  """Merge two partial string statistics and return the merged statistics."""
  result = _PartialStringStats()
  result.total_bytes_length = (left.total_bytes_length +
                               right.total_bytes_length)
  result.total_num_values = (left.total_num_values +
                             right.total_num_values)
  return result


def _make_feature_stats_proto(
    string_stats, feature_name,
    is_categorical):
  """Convert the partial string statistics into FeatureNameStatistics proto."""
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (statistics_pb2.FeatureNameStatistics.INT if is_categorical
                 else statistics_pb2.FeatureNameStatistics.STRING)
  result.string_stats.avg_length = (string_stats.total_bytes_length /
                                    string_stats.total_num_values)
  return result


class StringStatsGenerator(stats_generator.CombinerStatsGenerator):
  """A combiner statistics generator that computes the statistics
  for features of string type."""

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name = 'StringStatsGenerator',
      schema = None):
    """Initializes a string statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
    """
    super(StringStatsGenerator, self).__init__(name, schema)
    self._categorical_features = set(
        stats_util.get_categorical_numeric_features(schema) if schema else [])

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self):
    return {}

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are numpy arrays representing a batch of examples) into the
  # accumulator.
  def add_input(self, accumulator,
                input_batch
               ):
    # Iterate through each feature and update the partial string stats.
    for feature_name, values in six.iteritems(input_batch):
      # Update the string statistics for every example in the batch.
      for value in values:
        # Check if we have a numpy array with at least one value.
        if not isinstance(value, np.ndarray) or value.size == 0:
          continue

        # If the feature is neither categorical nor of string type, then
        # skip the feature.
        if not (feature_name in self._categorical_features or
                stats_util.make_feature_type(value.dtype) ==
                statistics_pb2.FeatureNameStatistics.STRING):
          continue

        # If we encounter this feature for the first time, create a
        # new partial string stats.
        if feature_name not in accumulator:
          accumulator[feature_name] = _PartialStringStats()

        # If we have a categorical feature, convert the value to string type.
        if feature_name in self._categorical_features:
          value = value.astype(str)

        # Update the partial string stats.
        for v in value:
          accumulator[feature_name].total_bytes_length += len(v)
        accumulator[feature_name].total_num_values += len(value)

    return accumulator

  # Merge together a list of partial string statistics.
  def merge_accumulators(
      self, accumulators
  ):
    result = {}

    for accumulator in accumulators:
      for feature_name, string_stats in accumulator.items():
        if feature_name not in result:
          result[feature_name] = string_stats
        else:
          result[feature_name] = _merge_string_stats(
              result[feature_name], string_stats)
    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator
                    ):
    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, string_stats in accumulator.items():
      # Construct the FeatureNameStatistics proto from the partial
      # string stats.
      feature_stats_proto = _make_feature_stats_proto(
          string_stats, feature_name,
          feature_name in self._categorical_features)
      # Copy the constructed FeatureNameStatistics proto into the
      # DatasetFeatureStatistics proto.
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result
