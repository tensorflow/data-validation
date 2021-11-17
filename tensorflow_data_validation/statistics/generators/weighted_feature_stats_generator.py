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
"""Module that computes statistics used to validate weighted features.

Currently, this module generates the following statistics for each
weighted feature:
- missing_value: Number of examples missing the value_feature.
- missing_weight: Number of examples missing the weight_feature.
- min_weight_length_diff: The minimum of len(weight_feature) -
len(value_feature).
- max_weight_length_diff: The maximum of len(weight_feature) -
len(value_feature).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators.constituents import count_missing_generator
from tensorflow_data_validation.statistics.generators.constituents import length_diff_generator

from typing import Any, Dict, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# LINT.IfChange(custom_stat_names)
_MAX_WEIGHT_LENGTH_DIFF_NAME = 'max_weight_length_diff'
_MIN_WEIGHT_LENGTH_DIFF_NAME = 'min_weight_length_diff'
_MISSING_WEIGHT_NAME = 'missing_weight'
_MISSING_VALUE_NAME = 'missing_value'
# LINT.ThenChange(../../anomalies/schema.cc:weighted_feature_custom_stat_names)


class WeightedFeatureStatsGenerator(stats_generator.CompositeStatsGenerator):
  """Generates statistics for weighted features."""

  def __init__(self,
               schema: schema_pb2.Schema,
               name: Text = 'WeightedFeatureStatsGenerator') -> None:
    constituents = []
    for weighted_feature in schema.weighted_feature:
      weight = types.FeaturePath.from_proto(weighted_feature.weight_feature)
      value = types.FeaturePath.from_proto(weighted_feature.feature)
      component_paths = [weight, value]
      constituents.append(length_diff_generator.LengthDiffGenerator(
          weight, value, required_paths=component_paths))
      constituents.append(count_missing_generator.CountMissingGenerator(
          value, required_paths=component_paths))
      constituents.append(count_missing_generator.CountMissingGenerator(
          weight, required_paths=component_paths))
    super(WeightedFeatureStatsGenerator, self).__init__(name, constituents,
                                                        schema)

  def extract_composite_output(
      self, accumulator: Dict[Text,
                              Any]) -> statistics_pb2.DatasetFeatureStatistics:
    """Populates and returns a stats proto containing custom stats.

    Args:
      accumulator: The final accumulator representing the global combine state.

    Returns:
      A DatasetFeatureStatistics proto.
    """
    result = statistics_pb2.DatasetFeatureStatistics()
    for weighted_feature in self._schema.weighted_feature:
      feature_result = result.features.add(
          path=types.FeaturePath([weighted_feature.name]).to_proto())
      weight = types.FeaturePath.from_proto(weighted_feature.weight_feature)
      value = types.FeaturePath.from_proto(weighted_feature.feature)
      required_paths = [weight, value]

      weight_count_missing = accumulator[
          count_missing_generator.CountMissingGenerator.key(
              weight, required_paths)]
      feature_result.custom_stats.add(
          name=_MISSING_WEIGHT_NAME, num=weight_count_missing)

      value_count_missing = accumulator[
          count_missing_generator.CountMissingGenerator.key(
              value, required_paths)]
      feature_result.custom_stats.add(
          name=_MISSING_VALUE_NAME, num=value_count_missing)

      min_weight_length_diff, max_weight_length_diff = accumulator[
          length_diff_generator.LengthDiffGenerator.key(
              weight, value, required_paths)]
      feature_result.custom_stats.add(
          name=_MIN_WEIGHT_LENGTH_DIFF_NAME, num=min_weight_length_diff)
      feature_result.custom_stats.add(
          name=_MAX_WEIGHT_LENGTH_DIFF_NAME, num=max_weight_length_diff)
    return result
