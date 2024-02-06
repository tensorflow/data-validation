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
"""Module that computes cross feature statistics.

We compute the following statistics for numeric feature crosses (only univalent
feature values are considered):
- Standard covariance. E[(X-E[X])*(Y-E[Y])]
- Pearson product-moment correlation coefficient.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import random

from typing import Dict, Iterable, List, Optional, Text
from absl import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # pylint: disable=g-multiple-import
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util

from tensorflow_metadata.proto.v0 import path_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialCrossFeatureStats(object):
  """Holds partial cross feature statistics for a feature cross."""

  __slots__ = ['sum_x', 'sum_y', 'sum_square_x', 'sum_square_y', 'sum_xy',
               'count']

  def __init__(self):
    self.sum_x = 0
    self.sum_y = 0
    self.sum_square_x = 0
    self.sum_square_y = 0
    self.sum_xy = 0
    self.count = 0

  def __iadd__(self, other: '_PartialCrossFeatureStats'
              ) -> '_PartialCrossFeatureStats':
    """Merges two partial cross feature statistics."""
    self.sum_x += other.sum_x
    self.sum_y += other.sum_y
    self.sum_square_x += other.sum_square_x
    self.sum_square_y += other.sum_square_y
    self.sum_xy += other.sum_xy
    self.count += other.count
    return self

  def update(self, feature_x: Series, feature_y: Series) -> None:
    """Updates partial cross feature statistics."""
    self.sum_x += feature_x.sum()
    self.sum_y += feature_y.sum()
    # pytype: disable=unsupported-operands  # typed-pandas
    self.sum_square_x += (feature_x ** 2).sum()
    self.sum_square_y += (feature_y ** 2).sum()
    self.sum_xy += (feature_x * feature_y).sum()
    # pytype: enable=unsupported-operands  # typed-pandas
    self.count += len(feature_x)


CrossFeatureStatsGeneratorAccumulator = Dict[types.FeatureCross,
                                             _PartialCrossFeatureStats]


class CrossFeatureStatsGenerator(stats_generator.CombinerStatsGenerator):
  """A combiner statistics generator that computes cross feature statistics.
  """

  def __init__(
      self,  # pylint: disable=useless-super-delegation
      name: Text = 'CrossFeatureStatsGenerator',
      feature_crosses: Optional[List[types.FeatureCross]] = None,
      sample_rate: float = 0.1) -> None:
    """Initializes cross feature statistics generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      feature_crosses: List of numeric feature crosses for which to compute
        statistics. If None, we compute statistics for all numeric crosses.
      sample_rate: Sample rate.
    """
    super(CrossFeatureStatsGenerator, self).__init__(name, None)
    self._feature_crosses = feature_crosses
    self._features_needed = None
    if self._feature_crosses:
      self._features_needed = set()
      for (feat_x, feat_y) in self._feature_crosses:
        self._features_needed.add(feat_x)
        self._features_needed.add(feat_y)
    self._sample_rate = sample_rate

  # Create an accumulator, which maps feature name to the partial stats
  # associated with the feature.
  def create_accumulator(self) -> CrossFeatureStatsGeneratorAccumulator:
    return {}

  def _get_univalent_values_with_parent_indices(
      self, examples: pa.RecordBatch) -> Dict[types.FeatureName, DataFrame]:
    """Extracts univalent values for each feature along with parent indices."""
    result = {}
    for feature_name, feat_arr in zip(examples.schema.names, examples.columns):
      if (self._features_needed is not None and
          feature_name not in self._features_needed):
        continue
      feature_type = stats_util.get_feature_type_from_arrow_type(
          feature_name, feat_arr.type)
      # Only consider crosses of numeric features.
      # TODO(zhuo): Support numeric features nested under structs.
      if feature_type in (None, statistics_pb2.FeatureNameStatistics.STRING,
                          statistics_pb2.FeatureNameStatistics.STRUCT):
        continue
      value_lengths = np.asarray(array_util.ListLengthsFromListArray(feat_arr))
      univalent_parent_indices = set((value_lengths == 1).nonzero()[0])
      # If there are no univalent values, continue to the next feature.
      if not univalent_parent_indices:
        continue
      flattened, value_parent_indices = array_util.flatten_nested(
          feat_arr, True)
      non_missing_values = np.asarray(flattened)
      if feature_type == statistics_pb2.FeatureNameStatistics.FLOAT:
        # Remove any NaN values if present.
        non_nan_mask = ~np.isnan(non_missing_values)
        non_missing_values = non_missing_values[non_nan_mask]
        value_parent_indices = value_parent_indices[non_nan_mask]
      df = pd.DataFrame({feature_name: non_missing_values,
                         'parent_index': value_parent_indices})
      # Only keep the univalent feature values.
      df = df[df['parent_index'].isin(univalent_parent_indices)]

      result[feature_name] = df

    return result

  # Incorporates the input (an arrow RecordBatch) into the accumulator.
  def add_input(
      self, accumulator: CrossFeatureStatsGeneratorAccumulator,
      examples: pa.RecordBatch
  ) -> Dict[types.FeatureCross, _PartialCrossFeatureStats]:
    if random.random() > self._sample_rate:
      return accumulator
    # Cache the values and parent indices for each feature. We cache this to
    # avoid doing the same computation for a feature multiple times in
    # each cross.
    features_for_cross = self._get_univalent_values_with_parent_indices(
        examples)

    # Generate crosses of numeric univalent features and update the partial
    # cross stats.
    feature_crosses = itertools.combinations(
        sorted(list(features_for_cross.keys())), 2
    )
    if self._feature_crosses is not None:
      # If the config includes a list of feature crosses to compute, limit the
      # crosses generated to those in that list.
      configured_crosses = set(self._feature_crosses)
      valid_crosses = set(feature_crosses)
      feature_crosses = configured_crosses.intersection(valid_crosses)

      skipped_crosses = configured_crosses.difference(valid_crosses)
      if skipped_crosses:
        logging.warn(
            'Skipping the following configured feature crosses: %s\n Feature'
            ' crosses can be computed only for univalent numeric features. ',
            ', '.join(
                sorted([
                    '_'.join([cross[0], cross[1]]) for cross in skipped_crosses
                ])
            ),
        )
    for feat_name_x, feat_name_y in feature_crosses:
      feat_cross = (feat_name_x, feat_name_y)
      if feat_cross not in accumulator:
        accumulator[feat_cross] = _PartialCrossFeatureStats()
      df_x, df_y = (features_for_cross[feat_name_x],
                    features_for_cross[feat_name_y])
      # Join based on parent index so that we have the value pairs
      # corresponding to each example.
      merged_df = pd.merge(df_x, df_y, on='parent_index')
      # Update the partial cross stats.
      accumulator[feat_cross].update(merged_df[feat_name_x],
                                     merged_df[feat_name_y])

    return accumulator

  # Merge together a list of cross feature statistics.
  def merge_accumulators(
      self, accumulators: Iterable[CrossFeatureStatsGeneratorAccumulator]
  ) -> CrossFeatureStatsGeneratorAccumulator:
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      for feat_cross, cross_feat_stats in accumulator.items():
        if feat_cross not in result:
          result[feat_cross] = cross_feat_stats
        else:
          result[feat_cross] += cross_feat_stats
    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator: CrossFeatureStatsGeneratorAccumulator
                    ) -> statistics_pb2.DatasetFeatureStatistics:
    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feat_cross, cross_feat_stats in accumulator.items():
      # Construct the CrossFeatureStatistics proto from the partial
      # cross feature stats.
      cross_feat_stats_proto = result.cross_features.add()
      path_x = path_pb2.Path()
      path_x.step.append(feat_cross[0])
      path_y = path_pb2.Path()
      path_y.step.append(feat_cross[1])
      cross_feat_stats_proto.path_x.CopyFrom(path_x)
      cross_feat_stats_proto.path_y.CopyFrom(path_y)
      cross_feat_stats_proto.count = cross_feat_stats.count
      if cross_feat_stats.count > 0:
        num_cross_stats_proto = statistics_pb2.NumericCrossStatistics()
        covariance = (cross_feat_stats.sum_xy / cross_feat_stats.count) -\
            (cross_feat_stats.sum_x / cross_feat_stats.count) *\
            (cross_feat_stats.sum_y / cross_feat_stats.count)
        num_cross_stats_proto.covariance = covariance
        std_dev_x = math.sqrt(max(
            0, (cross_feat_stats.sum_square_x / cross_feat_stats.count) -
            math.pow(cross_feat_stats.sum_x / cross_feat_stats.count, 2)))
        std_dev_y = math.sqrt(max(
            0, (cross_feat_stats.sum_square_y / cross_feat_stats.count) -
            math.pow(cross_feat_stats.sum_y / cross_feat_stats.count, 2)))
        if std_dev_x != 0 and std_dev_y != 0:
          correlation = covariance / (std_dev_x * std_dev_y)
          num_cross_stats_proto.correlation = correlation
        cross_feat_stats_proto.num_cross_stats.CopyFrom(num_cross_stats_proto)

    return result
