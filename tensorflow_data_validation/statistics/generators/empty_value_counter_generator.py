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
"""Module that counts rows with given empty value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Iterable

from absl import logging
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util

from tensorflow_metadata.proto.v0 import statistics_pb2


class _PartialCounterStats(object):
  """Partial feature stats for dates/times."""

  def __init__(self) -> None:
    self.counter = collections.Counter(
        {'int_-1': 0, 'str_empty': 0, 'list_empty': 0}
    )

  def __add__(self, other: '_PartialCounterStats') -> '_PartialCounterStats':
    """Merges two partial stats."""
    self.counter.update(other.counter)
    return self

  def update(
      self,
      values: np.ndarray,
      value_type: types.FeatureNameStatisticsType,
      is_multivalent: bool = False,
  ) -> None:
    """Updates the partial  statistics using the values.

    Args:
      values: A numpy array of values in a batch.
      value_type: The type of the values.
      is_multivalent: If the feature is multivalent.
    """

    # Multivalent feature handling.
    if is_multivalent:
      empty_list = (values == 0).sum()
      self.counter.update({'list_empty': empty_list})
    elif (
        value_type == statistics_pb2.FeatureNameStatistics.STRING
        or value_type == statistics_pb2.FeatureNameStatistics.BYTES
    ):
      empty_str = 0
      for value in values:
        if value is not None and not value:
          empty_str += 1
      self.counter.update({'str_empty': empty_str})

    elif (
        value_type == statistics_pb2.FeatureNameStatistics.FLOAT
        or value_type == statistics_pb2.FeatureNameStatistics.INT
    ):
      empty_neg_1 = 0
      for value in values:
        if value == -1:
          empty_neg_1 += 1
      self.counter.update({'int_-1': empty_neg_1})
    else:
      logging.warning('Unsupported type: %s , %s', values[0].dtype, value_type)
      raise ValueError(
          'Attempt to update partial time stats with values of an '
          'unsupported type.'
      )


class EmptyValueCounterGenerator(stats_generator.CombinerFeatureStatsGenerator):
  """Counts rows with given empty values."""

  def __init__(self) -> None:
    """Initializes a EmptyValueCounterGenerator."""

    super(EmptyValueCounterGenerator, self).__init__(
        'EmptyValueCounterGenerator'
    )

  def create_accumulator(self) -> _PartialCounterStats:
    """Returns a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    return _PartialCounterStats()

  def add_input(
      self,
      accumulator: _PartialCounterStats,
      feature_path: types.FeaturePath,
      feature_array: pa.Array,
  ) -> _PartialCounterStats:
    """Returns result of folding a batch of inputs into the current accumulator.

    Args:
      accumulator: The current accumulator.
      feature_path: The path of the feature.
      feature_array: An arrow Array representing a batch of feature values which
        should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """

    feature_type = stats_util.get_feature_type_from_arrow_type(
        feature_path, feature_array.type
    )
    # Ignore null array.
    if feature_type is None or not feature_array:
      return accumulator

    nest_level = arrow_util.get_nest_level(feature_array.type)
    if nest_level > 1:
      # Flatten removes top level nulls.
      feature_array = feature_array.flatten()
      list_lengths = array_util.ListLengthsFromListArray(feature_array)
      accumulator.update(
          np.asarray(list_lengths), feature_type, is_multivalent=True
      )
    elif (
        feature_type == statistics_pb2.FeatureNameStatistics.STRING
        or feature_type == statistics_pb2.FeatureNameStatistics.BYTES
    ):

      def _maybe_get_utf8(val):
        return stats_util.maybe_get_utf8(val) if isinstance(val, bytes) else val

      values = np.asarray(array_util.flatten_nested(feature_array)[0])
      maybe_utf8 = np.vectorize(_maybe_get_utf8, otypes=[object])(values)
      accumulator.update(maybe_utf8, feature_type)
    elif (
        feature_type == statistics_pb2.FeatureNameStatistics.INT
        or feature_type == statistics_pb2.FeatureNameStatistics.FLOAT
    ):
      values = np.asarray(array_util.flatten_nested(feature_array)[0])
      accumulator.update(values, feature_type)
    else:
      logging.warning('Unsupported type: %s', feature_type)
      raise ValueError(
          'Attempt to update partial time stats with values of an '
          'unsupported type.'
      )

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_PartialCounterStats]
  ) -> _PartialCounterStats:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    it = iter(accumulators)
    result = next(it)
    for acc in it:
      result += acc
    return result

  def extract_output(
      self, accumulator: _PartialCounterStats
  ) -> statistics_pb2.FeatureNameStatistics:
    """Returns the result of converting accumulator into the output value.

    This method will add the time_domain custom stat to the proto if the match
    ratio is at least self._match_ratio. The match ratio is determined by
    dividing the number of values that have the most common valid format by the
    total number of values considered. If this method adds the time_domain
    custom stat, it also adds the match ratio and the most common valid format
    to the proto as custom stats.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.FeatureNameStatistics()
    for name, count in accumulator.counter.items():
      if count:
        result.custom_stats.add(name=name, num=count)
    return result
