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
"""A generator for computing the min/max list length diffs between two paths.

This stats generator is useful for ensuring that two separate paths in a record
batch have equal list lengths for all the rows in which they appear. It also
supports restricting these comparison to rows in which all of the
`required_paths` are non-null. This prevents rows in which both the `left_path`
and `right_path` are missing from contributing the length diff 0 (0 - 0) to the
accumulated min and max.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch
from tensorflow_data_validation.statistics.generators import stats_generator
from typing import Iterable, Optional, Text, Tuple, Union

# Accumulator type
MinMax = Tuple[float, float]


class LengthDiffGenerator(stats_generator.ConstituentStatsGenerator):
  """A generator which tracks the min/max list length diffs for two paths."""

  def __init__(self,
               left_path: types.FeaturePath,
               right_path: types.FeaturePath,
               required_paths: Optional[Iterable[types.FeaturePath]] = None):
    """Initializes LengthDiffGenerator for a specific pair of paths.

    Args:
      left_path: The path whose list lengths should be treated as the left side
        of the difference (lengths(left_path) - lengths(right_path)).
      right_path: The path whose list lengths should be treated as the right
        side of the difference (lengths(left_path) - lengths(right_path)).
      required_paths: The set of paths which must all be non-null in order for a
        length diff at a given row to contribute to the min or max.
    """
    self._left_path = left_path
    self._right_path = right_path
    if required_paths:
      self._required_paths = tuple(sorted(required_paths))
    else:
      self._required_paths = None

  @classmethod
  def key(
      cls,
      left_path: types.FeaturePath,
      right_path: types.FeaturePath,
      required_paths: Optional[Iterable[types.FeaturePath]] = None
  ) -> Tuple[Union[Text, types.FeaturePath], ...]:
    """Generates key for an instance created with the same args passed to init.

    Args:
      left_path: The path whose list lengths should be treated as the left side
        of the difference.
      right_path: The path whose list lengths should be treated as the right
        side of the difference.
      required_paths: The set of paths which must all be non-null in order for a
        length diff in the arrays for `left_path` and `right_path` to contribute
        to the accumulated min and max.

    Returns:
      The unique key for this set of init args.
    """
    key_tuple = ('LengthDiffGenerator', left_path, right_path)
    if required_paths:
      key_tuple += tuple(sorted(required_paths))
    return key_tuple

  def get_key(self) -> Tuple[Union[Text, types.FeaturePath], ...]:
    """Generates a unique ID for this instance.

    Returns:
      The unique key for this set of init args.
    """
    return LengthDiffGenerator.key(self._left_path, self._right_path,
                                   self._required_paths)

  def create_accumulator(self) -> MinMax:
    return float('inf'), float('-inf')

  def add_input(self, accumulator: MinMax,
                batch: input_batch.InputBatch) -> MinMax:
    """Updates the min and max lengths from new batch."""
    try:
      left_lengths = batch.list_lengths(self._left_path)
    except KeyError:
      left_lengths = np.full(batch.record_batch.num_rows, 0)
    try:
      right_lengths = batch.list_lengths(self._right_path)
    except KeyError:
      right_lengths = np.full(batch.record_batch.num_rows, 0)
    diffs = left_lengths - right_lengths

    if self._required_paths:
      diffs = diffs[~batch.all_null_mask(*self._required_paths)]

    min_diff, max_diff = accumulator
    if diffs.size:
      min_diff = min(min_diff, np.min(diffs))
      max_diff = max(max_diff, np.max(diffs))
    return min_diff, max_diff

  def merge_accumulators(self, accumulators: Iterable[MinMax]) -> MinMax:
    result_min, result_max = self.create_accumulator()
    for acc_min, acc_max in accumulators:
      result_min = min(result_min, acc_min)
      result_max = max(result_max, acc_max)
    return result_min, result_max

  def extract_output(self, accumulator: MinMax) -> MinMax:
    """Returns the length differences as the tuple (min_diff, max_diff).

    If no rows have ever been observed in which all the `required_paths` were
    non-null, the min and max will be set to 0.

    Args:
      accumulator: The input accumulator of the form (min_diff, max_diff).

    Returns:
      A tuple of (min_diff, max_diff).
    """
    min_diff, max_diff = accumulator
    min_diff = min_diff if min_diff != float('inf') else 0
    max_diff = max_diff if max_diff != float('-inf') else 0
    return (min_diff, max_diff)
