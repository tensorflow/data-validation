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
"""A stats generator which counts the number of missing (null) values in a path.

This constituent stats generator counts the total number of rows in all batches
which are null. If a set of `required_paths` are also provided, only those rows
in which at least one of the `required paths` is present will be counted. This
is useful in the case where a set of features should be considered holistically
(like weighted features or sparse features). In this case, if the whole feature
is missing (i.e. all components of the weighted or sparse feature) then it is
not useful to report the absence of a single component.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch
from tensorflow_data_validation.statistics.generators import stats_generator
from typing import Iterable, Optional, Text, Tuple, Union


class CountMissingGenerator(stats_generator.ConstituentStatsGenerator):
  """A stats generator which counts the number of missing values in a path."""

  def __init__(self,
               path: types.FeaturePath,
               required_paths: Optional[Iterable[types.FeaturePath]] = None):
    """Initializes to count the number of null lists in a specific feature path.

    When required_paths is also passed, rows which are null for all of
    the required paths will not be counted as missing.

    Args:
      path: The path in which to count missing rows.
      required_paths: The set of paths among which at least one must be non-null
        in order for a null entry in the array for `path` to contribute to the
        missing count.
    """
    self._path = path
    if required_paths:
      self._required_paths = tuple(sorted(required_paths))
    else:
      self._required_paths = None

  @classmethod
  def key(
      cls,
      path: types.FeaturePath,
      required_paths: Optional[Iterable[types.FeaturePath]] = None
  ) -> Tuple[Union[Text, types.FeaturePath], ...]:
    """Generates a key for instances created with the same args passed to init.

    Args:
      path: The path in which to count missing rows.
      required_paths: The set of paths among which at least one must be non-null
        in order for a null entry in the array for `path` to contribute to the
        missing count.

    Returns:
      The unique key for this set of init args.
    """
    key_tuple = ('CountMissingGenerator', path)
    if required_paths:
      key_tuple += tuple(sorted(required_paths))
    return key_tuple

  def get_key(self) -> Tuple[Union[Text, types.FeaturePath], ...]:
    """Generates a unique key for this instance.

    Returns:
      The unique key for this set of init args.
    """
    return CountMissingGenerator.key(self._path, self._required_paths)

  def create_accumulator(self) -> int:
    return 0

  def add_input(self, accumulator, batch: input_batch.InputBatch) -> int:
    """Accumulates the number of missing rows from new batch."""
    null_mask = batch.null_mask(self._path)
    if self._required_paths:
      required_null_mask = batch.all_null_mask(*self._required_paths)
      null_mask = null_mask & ~required_null_mask
    return accumulator + np.sum(null_mask)

  def merge_accumulators(self, accumulators: Iterable[int]) -> int:
    return sum(accumulators)

  def extract_output(self, accumulator: int) -> int:
    """Returns the count of missing values for this stats generator."""
    return accumulator
