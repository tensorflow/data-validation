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
"""An abstraction for interacting with a pyarrow.RecordBatch.

An input batch is a thin wrapper around a pyarrow.RecordBatch that implements
various common operations, and handles caching for some.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pyarrow as pa

from tensorflow_data_validation import types
from tfx_bsl.arrow import array_util
from tfx_bsl.arrow import path as tfx_bsl_path
from tfx_bsl.arrow import table_util


class InputBatch(object):
  """A Batch wraps a pyarrow.RecordBatch and provides caching functionality.

  This is useful when several different generators need to apply the same
  computation to the same input record batch. A CompositeCombinerStatsGenerator
  instantiates an InputBatch and then passes it to the add_input method of each
  constituent generator. This allows the constituent generators to reuse
  expensive operations that have already been computed by other constituents.
  """

  def __init__(self, record_batch: pa.RecordBatch):
    self._record_batch = record_batch
    self._cache = {}

  @property
  def record_batch(self) -> pa.RecordBatch:
    return self._record_batch

  def null_mask(self, path: types.FeaturePath) -> np.ndarray:
    """Returns a boolean mask of rows which are null in the referenced array.

    If the requested path cannot be found in the record batch, it will be
    considered null in all rows in the record batch.

    Args:
      path: The path corresponding to the array from which to generate the null
        mask.
    """
    try:
      array, _ = table_util.get_array(
          self._record_batch,
          tfx_bsl_path.ColumnPath(path.steps()),
          return_example_indices=False,
      )
      # GetArrayNullBitmapAsByteArray is only useful for non-null type arrays.
      if pa.types.is_null(array.type):
        return np.full(self._record_batch.num_rows, True)
      return np.asarray(
          array_util.GetArrayNullBitmapAsByteArray(array), dtype=bool)
    except KeyError:
      return np.full(self._record_batch.num_rows, True)

  def all_null_mask(self, *paths: types.FeaturePath) -> np.ndarray:
    """Returns a boolean mask of rows which are null in all provided paths.

    All provided paths must correspond to array of the same length.

    Args:
      *paths: Any number of paths for which to compute the all null mask.

    Returns:
      A boolean numpy array of shape (N,), where N is the size of all arrays
      referenced by paths.
    """
    key = ('all_null_mask',) + paths
    if key in self._cache:
      return self._cache[key]
    if not paths:
      raise ValueError('Paths cannot be empty.')
    mask = self.null_mask(paths[0])
    for path in paths[1:]:
      path_mask = self.null_mask(path)
      if mask.size != path_mask.size:
        raise ValueError('All array lengths must be equal. '
                         'other_null_mask.size != null_mask({}).size '
                         '({} != {}).'.format(path, mask.size, path_mask.size))
      mask = mask & path_mask
    self._cache[key] = mask
    return mask

  def list_lengths(self, path: types.FeaturePath) -> np.ndarray:
    """Returns a numpy array containing the length of each feature list.

    If the requested path is not present in the record batch wrapped by the
    InputBatch, the returned array will consist of zeros, and be of length equal
    to the number of rows in the record batch.

    Args:
      path: The path for which to return list lengths.

    Returns:
      An ndarray containing the lengths of each nested list. The returned
      ndarray will be of shape (N,) where N is the number of rows in the
      referenced array (or in the record batch, if the path cannot be found).

    Raises:
      ValueError: When the referenced array is neither a ListArray nor null.
    """
    key = ('list_lengths({})', path)
    if key in self._cache:
      return self._cache[key]
    try:
      array, _ = table_util.get_array(
          self._record_batch,
          tfx_bsl_path.ColumnPath(path.steps()),
          return_example_indices=False,
      )
      if pa.types.is_null(array.type):
        lengths = np.full(self._record_batch.num_rows, 0)
      elif not array_util.is_list_like(array.type):
        raise ValueError('Can only compute list lengths on list arrays, found '
                         '{}'.format(array.type))
      else:
        lengths = np.asarray(array_util.ListLengthsFromListArray(array))
    except KeyError:
      lengths = np.full(self._record_batch.num_rows, 0)
    self._cache[key] = lengths
    return lengths
