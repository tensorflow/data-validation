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

"""Utilities for binning numeric arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pyarrow as pa

from typing import Sequence, Tuple


def bin_array(array: pa.Array,
              boundaries: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
  """Converts an array to an array of bin indices using provided boundaries.

  Provided n boundaries, bin will return bin indices in [-1, n]. Bin index
  0 corresponds to the bin [-infinity, boundaries[0]] and bin index
  len(boundaries) corresponds to the bin [boundaries[-1], infinity). Bin index
  of np.nan or None means that the value is null.

  To convert bin indices back into a useful form, see _get_bucket().

  Args:
    array: An ascending sorted array of numeric values to convert to bin
      indices.
    boundaries: A list of bin boundaries to use, excluding the implicit lower
      bound (-infinity) and upper bound (infinity).

  Returns:
    (element_indices, bins): A pair of numpy arrays in which the first element
    is the indices of input array elements with well-defined bins (i.e.
    non-null) and the second element is the bin index for the element at the
    corresponding index within the element indices array. In other words, the
    bin for array[element_indices[i]] is bins[i].
  """
  if pa.types.is_null(array.type):
    return np.array([]), np.array([])

  # Given an array with shape (n, 1) and a list of boundaries of shape (1, b),
  # np.less (and np.greater_equal) returns an (n, b) shape matrix of boolean
  # values where the entry at (i, j) indicates whether the ith array element is
  # less than (or greater than or equal to) the jth boundary.
  array_column = np.expand_dims(np.asarray(array, dtype=float), axis=1)
  lower_bound_masks = np.greater_equal(array_column, boundaries)
  upper_bound_masks = np.less(array_column, boundaries)

  # Add two open interval buckets on the ends and shift mask indexing so that
  # lower_bound_masks[i, j] indicates that array[i] >= boundaries[j-1]
  # and upper_bound_masks[i,j] indicates that array[i] < boundaries[j], where
  # the first boundary is implicitly negative infinity and the last boundary is
  # implicitly positive infinity.
  true_mask = np.ones(array_column.shape, dtype=bool)
  lower_bound_masks = np.hstack([true_mask, lower_bound_masks])
  upper_bound_masks = np.hstack([upper_bound_masks, true_mask])

  # bin_mask[i,j] = (array[i] >= boundaries[j-1]) && (array[i] < boundaries[j])
  bin_masks = lower_bound_masks & upper_bound_masks

  # Find the indices of the nonzero elements.
  return bin_masks.nonzero()


def get_boundaries(bin_index: int,
                   boundaries: Sequence[float]) -> Tuple[float, float]:
  """Returns a the bucket [min, max) corresponding to the provided bin_index.

  Args:
    bin_index: A bin index returned by bin_array.
    boundaries: The same boundaries passed to bin_array.

  Returns:
    The low and high boundaries of the bin corresponding to bin_index.
  """
  inf = float('inf')
  low_value = -inf if bin_index == 0 else boundaries[bin_index - 1]
  high_value = inf if bin_index == len(boundaries) else boundaries[bin_index]
  return low_value, high_value
