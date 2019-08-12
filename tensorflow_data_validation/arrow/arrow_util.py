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
# limitations under the License
"""Util functions regarding to Arrow objects."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from tensorflow_data_validation.pywrap import pywrap_tensorflow_data_validation as pywrap
from typing import Iterable, Optional, Text, Tuple

# The following are function aliases thus valid function names.
# pylint: disable=invalid-name
ListLengthsFromListArray = pywrap.TFDV_Arrow_ListLengthsFromListArray
GetFlattenedArrayParentIndices = pywrap.TFDV_Arrow_GetFlattenedArrayParentIndices
GetArrayNullBitmapAsByteArray = pywrap.TFDV_Arrow_GetArrayNullBitmapAsByteArray
GetBinaryArrayTotalByteSize = pywrap.TFDV_Arrow_GetBinaryArrayTotalByteSize
ValueCounts = pywrap.TFDV_Arrow_ValueCounts
MakeListArrayFromParentIndicesAndValues = (
    pywrap.TFDV_Arrow_MakeListArrayFromParentIndicesAndValues)


def primitive_array_to_numpy(primitive_array: pa.Array) -> np.ndarray:
  """Converts a primitive Arrow array to a numpy 1-D ndarray.

  Copying is avoided as much as possible.

  Args:
    primitive_array: a primitive Arrow array.

  Returns:
    A 1-D ndarray of corresponding type (for string/binary arrays, the
    corresponding numpy type is np.object (str, bytes or unicode)).
  """
  array_type = primitive_array.type
  if (pa.types.is_binary(array_type) or
      pa.types.is_string(array_type) or
      primitive_array.null_count > 0):
    # no free conversion.
    return primitive_array.to_pandas()
  return primitive_array.to_numpy()


def enumerate_arrays(
    table: pa.Table, weight_column: Optional[Text], enumerate_leaves_only: bool
) -> Iterable[Tuple[types.FeaturePath, pa.Array, Optional[np.ndarray]]]:
  """Enumerates arrays in a Table.

  It assumes all the columns in `table` has only one chunk.
  It assumes `table` contains only arrays of the following supported types:
    - list<primitive>
    - list<struct<[Ts]>> where Ts are the types of the fields in the struct
      type, and they can only be one of the supported types
      (recursion intended).

  It enumerates each column (i.e. array, because there is only one chunk) in
  the table (also see `enumerate_leaves_only`) If an array is of type
  list<struct<[Ts]>>, then it flattens the outermost list, then enumerates the
  array of each field in the result struct<[Ts]> array, and continues
  recursively. The weights get "aligned" automatically in this process,
  therefore weights, the third term in the returned tuple always has array[i]'s
  weight being weights[i].

  Args:
    table: The Table whose arrays to be visited. It is assumed that the table
      contains only one chunk.
    weight_column: The name of the weight column, or None. The elements of
      the weight column should be lists of numerics, and each list should
      contain only one value. The weight column is not enumarated.
    enumerate_leaves_only: If True, only enumerate "leaf" arrays.
      Otherwise, also enumerate the struct arrays where the leaf arrays are
      contained.

  Yields:
    A tuple. The first term is the path of the feature; the second term is
    the feature array and the third term is the weight array for the feature
    array (i.e. weights[i] is the weight for array[i]).

  Raises:
    ValueError: When the weight column is not a list array whose elements are
      not 1-element lists.
  """

  def _recursion_helper(
      feature_path: types.FeaturePath, array: pa.Array,
      weights: Optional[np.ndarray]
  ) -> Iterable[Tuple[types.FeaturePath, pa.Array, Optional[np.ndarray]]]:
    """Recursion helper."""
    array_type = array.type
    if (pa.types.is_list(array_type) and
        pa.types.is_struct(array_type.value_type)):
      if not enumerate_leaves_only:
        yield (feature_path, array, weights)
      flat_struct_array = array.flatten()
      flat_weights = None
      if weights is not None:
        flat_weights = weights[GetFlattenedArrayParentIndices(array).to_numpy()]
      for field in flat_struct_array.type:
        field_name = field.name
        # use "yield from" after PY 3.3.
        for e in _recursion_helper(
            feature_path.child(field_name),
            flat_struct_array.field(field_name), flat_weights):
          yield e
    else:
      yield (feature_path, array, weights)

  weights = None
  if weight_column is not None:
    weights = table.column(weight_column).data.chunk(0).flatten().to_numpy()
    if weights.size != table.num_rows:
      raise ValueError(
          'The weight feature must have exactly one value in each example')
  for column in table.columns:
    column_name = column.name
    if column_name == weight_column:
      continue
    # use "yield from" after PY 3.3.
    for e in _recursion_helper(
        types.FeaturePath([column_name]), column.data.chunk(0), weights):
      yield e
