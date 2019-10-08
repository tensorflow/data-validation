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
from tfx_bsl.arrow import array_util
from typing import Iterable, Optional, Text, Tuple


def _get_weight_feature(input_table: pa.Table,
                        weight_feature: Text) -> np.ndarray:
  """Gets the weight column from the input table.

  Args:
    input_table: Input table.
    weight_feature: Name of the weight feature.

  Returns:
    A numpy array containing the weights of the examples in the input table.

  Raises:
    ValueError: If the weight feature is not present in the input table or is
        not a valid weight feature (must be of numeric type and have a
        single value for each example).
  """
  try:
    weights = input_table.column(weight_feature).data.chunk(0)
  except KeyError:
    raise ValueError('Weight feature "{}" not present in the input '
                     'table.'.format(weight_feature))

  # Before flattening, check that there is a single value for each example.
  weight_lengths = array_util.ListLengthsFromListArray(weights).to_numpy()
  if not np.all(weight_lengths == 1):
    raise ValueError(
        'Weight feature "{}" must have exactly one value in each example.'
        .format(weight_feature))
  weights = weights.flatten()
  # Before converting to numpy view, check the type (cannot convert string and
  # binary arrays to numpy view).
  weights_type = weights.type
  if pa.types.is_string(weights_type) or pa.types.is_binary(weights_type):
    raise ValueError(
        'Weight feature "{}" must be of numeric type. Found {}.'.format(
            weight_feature, weights_type))
  return weights.to_numpy()


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
      contain only one value.
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
        flat_weights = weights[
            array_util.GetFlattenedArrayParentIndices(array).to_numpy()]
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
    weights = _get_weight_feature(table, weight_column)
  for column in table.columns:
    column_name = column.name
    # use "yield from" after PY 3.3.
    for e in _recursion_helper(
        types.FeaturePath([column_name]), column.data.chunk(0), weights):
      yield e
