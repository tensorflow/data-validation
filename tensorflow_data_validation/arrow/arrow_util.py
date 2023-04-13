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

from typing import Callable, Dict, Iterable, Optional, Text, Tuple

import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl.arrow import array_util


def get_weight_feature(input_record_batch: pa.RecordBatch,
                       weight_column: Text) -> np.ndarray:
  """Gets the weight column from the input record batch.

  Args:
    input_record_batch: Input record batch.
    weight_column: Name of the column containing the weight.

  Returns:
    A numpy array containing the weights of the examples in the input
    record_batch.

  Raises:
    ValueError: If the weight feature is not present in the input record_batch
    or is not a valid weight feature (must be of numeric type and have a
    single value for each example).
  """
  weights_field_index = input_record_batch.schema.get_field_index(weight_column)
  if weights_field_index < 0:
    raise ValueError('Weight column "{}" not present in the input '
                     'record batch.'.format(weight_column))
  weights = input_record_batch.column(weights_field_index)

  if pa.types.is_null(weights.type):
    raise ValueError('Weight column "{}" cannot be null.'.format(weight_column))
  # Before flattening, check that there is a single value for each example.
  weight_lengths = array_util.ListLengthsFromListArray(weights).to_numpy()
  if not np.all(weight_lengths == 1):
    raise ValueError(
        'Weight column "{}" must have exactly one value in each example.'
        .format(weight_column))
  flat_weights = weights.flatten()
  # Before converting to numpy view, check the type (cannot convert string and
  # binary arrays to numpy view).
  flat_weights_type = flat_weights.type
  if (not pa.types.is_floating(flat_weights_type) and
      not pa.types.is_integer(flat_weights_type)):
    raise ValueError(
        'Weight column "{}" must be of numeric type. Found {}.'.format(
            weight_column, flat_weights_type))
  return np.asarray(flat_weights)


def is_binary_like(data_type: pa.DataType) -> bool:
  """Returns true if an Arrow type is binary-like.

  Qualified types are {Large,}BinaryArray, {Large,}StringArray.

  Args:
    data_type: a pa.Array.

  Returns:
    bool.
  """
  return (pa.types.is_binary(data_type) or
          pa.types.is_large_binary(data_type) or
          pa.types.is_unicode(data_type) or
          pa.types.is_large_unicode(data_type))


def enumerate_arrays(
    record_batch: pa.RecordBatch,
    example_weight_map: Optional[ExampleWeightMap],
    enumerate_leaves_only: bool,
    wrap_flat_struct_in_list: bool = True,
    column_select_fn: Optional[Callable[[types.FeatureName], bool]] = None
) -> Iterable[Tuple[types.FeaturePath, pa.Array, Optional[np.ndarray]]]:
  """Enumerates arrays in a RecordBatch.

  Define:
    primitive: primitive arrow arrays (e.g. Int64Array).
    nested_list := list<nested_list> | list<primitive> | null
    # note: a null array can be seen as a list<primitive>, which contains only
    #   nulls and the type of the primitive is unknown.
    # example:
    #   null,
    #   list<null>,  # like list<list<unknown_type>> with only null values.
    #   list<list<int64>>,
    struct := struct<{field: nested_list | struct}> | list<struct>
    # example:
    #   struct<{"foo": list<int64>},
    #   list<struct<{"foo": list<int64>}>>,
    #   struct<{"foo": struct<{"bar": list<list<int64>>}>}>

  This function assumes `record_batch` contains only nested_list and struct
  columns. It enumerates each column in `record_batch`, and if that column is
  a struct, it flattens the outer lists wrapping it (if any), and recursively
  enumerates the array of each field in the struct (also see
  `enumerate_leaves_only`).

  The weights get "aligned" automatically in this process, therefore weights,
  the third term in the returned tuple always has enumerated_array[i]'s weight
  being weights[i].

  A FeaturePath is included in the result to address the enumerated array.
  Note that the FeaturePath merely addresses in the `record_batch` and struct
  arrays. It does not indicate whether / how a struct array is nested.

  Args:
    record_batch: The RecordBatch whose arrays to be visited.
    example_weight_map: an ExampleWeightMap that maps a FeaturePath to its
      corresponding weight column.
    enumerate_leaves_only: If True, only enumerate leaf arrays. A leaf array
      is an array whose type does not have any struct nested in.
      Otherwise, also enumerate the struct arrays where the leaf arrays are
      contained.
    wrap_flat_struct_in_list: if True, and if a struct<[Ts]> array is
      encountered, it will be wrapped in a list array, so it becomes a
      list<struct<[Ts]>>, in which each sub-list contains one element.
      A caller can make use of this option to assume all the arrays enumerated
      here are list<inner_type>.
    column_select_fn: If provided, only enumerates leaf arrays of columns with
      names for which this function evaluates to True.
  Yields:
    A tuple. The first term is the path of the feature; the second term is
    the feature array and the third term is the weight array for the feature
    array (i.e. weights[i] is the weight for array[i]).

  Raises:
    ValueError: When the weight column is not a list array whose elements are
      1-element lists.
  """

  def _recursion_helper(
      feature_path: types.FeaturePath, array: pa.Array,
      all_weights: Dict[types.FeatureName, np.ndarray],
  ) -> Iterable[Tuple[types.FeaturePath, pa.Array, Optional[np.ndarray]]]:
    """Recursion helper."""
    array_type = array.type
    innermost_nested_type = array_util.get_innermost_nested_type(array_type)
    if pa.types.is_struct(innermost_nested_type):
      if not enumerate_leaves_only:
        weights = all_weights.get(example_weight_map.get(feature_path))
        # special handing for a flat struct array -- wrap it in a ListArray
        # whose elements are singleton lists. This way downstream can keep
        # assuming the enumerated arrays are list<*>.
        to_yield = array
        if pa.types.is_struct(array_type) and wrap_flat_struct_in_list:
          to_yield = array_util.ToSingletonListArray(array)
        yield (feature_path, to_yield, weights)
      flat_struct_array, parent_indices = array_util.flatten_nested(
          array, bool(all_weights))
      # Potential optimization:
      # Only flatten weights that we know will be used in the recursion.
      flat_all_weights = {
          weight_feature_name: w[parent_indices]
          for weight_feature_name, w in all_weights.items()
      }
      for field in flat_struct_array.type:
        field_name = field.name
        yield from _recursion_helper(
            feature_path.child(field_name),
            array_util.get_field(flat_struct_array, field_name),
            flat_all_weights,
        )
    else:
      weights = all_weights.get(example_weight_map.get(feature_path))
      yield (feature_path, array, weights)

  if example_weight_map is None:
    example_weight_map = ExampleWeightMap(
        weight_feature=None, per_feature_override=None)
  all_weights = {
      weight_column: get_weight_feature(record_batch, weight_column)
      for weight_column in example_weight_map.all_weight_features()
  }

  for column_name, column in zip(record_batch.schema.names,
                                 record_batch.columns):
    if column_select_fn and not column_select_fn(column_name):
      continue
    yield from _recursion_helper(
        types.FeaturePath([column_name]), column, all_weights)


def get_nest_level(array_type: pa.DataType) -> int:
  """Returns the nest level of an array type.

  The nest level of primitive types is 0.
  The nest level of null is 1, because an null array is to represent
    list<unknown_type>.
  The nest level of list<inner_type> is get_nest_level(inner_type) + 1

  Args:
    array_type: pa.DataType

  Returns:
    the nest level.
  """
  result = 0
  while array_util.is_list_like(array_type):
    result += 1
    array_type = array_type.value_type

  # null is like list<unkown_primitive>
  if pa.types.is_null(array_type):
    result += 1
  return result


def get_column(record_batch: pa.RecordBatch,
               feature_name: types.FeatureName,
               missing_ok: bool = False) -> Optional[pa.Array]:
  """Get a column by feature name.

  Args:
    record_batch: A pa.RecordBatch.
    feature_name: The name of a feature (column) within record_batch.
    missing_ok: If True, returns None for missing feature names.

  Returns:
    The column with the specified name, or None if missing_ok is true and
    a column with the specified name is missing, or more than one exist.

  Raises:
    KeyError: If a column with the specified name is missing, or more than
    one exist, and missing_ok is False.
  """
  idx = record_batch.schema.get_field_index(feature_name)
  if idx < 0:
    if missing_ok:
      return None
    raise KeyError('missing column %s' % feature_name)
  return record_batch.column(idx)
