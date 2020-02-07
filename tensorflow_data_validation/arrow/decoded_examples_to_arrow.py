# Copyright 2018 Google LLC
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

"""Util to convert a list of decoded examples to an Arrow Table."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from typing import List


def _GetEmptyTable(num_rows: int) -> pa.Table:
  # pyarrow doesn't provide an API to create a table with zero column but non
  # zero rows. We work around it by adding a dummy column first and then
  # removing it.
  t = pa.Table.from_arrays(
      [pa.array([None] * num_rows, type=pa.null())], ["dummy"])
  return t.remove_column(0)


def DecodedExamplesToTable(decoded_examples: List[types.Example]) -> pa.Table:
  """Converts a list of types.Example to an Arrow Table.

  where types.Example is Dict[Union[bytes, unicode], Union[None, np.ndarray]]
  The result table has M rows and N columns where M is the number of examples
  in the list and N is the number of unique features in the examples.
  Each column is either a ListArray<primitive|string|binary> or a NullArray.
  None and missing feature handling:
    - if a feature's value is None in an example, then its corresponding column
      in the result table will have a null at the corresponding position.
    - if a feature's value is always None across all the examples in the input
      list, then its corresponding column in the result table will be a
      NullArray.
    - if an example does not contain a feature (in the universe of features),
      then the column of that feature will have a null at the corresponding
      position.

  Args:
    decoded_examples: a Dict[Union[bytes, unicode], Union[None, np.ndarray]]

  Returns:
    a pa.Table.

  Raises:
    ValueError: when the conversion fails.
    TypeError: when some of the output columns are not of supported types.
  """
  if not decoded_examples:
    return _GetEmptyTable(0)

  struct_array = pa.array(decoded_examples)
  if not pa.types.is_struct(struct_array.type):
    raise ValueError("Unexpected Arrow type created from input")
  field_names = [f.name for f in list(struct_array.type)]
  if not field_names:
    return _GetEmptyTable(len(decoded_examples))
  value_arrays = struct_array.flatten()
  for name, array in six.moves.zip(field_names, value_arrays):
    if pa.types.is_null(array.type):
      continue
    if not arrow_util.is_list_like(array.type):
      raise TypeError("Expected list arrays for field {} but got {}".format(
          name, array.type))
    value_type = array.type.value_type
    if (not pa.types.is_integer(value_type) and
        not pa.types.is_floating(value_type) and
        not arrow_util.is_binary_like(value_type) and
        not pa.types.is_null(value_type)):
      raise TypeError("Type not supported: {} {}".format(name, array.type))

  return pa.Table.from_arrays(value_arrays, field_names)
