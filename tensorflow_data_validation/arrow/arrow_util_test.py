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
"""Tests for tensorflow_data_validation.arrow.arrow_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest
import numpy as np
import pyarrow as pa
from tensorflow_data_validation.arrow import arrow_util


class ArrowUtilTest(absltest.TestCase):

  def test_invalid_input_type(self):

    functions_expecting_list_array = [
        arrow_util.FlattenListArray,
        arrow_util.ListLengthsFromListArray,
        arrow_util.GetFlattenedArrayParentIndices,
    ]
    functions_expecting_array = [arrow_util.GetArrayNullBitmapAsByteArray]
    for f in itertools.chain(functions_expecting_list_array,
                             functions_expecting_array):
      with self.assertRaisesRegexp(RuntimeError, "Could not unwrap Array"):
        f(1)

    for f in functions_expecting_list_array:
      with self.assertRaisesRegexp(RuntimeError, "Expected ListArray but got"):
        f(pa.array([1, 2, 3]))

  def test_flatten_list_array(self):
    flattened = arrow_util.FlattenListArray(
        pa.array([], type=pa.list_(pa.int64())))
    self.assertTrue(flattened.equals(pa.array([], type=pa.int64())))

    flattened = arrow_util.FlattenListArray(
        pa.array([[1.], [2.], [], [3.]]))
    self.assertTrue(flattened.equals(pa.array([1., 2., 3.])))

  def test_list_lengths(self):
    list_lengths = arrow_util.ListLengthsFromListArray(
        pa.array([], type=pa.list_(pa.int64())))
    self.assertTrue(list_lengths.equals(pa.array([], type=pa.int32())))
    list_lengths = arrow_util.ListLengthsFromListArray(
        pa.array([[1., 2.], [], [3.]]))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int32())))
    list_lengths = arrow_util.ListLengthsFromListArray(
        pa.array([[1., 2.], None, [3.]]))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int32())))

  def test_get_array_null_bitmap_as_byte_array(self):
    array = pa.array([], type=pa.int32())
    null_masks = arrow_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([], type=pa.uint8())))

    array = pa.array([1, 2, None, 3, None], type=pa.int32())
    null_masks = arrow_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(
        null_masks.equals(pa.array([0, 0, 1, 0, 1], type=pa.uint8())))

    array = pa.array([1, 2, 3])
    null_masks = arrow_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([0, 0, 0], type=pa.uint8())))

    array = pa.array([None, None, None], type=pa.int32())
    null_masks = arrow_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([1, 1, 1], type=pa.uint8())))
    # Demonstrate that the returned array can be converted to a numpy boolean
    # array w/o copying
    np.testing.assert_equal(
        np.array([True, True, True]), null_masks.to_numpy().view(np.bool))

  def test_get_flattened_array_parent_indices(self):
    indices = arrow_util.GetFlattenedArrayParentIndices(
        pa.array([], type=pa.list_(pa.int32())))
    self.assertTrue(indices.equals(pa.array([], type=pa.int32())))

    indices = arrow_util.GetFlattenedArrayParentIndices(
        pa.array([[1.], [2.], [], [3.]]))
    self.assertTrue(indices.equals(pa.array([0, 1, 3], type=pa.int32())))


if __name__ == "__main__":
  absltest.main()
