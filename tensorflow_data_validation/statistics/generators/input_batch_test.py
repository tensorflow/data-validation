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
"""Tests for tensorflow_data_validation.statistics.input_batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np
import pyarrow as pa

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch


class InputBatchTest(absltest.TestCase):

  def test_null_mask(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([[1], None, []])], ['feature']))
    path = types.FeaturePath(['feature'])
    expected_mask = np.array([False, True, False])
    np.testing.assert_array_equal(batch.null_mask(path), expected_mask)

  def test_null_mask_path_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([[1], None, []])], ['feature']))
    path = types.FeaturePath(['feature2'])
    expected_mask = np.array([True, True, True])
    np.testing.assert_array_equal(batch.null_mask(path), expected_mask)

  def test_null_mask_empty_array(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([])], ['feature']))
    path = types.FeaturePath(['feature'])
    expected_mask = np.array([], dtype=bool)
    np.testing.assert_array_equal(batch.null_mask(path), expected_mask)

  def test_null_mask_null_array(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([None], type=pa.null())],
                                   ['feature']))
    path = types.FeaturePath(['feature'])
    expected_mask = np.array([True])
    np.testing.assert_array_equal(batch.null_mask(path), expected_mask)

  def test_all_null_mask(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1], None, []]),
            pa.array([[1], None, None]),
            pa.array([[1], None, None])
        ], ['f1', 'f2', 'f3']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    path3 = types.FeaturePath(['f3'])
    expected_mask = np.array([False, True, False])
    np.testing.assert_array_equal(
        batch.all_null_mask(path1, path2, path3), expected_mask)

  def test_all_null_mask_all_null(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([None, None], type=pa.null()),
            pa.array([None, None], type=pa.null())
        ], ['f1', 'f2']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    expected_mask = np.array([True, True])
    np.testing.assert_array_equal(
        batch.all_null_mask(path1, path2), expected_mask)

  def test_all_null_mask_one_null(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays(
            [pa.array([[1], [1]]),
             pa.array([None, None], type=pa.null())], ['f1', 'f2']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    expected_mask = np.array([False, False])
    np.testing.assert_array_equal(
        batch.all_null_mask(path1, path2), expected_mask)

  def test_all_null_mask_one_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([None, [1]])], ['f2']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    expected_mask = np.array([True, False])
    np.testing.assert_array_equal(
        batch.all_null_mask(path1, path2), expected_mask)

  def test_all_null_mask_all_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([None, None], type=pa.null())],
                                   ['f3']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    expected_mask = np.array([True, True])
    np.testing.assert_array_equal(
        batch.all_null_mask(path1, path2), expected_mask)

  def test_all_null_mask_no_paths(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([None, None], type=pa.null())],
                                   ['f3']))
    with self.assertRaisesRegex(ValueError, r'Paths cannot be empty.*'):
      batch.all_null_mask()

  def test_list_lengths(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1], None, [1, 2]]),
        ], ['f1']))
    np.testing.assert_array_equal(
        batch.list_lengths(types.FeaturePath(['f1'])), [1, 0, 2])

  def test_list_lengths_empty_array(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([])], ['f1']))
    np.testing.assert_array_equal(
        batch.list_lengths(types.FeaturePath(['f1'])), [])

  def test_list_lengths_path_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([1, None, 1]),
        ], ['f1']))
    np.testing.assert_array_equal(
        batch.list_lengths(types.FeaturePath(['f2'])), [0, 0, 0])

  def test_list_lengths_null_array(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([None, None, None], type=pa.null()),
        ], ['f1']))
    np.testing.assert_array_equal(
        batch.list_lengths(types.FeaturePath(['f1'])), [0, 0, 0])

  def test_all_null_mask_unequal_lengths(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1]]),
            pa.array([[{
                'sf1': [[1]]
            }, {
                'sf1': [[1]]
            }]]),
        ], ['f1', 'f2']))
    with self.assertRaisesRegex(ValueError,
                                r'.*null_mask\(f2.sf1\).size.*\(1 != 2\).*'):
      batch.all_null_mask(
          types.FeaturePath(['f1']), types.FeaturePath(['f2', 'sf1']))

  def test_list_lengths_non_list(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([1, None, 1]),
        ], ['f1']))
    with self.assertRaisesRegex(
        ValueError, r'Can only compute list lengths on list arrays, found.*'):
      batch.list_lengths(types.FeaturePath(['f1']))


if __name__ == '__main__':
  absltest.main()
