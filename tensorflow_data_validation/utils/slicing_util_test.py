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
"""Tests for the slicing utilities."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation.utils import slicing_util


class SlicingUtilTest(absltest.TestCase):

  # This should be simply self.assertCountEqual(), but
  # RecordBatch.__eq__ is not implemented.
  # TODO(zhuo): clean-up after ARROW-8277 is available.
  def _check_results(self, got, expected):
    got_dict = {g[0]: g[1] for g in got}
    expected_dict = {e[0]: e[1] for e in expected}
    self.assertCountEqual(got_dict.keys(), expected_dict.keys())
    for k, got_record_batch in got_dict.items():
      expected_record_batch = expected_dict[k]
      self.assertTrue(got_record_batch.equals(expected_record_batch))

  def test_get_feature_value_slicer(self):
    features = {'a': None, 'b': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
        pa.array([['dog'], ['cat'], ['wolf'], ['dog', 'wolf'], ['wolf']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_1_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[1], [2, 1, 1]]), pa.array([['dog'], ['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_1_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])], ['a', 'b'])
        ),
        (u'a_1_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_2_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_3_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[3], [3]]), pa.array([['wolf'], ['wolf']])],
             ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_single_feature(self):
    features = {'a': [2]}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_no_slice(self):
    features = {'a': [3]}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = []
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_bytes_feature_valid_utf8(self):
    features = {'b': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([[b'dog'], [b'cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[1]]), pa.array([[b'dog']])], ['a', 'b'])
        ),
        (u'b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([[b'cat']])], ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_non_utf8_slice_key(self):
    features = {'a': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[b'\xF0'], ['cat']]),
    ], ['a'])
    with self.assertRaisesRegexp(ValueError, 'must be valid UTF-8'):
      _ = list(
          slicing_util.get_feature_value_slicer(features)(input_record_batch))


if __name__ == '__main__':
  absltest.main()
