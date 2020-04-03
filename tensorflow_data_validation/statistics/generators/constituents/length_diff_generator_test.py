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
"""Tests for tensorflow_data_validation.statistics.constituents.length_diff_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch
from tensorflow_data_validation.statistics.generators.constituents import length_diff_generator


class LengthDiffGeneratorTest(absltest.TestCase):

  def test_length_diff_generator_key(self):
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    generator = length_diff_generator.LengthDiffGenerator(path1, path2)
    expected_key = ('LengthDiffGenerator', path1, path2)
    self.assertDictEqual({expected_key: None}, {generator.get_key(): None})
    self.assertDictEqual(
        {expected_key: None},
        {length_diff_generator.LengthDiffGenerator.key(path1, path2): None})

  def test_length_diff_generator_key_with_required(self):
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath(['required'])
    required_paths = [path1, path2, required_path]
    generator = length_diff_generator.LengthDiffGenerator(
        path1, path2, required_paths)
    expected_key = ('LengthDiffGenerator', path1, path2, path1, path2,
                    required_path)
    self.assertDictEqual({expected_key: None}, {generator.get_key(): None})
    self.assertDictEqual({expected_key: None}, {
        length_diff_generator.LengthDiffGenerator.key(path1, path2,
                                                      required_paths):
            None
    })

  def test_length_diff_generator_positive_min_max(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1, 2, 3], None, [1]]),
            pa.array([[1], None, []]),
            pa.array([[1], None, [1]])
        ], ['f1', 'f2', 'required']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath('required')
    required_paths = [path1, path2, required_path]
    generator = length_diff_generator.LengthDiffGenerator(
        path1, path2, required_paths)
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual((1, 2), generator.extract_output(accumulator))

  def test_length_diff_generator_negative_min_max(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1, 2, 3], None, [1]]),
            pa.array([[1], None, []]),
            pa.array([[1], None, [1]])
        ], ['f1', 'f2', 'required']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath('required')
    generator = length_diff_generator.LengthDiffGenerator(
        path2, path1, required_paths=[path1, path2, required_path])
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual((-2, -1), generator.extract_output(accumulator))

  def test_length_diff_generator_both_null(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([None, None, None]),
            pa.array([None, None, None]),
            pa.array([[1], [1], [1]])
        ], ['f1', 'f2', 'required']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath('required')
    generator = length_diff_generator.LengthDiffGenerator(
        path1, path2, required_paths=[required_path])
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual((0, 0), generator.extract_output(accumulator))

  def test_length_diff_generator_both_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([[1], [1], [1]])], ['required']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath('required')
    generator = length_diff_generator.LengthDiffGenerator(
        path1, path2, required_paths=[required_path])
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual((0, 0), generator.extract_output(accumulator))

  def test_length_diff_generator_required_missing(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([
            pa.array([[1, 2, 3], None, [1]]),
            pa.array([[1], None, []]),
            pa.array([None, None, None])
        ], ['f1', 'f2', 'required']))
    path1 = types.FeaturePath(['f1'])
    path2 = types.FeaturePath(['f2'])
    required_path = types.FeaturePath('required')
    generator = length_diff_generator.LengthDiffGenerator(
        path1, path2, required_paths=[required_path])
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual((0, 0), generator.extract_output(accumulator))


if __name__ == '__main__':
  absltest.main()
