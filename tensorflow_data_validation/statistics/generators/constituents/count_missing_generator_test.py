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
"""Tests for tensorflow_data_validation.statistics.constituents.count_missing_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch
from tensorflow_data_validation.statistics.generators.constituents import count_missing_generator


class CountMissingGeneratorTest(absltest.TestCase):

  def test_count_missing_generator_key(self):
    path = types.FeaturePath(['feature'])
    generator = count_missing_generator.CountMissingGenerator(path)
    expected_key = ('CountMissingGenerator', path)
    # use assertDictEqual to make failures readable while checking hash value.
    self.assertDictEqual({expected_key: None}, {generator.get_key(): None})
    self.assertDictEqual(
        {expected_key: None},
        {count_missing_generator.CountMissingGenerator.key(path): None})

  def test_count_missing_generator_key_with_required(self):
    path = types.FeaturePath(['index'])
    required = types.FeaturePath(['value'])
    generator = count_missing_generator.CountMissingGenerator(
        path, [required])
    expected_key = ('CountMissingGenerator', path, required)
    self.assertDictEqual({expected_key: None}, {generator.get_key(): None})
    self.assertDictEqual({expected_key: None}, {
        count_missing_generator.CountMissingGenerator.key(path, [required]):
            None
    })

  def test_count_missing_generator_single_batch(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays([pa.array([[1], None, []])], ['feature']))
    path = types.FeaturePath(['feature'])
    generator = count_missing_generator.CountMissingGenerator(path)
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual(1, generator.extract_output(accumulator))

  def test_count_missing_generator_required_path(self):
    batch = input_batch.InputBatch(
        pa.RecordBatch.from_arrays(
            [pa.array([[1], None, []]),
             pa.array([[1], None, []])], ['index', 'value']))
    path = types.FeaturePath(['index'])
    required_path = types.FeaturePath(['value'])
    generator = count_missing_generator.CountMissingGenerator(
        path, [required_path])
    accumulator = generator.create_accumulator()
    accumulator = generator.add_input(accumulator, batch)
    self.assertEqual(0, generator.extract_output(accumulator))


if __name__ == '__main__':
  absltest.main()
