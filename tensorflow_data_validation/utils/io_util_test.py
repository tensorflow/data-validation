# Copyright 2021 Google LLC
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
"""Tests for io_util."""

import tempfile

from absl.testing import absltest
import apache_beam as beam
from tensorflow_data_validation.utils import io_util


class MaterializerTest(absltest.TestCase):

  def test_write_then_read(self):
    values = ['abcd', 91, {'x': 'y'}]
    temp_dir = tempfile.mkdtemp()
    materializer = io_util.Materializer(temp_dir)
    with beam.Pipeline() as p:
      _ = p | beam.Create(values) | materializer.writer()
    got_values = []
    for val in materializer.reader():
      got_values.append(val)
    self.assertCountEqual(values, got_values)

  def test_cleanup(self):
    values = ['abcd', 91, {'x': 'y'}]
    temp_dir = tempfile.mkdtemp()
    materializer = io_util.Materializer(temp_dir)
    with beam.Pipeline() as p:
      _ = p | beam.Create(values) | materializer.writer()
    self.assertNotEmpty(materializer._output_files())
    materializer.cleanup()
    self.assertEmpty(materializer._output_files())
    with self.assertRaisesRegex(ValueError,
                                'Materializer must not be used after cleanup.'):
      materializer.reader()

  def test_context_manager(self):
    with io_util.Materializer(tempfile.mkdtemp()) as materializer:
      values = ['abcd', 91, {'x': 'y'}]
      with beam.Pipeline() as p:
        _ = p | beam.Create(values) | materializer.writer()
      got_values = []
      for val in materializer.reader():
        got_values.append(val)
    self.assertCountEqual(values, got_values)
    self.assertEmpty(materializer._output_files())


if __name__ == '__main__':
  absltest.main()
