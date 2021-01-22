# Copyright 2020 Google LLC
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
"""Tests for schema utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl.testing import absltest
import tensorflow as tf
from tensorflow_data_validation.utils import vocab_util


class VocabUtilTest(absltest.TestCase):

  def test_text_file(self):
    with tempfile.NamedTemporaryFile() as f:
      f.write(b'Foo\nBar\n')
      f.flush()

      vocab, reverse_vocab = vocab_util.load_vocab(f.name)
      self.assertEqual(vocab, {'Foo': 0, 'Bar': 1})
      self.assertEqual(reverse_vocab, {0: 'Foo', 1: 'Bar'})

  def test_gz_recordio_file(self):
    with tempfile.NamedTemporaryFile(suffix='.tfrecord.gz') as f:
      writer = tf.io.TFRecordWriter(f.name, options='GZIP')
      for element in [b'Foo', b'Bar']:
        writer.write(element)
      writer.flush()
      f.flush()

      vocab, reverse_vocab = vocab_util.load_vocab(f.name)
      self.assertEqual(vocab, {'Foo': 0, 'Bar': 1})
      self.assertEqual(reverse_vocab, {0: 'Foo', 1: 'Bar'})

if __name__ == '__main__':
  absltest.main()
