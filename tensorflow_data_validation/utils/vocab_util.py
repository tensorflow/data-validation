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
"""Utilities for retrieving the vocabulary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Text, Tuple
import six
import tensorflow as tf


def load_vocab(path: Text) -> Tuple[Dict[Text, int], Dict[int, Text]]:
  """Loads the vocabulary from the specified path.

  Args:
    path: The path to the vocabulary file. If the file has a tfrecord.gz suffix,
      we assume it is a GZIP-compressed TFRecord file. Otherwise, we assume it
      is a text file.

  Returns:
    A tuple where the first element is a dictionary specifying the string token
    to integer mapping and the second element represents the reverse lookup
    (i.e. integer token to string mapping).

  Raises:
    ValueError: Vocabulary path does not exist.
  """
  vocab = {}
  reverse_vocab = {}

  if not tf.io.gfile.exists(path):
    raise ValueError('Vocabulary path: %s does not exist' % path)

  def populate_entry(index, entry):
    entry = six.ensure_text(entry).strip()
    vocab[entry] = index
    reverse_vocab[index] = entry

  if path.endswith('tfrecord.gz'):
    data_iter = tf.compat.v1.io.tf_record_iterator(
                path,
                tf.io.TFRecordOptions(compression_type='GZIP'))
    for index, entry in enumerate(data_iter):
      populate_entry(index, entry)
  else:
    with tf.io.gfile.GFile(path) as f:
      for index, entry in enumerate(f):
        populate_entry(index, entry)
  return vocab, reverse_vocab

