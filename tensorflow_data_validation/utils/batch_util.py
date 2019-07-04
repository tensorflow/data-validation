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
# limitations under the License.

"""Utilities for batching input examples."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from apache_beam.transforms import window
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tensorflow_data_validation.types_compat import List


def merge_single_batch(batch):
  """Merges batched input examples to proper batch format."""
  batch_size = len(batch)
  result = {}
  for idx, example in enumerate(batch):
    for feature in example:
      if feature not in result:
        # New feature. Initialize the list with None.
        result[feature] = [None] * batch_size
      result[feature][idx] = example[feature]
  return result


@beam.typehints.with_input_types(types.Example)
@beam.typehints.with_output_types(pa.Table)
class BatchExamplesDoFn(beam.DoFn):
  """A DoFn which batches input example dicts into an arrow table."""

  def __init__(self, desired_batch_size):
    self._desired_batch_size = desired_batch_size
    self._buffer = []

  def _flush_buffer(self):
    arrow_table = decoded_examples_to_arrow.DecodedExamplesToTable(
        self._buffer)
    del self._buffer[:]
    return arrow_table

  def finish_bundle(self):
    if self._buffer:
      yield window.GlobalWindows.windowed_value(self._flush_buffer())

  def process(self, example):
    self._buffer.append(example)
    if len(self._buffer) >= self._desired_batch_size:
      yield self._flush_buffer()
