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

import abc
import apache_beam as beam
from apache_beam.transforms import window
import six
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from tfx_bsl.coders import example_coder
from typing import Any, List, Iterable


# TODO(pachristopher): Debug why beam.BatchElements is expensive than
# the custom DoFn and consider using it instead.
@six.add_metaclass(abc.ABCMeta)
class BatchDoFn(beam.DoFn):
  """Base class for batched DoFns."""

  def __init__(self, desired_batch_size: int):
    self._desired_batch_size = desired_batch_size
    self._buffer = []

  def _flush_buffer(self) -> List[Any]:
    result = self.process_batch(self._buffer)
    del self._buffer[:]
    return result

  @abc.abstractmethod
  def process_batch(self, batch: List[Any]) -> Any:
    """Sub-classes should implement this method."""

  def finish_bundle(self):
    if self._buffer:
      yield window.GlobalWindows.windowed_value(self._flush_buffer())

  def process(self, element: Any) -> Iterable[Any]:
    self._buffer.append(element)
    if len(self._buffer) >= self._desired_batch_size:
      yield self._flush_buffer()


@beam.typehints.with_input_types(types.BeamExample)
@beam.typehints.with_output_types(pa.Table)
class _BatchExamplesDoFn(BatchDoFn):
  """A DoFn which batches input example dicts into an arrow table."""

  def process_batch(self, batch: List[Any]) -> pa.Table:
    return decoded_examples_to_arrow.DecodedExamplesToTable(batch)


# TODO(pachristopher): Deprecate this.
@beam.ptransform_fn
@beam.typehints.with_input_types(types.BeamExample)
@beam.typehints.with_output_types(pa.Table)
def BatchExamplesToArrowTables(
    examples: beam.pvalue.PCollection,
    desired_batch_size: int = constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.pvalue.PCollection:  # pylint: disable=invalid-name
  """Batches example dicts into Arrow tables.

  Args:
    examples: A PCollection of example dicts.
    desired_batch_size: Batch size. The output Arrow tables will have as many
      rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow tables.
  """
  # Check if we have the default windowing behavior. The _BatchExamplesDoFn
  # is expected to be called under a Global window.
  assert examples.windowing.is_default()

  return (examples
          | 'BatchExamplesToArrowTables' >> beam.ParDo(
              _BatchExamplesDoFn(desired_batch_size=desired_batch_size)))


@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(pa.Table)
class _BatchSerializedExamplesDoFn(BatchDoFn):
  """A DoFn which batches input serialized examples into an arrow table."""

  def __init__(self, desired_batch_size: int):
    super(_BatchSerializedExamplesDoFn, self).__init__(desired_batch_size)
    self._decoder = None

  def setup(self):
    self._decoder = example_coder.ExamplesToRecordBatchDecoder()

  def process_batch(self, batch: List[bytes]) -> pa.Table:
    return pa.Table.from_batches([self._decoder.DecodeBatch(self._buffer)])


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(pa.Table)
def BatchSerializedExamplesToArrowTables(
    examples: beam.pvalue.PCollection,
    desired_batch_size: int = constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.pvalue.PCollection:  # pylint: disable=invalid-name
  """Batches serialized examples into Arrow tables.

  Args:
    examples: A PCollection of serialized tf.Examples.
    desired_batch_size: Batch size. The output Arrow tables will have as many
      rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow tables.
  """
  # Check if we have the default windowing behavior. The _BatchExamplesDoFn
  # is expected to be called under a Global window.
  assert examples.windowing.is_default()

  return (examples
          | 'BatchSerializedExamplesToArrowTables' >> beam.ParDo(
              _BatchSerializedExamplesDoFn(
                  desired_batch_size=desired_batch_size)))
