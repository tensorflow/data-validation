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
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tfx_bsl.coders import example_coder
from typing import Dict, List, Iterable, Optional, Text


def GetBeamBatchKwargs(desired_batch_size: Optional[int]) -> Dict[Text, int]:
  """Returns the kwargs to be passed to beam.BatchElements."""
  if desired_batch_size is None:
    return {}
  return {
      "min_batch_size": desired_batch_size,
      "max_batch_size": desired_batch_size,
  }


# TODO(pachristopher): Deprecate this.
@beam.ptransform_fn
@beam.typehints.with_input_types(types.BeamExample)
@beam.typehints.with_output_types(pa.Table)
def BatchExamplesToArrowTables(
    examples: beam.pvalue.PCollection,
    desired_batch_size: Optional[int] = constants
    .DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.pvalue.PCollection:
  """Batches example dicts into Arrow tables.

  Args:
    examples: A PCollection of example dicts.
    desired_batch_size: Batch size. The output Arrow tables will have as many
      rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow tables.
  """
  # DecodedExamplesToTable should be called within a lambda function instead of
  # specifying the function name in beam.Map for the reasons discussed in
  # b/143648957.
  # TODO(b/131315065): Remove the comment above when the CSV decoder no longer
  # uses BatchExamplesToArrowTables.
  return (
      examples
      | "BatchBeamExamples" >>
      beam.BatchElements(**GetBeamBatchKwargs(desired_batch_size))
      | "DecodeExamplesToTable" >>
      # pylint: disable=unnecessary-lambda
      beam.Map(lambda x: decoded_examples_to_arrow.DecodedExamplesToTable(x)))
      # pylint: enable=unnecessary-lambda


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(pa.Table)
def BatchSerializedExamplesToArrowTables(
    examples: beam.pvalue.PCollection,
    desired_batch_size: Optional[int] = constants
    .DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.pvalue.PCollection:
  """Batches serialized examples into Arrow tables.

  Args:
    examples: A PCollection of serialized tf.Examples.
    desired_batch_size: Batch size. The output Arrow tables will have as many
      rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow tables.
  """
  return (examples
          | "BatchSerializedExamples" >>
          beam.BatchElements(**GetBeamBatchKwargs(desired_batch_size))
          | "BatchDecodeExamples" >> beam.ParDo(_BatchDecodeExamplesDoFn()))


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.Table)
class _BatchDecodeExamplesDoFn(beam.DoFn):
  """A DoFn which batches input serialized examples into an arrow table."""

  def __init__(self):
    self._decoder = None
    self._example_size = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, "example_size")

  def setup(self):
    self._decoder = example_coder.ExamplesToRecordBatchDecoder()

  def process(self, batch: List[bytes]) -> Iterable[pa.Table]:
    if batch:
      self._example_size.inc(sum(map(len, batch)))
    yield pa.Table.from_batches([self._decoder.DecodeBatch(batch)])
