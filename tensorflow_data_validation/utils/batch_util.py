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

from typing import Optional

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tfx_bsl.coders import batch_util


# TODO(b/221152546): Deprecate this.
@beam.ptransform_fn
def BatchExamplesToArrowRecordBatches(
    examples: beam.PCollection[types.LegacyExample],
    desired_batch_size: Optional[int] = constants
    .DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.PCollection[pa.RecordBatch]:
  """Batches example dicts into Arrow record batches.

  Args:
    examples: A PCollection of example dicts.
    desired_batch_size: Batch size. The output Arrow record batches will have as
      many rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow record batches.
  """
  return (
      examples
      | "BatchBeamExamples" >> beam.BatchElements(
          **batch_util.GetBatchElementsKwargs(desired_batch_size))
      | "DecodeExamplesToRecordBatch" >> beam.Map(
          # pylint: disable=unnecessary-lambda
          lambda x: decoded_examples_to_arrow.DecodedExamplesToRecordBatch(x)))
  # pylint: enable=unnecessary-lambda
