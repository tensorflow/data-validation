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

"""Decode TF Examples into in-memory representation for tf data validation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import batch_util
from tfx_bsl.coders import example_coder


# TODO(pachristopher): Deprecate this in 0.16.
class TFExampleDecoder(object):
  """A decoder for decoding TF examples into tf data validation datasets.
  """

  def decode(self, serialized_example_proto: bytes) -> types.Example:
    """Decodes serialized tf.Example to tf data validation input dict."""
    return example_coder.ExampleToNumpyDict(serialized_example_proto)


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(pa.Table)
def DecodeTFExample(
    examples: beam.pvalue.PCollection,
    desired_batch_size: int = constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE
) -> beam.pvalue.PCollection:  # pylint: disable=invalid-name
  """Decodes serialized TF examples into Arrow tables.

  Args:
    examples: A PCollection of strings representing serialized TF examples.
    desired_batch_size: Batch size. The output Arrow tables will have as many
      rows as the `desired_batch_size`.

  Returns:
    A PCollection of Arrow tables.
  """
  return (examples
          | 'BatchSerializedExamplesToArrowTables' >>
          batch_util.BatchSerializedExamplesToArrowTables(
              desired_batch_size=desired_batch_size))
