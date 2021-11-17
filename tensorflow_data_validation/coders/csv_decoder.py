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
"""Decode CSV records into in-memory representation for tf data validation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Text, Union

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tfx_bsl.coders import csv_decoder

from tensorflow_metadata.proto.v0 import schema_pb2


@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(pa.RecordBatch)
class DecodeCSV(beam.PTransform):
  """Decodes CSV records into Arrow RecordBatches.

  DEPRECATED: please use tfx_bsl.public.CsvTFXIO instead.
  """

  def __init__(self,
               column_names: List[types.FeatureName],
               delimiter: Text = ',',
               skip_blank_lines: bool = True,
               schema: Optional[schema_pb2.Schema] = None,
               desired_batch_size: Optional[int] = constants
               .DEFAULT_DESIRED_INPUT_BATCH_SIZE,
               multivalent_columns: Optional[List[types.FeatureName]] = None,
               secondary_delimiter: Optional[Union[Text, bytes]] = None):
    """Initializes the CSV decoder.

    Args:
      column_names: List of feature names. Order must match the order in the CSV
        file.
      delimiter: A one-character string used to separate fields.
      skip_blank_lines: A boolean to indicate whether to skip over blank lines
        rather than interpreting them as missing values.
      schema: An optional schema of the input data. If provided, types
        will be inferred from the schema. If this is provided, the feature names
        must equal column_names.
      desired_batch_size: Batch size. The output Arrow RecordBatches will have
        as many rows as the `desired_batch_size`.
      multivalent_columns: Name of column that can contain multiple
        values.
      secondary_delimiter: Delimiter used for parsing multivalent columns.
    """
    if not isinstance(column_names, list):
      raise TypeError('column_names is of type %s, should be a list' %
                      type(column_names).__name__)

    self._column_names = column_names
    self._delimiter = delimiter
    self._skip_blank_lines = skip_blank_lines
    self._schema = schema
    self._desired_batch_size = desired_batch_size
    self._multivalent_columns = multivalent_columns
    self._secondary_delimiter = secondary_delimiter

  def expand(self, lines: beam.pvalue.PCollection):
    """Decodes the input CSV records into RecordBatches.

    Args:
      lines: A PCollection of strings representing the lines in the CSV file.

    Returns:
      A PCollection of RecordBatches representing the CSV records.
    """
    return (lines | 'CSVToRecordBatch' >> csv_decoder.CSVToRecordBatch(
        column_names=self._column_names,
        delimiter=self._delimiter,
        skip_blank_lines=self._skip_blank_lines,
        schema=self._schema,
        desired_batch_size=self._desired_batch_size,
        multivalent_columns=self._multivalent_columns,
        secondary_delimiter=self._secondary_delimiter))
