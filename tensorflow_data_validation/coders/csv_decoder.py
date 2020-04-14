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

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tfx_bsl.coders import csv_decoder as csv_decoder
from tfx_bsl.tfxio import record_based_tfxio
from typing import List, Optional, Text, Union

from tensorflow_metadata.proto.v0 import schema_pb2


@beam.typehints.with_input_types(types.BeamCSVRecord)
@beam.typehints.with_output_types(pa.RecordBatch)
class DecodeCSV(beam.PTransform):
  """Decodes CSV records into Arrow RecordBatches."""

  def __init__(
      self,
      column_names: List[types.FeatureName],
      delimiter: Text = ',',
      skip_blank_lines: bool = True,
      schema: Optional[schema_pb2.Schema] = None,
      infer_type_from_schema: bool = False,
      desired_batch_size: Optional[int] = constants
      .DEFAULT_DESIRED_INPUT_BATCH_SIZE,
      multivalent_columns_names: Optional[List[types.FeatureName]] = None,
      secondary_delimiter: Optional[Union[Text, bytes]] = None):
    """Initializes the CSV decoder.

    Args:
      column_names: List of feature names. Order must match the order in the CSV
        file.
      delimiter: A one-character string used to separate fields.
      skip_blank_lines: A boolean to indicate whether to skip over blank lines
        rather than interpreting them as missing values.
      schema: An optional schema of the input data.
      infer_type_from_schema: A boolean to indicate whether the feature types
        should be inferred from the schema. If set to True, an input schema must
        be provided.
      desired_batch_size: Batch size. The output Arrow RecordBatches will have
        as many rows as the `desired_batch_size`.
      multivalent_columns_names: Name of column that can contain multiple
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
    self._infer_type_from_schema = infer_type_from_schema
    self._desired_batch_size = desired_batch_size
    self._multivalent_columns_names = multivalent_columns_names
    self._secondary_delimiter = secondary_delimiter

  def expand(self, lines: beam.pvalue.PCollection):
    """Decodes the input CSV records into an in-memory dict representation.

    Args:
      lines: A PCollection of strings representing the lines in the CSV file.

    Returns:
      A PCollection of dicts representing the CSV records.
    """
    csv_lines = (
        lines | 'ParseCSVLines' >> beam.ParDo(csv_decoder.ParseCSVLine(
            self._delimiter)))

    if self._infer_type_from_schema:
      column_infos = _get_feature_types_from_schema(self._schema,
                                                    self._column_names)
    else:
      # TODO(b/72746442): Consider using a DeepCopy optimization similar to TFT.
      # Do first pass to infer the feature types.
      column_infos = beam.pvalue.AsSingleton(
          csv_lines | 'InferColumnTypes' >> beam.CombineGlobally(
              csv_decoder.ColumnTypeInferrer(
                  column_names=self._column_names,
                  skip_blank_lines=self._skip_blank_lines,
                  multivalent_columns=self._multivalent_columns_names,
                  secondary_delimiter=self._secondary_delimiter)))

    # Do second pass to generate the in-memory dict representation.
    return (
        csv_lines
        | 'BatchCSVLines' >>
        beam.BatchElements(**record_based_tfxio.GetBatchElementsKwargs(
            self._desired_batch_size))
        | 'BatchedCSVRowsToArrow' >> beam.ParDo(
            csv_decoder.BatchedCSVRowsToRecordBatch(
                skip_blank_lines=self._skip_blank_lines,
                multivalent_columns=self._multivalent_columns_names,
                secondary_delimiter=self._secondary_delimiter), column_infos))


def _get_feature_types_from_schema(
    schema: schema_pb2.Schema,
    column_names: List[types.FeatureName]) -> List[csv_decoder.ColumnInfo]:
  """Get statistics feature types from the input schema."""
  schema_type_to_stats_type = {
      schema_pb2.INT: csv_decoder.ColumnType.INT,
      schema_pb2.FLOAT: csv_decoder.ColumnType.FLOAT,
      schema_pb2.BYTES: csv_decoder.ColumnType.STRING
  }
  feature_type_map = {}
  for feature in schema.feature:
    feature_type_map[feature.name] = schema_type_to_stats_type[feature.type]

  return [
      csv_decoder.ColumnInfo(col_name, feature_type_map.get(col_name, None))
      for col_name in column_names
  ]
