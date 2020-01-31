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
from tensorflow_data_validation.utils import batch_util
from tfx_bsl.coders import csv_decoder as csv_decoder
from typing import List, Iterable, Optional, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# TODO(b/111831548): Add support for a secondary delimiter to parse
# value lists.
@beam.typehints.with_input_types(types.BeamCSVRecord)
@beam.typehints.with_output_types(pa.Table)
class DecodeCSV(beam.PTransform):
  """Decodes CSV records into Arrow tables.

  Currently we assume each column in the input CSV has only a single value.
  """

  def __init__(self,
               column_names: List[types.FeatureName],
               delimiter: Text = ',',
               skip_blank_lines: bool = True,
               schema: Optional[schema_pb2.Schema] = None,
               infer_type_from_schema: bool = False,
               desired_batch_size: Optional[int] = constants
               .DEFAULT_DESIRED_INPUT_BATCH_SIZE):
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
      desired_batch_size: Batch size. The output Arrow tables will have as many
        rows as the `desired_batch_size`.
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
                  skip_blank_lines=self._skip_blank_lines)))

    # Do second pass to generate the in-memory dict representation.
    return (csv_lines
            | 'BatchCSVLines' >> beam.BatchElements(
                **batch_util.GetBeamBatchKwargs(self._desired_batch_size))
            | 'BatchedCSVRowsToArrow' >> beam.ParDo(
                _BatchedCSVRowsToArrow(skip_blank_lines=self._skip_blank_lines),
                column_infos))


@beam.typehints.with_input_types(List[List[csv_decoder.CSVCell]],
                                 List[csv_decoder.ColumnInfo])
@beam.typehints.with_output_types(pa.Table)
class _BatchedCSVRowsToArrow(beam.DoFn):
  """DoFn to convert a batch of csv rows to a pa.Table."""

  __slots__ = [
      '_skip_blank_lines', '_column_handlers', '_column_arrow_types',
      '_column_names'
  ]

  def __init__(self, skip_blank_lines: bool):
    self._skip_blank_lines = skip_blank_lines
    self._column_handlers = None
    self._column_names = None
    self._column_arrow_types = None

  def _process_column_infos(self, column_infos: List[csv_decoder.ColumnInfo]):
    column_handlers = []
    column_arrow_types = []
    for c in column_infos:
      if c.type == statistics_pb2.FeatureNameStatistics.INT:
        column_handlers.append(lambda v: (int(v),))
        column_arrow_types.append(pa.list_(pa.int64()))
      elif c.type == statistics_pb2.FeatureNameStatistics.FLOAT:
        column_handlers.append(lambda v: (float(v),))
        column_arrow_types.append(pa.list_(pa.float32()))
      elif c.type == statistics_pb2.FeatureNameStatistics.STRING:
        column_handlers.append(lambda v: (v,))
        column_arrow_types.append(pa.list_(pa.binary()))
      else:
        column_handlers.append(lambda _: None)
        column_arrow_types.append(pa.null())
    self._column_handlers = column_handlers
    self._column_arrow_types = column_arrow_types
    self._column_names = [c.name for c in column_infos]

  def process(self, batch: List[List[types.CSVCell]],
              column_infos: List[csv_decoder.ColumnInfo]) -> Iterable[pa.Table]:
    if self._column_names is None:
      self._process_column_infos(column_infos)

    values_list_by_column = [[] for _ in self._column_handlers]
    for csv_row in batch:
      if not csv_row:
        if not self._skip_blank_lines:
          for l in values_list_by_column:
            l.append(None)
        continue
      if len(csv_row) != len(self._column_handlers):
        raise ValueError('Encountered a row of unexpected number of columns')

      for value, handler, values_list in (
          zip(csv_row, self._column_handlers, values_list_by_column)):
        values_list.append(handler(value) if value else None)

    arrow_arrays = [
        pa.array(l, type=t) for l, t in
        zip(values_list_by_column, self._column_arrow_types)
    ]
    yield pa.Table.from_arrays(arrow_arrays, self._column_names)


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
