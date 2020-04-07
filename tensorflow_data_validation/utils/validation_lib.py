# Copyright 2019 Google LLC
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
# limitations under the License
"""Convenient library for detecting anomalies on a per-example basis."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import tempfile

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
from tensorflow_data_validation import types
from tensorflow_data_validation.api import validation_api
from tensorflow_data_validation.coders import csv_decoder
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.utils import stats_gen_lib
from typing import List, Optional, Text

from tensorflow_metadata.proto.v0 import statistics_pb2


def validate_examples_in_tfrecord(
    data_location: Text,
    stats_options: options.StatsOptions,
    output_path: Optional[Text] = None,
    # TODO(b/131719250): Add option to output a sample of anomalous examples for
    # each anomaly reason.
    pipeline_options: Optional[PipelineOptions] = None,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Validates TFExamples in TFRecord files.

  Runs a Beam pipeline to detect anomalies on a per-example basis. If this
  function detects anomalous examples, it generates summary statistics regarding
  the set of examples that exhibit each anomaly.

  This is a convenience function for users with data in TFRecord format.
  Users with data in unsupported file/data formats, or users who wish
  to create their own Beam pipelines need to use the 'IdentifyAnomalousExamples'
  PTransform API directly instead.

  Args:
    data_location: The location of the input data files.
    stats_options: `tfdv.StatsOptions` for generating data statistics. This must
      contain a schema.
    output_path: The file path to output data statistics result to. If None, the
      function uses a temporary directory. The output will be a TFRecord file
      containing a single data statistics list proto, and can be read with the
      'load_statistics' function.
      If you run this function on Google Cloud, you must specify an
      output_path. Specifying None may cause an error.
    pipeline_options: Optional beam pipeline options. This allows users to
      specify various beam pipeline execution parameters like pipeline runner
      (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
      See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
      more details.

  Returns:
    A DatasetFeatureStatisticsList proto in which each dataset consists of the
      set of examples that exhibit a particular anomaly.

  Raises:
    ValueError: If the specified stats_options does not include a schema.
  """
  if stats_options.schema is None:
    raise ValueError('The specified stats_options must include a schema.')
  if output_path is None:
    output_path = os.path.join(tempfile.mkdtemp(), 'anomaly_stats.tfrecord')
  output_dir_path = os.path.dirname(output_path)
  if not tf.io.gfile.exists(output_dir_path):
    tf.io.gfile.makedirs(output_dir_path)

  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p
        | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=data_location)
        | 'DecodeData' >> tf_example_decoder.DecodeTFExample(
            desired_batch_size=1)
        | 'DetectAnomalies' >>
        validation_api.IdentifyAnomalousExamples(stats_options)
        |
        'GenerateSummaryStatistics' >> stats_impl.GenerateSlicedStatisticsImpl(
            stats_options, is_slicing_enabled=True)
        # TODO(b/112014711) Implement a custom sink to write the stats proto.
        | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
            output_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(
                statistics_pb2.DatasetFeatureStatisticsList)))

  return stats_gen_lib.load_statistics(output_path)


def validate_examples_in_csv(
    data_location: Text,
    stats_options: options.StatsOptions,
    column_names: Optional[List[types.FeatureName]] = None,
    delimiter: Text = ',',
    output_path: Optional[Text] = None,
    # TODO(b/131719250): Add option to output a sample of anomalous examples for
    # each anomaly reason.
    pipeline_options: Optional[PipelineOptions] = None,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Validates examples in csv files.

  Runs a Beam pipeline to detect anomalies on a per-example basis. If this
  function detects anomalous examples, it generates summary statistics regarding
  the set of examples that exhibit each anomaly.

  This is a convenience function for users with data in CSV format.
  Users with data in unsupported file/data formats, or users who wish
  to create their own Beam pipelines need to use the 'IdentifyAnomalousExamples'
  PTransform API directly instead.

  Args:
    data_location: The location of the input data files.
    stats_options: `tfdv.StatsOptions` for generating data statistics. This must
      contain a schema.
    column_names: A list of column names to be treated as the CSV header. Order
      must match the order in the input CSV files. If this argument is not
      specified, we assume the first line in the input CSV files as the header.
      Note that this option is valid only for 'csv' input file format.
    delimiter: A one-character string used to separate fields in a CSV file.
    output_path: The file path to output data statistics result to. If None, the
      function uses a temporary directory. The output will be a TFRecord file
      containing a single data statistics list proto, and can be read with the
      'load_statistics' function.
      If you run this function on Google Cloud, you must specify an
      output_path. Specifying None may cause an error.
    pipeline_options: Optional beam pipeline options. This allows users to
      specify various beam pipeline execution parameters like pipeline runner
      (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
      See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
        more details.

  Returns:
    A DatasetFeatureStatisticsList proto in which each dataset consists of the
      set of examples that exhibit a particular anomaly.

  Raises:
    ValueError: If the specified stats_options does not include a schema.
  """
  if stats_options.schema is None:
    raise ValueError('The specified stats_options must include a schema.')
  if output_path is None:
    output_path = os.path.join(tempfile.mkdtemp(), 'anomaly_stats.tfrecord')
  output_dir_path = os.path.dirname(output_path)
  if not tf.io.gfile.exists(output_dir_path):
    tf.io.gfile.makedirs(output_dir_path)

  # If a header is not provided, assume the first line in a file
  # to be the header.
  skip_header_lines = 1 if column_names is None else 0
  if column_names is None:
    column_names = stats_gen_lib.get_csv_header(data_location, delimiter)

  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p
        | 'ReadData' >> beam.io.textio.ReadFromText(
            file_pattern=data_location, skip_header_lines=skip_header_lines)
        | 'DecodeData' >> csv_decoder.DecodeCSV(
            column_names=column_names, delimiter=delimiter,
            schema=stats_options.schema,
            infer_type_from_schema=stats_options.infer_type_from_schema,
            desired_batch_size=1)
        | 'DetectAnomalies' >>
        validation_api.IdentifyAnomalousExamples(stats_options)
        |
        'GenerateSummaryStatistics' >> stats_impl.GenerateSlicedStatisticsImpl(
            stats_options, is_slicing_enabled=True)
        # TODO(b/112014711) Implement a custom sink to write the stats proto.
        | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
            output_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(
                statistics_pb2.DatasetFeatureStatisticsList)))

  return stats_gen_lib.load_statistics(output_path)
