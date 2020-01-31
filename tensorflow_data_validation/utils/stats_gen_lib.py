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
# limitations under the License
"""Convenient library for data statistics generation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import csv
import logging
import multiprocessing
import os
import tempfile

import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import PipelineOptions
from joblib import delayed
from joblib import Parallel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tensorflow_data_validation.coders import csv_decoder
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.statistics.generators import stats_generator
from typing import Any, List, Optional, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def generate_statistics_from_tfrecord(
    data_location: Text,
    output_path: Optional[bytes] = None,
    stats_options: options.StatsOptions = options.StatsOptions(),
    pipeline_options: Optional[PipelineOptions] = None,
    compression_type: Text = CompressionTypes.AUTO,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Compute data statistics from TFRecord files containing TFExamples.

  Runs a Beam pipeline to compute the data statistics and return the result
  data statistics proto.

  This is a convenience method for users with data in TFRecord format.
  Users with data in unsupported file/data formats, or users who wish
  to create their own Beam pipelines need to use the 'GenerateStatistics'
  PTransform API directly instead.

  Args:
    data_location: The location of the input data files.
    output_path: The file path to output data statistics result to. If None, we
      use a temporary directory. It will be a TFRecord file containing a single
      data statistics proto, and can be read with the 'load_statistics' API.
      If you run this function on Google Cloud, you must specify an
      output_path. Specifying None may cause an error.
    stats_options: `tfdv.StatsOptions` for generating data statistics.
    pipeline_options: Optional beam pipeline options. This allows users to
      specify various beam pipeline execution parameters like pipeline runner
      (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
      See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
      more details.
    compression_type: Used to handle compressed input files. Default value is
      CompressionTypes.AUTO, in which case the file_path's extension will be
      used to detect the compression.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  if output_path is None:
    output_path = os.path.join(tempfile.mkdtemp(), 'data_stats.tfrecord')
  output_dir_path = os.path.dirname(output_path)
  if not tf.io.gfile.exists(output_dir_path):
    tf.io.gfile.makedirs(output_dir_path)

  batch_size = (
      stats_options.desired_batch_size if stats_options.desired_batch_size
      and stats_options.desired_batch_size > 0 else
      constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE)
  # PyLint doesn't understand Beam PTransforms.
  # pylint: disable=no-value-for-parameter
  with beam.Pipeline(options=pipeline_options) as p:
    # Auto detect tfrecord file compression format based on input data
    # path suffix.
    _ = (
        p
        | 'ReadData' >> beam.io.ReadFromTFRecord(
            file_pattern=data_location, compression_type=compression_type)
        | 'DecodeData' >> tf_example_decoder.DecodeTFExample(
            desired_batch_size=batch_size)
        | 'GenerateStatistics' >> stats_api.GenerateStatistics(stats_options)
        # TODO(b/112014711) Implement a custom sink to write the stats proto.
        | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
            output_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(
                statistics_pb2.DatasetFeatureStatisticsList)))
  return load_statistics(output_path)


def generate_statistics_from_csv(
    data_location: Text,
    column_names: Optional[List[types.FeatureName]] = None,
    delimiter: Text = ',',
    output_path: Optional[bytes] = None,
    stats_options: options.StatsOptions = options.StatsOptions(),
    pipeline_options: Optional[PipelineOptions] = None,
    compression_type: Text = CompressionTypes.AUTO,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Compute data statistics from CSV files.

  Runs a Beam pipeline to compute the data statistics and return the result
  data statistics proto.

  This is a convenience method for users with data in CSV format.
  Users with data in unsupported file/data formats, or users who wish
  to create their own Beam pipelines need to use the 'GenerateStatistics'
  PTransform API directly instead.

  Args:
    data_location: The location of the input data files.
    column_names: A list of column names to be treated as the CSV header. Order
      must match the order in the input CSV files. If this argument is not
      specified, we assume the first line in the input CSV files as the
      header. Note that this option is valid only for 'csv' input file format.
    delimiter: A one-character string used to separate fields in a CSV file.
    output_path: The file path to output data statistics result to. If None, we
      use a temporary directory. It will be a TFRecord file containing a single
      data statistics proto, and can be read with the 'load_statistics' API.
      If you run this function on Google Cloud, you must specify an
      output_path. Specifying None may cause an error.
    stats_options: `tfdv.StatsOptions` for generating data statistics.
    pipeline_options: Optional beam pipeline options. This allows users to
      specify various beam pipeline execution parameters like pipeline runner
      (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
      See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
      more details.
    compression_type: Used to handle compressed input files. Default value is
      CompressionTypes.AUTO, in which case the file_path's extension will be
      used to detect the compression.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  if output_path is None:
    output_path = os.path.join(tempfile.mkdtemp(), 'data_stats.tfrecord')
  output_dir_path = os.path.dirname(output_path)
  if not tf.io.gfile.exists(output_dir_path):
    tf.io.gfile.makedirs(output_dir_path)

  batch_size = (
      stats_options.desired_batch_size if stats_options.desired_batch_size
      and stats_options.desired_batch_size > 0 else
      constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE)
  # PyLint doesn't understand Beam PTransforms.
  # pylint: disable=no-value-for-parameter
  with beam.Pipeline(options=pipeline_options) as p:
    # If a header is not provided, assume the first line in a file
    # to be the header.
    skip_header_lines = 1 if column_names is None else 0
    if column_names is None:
      column_names = get_csv_header(data_location, delimiter)
    _ = (
        p
        | 'ReadData' >> beam.io.textio.ReadFromText(
            file_pattern=data_location, skip_header_lines=skip_header_lines,
            compression_type=compression_type)
        | 'DecodeData' >> csv_decoder.DecodeCSV(
            column_names=column_names, delimiter=delimiter,
            schema=stats_options.schema,
            infer_type_from_schema=stats_options.infer_type_from_schema,
            desired_batch_size=batch_size)
        | 'GenerateStatistics' >> stats_api.GenerateStatistics(stats_options)
        # TODO(b/112014711) Implement a custom sink to write the stats proto.
        | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
            output_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(
                statistics_pb2.DatasetFeatureStatisticsList)))
  return load_statistics(output_path)


def generate_statistics_from_dataframe(
    dataframe: pd.DataFrame,
    stats_options: options.StatsOptions = options.StatsOptions(),
    n_jobs: int = 1
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Compute data statistics for the input pandas DataFrame.

  This is a utility method for users with in-memory data represented
  as a pandas DataFrame.

  Args:
    dataframe: Input pandas DataFrame.
    stats_options: `tfdv.StatsOptions` for generating data statistics.
    n_jobs: Number of processes to run (defaults to 1). If -1 is provided,
      uses the same number of processes as the number of CPU cores.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  if not isinstance(dataframe, pd.DataFrame):
    raise TypeError('dataframe argument is of type {}. Must be a '
                    'pandas DataFrame.'.format(type(dataframe).__name__))

  stats_generators = stats_impl.get_generators(stats_options, in_memory=True)  # type: List[stats_generator.CombinerStatsGenerator]
  if n_jobs < -1 or n_jobs == 0:
    raise ValueError('Invalid n_jobs parameter {}. Should be either '
                     ' -1 or >= 1.'.format(n_jobs))

  if n_jobs == -1:
    n_jobs = multiprocessing.cpu_count()
  n_jobs = max(min(n_jobs, multiprocessing.cpu_count()), 1)

  if n_jobs == 1:
    merged_partial_stats = _generate_partial_statistics_from_df(
        dataframe, stats_options, stats_generators)
  else:
    # TODO(pachristopher): Investigate why we don't observe linear speedup after
    # a certain number of processes.
    splits = np.array_split(dataframe, n_jobs)
    partial_stats = Parallel(n_jobs=n_jobs)(
        delayed(_generate_partial_statistics_from_df)(
            splits[i], stats_options, stats_generators) for i in range(n_jobs))
    merged_partial_stats = [
        gen.merge_accumulators(stats)
        for gen, stats in zip(stats_generators, zip(*partial_stats))
    ]
  return stats_impl.extract_statistics_output(
      merged_partial_stats, stats_generators)


def _generate_partial_statistics_from_df(
    dataframe: pd.DataFrame,
    stats_options: options.StatsOptions,
    stats_generators: List[stats_generator.CombinerStatsGenerator]
) -> List[Any]:
  """Generate accumulators containing partial stats."""
  inmemory_dicts = [{} for _ in range(len(dataframe))]
  isnull = pd.isnull
  # Initialize decoding fn based on column type.
  int_fn = lambda x: np.array([x], dtype=np.integer)
  float_fn = lambda x: None if isnull(x) else np.array([x], dtype=np.floating)
  str_fn = lambda x: None if isnull(x) else np.array([x], dtype=np.object)
  decode_fn = {
      # int type.
      'i': int_fn,
      'u': int_fn,
      # float type.
      'f': float_fn,
      # bool type.
      'b': int_fn,
      # string type.
      'S': str_fn,
      'O': str_fn,
      'U': str_fn,
  }

  feature_whitelist = set()
  if stats_options.feature_whitelist:
    feature_whitelist.update(stats_options.feature_whitelist)
  # Create a copy of the stats options so that we don't modify the input object.
  stats_options_modified = copy.copy(stats_options)
  # Remove feature_whitelist option as it is no longer needed.
  stats_options_modified.feature_whitelist = None
  schema = schema_pb2.Schema()
  for col_name, col_type in zip(dataframe.columns, dataframe.dtypes):
    kind = col_type.kind
    if (kind not in decode_fn or
        (feature_whitelist and col_name not in feature_whitelist)):
      logging.warning('Ignoring feature %s of type %s', col_name, col_type)
      continue
    if kind == 'b':
      # Track bool type feature as categorical.
      schema.feature.add(
          name=col_name, type=schema_pb2.INT,
          bool_domain=schema_pb2.BoolDomain())

    # Get decoding fn based on column type.
    fn = decode_fn[kind]
    # Iterate over the column and apply the decoding fn.
    j = 0
    for val in dataframe[col_name]:
      inmemory_dicts[j][col_name] = fn(val)
      j += 1
  if schema.feature:
    stats_options_modified.schema = schema
  return stats_impl.generate_partial_statistics_in_memory(
      decoded_examples_to_arrow.DecodedExamplesToTable(inmemory_dicts),
      stats_options_modified, stats_generators)


def get_csv_header(data_location: Text,
                   delimiter: Text) -> List[types.FeatureName]:
  """Gets the CSV header from the input files.

  This function assumes that the header is present as the first line in all
  the files in the input path.

  Args:
    data_location: Glob pattern(s) specifying the location of the input data
      files.
    delimiter: A one-character string used to separate fields in a CSV file.

  Returns:
    The list of column names.

  Raises:
    ValueError: If any of the input files is not found or empty, or if the files
      have different headers.
  """
  matched_files = tf.io.gfile.glob(data_location)
  if not matched_files:
    raise ValueError(
        'No file found in the input data location: %s' % data_location)

  # Read the header line in the first file.
  with tf.io.gfile.GFile(matched_files[0], 'r') as reader:
    try:
      result = next(csv.reader(reader, delimiter=delimiter))
    except StopIteration:
      raise ValueError('Found empty file when reading the header line: %s' %
                       matched_files[0])

  # Make sure that all files have the same header.
  for filename in matched_files[1:]:
    with tf.io.gfile.GFile(filename, 'r') as reader:
      try:
        if next(csv.reader(reader, delimiter=delimiter)) != result:
          raise ValueError('Files have different headers.')
      except StopIteration:
        raise ValueError(
            'Found empty file when reading the header line: %s' % filename)

  return result


def load_statistics(
    input_path: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Loads data statistics proto from file.

  Args:
    input_path: Data statistics file path.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  serialized_stats = next(tf.compat.v1.io.tf_record_iterator(input_path))
  result = statistics_pb2.DatasetFeatureStatisticsList()
  result.ParseFromString(serialized_stats)
  return result
