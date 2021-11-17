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

import collections
import os
import tempfile

from typing import List, Mapping, Optional, Text, Tuple, Union
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd
import pyarrow as pa
import tensorflow as tf
from tensorflow_data_validation import types
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.api import validation_api
from tensorflow_data_validation.coders import csv_decoder

from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.utils import io_util
from tensorflow_data_validation.utils import stats_gen_lib
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tf_example_record
from tensorflow_metadata.proto.v0 import statistics_pb2


def _encode_example_and_key(coder: example_coder.RecordBatchToExamplesEncoder,
                            kv):
  """Converts a (key, RecordBatch) tuple to a list of (key, tf.Example)."""
  k, v = kv
  result = []
  for record_batch in v:
    for serialized_example in coder.encode(record_batch):
      result.append((k, serialized_example))
  return result


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.KV[types.SliceKey,
                                                   List[pa.RecordBatch]])
@beam.typehints.with_output_types(beam.typehints.KV[types.SliceKey,
                                                    List[bytes]])
def _record_batch_to_example_fn(
    pcoll: beam.pvalue.PCollection,
    coder: example_coder.RecordBatchToExamplesEncoder):
  return pcoll | beam.FlatMap(lambda kv: _encode_example_and_key(coder, kv))


def validate_examples_in_tfrecord(
    data_location: Text,
    stats_options: options.StatsOptions,
    output_path: Optional[Text] = None,
    pipeline_options: Optional[PipelineOptions] = None,
    num_sampled_examples=0,
) -> Union[statistics_pb2.DatasetFeatureStatisticsList, Tuple[
    statistics_pb2.DatasetFeatureStatisticsList, Mapping[
        str, List[tf.train.Example]]]]:
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
    num_sampled_examples: If set, returns up to this many examples
      of each anomaly type as a map from anomaly reason string to a list of
      tf.Examples.

  Returns:
    If num_sampled_examples is zero, returns a single
    DatasetFeatureStatisticsList proto in which each dataset consists of the
    set of examples that exhibit a particular anomaly. If
    num_sampled_examples is nonzero, returns the same statistics
    proto as well as a mapping from anomaly to a list of tf.Examples that
    exhibited that anomaly.

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
  with io_util.Materializer(output_dir_path) as sample_materializer:
    with beam.Pipeline(options=pipeline_options) as p:
      anomalous_examples = (
          p
          | 'ReadData' >> (tf_example_record.TFExampleRecord(
              file_pattern=data_location,
              schema=None,
              telemetry_descriptors=['tfdv', 'validate_examples_in_tfrecord'
                                    ]).BeamSource(batch_size=1))
          | 'DetectAnomalies' >>
          validation_api.IdentifyAnomalousExamples(stats_options))
      _ = (
          anomalous_examples | 'GenerateSummaryStatistics' >>
          stats_impl.GenerateSlicedStatisticsImpl(
              stats_options, is_slicing_enabled=True)
          | 'WriteStatsOutput' >>
          stats_api.WriteStatisticsToTFRecord(output_path))
      if num_sampled_examples:
        # TODO(b/68154497): Relint
        # pylint: disable=no-value-for-parameter
        _ = (
            anomalous_examples
            | 'Sample' >>
            beam.combiners.Sample.FixedSizePerKey(num_sampled_examples)
            | 'ToExample' >> _record_batch_to_example_fn(
                example_coder.RecordBatchToExamplesEncoder(
                    stats_options.schema))
            | 'WriteSamples' >> sample_materializer.writer())
        # pylint: enable=no-value-for-parameter
    if num_sampled_examples:
      samples_per_reason = collections.defaultdict(list)
      for reason, serialized_example in sample_materializer.reader():
        samples_per_reason[reason].append(
            tf.train.Example.FromString(serialized_example))
      return stats_util.load_statistics(output_path), samples_per_reason
  return stats_util.load_statistics(output_path)


def _try_unwrap(maybe_collection):
  """If input is a collection of one item, return that, or return input."""
  if isinstance(maybe_collection, str) or isinstance(maybe_collection, bytes):
    return maybe_collection
  try:
    if len(maybe_collection) == 1:
      return next(iter(maybe_collection))
  except TypeError:
    return maybe_collection


def _encode_pandas_and_key(kv):
  """Converts a (key, RecordBatch) tuple to a list of (key, pd.DataFrame)."""
  k, v = kv
  result = []
  for record_batch in v:
    # to_pandas() returns a DF that may (or always?) contain lists of
    # RecordBatch array contents per-cell. When converting from a CSV there
    # should be exactly one item; this function best-effort unwraps the
    # collection in that case.
    df = record_batch.to_pandas().applymap(_try_unwrap)
    result.append((k, df))
  return result


def validate_examples_in_csv(
    data_location: Text,
    stats_options: options.StatsOptions,
    column_names: Optional[List[types.FeatureName]] = None,
    delimiter: Text = ',',
    output_path: Optional[Text] = None,
    pipeline_options: Optional[PipelineOptions] = None,
    num_sampled_examples=0,
) -> Union[statistics_pb2.DatasetFeatureStatisticsList, Tuple[
    statistics_pb2.DatasetFeatureStatisticsList, Mapping[str, pd.DataFrame]]]:
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
      'load_statistics' function. If you run this function on Google Cloud, you
      must specify an output_path. Specifying None may cause an error.
    pipeline_options: Optional beam pipeline options. This allows users to
      specify various beam pipeline execution parameters like pipeline runner
      (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
      See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
        more details.
    num_sampled_examples: If set, returns up to this many examples of each
      anomaly type as a map from anomaly reason string to pd.DataFrame.

  Returns:
    If num_sampled_examples is zero, returns a single
    DatasetFeatureStatisticsList proto in which each dataset consists of the
    set of examples that exhibit a particular anomaly. If
    num_sampled_examples is nonzero, returns the same statistics
    proto as well as a mapping from anomaly to a pd.DataFrame of CSV rows
    exhibiting that anomaly.

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
  if num_sampled_examples:
    sample_materializer = io_util.Materializer(output_dir_path)

  # If a header is not provided, assume the first line in a file
  # to be the header.
  skip_header_lines = 1 if column_names is None else 0
  if column_names is None:
    column_names = stats_gen_lib.get_csv_header(data_location, delimiter)

  with beam.Pipeline(options=pipeline_options) as p:

    anomalous_examples = (
        p
        | 'ReadData' >> beam.io.textio.ReadFromText(
            file_pattern=data_location, skip_header_lines=skip_header_lines)
        | 'DecodeData' >> csv_decoder.DecodeCSV(
            column_names=column_names,
            delimiter=delimiter,
            schema=stats_options.schema
            if stats_options.infer_type_from_schema else None,
            desired_batch_size=1)
        | 'DetectAnomalies' >>
        validation_api.IdentifyAnomalousExamples(stats_options))
    _ = (
        anomalous_examples
        |
        'GenerateSummaryStatistics' >> stats_impl.GenerateSlicedStatisticsImpl(
            stats_options, is_slicing_enabled=True)
        |
        'WriteStatsOutput' >> stats_api.WriteStatisticsToTFRecord(output_path))
    if num_sampled_examples:
      _ = (
          anomalous_examples
          | 'Sample' >>
          beam.combiners.Sample.FixedSizePerKey(num_sampled_examples)
          | 'ToPandas' >> beam.FlatMap(_encode_pandas_and_key)
          | 'WriteSamples' >> sample_materializer.writer())

  if num_sampled_examples:
    samples_per_reason_acc = collections.defaultdict(list)
    for reason, pandas_dataframe in sample_materializer.reader():
      samples_per_reason_acc[reason].append(pandas_dataframe)
    samples_per_reason = {}
    for reason, dataframes in samples_per_reason_acc.items():
      samples_per_reason[reason] = pd.concat(dataframes)
    sample_materializer.cleanup()
    return stats_util.load_statistics(output_path), samples_per_reason
  return stats_util.load_statistics(output_path)
