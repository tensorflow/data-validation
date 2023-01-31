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
"""TensorFlow Data Validation Statistics API.

The Statistics API for TF Data Validation consists of a single beam.PTransform,
GenerateStatistics, that computes a set of statistics on an input set of
examples in a single pass over the examples.

GenerateStatistics applies a set of statistics generators, each of which
computes different types of statistics.  Specifically, we have two default
generators:
  1) BasicStatsGenerator, which computes the common statistics for all features,
     numeric statistics for features of numeric type (INT or FLOAT), and common
     string statistics for features of string type.
  2) TopKUniquesStatsGenerator, which computes the top-k and number of unique
     values for features of string type.

If the enable_semantic_domain_stats option in `StatsOptions` is True,
GenerateStatistics will also apply generators that compute statistics for
semantic domains (e.g., ImageStatsGenerator).

Additional generators can be implemented and added to the default set to
compute additional custom statistics.

The stats generators process a batch of examples at a time. All the stats
generators are run together in the same pass. At the end, their
outputs are combined and converted to a DatasetFeatureStatisticsList proto
(https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/statistics.proto).  # pylint: disable=line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from typing import Generator, Text, Optional

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation.utils import artifacts_io_impl
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tfx_bsl.statistics import merge_util

from tensorflow_metadata.proto.v0 import statistics_pb2


class GenerateStatistics(beam.PTransform):
  """API for generating data statistics.

  Example:

  ```python
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> tfx_bsl.public.tfxio.TFExampleRecord(data_location)
               .BeamSource()
           | 'GenerateStatistics' >> GenerateStatistics()
           | 'WriteStatsOutput' >> tfdv.WriteStatisticsToTFRecord(output_path))
  ```
  """

  def __init__(
      self,
      options: stats_options.StatsOptions = stats_options.StatsOptions()
  ) -> None:
    """Initializes the transform.

    Args:
      options: `tfdv.StatsOptions` for generating data statistics.

    Raises:
      TypeError: If options is not of the expected type.
    """
    if not isinstance(options, stats_options.StatsOptions):
      raise TypeError('options is of type %s, should be a StatsOptions.' %
                      type(options).__name__)
    self._options = options

  def expand(
      self, dataset: beam.PCollection[pa.RecordBatch]
  ) -> beam.PCollection[statistics_pb2.DatasetFeatureStatisticsList]:
    if self._options.sample_rate is not None:
      dataset |= ('SampleExamplesAtRate(%s)' % self._options.sample_rate >>
                  beam.FlatMap(_sample_at_rate,
                               sample_rate=self._options.sample_rate))

    return (dataset | 'RunStatsGenerators' >>
            stats_impl.GenerateStatisticsImpl(self._options))


def _sample_at_rate(example: pa.RecordBatch, sample_rate: float
                   ) -> Generator[pa.RecordBatch, None, None]:
  """Sample examples at input sampling rate."""
  if random.random() <= sample_rate:
    yield example


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
class WriteStatisticsToBinaryFile(beam.PTransform):
  """API for writing serialized data statistics to a binary file."""

  def __init__(self, output_path: Text) -> None:
    """Initializes the transform.

    Args:
      output_path: Output path for writing data statistics.
    """
    self._output_path = output_path

  # TODO(b/202910677): Find a way to check that the PCollection passed here
  # has only one element.
  def expand(self, stats: beam.PCollection) -> beam.pvalue.PDone:
    return (stats
            | 'WriteStats' >> beam.io.WriteToText(
                self._output_path,
                shard_name_template='',
                append_trailing_newlines=False,
                coder=beam.coders.ProtoCoder(
                    statistics_pb2.DatasetFeatureStatisticsList)))


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
class WriteStatisticsToTFRecord(beam.PTransform):
  """API for writing serialized data statistics to TFRecord file."""

  def __init__(self, output_path: Text, sharded_output=False) -> None:
    """Initializes the transform.

    Args:
      output_path: The output path or path prefix (if sharded_output=True).
      sharded_output: If true, writes sharded TFRecords files in the form
        output_path-SSSSS-of-NNNNN.
    """
    self._output_path = output_path
    self._sharded_output = sharded_output

  def expand(self, stats: beam.PCollection) -> beam.pvalue.PDone:
    return (stats
            | 'WriteStats' >> beam.io.WriteToTFRecord(
                self._output_path,
                shard_name_template='' if not self._sharded_output else None,
                coder=beam.coders.ProtoCoder(
                    statistics_pb2.DatasetFeatureStatisticsList)))


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class MergeDatasetFeatureStatisticsList(beam.PTransform):
  """API for merging sharded DatasetFeatureStatisticsList."""
  # TODO(b/202910677): Replace this with a more efficient CombineFn.

  def expand(self, stats: beam.PCollection):
    return stats | 'MergeDatasetFeatureStatisticsProtos' >> beam.CombineGlobally(
                merge_util.merge_dataset_feature_statistics_list)


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
class WriteStatisticsToRecordsAndBinaryFile(beam.PTransform):
  """API for writing statistics to both sharded records and binary pb.

  This PTransform assumes that input represents sharded statistics, which are
  written directly. These statistics are also merged and written to a binary
  proto.

  Currently Experimental.

  TODO(b/202910677): After full migration to sharded stats, clean this up.
  """

  def __init__(
      self,
      binary_proto_path: str,
      records_path_prefix: str,
      columnar_path_prefix: Optional[str] = None,
  ) -> None:
    """Initializes the transform.

    Args:
      binary_proto_path: Output path for writing statistics as a binary proto.
      records_path_prefix: File pattern for writing statistics to sharded
        records.
      columnar_path_prefix: Optional file pattern for writing statistics to
        columnar outputs. If provided, columnar outputs will be written when
        supported.
    """
    self._binary_proto_path = binary_proto_path
    self._records_path_prefix = records_path_prefix
    self._io_provider = artifacts_io_impl.get_io_provider()
    self._columnar_path_prefix = columnar_path_prefix

  def expand(self, stats: beam.PCollection) -> beam.pvalue.PDone:
    # Write sharded outputs, ignoring PDone.
    _ = (
        stats | 'WriteShardedStats' >> self._io_provider.record_sink_impl(
            output_path_prefix=self._records_path_prefix))
    if self._columnar_path_prefix is not None:
      columnar_provider = artifacts_io_impl.get_default_columnar_provider()
      if columnar_provider is not None:
        _ = (
            stats | 'WriteColumnarStats' >> columnar_provider.record_sink_impl(
                self._columnar_path_prefix))
    return (stats
            | 'MergeDatasetFeatureStatisticsProtos' >> beam.CombineGlobally(
                merge_util.merge_dataset_feature_statistics_list)
            | 'WriteBinaryStats' >> WriteStatisticsToBinaryFile(
                self._binary_proto_path))


def default_sharded_output_supported() -> bool:
  """True if sharded output is supported by default."""
  return artifacts_io_impl.should_write_sharded()


def default_sharded_output_suffix() -> str:
  """Returns the default sharded output suffix."""
  return artifacts_io_impl.get_io_provider().file_suffix()
