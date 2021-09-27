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
from typing import Generator, Text
import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options

from tensorflow_metadata.proto.v0 import statistics_pb2


# TODO(b/112146483): Test the Stats API with unicode input.
@beam.typehints.with_input_types(pa.RecordBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
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

  def expand(self, dataset: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    if self._options.sample_rate is not None:
      dataset |= ('SampleExamplesAtRate(%s)' % self._options.sample_rate >>
                  beam.FlatMap(_sample_at_rate,
                               sample_rate=self._options.sample_rate))

    return (dataset | 'RunStatsGenerators' >>
            stats_impl.GenerateStatisticsImpl(self._options))


def _sample_at_rate(example: pa.RecordBatch, sample_rate: float
                   ) -> Generator[pa.RecordBatch, None, None]:
  """Sample examples at input sampling rate."""
  # TODO(pachristopher): Revisit this to decide if we need to fix a seed
  # or add an optional seed argument.
  if random.random() <= sample_rate:
    yield example


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(beam.pvalue.PDone)
class WriteStatisticsToBinaryFile(beam.PTransform):
  """API for writing serialized data statistics to a binary file."""

  def __init__(self, output_path: Text) -> None:
    """Initializes the transform.

    Args:
      output_path: Output path for writing data statistics.
    """
    self._output_path = output_path

  def expand(self, stats: beam.pvalue.PCollection) -> beam.pvalue.PDone:
    return (stats
            | 'WriteStats' >> beam.io.WriteToText(
                self._output_path,
                shard_name_template='',
                append_trailing_newlines=False,
                coder=beam.coders.ProtoCoder(
                    statistics_pb2.DatasetFeatureStatisticsList)))


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(beam.pvalue.PDone)
class WriteStatisticsToTFRecord(beam.PTransform):
  """API for writing serialized data statistics to TFRecord file."""

  def __init__(self, output_path: Text) -> None:
    """Initializes the transform.

    Args:
      output_path: Output path for writing data statistics.
    """
    self._output_path = output_path

  def expand(self, stats: beam.pvalue.PCollection) -> beam.pvalue.PDone:
    return (stats
            | 'WriteStats' >> beam.io.WriteToTFRecord(
                self._output_path,
                shard_name_template='',
                coder=beam.coders.ProtoCoder(
                    statistics_pb2.DatasetFeatureStatisticsList)))
