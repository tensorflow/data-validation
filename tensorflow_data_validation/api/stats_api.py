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
import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from typing import Generator

from tensorflow_metadata.proto.v0 import statistics_pb2


# TODO(b/112146483): Test the Stats API with unicode input.
@beam.typehints.with_input_types(pa.Table)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateStatistics(beam.PTransform):
  """API for generating data statistics.

  Example:

  ```python
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'DecodeData' >> beam.Map(TFExampleDecoder().decode)
           | 'GenerateStatistics' >> GenerateStatistics()
           | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
               output_path, shard_name_template='',
               coder=beam.coders.ProtoCoder(
                   statistics_pb2.DatasetFeatureStatisticsList)))
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
    # Sample input data if sample_count option is provided.
    # TODO(b/117229955): Consider providing an option to write the sample
    # to a file.
    if self._options.sample_count is not None:
      # TODO(pachristopher): Consider moving the sampling logic to decoders.
      # beam.combiners.Sample.FixedSizeGlobally returns a
      # PCollection[List[pa.Table]], which we then flatten to get a
      # PCollection[pa.Table].
      batch_size = (
          self._options.desired_batch_size if self._options.desired_batch_size
          and self._options.desired_batch_size > 0 else
          constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE)
      batch_count = (
          int(self._options.sample_count / batch_size) +
          (1 if self._options.sample_count % batch_size else 0))
      dataset |= ('SampleExamples(%s)' % self._options.sample_count >>
                  beam.combiners.Sample.FixedSizeGlobally(batch_count)
                  | 'FlattenExamples' >> beam.FlatMap(lambda lst: lst))
    elif self._options.sample_rate is not None:
      dataset |= ('SampleExamplesAtRate(%s)' % self._options.sample_rate >>
                  beam.FlatMap(_sample_at_rate,
                               sample_rate=self._options.sample_rate))

    return (dataset | 'RunStatsGenerators' >>
            stats_impl.GenerateStatisticsImpl(self._options))


def _sample_at_rate(example: types.Example, sample_rate: float
                   ) -> Generator[types.Example, None, None]:
  """Sample examples at input sampling rate."""
  # TODO(pachristopher): Revisit this to decide if we need to fix a seed
  # or add an optional seed argument.
  if random.random() <= sample_rate:
    yield example
