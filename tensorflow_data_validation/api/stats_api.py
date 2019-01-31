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

GenerateStatistics applies a set of statistics generator each of which
computes a specific type of statistic.  Specifically, we have five default
generators:
  1) CommonStatsGenerator, which computes the common statistics for all
     the features.
  2) NumericStatsGenerator, which computes the numeric statistics for features
     of numeric type (INT or FLOAT).
  3) StringStatsGenerator, which computes the common string statistics for
     features of string type.
  4) TopKStatsGenerator, which computes the top-k values for features
     of string type.
  5) UniqueStatsGenerator, which computes the number of unique values for
     features of string type.

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
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.types_compat import Generator

from tensorflow_metadata.proto.v0 import statistics_pb2


# TODO(b/112146483): Test the Stats API with unicode input.
@beam.typehints.with_input_types(types.BeamExample)
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
      options = stats_options.StatsOptions()
  ):
    """Initializes the transform.

    Args:
      options: Options for generating data statistics.

    Raises:
      TypeError: If options is not of the expected type.
    """
    if not isinstance(options, stats_options.StatsOptions):
      raise TypeError('options is of type %s, should be a StatsOptions.' %
                      type(options).__name__)
    self._options = options

  def expand(self, dataset):
    # Sample input data if sample_count option is provided.
    # TODO(b/117229955): Consider providing an option to write the sample
    # to a file.
    if self._options.sample_count is not None:
      # beam.combiners.Sample.FixedSizeGlobally returns a
      # PCollection[List[types.Example]], which we then flatten to get a
      # PCollection[types.Example].
      dataset |= ('SampleExamples(%s)' % self._options.sample_count >>
                  beam.combiners.Sample.FixedSizeGlobally(
                      self._options.sample_count)
                  | 'FlattenExamples' >> beam.FlatMap(lambda lst: lst))
    elif self._options.sample_rate is not None:
      dataset |= ('SampleExamplesAtRate(%s)' % self._options.sample_rate >>
                  beam.FlatMap(_sample_at_rate,
                               sample_rate=self._options.sample_rate))

    return (dataset | 'RunStatsGenerators' >>
            stats_impl.GenerateStatisticsImpl(self._options))


def _sample_at_rate(example, sample_rate
                   ):
  """Sample examples at input sampling rate."""
  # TODO(pachristopher): Revisit this to decide if we need to fix a seed
  # or add an optional seed argument.
  if random.random() <= sample_rate:
    yield example
