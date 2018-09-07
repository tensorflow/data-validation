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

import collections
import apache_beam as beam
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics.generators import common_stats_generator
from tensorflow_data_validation.statistics.generators import numeric_stats_generator
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators import string_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_stats_generator
from tensorflow_data_validation.statistics.generators import uniques_stats_generator
from tensorflow_data_validation.utils import batch_util
from tensorflow_data_validation.utils import profile_util
from tensorflow_data_validation.types_compat import List

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsOptions(
    collections.namedtuple('StatsOptions', [
        'generators', 'feature_whitelist', 'schema', 'num_top_values',
        'num_rank_histogram_buckets', 'num_values_histogram_buckets',
        'num_histogram_buckets', 'num_quantiles_histogram_buckets', 'epsilon'
    ])):
  """Options for generating data statistics.

  Attributes:
    generators: An optional list of statistics generators. A statistics
      generator must extend either CombinerStatsGenerator or
      TransformStatsGenerator.
    feature_whitelist: An optional list of names of the features to calculate
      statistics for.
    schema: An optional tensorflow_metadata Schema proto. Currently we use the
      schema to infer categorical and bytes features.
    num_top_values: An optional number of most frequent feature values to keep
      for string features.
    num_rank_histogram_buckets: An optional number of buckets in the rank
      histogram for string features.
    num_values_histogram_buckets: An optional number of buckets in a quantiles
      histogram for the number of values per Feature, which is stored in
      CommonStatistics.num_values_histogram.
    num_histogram_buckets: An optional number of buckets in a standard
      NumericStatistics.histogram with equal-width buckets.
    num_quantiles_histogram_buckets: An optional number of buckets in a
      quantiles NumericStatistics.histogram.
    epsilon: An optional error tolerance for the computation of quantiles,
      typically a small fraction close to zero (e.g. 0.01). Higher values of
      epsilon increase the quantile approximation, and hence result in more
      unequal buckets, but could improve performance, and resource consumption.
  """

  def __new__(cls,
              generators=None,
              feature_whitelist=None,
              schema=None,
              num_top_values=20,
              num_rank_histogram_buckets=1000,
              num_values_histogram_buckets=10,
              num_histogram_buckets=10,
              num_quantiles_histogram_buckets=10,
              epsilon=0.01):
    # Default generate statistics option values
    return super(StatsOptions, cls).__new__(
        cls,
        generators=generators,
        feature_whitelist=feature_whitelist,
        schema=schema,
        num_top_values=num_top_values,
        num_rank_histogram_buckets=num_rank_histogram_buckets,
        num_values_histogram_buckets=num_values_histogram_buckets,
        num_histogram_buckets=num_histogram_buckets,
        num_quantiles_histogram_buckets=num_quantiles_histogram_buckets,
        epsilon=epsilon)


@beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateStatistics(beam.PTransform):
  """Public API for generating data statistics.

  Example usage assuming the input is a tfrecord file of tf.Example:

    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'DecodeData' >> beam.Map(TFExampleDecoder().decode)
           | 'GenerateStatistics' >> GenerateStatistics()
           | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
               output_path, shard_name_template='',
               coder=beam.coders.ProtoCoder(
                   statistics_pb2.DatasetFeatureStatisticsList)))
  """

  def __init__(self, options = StatsOptions()):
    """Initializes the transform.

    Args:
      options: Options for generating data statistics.

    Raises:
      TypeError: If any of the input options is not of the expected type.
      ValueError: If any of the input options is invalid.
    """

    self._check_options(options)
    self._options = options

  def _check_options(self, options):
    if options.generators is not None:
      if not isinstance(options.generators, list):
        raise TypeError('generators is of type %s, should be a list' % type(
            options.generators).__name__)
      for generator in options.generators:
        if (not isinstance(generator,
                           (stats_generator.CombinerStatsGenerator,
                            stats_generator.TransformStatsGenerator))):
          raise TypeError(
              'Statistics generator must extend one of '
              'CombinerStatsGenerator or TransformStatsGenerator, '
              'found object of type %s' % type(generator).__class__.__name__)

    if (options.feature_whitelist is not None and
        not isinstance(options.feature_whitelist, list)):
      raise TypeError('feature_whitelist is of type %s, should be a list' %
                      type(options.feature_whitelist).__name__)

    if options.schema is not None and not isinstance(options.schema,
                                                     schema_pb2.Schema):
      raise TypeError('schema is of type %s, should be a Schema proto.' % type(
          options.schema).__name__)

    if options.num_values_histogram_buckets < 1:
      raise ValueError('Invalid num_values_histogram_buckets %d' %
                       options.num_values_histogram_buckets)

    if options.num_histogram_buckets < 1:
      raise ValueError(
          'Invalid num_histogram_buckets %d' % options.num_histogram_buckets)

    if options.num_quantiles_histogram_buckets < 1:
      raise ValueError('Invalid num_quantiles_histogram_buckets %d' %
                       options.num_quantiles_histogram_buckets)

  def expand(self, dataset):
    # Initialize a list of stats generators to run.
    stats_generators = [
        # Create common stats generator.
        common_stats_generator.CommonStatsGenerator(
            schema=self._options.schema,
            num_values_histogram_buckets=\
                self._options.num_values_histogram_buckets,
            epsilon=self._options.epsilon),

        # Create numeric stats generator.
        numeric_stats_generator.NumericStatsGenerator(
            schema=self._options.schema,
            num_histogram_buckets=self._options.num_histogram_buckets,
            num_quantiles_histogram_buckets=\
                self._options.num_quantiles_histogram_buckets,
            epsilon=self._options.epsilon),

        # Create string stats generator.
        string_stats_generator.StringStatsGenerator(
            schema=self._options.schema),

        # Create topk stats generator.
        top_k_stats_generator.TopKStatsGenerator(
            schema=self._options.schema,
            num_top_values=self._options.num_top_values,
            num_rank_histogram_buckets=\
                self._options.num_rank_histogram_buckets),

        # Create uniques stats generator.
        uniques_stats_generator.UniquesStatsGenerator(
            schema=self._options.schema)
    ]
    if self._options.generators is not None:
      # Add custom stats generators.
      stats_generators.extend(self._options.generators)

    # Profile and then batch input examples.
    batched_dataset = (
        dataset
        | 'Profile' >> profile_util.Profile()
        | 'BatchInputs' >> batch_util.BatchExamples())

    # If a set of whitelist features are provided, keep only those features.
    filtered_dataset = batched_dataset
    if self._options.feature_whitelist:
      filtered_dataset = (
          batched_dataset | 'RemoveNonWhitelistedFeatures' >> beam.Map(
              _filter_features,
              feature_whitelist=self._options.feature_whitelist))

    return (filtered_dataset | 'RunStatsGenerators' >>
            stats_impl.GenerateStatisticsImpl(stats_generators))


def _filter_features(
    batch,
    feature_whitelist):
  """Remove features that are not whitelisted.

  Args:
    batch: A dict containing the input batch of examples.
    feature_whitelist: A list of feature names to whitelist.

  Returns:
    A dict containing only the whitelisted features of the input batch.
  """
  return {
      feature_name: batch[feature_name]
      for feature_name in feature_whitelist
      if feature_name in batch
  }
