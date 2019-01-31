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
"""Module that estimates a custom non-streaming statistic.

A Beam transform will compute the custom statistic over multiple samples of the
data to estimate the true value of the statistic over the entire dataset.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from collections import defaultdict
from functools import partial
import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import batch_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Dict, List, Text, Tuple

from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(b/117270483): Memory concerns when accumulating ExampleBatches in future
# call to beam.transforms.combiners.ToList.


# Returns the number of examples in the batch.
def _num_examples(batch):
  return len(six.next(six.itervalues(batch))) if batch else 0


# Merges all of the batches in the accumulator into one batch.
def _flatten_examples(batches):
  """Flattens the list of ExampleBatches into one ExampleBatch.

  Args:
    batches: A list of ExampleBatches.

  Returns:
    A single ExampleBatch.
  """

  unique_features = set()
  for batch in batches:
    unique_features.update(batch.keys())

  num_examples_per_batch = [_num_examples(batch) for batch in batches]
  result = {}
  for feature_name in unique_features:
    feature_values = []
    for i, batch in enumerate(batches):
      if feature_name in batch:
        feature_values.extend(batch[feature_name])
      else:
        feature_values.extend([None] * num_examples_per_batch[i])
    result[feature_name] = np.array(feature_values)
  return result


# TODO(b/117937992): Seed RNG so MI and partitioner are determenistic in test
def _assign_to_partition(example, num_partitions
                        ):
  """Assigns an example to a partition key."""
  return np.random.randint(num_partitions), example


def _get_partitioned_statistics_summary(
    statistics
):
  """Computes meta-statistics over the custom stats in the input dict."""

  summary = defaultdict(defaultdict)
  for feature_name, feature_statistics in statistics.items():
    for stat_name, stat_values in feature_statistics.items():
      summary[feature_name]['min_' + stat_name] = np.min(stat_values)
      summary[feature_name]['max_' + stat_name] = np.max(stat_values)
      summary[feature_name]['mean_' + stat_name] = np.mean(stat_values)
      summary[feature_name]['median_' + stat_name] = np.median(stat_values)
      summary[feature_name]['std_dev_' + stat_name] = np.std(stat_values)
      summary[feature_name]['num_partitions_' + stat_name] = stat_values.size
  return summary


def get_valid_statistics(
    statistics,
    min_partitions_stat_presence
):
  """Filters out statistics that were not computed over all partitions."""
  valid_statistics = defaultdict(defaultdict)
  for feature_name, feature_statistics in statistics.items():
    for stat_name, stat_values in feature_statistics.items():
      # Only keep statistics that appear min_partitions_stat_presence times
      if len(stat_values) >= min_partitions_stat_presence:
        valid_statistics[feature_name][stat_name] = np.array(stat_values)
  return valid_statistics


class PartitionedStatsFn(object):
  """A custom non-streaming statistic.

  A PartitionedStatsFn is a custom statistic that cannot be computed in a
  streaming fashion. A user is required to implement the compute function.

  NonStreamingCustomStatsGenerator are initialized with
  a PartitionedStatsFn to estimate the PartitionedStatsFn over a large dataset.
  Examples in the dataset will be randomly assigned to a partition. Then the
  compute method will be called on each partition. If the examples in the
  partition contain invalid feature values, implementations of
  PartitionedStatsFn also have the option to "gracefully fail" without returning
  a statistic value for any invalid features.
  """

  def compute(self, examples
             ):
    """Computes custom statistics over the batch of examples.

    Args:
      examples: The batch of examples.

    Returns:
      DatasetFeatureStatistics containing the custom statistics for
      each feature in the dataset.

      The DatasetFeatureStatistics proto can be constructed using the
      make_dataset_feature_stats_proto method.
    """
    raise NotImplementedError()


class _PartitionedStatisticsAnalyzerAccumulator(object):
  """Holds the partial state of partitioned statistics summaries."""

  def __init__(self):
    # A partial is used so that the class is pickleable.
    self.statistics = defaultdict(partial(defaultdict, list))


class PartitionedStatisticsAnalyzer(beam.CombineFn):
  """Computes meta-statistics for non-streaming partitioned statistics.

  This analyzer computes meta-statistics including the min, max, mean, median
  and std dev of numeric statistics that are calculated over partitions
  of the dataset. Statistics may be missing from some partitions if
  the partition contains invalid feature values causing PartitionedStatsFn to
  "gracefully fail". Meta-statistics for a feature are only calculated if the
  number of partitions in which the statistic is computed passes a configurable
  threshold.
  """

  def __init__(self, min_partitions_stat_presence):
    """Initializes the analyzer."""

    # Meta-stats are only computed if a stat is found in at least
    # min_partitions_stat_presence number of partitions.
    self._min_partitions_stat_presence = min_partitions_stat_presence

  def create_accumulator(self):
    """Creates an accumulator, which stores partial state of meta-statistics."""

    return _PartitionedStatisticsAnalyzerAccumulator()

  def add_input(self, accumulator,
                statistic
               ):
    """Adds the input (DatasetFeatureStatistics) into the accumulator."""

    for feature in statistic.features:
      for stat in feature.custom_stats:
        accumulator.statistics[feature.name][stat.name].append(stat.num)
    return accumulator

  def merge_accumulators(
      self, accumulators
  ):
    """Merges together a list of PartitionedStatisticsAnalyzerAccumulators."""

    result = _PartitionedStatisticsAnalyzerAccumulator()
    for accumulator in accumulators:
      for feature_name, feature_statistics in accumulator.statistics.items():
        for stat_name, stat_values in feature_statistics.items():
          result.statistics[feature_name][stat_name].extend(stat_values)
    return result

  def extract_output(self,
                     accumulator
                    ):
    """Returns meta-statistics as a DatasetFeatureStatistics proto."""

    valid_stats_summary = _get_partitioned_statistics_summary(
        get_valid_statistics(accumulator.statistics,
                             self._min_partitions_stat_presence))
    return stats_util.make_dataset_feature_stats_proto(valid_stats_summary)


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.Example)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatistics)
class _GenerateNonStreamingCustomStats(beam.PTransform):
  """A beam.PTransform that implements NonStreamingCustomStatsGenerator."""

  def __init__(self, stats_fn,
               num_partitions, min_partitions_stat_presence,
               max_examples_per_partition, seed, name):
    """Initializes _GenerateNonStreamingCustomStats."""

    self._stats_fn = stats_fn
    self._num_partitions = num_partitions
    self._min_partitions_stat_presence = min_partitions_stat_presence
    self._name = name
    self._seed = seed
    self._max_examples_per_partition = max_examples_per_partition
    # Seeds the random number generator used in the partitioner.
    np.random.seed(self._seed)

  def expand(self, pcoll):
    """Estimates the user defined statistic."""

    return (
        pcoll
        | 'AssignExampleToPartition' >> beam.Map(
            _assign_to_partition, num_partitions=self._num_partitions)
        | 'GroupPartitionsIntoList' >> beam.CombinePerKey(
            beam.combiners.SampleCombineFn(self._max_examples_per_partition))
        | 'RemovePartitionKey' >> beam.Values()
        | 'BatchExamples' >> beam.Map(batch_util.merge_single_batch)
        | 'ComputeStatsFn' >> beam.Map(self._stats_fn.compute)
        | 'ComputeMetaStats' >> beam.CombineGlobally(
            PartitionedStatisticsAnalyzer(min_partitions_stat_presence=self
                                          ._min_partitions_stat_presence)))


class NonStreamingCustomStatsGenerator(stats_generator.TransformStatsGenerator):
  """Estimates custom statistics in a non-streaming fashion.

  A TransformStatsGenerator which partitions the input data and calls the user
  specified stats_fn over each partition. Meta-statistics are calculated over
  the statistics returned by stats_fn to estimate the true value of the
  statistic. For invalid feature values, the worker computing PartitionedStatsFn
  over a partition may "gracefully fail" and not report that statistic (refer to
  PartitionedStatsFn for more information). Meta-statistics for a feature are
  only calculated if the number of partitions where the statistic is computed
  exceeds a configurable threshold.

  A large number of examples in a partition may result in worker OOM errors.
  This can be prevented by setting max_examples_per_partition.
  """

  def __init__(
      self,
      stats_fn,
      num_partitions,
      min_partitions_stat_presence,
      seed,
      max_examples_per_partition,
      name = 'NonStreamingCustomStatsGenerator'):
    """Initializes NonStreamingCustomStatsGenerator.

    Args:
      stats_fn: The PartitionedStatsFn that will be run on each sample.
      num_partitions: The number of partitions the stat will be calculated on.
      min_partitions_stat_presence: The minimum number of partitions a stat
        computation must succeed in for the result to be returned.
      seed: An int used to seed the numpy random number generator.
      max_examples_per_partition: An integer used to specify the maximum
        number of examples per partition to limit memory usage in a worker. If
        the number of examples per partition exceeds this value, the examples
        are randomly selected.
      name: An optional unique name associated with the statistics generator.
    """

    super(NonStreamingCustomStatsGenerator, self).__init__(
        name=name,
        ptransform=_GenerateNonStreamingCustomStats(
            stats_fn=stats_fn,
            num_partitions=num_partitions,
            min_partitions_stat_presence=min_partitions_stat_presence,
            seed=seed,
            max_examples_per_partition=max_examples_per_partition,
            name=name
            ))
