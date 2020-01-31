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

import collections
import functools
import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import table_util
from typing import Dict, Iterable, List, Text, Tuple

from tensorflow_metadata.proto.v0 import statistics_pb2


def _assign_to_partition(sliced_table: types.SlicedTable,
                         num_partitions: int
                        ) -> Tuple[Tuple[types.SliceKey, int], pa.Table]:
  """Assigns an example to a partition key."""
  slice_key, table = sliced_table
  return (slice_key, np.random.randint(num_partitions)), table


def _get_partitioned_statistics_summary(
    statistics: Dict[types.FeaturePath, Dict[Text, np.ndarray]]
) -> Dict[types.FeaturePath, Dict[Text, float]]:
  """Computes meta-statistics over the custom stats in the input dict."""

  summary = collections.defaultdict(collections.defaultdict)
  for feature_path, feature_statistics in statistics.items():
    summary_for_feature = summary[feature_path]
    for stat_name, stat_values in feature_statistics.items():
      summary_for_feature['min_' + stat_name] = np.min(stat_values)
      summary_for_feature['max_' + stat_name] = np.max(stat_values)
      summary_for_feature['mean_' + stat_name] = np.mean(stat_values)
      summary_for_feature['median_' + stat_name] = np.median(stat_values)
      summary_for_feature['std_dev_' + stat_name] = np.std(stat_values)
      summary_for_feature['num_partitions_' + stat_name] = stat_values.size
  return summary


def get_valid_statistics(
    statistics: Dict[types.FeaturePath, Dict[Text, np.ndarray]],
    min_partitions_stat_presence: int
) -> Dict[types.FeaturePath, Dict[Text, np.ndarray]]:
  """Filters out statistics that were not computed over all partitions."""
  valid_statistics = collections.defaultdict(collections.defaultdict)
  for feature_path, feature_statistics in statistics.items():
    for stat_name, stat_values in feature_statistics.items():
      # Only keep statistics that appear min_partitions_stat_presence times
      if len(stat_values) >= min_partitions_stat_presence:
        valid_statistics[feature_path][stat_name] = np.array(stat_values)
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

  def compute(self, examples: types.ExampleBatch
             ) -> statistics_pb2.DatasetFeatureStatistics:
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
    self.statistics = collections.defaultdict(
        functools.partial(collections.defaultdict, list))


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

  def __init__(self, min_partitions_stat_presence: int):
    """Initializes the analyzer."""

    # Meta-stats are only computed if a stat is found in at least
    # min_partitions_stat_presence number of partitions.
    self._min_partitions_stat_presence = min_partitions_stat_presence

  def create_accumulator(self) -> _PartitionedStatisticsAnalyzerAccumulator:
    """Creates an accumulator, which stores partial state of meta-statistics."""

    return _PartitionedStatisticsAnalyzerAccumulator()

  def add_input(self, accumulator: _PartitionedStatisticsAnalyzerAccumulator,
                statistic: statistics_pb2.DatasetFeatureStatistics
               ) -> _PartitionedStatisticsAnalyzerAccumulator:
    """Adds the input (DatasetFeatureStatistics) into the accumulator."""

    for feature in statistic.features:
      for stat in feature.custom_stats:
        accumulator.statistics[
            types.FeaturePath.from_proto(feature.path)][stat.name].append(
                stat.num)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_PartitionedStatisticsAnalyzerAccumulator]
  ) -> _PartitionedStatisticsAnalyzerAccumulator:
    """Merges together a list of PartitionedStatisticsAnalyzerAccumulators."""

    result = _PartitionedStatisticsAnalyzerAccumulator()
    for accumulator in accumulators:
      for feature_path, feature_statistics in accumulator.statistics.items():
        for stat_name, stat_values in feature_statistics.items():
          result.statistics[feature_path][stat_name].extend(stat_values)
    return result

  def extract_output(self,
                     accumulator: _PartitionedStatisticsAnalyzerAccumulator
                    ) -> statistics_pb2.DatasetFeatureStatistics:
    """Returns meta-statistics as a DatasetFeatureStatistics proto."""

    valid_stats_summary = _get_partitioned_statistics_summary(
        get_valid_statistics(accumulator.statistics,
                             self._min_partitions_stat_presence))
    return stats_util.make_dataset_feature_stats_proto(valid_stats_summary)


def _process_partition(
    partition: Tuple[Tuple[types.SliceKey, int], List[pa.Table]],
    stats_fn: PartitionedStatsFn
) -> Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]:
  """Process batches in a single partition."""
  (slice_key, _), tables = partition
  return slice_key, stats_fn.compute(table_util.MergeTables(tables))


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.SlicedExample)
@beam.typehints.with_output_types(
    Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics])
class _GenerateNonStreamingCustomStats(beam.PTransform):
  """A beam.PTransform that implements NonStreamingCustomStatsGenerator."""

  def __init__(self, stats_fn: PartitionedStatsFn,
               num_partitions: int, min_partitions_stat_presence: int,
               seed: int, max_examples_per_partition: int, batch_size: int,
               name: Text) -> None:
    """Initializes _GenerateNonStreamingCustomStats."""

    self._stats_fn = stats_fn
    self._num_partitions = num_partitions
    self._min_partitions_stat_presence = min_partitions_stat_presence
    self._name = name
    self._seed = seed
    self._max_batches_per_partition = int(max_examples_per_partition /
                                          batch_size)
    # Seeds the random number generator used in the partitioner.
    np.random.seed(self._seed)

  def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    """Estimates the user defined statistic."""

    return (
        pcoll
        | 'AssignBatchToPartition' >> beam.Map(
            _assign_to_partition, num_partitions=self._num_partitions)
        | 'GroupPartitionsIntoList' >> beam.CombinePerKey(
            beam.combiners.SampleCombineFn(self._max_batches_per_partition))
        | 'ProcessPartition' >> beam.Map(_process_partition,
                                         stats_fn=self._stats_fn)
        | 'ComputeMetaStats' >> beam.CombinePerKey(
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
      stats_fn: PartitionedStatsFn,
      num_partitions: int,
      min_partitions_stat_presence: int,
      seed: int,
      max_examples_per_partition: int,
      batch_size: int = 1000,
      name: Text = 'NonStreamingCustomStatsGenerator') -> None:
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
      batch_size: Number of examples per input batch.
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
            batch_size=batch_size,
            name=name
            ))
