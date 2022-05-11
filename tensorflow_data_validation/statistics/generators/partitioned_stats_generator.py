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

import collections
import functools
from typing import Dict, Iterable, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import table_util

from tensorflow_metadata.proto.v0 import statistics_pb2


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


def _default_assign_to_partition(
    sliced_record_batch: types.SlicedRecordBatch,
    num_partitions: int) -> Tuple[Tuple[types.SliceKey, int], pa.RecordBatch]:
  """Assigns an example to a partition key."""
  slice_key, record_batch = sliced_record_batch
  return (slice_key, np.random.randint(num_partitions)), record_batch


@beam.typehints.with_input_types(types.SlicedRecordBatch)
@beam.typehints.with_output_types(Tuple[Tuple[types.SliceKey, int],
                                        pa.RecordBatch])
@beam.ptransform_fn
def _DefaultPartitionTransform(pcol, num_partitions):  # pylint: disable=invalid-name
  """Ptransform wrapping _default_assign_to_partition."""
  return pcol | 'DefaultPartition' >> beam.Map(_default_assign_to_partition,
                                               num_partitions)


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

  def compute(self, examples: pa.RecordBatch
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

  def partitioner(self, num_partitions: int) -> beam.PTransform:
    """Optional PTransform to perform partition assignment.

    This may be overridden by subclasses to return a PTransform matching the
    signature of _default_partition_transform, which will be used if this method
    returns None.

    Args:
      num_partitions: The number of partitions to use. Overriding subclasses are
        free to use a different number of partitions.

    Returns:
      A PTransform.
    """
    return _DefaultPartitionTransform(num_partitions)  # pylint: disable=no-value-for-parameter


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
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
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


class _SampleRecordBatchRowsAccumulator(object):
  """Accumulator to keep track of the current (top-k) sample of records."""

  __slots__ = [
      'record_batches', 'curr_num_rows', 'curr_byte_size', 'random_ints'
  ]

  def __init__(self):
    # Record batches to sample.
    self.record_batches = []

    # The total number of rows (examples) in all of `record_batches`.
    self.curr_num_rows = 0

    # Current total byte size of all the pa.RecordBatches accumulated.
    self.curr_byte_size = 0

    # This is a list of numpy array of random integers. Each element maps to one
    # row in each record batch. Each row should only be assigned a random number
    # once, in order to avoid sampling bias. Thus, we need to preserve the
    # assigned number for each accumulator, across multiple `compacts`.
    self.random_ints = []


# TODO(b/192393883): move this to tfx_bsl.
@beam.typehints.with_input_types(pa.RecordBatch)
@beam.typehints.with_output_types(pa.RecordBatch)
class _SampleRecordBatchRows(beam.CombineFn):
  """Samples rows from record batches.

  The record batches in the partition can vary in the number of rows.
  SamplePartition guarantees that the sample returned is always going to be
  <= sample_size.

  The actual sampling occurs in `compact`. It uses np.partition to calculate
  the top-k of record batch's rows. Where the top-k is a random number assigned
  to each row. Given a uniform distribution of the random number, we can keep a
  running sample of the partition of size k. This gives each row an equal
  probability of being selected.
  """

  _BUFFER_SIZE_SCALAR = 5

  def __init__(self, sample_size: int):
    """Initializes the analyzer."""
    self._sample_size = sample_size
    # Number of record batches in accumulator when compacting.
    self._combine_num_record_batches = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE,
        'sample_record_batch_rows_combine_num_record_batches')
    self._combine_num_columns = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'sample_record_batch_num_columns')
    # Post compress byte size.
    self._combine_byte_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE,
        'sample_record_batch_rows_combine_byte_size')
    # Number of compacts.
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'sample_record_batch_rows_num_compacts')
    # Total number of rows.
    self._num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'sample_record_batch_rows_num_instances')

    # We allow our accumulators to keep a buffer of _BUFFER_SIZE_SCALAR x sample
    # size. With this threshold, OOM issues are possible, but unlikely.
    self._merge_record_batch_threshold = self._BUFFER_SIZE_SCALAR * sample_size

  def create_accumulator(self) -> _SampleRecordBatchRowsAccumulator:
    """Creates an accumulator."""
    return _SampleRecordBatchRowsAccumulator()

  def add_input(
      self, accumulator: _SampleRecordBatchRowsAccumulator,
      record_batch: pa.RecordBatch) -> _SampleRecordBatchRowsAccumulator:
    """Adds the input into the accumulator."""
    num_rows = record_batch.num_rows
    self._num_instances.inc(num_rows)
    self._combine_num_columns.update(len(record_batch.columns))
    accumulator.record_batches.append(record_batch)
    accumulator.curr_num_rows += num_rows
    accumulator.curr_byte_size += record_batch.nbytes

    curr_random_ints = np.random.randint(
        0,
        np.iinfo(np.int64).max,
        dtype=np.int64,
        size=(num_rows,))
    accumulator.random_ints.append(curr_random_ints)

    if accumulator.curr_num_rows > self._merge_record_batch_threshold:
      accumulator = self._compact_impl(accumulator)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_SampleRecordBatchRowsAccumulator]
  ) -> _SampleRecordBatchRowsAccumulator:
    """Merges together a list of _SampleRecordBatchRowsAccumulator."""
    result = _SampleRecordBatchRowsAccumulator()

    for acc in accumulators:
      result.record_batches.extend(acc.record_batches)
      result.curr_num_rows += acc.curr_num_rows
      result.curr_byte_size += acc.curr_byte_size
      result.random_ints.extend(acc.random_ints)
      # Compact if we are over the threshold.
      if result.curr_num_rows > self._merge_record_batch_threshold:
        result = self._compact_impl(result)

    result = self._compact_impl(result)
    return result

  def compact(
      self, accumulator: _SampleRecordBatchRowsAccumulator
  ) -> _SampleRecordBatchRowsAccumulator:
    return self._compact_impl(accumulator)

  def extract_output(self,
                     accumulator: _SampleRecordBatchRowsAccumulator
                    ) -> pa.RecordBatch:
    """Returns the sample as a record batch."""
    # We force the compact, to comply with the contract of outputting one record
    # batch.
    acc = self._compact_impl(accumulator)
    assert len(acc.record_batches) == 1
    return acc.record_batches[0]

  def _compact_impl(
      self, accumulator: _SampleRecordBatchRowsAccumulator
  ) -> _SampleRecordBatchRowsAccumulator:
    """Compacts the accumulator.

    This compact selects samples rows from the record batch, and merges them
    into one record batch. We can then clear the cache of all record batches
    seen so far. If the accumulator holds too few record batches, then nothing
    will be compacted.

    The sampling is done by assigning each row in the record batch a random
    number. Then we choose the top-k of the random numbers to get a sample of
    size k.

    Args:
      accumulator: The _SampleRecordBatchRowsAccumulator to compact.

    Returns:
      A _SampleRecordBatchRowsAccumulator that contains one or a list of record
      batch.
    """
    self._combine_num_record_batches.update(len(accumulator.record_batches))

    # There is nothing to compact.
    if accumulator.curr_num_rows <= 1:
      return accumulator

    # There is no need to compact yet.
    if (len(accumulator.record_batches) <= 1 and
        accumulator.curr_num_rows <= self._sample_size):
      return accumulator
    self._num_compacts.inc(1)
    k = min(self._sample_size, accumulator.curr_num_rows)

    rand_ints = np.concatenate(accumulator.random_ints)

    # Find the value that is the breakpoint for the top-k.
    kth_value = np.partition(rand_ints, k - 1)[k - 1]

    # This mask will always have >= 1 Trues.
    equals_to_kth = (rand_ints == kth_value)

    # This mask will always have < k Trues.
    less_than_kth = rand_ints < kth_value

    # Since there may be duplicate values, `equals_to_kth + less_than_kth` might
    # be greater than `k`. We need to keep track of how many to add, without
    # surpassing `k`.
    kth_to_add = k - np.sum(less_than_kth)

    # Preserve the random integers that we had assigned to each row.
    sample_random_ints = rand_ints[rand_ints <= kth_value][:k]

    beg = 0
    sample_indices = []
    for rb in accumulator.record_batches:
      size = rb.num_rows
      end = beg + size
      less_than_kth_indices = np.nonzero(less_than_kth[beg:end])[0]
      indices = less_than_kth_indices

      # Add indices of any duplicate values that are equal to `k`.
      if kth_to_add > 0:
        equals_to_kth_indices = np.nonzero(equals_to_kth[beg:end])[0]
        if equals_to_kth_indices.size > 0:
          if equals_to_kth_indices.size >= kth_to_add:
            indices = np.concatenate(
                [less_than_kth_indices, equals_to_kth_indices[:kth_to_add]])
            kth_to_add = 0
          else:
            indices = np.concatenate(
                [less_than_kth_indices, equals_to_kth_indices])
            kth_to_add -= equals_to_kth_indices.size

      sample_indices.append(indices)
      beg += size

    result = _SampleRecordBatchRowsAccumulator()

    # Take and merge the record batches, based on the sampled indices.
    rbs = []
    for rb, indices in zip(accumulator.record_batches, sample_indices):
      rbs.append(table_util.RecordBatchTake(rb, pa.array(indices)))
    compressed_rb = table_util.MergeRecordBatches(rbs)
    result.record_batches = [compressed_rb]
    result.curr_num_rows = compressed_rb.num_rows
    result.curr_byte_size = compressed_rb.nbytes
    result.random_ints = [sample_random_ints]

    self._combine_byte_size.update(result.curr_byte_size)

    return result


def _process_partition(
    partition: Tuple[Tuple[types.SliceKey, int], pa.RecordBatch],
    stats_fn: PartitionedStatsFn
) -> Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]:
  """Process batch in a single partition."""
  (slice_key, _), record_batch = partition
  return slice_key, stats_fn.compute(record_batch)


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
    self._max_examples_per_partition = max_examples_per_partition

    # Seeds the random number generator used in the partitioner.
    np.random.seed(self._seed)

  def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    """Estimates the user defined statistic."""

    return (pcoll
            | 'AssignBatchToPartition' >> self._stats_fn.partitioner(
                self._num_partitions)
            | 'GroupPartitionsIntoList' >> beam.CombinePerKey(
                _SampleRecordBatchRows(self._max_examples_per_partition))
            | 'ProcessPartition' >> beam.Map(
                _process_partition, stats_fn=self._stats_fn)
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
