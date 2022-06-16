# Copyright 2021 Google LLC
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
"""Utility for partitioning RecordBatches by features."""

import collections
from typing import Any, FrozenSet, Iterable, Mapping, Tuple, Union

import apache_beam as beam
import farmhash
import pyarrow as pa
from tensorflow_data_validation import types

from tensorflow_metadata.proto.v0 import statistics_pb2


class ColumnHasher(object):
  """Assigns column names to feature partitions."""

  def __init__(self, partitions: int):
    self.num_partitions = partitions

  def assign(self, feature_name: Union[bytes, str]) -> int:
    """Assigns a feature partition based on the name of a feature."""
    if isinstance(feature_name, bytes):
      feature_name = feature_name.decode('utf8')
    # TODO(b/236190177): Remove when binding is fixed.
    if '\x00' in feature_name:
      feature_name = feature_name.replace('\x00', '?')

    partition = farmhash.fingerprint32(feature_name) % self.num_partitions
    return partition

  def assign_sequence(self, *parts: Union[bytes, str]) -> int:
    """Assigns a feature partition based on a sequence of bytes or strings."""
    partition = 0
    for part in parts:
      partition += self.assign(part)
      partition = partition % self.num_partitions
    return partition

  def __eq__(self, o):
    return self.num_partitions == o.num_partitions


def generate_feature_partitions(
    sliced_record_batch: types.SlicedRecordBatch,
    partitioner: ColumnHasher,
    universal_features: FrozenSet[str],
) -> Iterable[Tuple[Tuple[types.SliceKey, int], pa.RecordBatch]]:
  """Partitions an input RecordBatch by feature name.

  The provided partitioner returns a value [0, k) deterministically for each
  feature name. Given an input containing multiple column names, up to k
  partitions are generated, with each partition containing the subset of
  features that were assigned by the provided partitioner to that value.


  Args:
    sliced_record_batch: An input RecordBatch. The slice-key of this input will
      be present in each output as part of the SliceKeyAndFeaturePartition.
    partitioner: A FeaturePartitioner instance.
    universal_features: Features that fall in every output partition.

  Yields:
    A sequence of partitions, each containing a subset of features.
  """
  slice_key = sliced_record_batch[0]
  partition_to_features = collections.defaultdict(
      lambda: ([], []))  # type: Mapping[int, Any]
  # Arrange output columns normal, universal, with columns within each in their
  # original order.
  for column_name, column in zip(sliced_record_batch[1].schema.names,
                                 sliced_record_batch[1].columns):
    if column_name in universal_features:
      continue
    entry = partition_to_features[partitioner.assign(column_name)]
    entry[0].append(column_name)
    entry[1].append(column)
  for column_name, column in zip(sliced_record_batch[1].schema.names,
                                 sliced_record_batch[1].columns):
    if column_name not in universal_features:
      continue
    for partition in range(partitioner.num_partitions):
      entry = partition_to_features[partition]
      entry[0].append(column_name)
      entry[1].append(column)

  for partition, features in partition_to_features.items():
    key = (slice_key, partition)
    column_names, columns = features
    yield (key, pa.RecordBatch.from_arrays(columns, column_names))


def _copy_with_no_features(
    statistics: statistics_pb2.DatasetFeatureStatistics
) -> statistics_pb2.DatasetFeatureStatistics:
  """Return a copy of 'statistics' with no features or cross-features."""
  return statistics_pb2.DatasetFeatureStatistics(
      name=statistics.name,
      num_examples=statistics.num_examples,
      weighted_num_examples=statistics.weighted_num_examples)


@beam.typehints.with_input_types(statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(
    beam.typehints.KV[int, statistics_pb2.DatasetFeatureStatisticsList])
class KeyAndSplitByFeatureFn(beam.DoFn):
  """Breaks a DatasetFeatureStatisticsList into shards keyed by partition index.

  Each partition index contains a random (but deterministic across workers)
  subset of features and cross features.
  """

  def __init__(self, num_partitions: int):
    """Initializes KeyAndSplitByFeatureFn.

    Args:
      num_partitions: The number of partitions to divide features/cross-features
        into. Must be >= 1.
    """
    if num_partitions < 1:
      raise ValueError('num_partitions must be >= 1.')
    if num_partitions != 1:
      self._hasher = ColumnHasher(num_partitions)
    else:
      self._hasher = None

  def process(self, statistics: statistics_pb2.DatasetFeatureStatisticsList):
    # If the number of partitions is one, or there are no datasets, yield the
    # full statistics proto with a placeholder key.
    if self._hasher is None or not statistics.datasets:
      yield (0, statistics)
      return
    for dataset in statistics.datasets:
      for feature in dataset.features:
        if feature.name:
          partition = self._hasher.assign_sequence(dataset.name, feature.name)
        else:
          partition = self._hasher.assign_sequence(dataset.name,
                                                   *feature.path.step)
        dataset_copy = _copy_with_no_features(dataset)
        dataset_copy.features.append(feature)
        yield (partition,
               statistics_pb2.DatasetFeatureStatisticsList(
                   datasets=[dataset_copy]))
      for cross_feature in dataset.cross_features:
        partition = self._hasher.assign_sequence(dataset.name,
                                                 *cross_feature.path_x.step,
                                                 *cross_feature.path_y.step)
        dataset_copy = _copy_with_no_features(dataset)
        dataset_copy.cross_features.append(cross_feature)
        yield (partition,
               statistics_pb2.DatasetFeatureStatisticsList(
                   datasets=[dataset_copy]))
      # If there were no features or cross-features, yield the dataset itself
      # into shard 0 to ensure it's not dropped entirely.
      if not dataset.features and not dataset.cross_features:
        yield (0,
               statistics_pb2.DatasetFeatureStatisticsList(datasets=[dataset]))
