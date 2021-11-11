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
import hashlib
from typing import Any, Iterable, Tuple, Union, FrozenSet, Mapping
import pyarrow as pa
from tensorflow_data_validation import types


class ColumnHasher(object):
  """Assigns column names to feature partitions."""

  def __init__(self, partitions: int):
    self.num_partitions = partitions

    # md5 is pretty slow compared to hash(), but we need stability, so we use
    # md5 and cache the result.
    # TODO(b/187961534): See if there's a faster stable hash available.
    self._cache = {}

  def assign(self, feature_name: Union[bytes, str]) -> int:
    """Assigns a feature partition based on the name of a feature."""
    if feature_name in self._cache:
      return self._cache[feature_name]
    if isinstance(feature_name, str):
      md5_hash = hashlib.md5(feature_name.encode('utf-8')).hexdigest()
    else:
      md5_hash = hashlib.md5(bytes(feature_name)).hexdigest()
    partition = int(md5_hash[0:8], 16) % self.num_partitions
    partition = (partition + int(md5_hash[8:], 16)) % self.num_partitions
    self._cache[feature_name] = partition
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
