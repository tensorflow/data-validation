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
"""Tests for feature_partition_util."""

from unittest import mock

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation.utils import feature_partition_util
from tensorflow_data_validation.utils import test_util


class FeaturePartitionUtilTest(absltest.TestCase):

  def test_splits_record_batch(self):
    feature1 = pa.array([1.0])
    feature2 = pa.array([2.0])
    feature3 = pa.array([3.0])
    record_batch = pa.RecordBatch.from_arrays([feature1, feature2, feature3],
                                              ['a', 'b', 'c'])
    sliced_record_batch = ('slice_key', record_batch)

    partitioner = mock.create_autospec(feature_partition_util.ColumnHasher(0))
    partitioner.assign.side_effect = [99, 43, 99]

    # Verify we saw the right features.
    partitions = list(
        feature_partition_util.generate_feature_partitions(
            sliced_record_batch, partitioner, frozenset([])))
    self.assertCountEqual(
        [mock.call('a'), mock.call('b'),
         mock.call('c')], partitioner.assign.call_args_list)

    # Verify we got the right output slices.
    expected = {
        ('slice_key', 99):
            pa.RecordBatch.from_arrays([feature1, feature3], ['a', 'c']),
        ('slice_key', 43):
            pa.RecordBatch.from_arrays([feature2], ['b']),
    }
    self.assertCountEqual(expected.keys(), [x[0] for x in partitions])
    for key, partitioned_record_batch in partitions:
      expected_batch = expected[key]
      test_util.make_arrow_record_batches_equal_fn(
          self, [expected_batch])([partitioned_record_batch])

  def test_splits_record_batch_with_universal_features(self):
    feature1 = pa.array([1.0])
    feature2 = pa.array([2.0])
    feature3 = pa.array([3.0])
    record_batch = pa.RecordBatch.from_arrays([feature1, feature2, feature3],
                                              ['a', 'b', 'c'])
    sliced_record_batch = ('slice_key', record_batch)

    partitioner = mock.create_autospec(feature_partition_util.ColumnHasher(0))
    partitioner.num_partitions = 4
    partitioner.assign.side_effect = [0, 1]

    # Verify we saw the right features.
    partitions = list(
        feature_partition_util.generate_feature_partitions(
            sliced_record_batch, partitioner, frozenset(['c'])))
    self.assertCountEqual(
        [mock.call('a'), mock.call('b')], partitioner.assign.call_args_list)

    # Verify we got the right output slices.
    expected = {
        ('slice_key', 0):
            pa.RecordBatch.from_arrays([feature1, feature3], ['a', 'c']),
        ('slice_key', 1):
            pa.RecordBatch.from_arrays([feature2, feature3], ['b', 'c']),
        ('slice_key', 2):
            pa.RecordBatch.from_arrays([feature3], ['c']),
        ('slice_key', 3):
            pa.RecordBatch.from_arrays([feature3], ['c']),
    }
    self.assertCountEqual(expected.keys(), [x[0] for x in partitions])
    for key, partitioned_record_batch in partitions:
      expected_batch = expected[key]
      test_util.make_arrow_record_batches_equal_fn(
          self, [expected_batch])([partitioned_record_batch])


class ColumnHasherTest(absltest.TestCase):

  def test_partitions_stable(self):
    column_names = ['rats', 'live', 'on', 'no', 'evil', 'star']
    # These values can be updated if the hasher changes.
    expected = [34, 16, 41, 14, 17, 40]
    hasher = feature_partition_util.ColumnHasher(44)
    for i, column_name in enumerate(column_names):
      self.assertEqual(expected[i], hasher.assign(column_name))


if __name__ == '__main__':
  absltest.main()
