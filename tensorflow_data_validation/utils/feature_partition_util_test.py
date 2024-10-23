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

from typing import Iterable, List, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation.utils import feature_partition_util
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


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

  def test_partitions_stable_strings(self):
    column_names = ['rats', 'live', 'on', 'no', 'evil', 'star']
    # These values can be updated if the hasher changes.
    expected = [14, 9, 28, 42, 3, 18]
    hasher = feature_partition_util.ColumnHasher(44)
    got = [hasher.assign(column_name) for column_name in column_names]
    self.assertEqual(expected, got)

  def test_partitions_stable_bytes(self):
    column_names = [b'rats', b'live', b'on', b'no', b'evil', b'star']
    # These values can be updated if the hasher changes.
    expected = [14, 9, 28, 42, 3, 18]
    hasher = feature_partition_util.ColumnHasher(44)
    got = [hasher.assign(column_name) for column_name in column_names]
    self.assertEqual(expected, got)


_BASE_PROTO_KEY_AND_SPLIT = """
          datasets: {
            name: 'abc'
            num_examples: 10
            weighted_num_examples: 3.4
            features: {
              path: {
                step: 'f1'
              }
            }
            features: {
              path: {
                step: 'f2'
              }
            }
            features: {
              path: {
                step: 'f3'
              }
            }
            cross_features: {
              path_x: {
                step: 'c1'
              }
              path_y: {
                step: 'c2'
              }
            }
          }
        """

_KEY_AND_SPLIT_TEST_CASES = [{
    'testcase_name': 'one_partition',
    'num_partitions': 1,
    'statistics': [_BASE_PROTO_KEY_AND_SPLIT],
    'expected': [(
        0,
        _BASE_PROTO_KEY_AND_SPLIT,
    )]
}, {
    'testcase_name':
        'two_partitions',
    'num_partitions':
        2,
    'statistics': [_BASE_PROTO_KEY_AND_SPLIT],
    'expected': [(0, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f1"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (0, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f2"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (0, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f3"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (1, """datasets {
  name: "abc"
  num_examples: 10
  weighted_num_examples: 3.4
  cross_features {
    path_x {
      step: "c1"
    }
    path_y {
      step: "c2"
    }
  }
}""")]
}, {
    'testcase_name':
        'many_partitions',
    'num_partitions':
        9999,
    'statistics': [_BASE_PROTO_KEY_AND_SPLIT],
    'expected': [(43, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f1"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (8454, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f2"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (316, """datasets {
  name: "abc"
  num_examples: 10
  features {
    path {
      step: "f3"
    }
  }
  weighted_num_examples: 3.4
}"""),
                 (2701, """datasets {
  name: "abc"
  num_examples: 10
  weighted_num_examples: 3.4
  cross_features {
    path_x {
      step: "c1"
    }
    path_y {
      step: "c2"
    }
  }
}""")]
}, {
    'testcase_name':
        'two_datasets_same_name_same_feature',
    'num_partitions':
        9999,
    'statistics': [
        """
        datasets: {
            name: 'abc'
            features: {
              path: {
                step: 'f1'
              }
            }
        }
        """, """
        datasets: {
            name: 'abc'
            features: {
              path: {
                step: 'f1'
              }
              type: STRING
            }
        }
        """
    ],
    'expected': [(43, """datasets {
  name: "abc"
  features {
    path {
      step: "f1"
    }
  }
}"""),
                 (43, """datasets {
  name: "abc"
  features {
    path {
      step: "f1"
    }
    type: STRING
  }
}""")]
}, {
    'testcase_name':
        'two_datasets_different_name_same_feature',
    'num_partitions':
        9999,
    'statistics': [
        """
        datasets: {
            name: 'abc'
            features: {
              path: {
                step: 'f1'
              }
            }
        }
        """, """
        datasets: {
            name: 'xyz'
            features: {
              path: {
                step: 'f1'
              }
            }
        }
        """
    ],
    'expected': [(43, """datasets {
  name: "abc"
  features {
    path {
      step: "f1"
    }
  }
}"""),
                 (6259, """datasets {
  name: "xyz"
  features {
    path {
      step: "f1"
    }
  }
}""")]
}, {
    'testcase_name':
        'does_not_crash_embedded_null_b236190177',
    'num_partitions':
        10,
    'statistics': [
        """
        datasets: {
            name: 'abc'
            features: {
              path: {
                step: '\x00'
              }
            }
        }
        """
    ],
    'expected': [(6, """
        datasets: {
            name: 'abc'
            features: {
              path: {
                step: '\x00'
              }
            }
        }
        """)]
}]


class KeyAndSplitByFeatureFnTest(parameterized.TestCase):

  @parameterized.named_parameters(_KEY_AND_SPLIT_TEST_CASES)
  def test_splits_statistics(
      self, num_partitions: int,
      statistics: List[statistics_pb2.DatasetFeatureStatisticsList],
      expected: List[Tuple[int, statistics_pb2.DatasetFeatureStatisticsList]]):
    statistics = list(
        text_format.Parse(s, statistics_pb2.DatasetFeatureStatisticsList())
        for s in statistics)
    expected = list(
        (x, text_format.Parse(s, statistics_pb2.DatasetFeatureStatisticsList()))
        for x, s in expected)

    def matcher(
        got: Iterable[Tuple[int, statistics_pb2.DatasetFeatureStatisticsList]]):
      self.assertCountEqual(got, expected)

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(statistics) | 'KeyAndSplit' >> beam.ParDo(
              feature_partition_util.KeyAndSplitByFeatureFn(num_partitions)))
      util.assert_that(result, matcher)

if __name__ == '__main__':
  absltest.main()
