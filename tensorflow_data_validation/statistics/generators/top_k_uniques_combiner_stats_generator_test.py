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
"""Tests for TopK and Uniques statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import top_k_uniques_combiner_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class TopKUniquesCombinerStatsGeneratorTest(
    test_util.CombinerStatsGeneratorTest):
  """Tests for TopKUniquesCombinerStatsGenerator."""

  def test_topk_uniques_combiner_with_single_bytes_feature(self):
    # 'fa': 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], ['a', 'c', 'd', 'a']],
                     type=pa.list_(pa.binary()))
        ], ['fa']),
        pa.Table.from_arrays(
            [pa.array([['a', 'b', 'c', 'd']], type=pa.list_(pa.binary()))],
            ['fa'])
    ]
    # Note that if two feature values have the same frequency, the one with the
    # lexicographically larger feature value will be higher in the order.
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 5
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 4.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "c"
              sample_count: 3.0
            }
            buckets {
              low_rank: 2
              high_rank: 2
              label: "d"
              sample_count: 2.0
            }
          }
      }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_weights(self):
    # non-weighted ordering
    # 3 'a', 2 'e', 2 'd', 2 'c', 1 'b'
    # weighted ordering
    # fa: 20 'e', 20 'd', 15 'a', 10 'c', 5 'b'
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], ['a', 'c', 'd', 'a']]),
            pa.array([[5.0], [5.0]]),
        ], ['fa', 'w']),
        pa.Table.from_arrays([
            pa.array([['d', 'e']]),
            pa.array([[15.0]]),
        ], ['fa', 'w']),
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: STRING
                string_stats {
                  unique: 5
                  top_values {
                    value: 'a'
                    frequency: 3.0
                  }
                  top_values {
                    value: 'e'
                    frequency: 2.0
                  }
                  top_values {
                    value: 'd'
                    frequency: 2.0
                  }
                  top_values {
                    value: 'c'
                    frequency: 2.0
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "a"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "e"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "d"
                      sample_count: 2.0
                    }
                  }
                  weighted_string_stats {
                    top_values {
                      value: 'e'
                      frequency: 20.0
                    }
                    top_values {
                      value: 'd'
                      frequency: 20.0
                    }
                    top_values {
                      value: 'a'
                      frequency: 15.0
                    }
                    top_values {
                      value: 'c'
                      frequency: 10.0
                    }
                    rank_histogram {
                      buckets {
                        low_rank: 0
                        high_rank: 0
                        label: "e"
                        sample_count: 20.0
                      }
                      buckets {
                        low_rank: 1
                        high_rank: 1
                        label: "d"
                        sample_count: 20.0
                      }
                      buckets {
                        low_rank: 2
                        high_rank: 2
                        label: "a"
                        sample_count: 15.0
                      }
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            weight_feature='w', num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_single_unicode_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    batches = [
        pa.Table.from_arrays(
            [pa.array([[u'a', u'b', u'c', u'e'], [u'a', u'c', u'd', u'a']])],
            ['fa']),
        pa.Table.from_arrays([pa.array([[u'a', u'b', u'c', u'd']])], ['fa']),
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: STRING
                string_stats {
                  unique: 5
                  top_values {
                    value: 'a'
                    frequency: 4
                  }
                  top_values {
                    value: 'c'
                    frequency: 3
                  }
                  top_values {
                    value: 'd'
                    frequency: 2
                  }
                  top_values {
                    value: 'b'
                    frequency: 2
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "a"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "d"
                      sample_count: 2.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_multiple_features(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 2 'b', 3 'c'
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd']]),
            pa.array([['a', 'c', 'c'], ['b'], None]),
        ], ['fa', 'fb']),
        pa.Table.from_arrays([
            pa.array([['a', 'a', 'b', 'c', 'd'], None]),
            pa.array([None, ['b', 'c']])
        ], ['fa', 'fb']),
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: STRING
                string_stats {
                  unique: 5
                  top_values {
                    value: 'a'
                    frequency: 4
                  }
                  top_values {
                    value: 'c'
                    frequency: 3
                  }
                  top_values {
                    value: 'd'
                    frequency: 2
                  }
                  top_values {
                    value: 'b'
                    frequency: 2
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "a"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "d"
                      sample_count: 2.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['fb']):
            text_format.Parse(
                """
                path {
                  step: 'fb'
                }
                type: STRING
                string_stats {
                  unique: 3
                  top_values {
                    value: 'c'
                    frequency: 3
                  }
                  top_values {
                    value: 'b'
                    frequency: 2
                  }
                  top_values {
                    value: 'a'
                    frequency: 1
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "c"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "b"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "a"
                      sample_count: 1.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_zero_row(self):
    batches = [
        pa.Table.from_arrays([pa.array([], type=pa.list_(pa.binary()))],
                             ['f1'])
    ]
    expected_result = {}
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiners_empty_table(self):
    batches = [pa.Table.from_arrays([], [])]
    expected_result = {}
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_missing_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 1 'b', 2 'c'
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd']]),
            pa.array([['a', 'c', 'c'], ['b'], None]),
        ], ['fa', 'fb']),
        pa.Table.from_arrays([
            pa.array([['a', 'a', 'b', 'c', 'd'], None]),
        ], ['fa'])
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: STRING
                string_stats {
                  unique: 5
                  top_values {
                    value: 'a'
                    frequency: 4
                  }
                  top_values {
                    value: 'c'
                    frequency: 3
                  }
                  top_values {
                    value: 'd'
                    frequency: 2
                  }
                  top_values {
                    value: 'b'
                    frequency: 2
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "a"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "d"
                      sample_count: 2.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['fb']):
            text_format.Parse(
                """
                path {
                  step: 'fb'
                }
                type: STRING
                string_stats {
                  unique: 3
                  top_values {
                    value: 'c'
                    frequency: 2
                  }
                  top_values {
                    value: 'b'
                    frequency: 1
                  }
                  top_values {
                    value: 'a'
                    frequency: 1
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "c"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "b"
                      sample_count: 1.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "a"
                      sample_count: 1.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_numeric_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd']]),
            pa.array([[1.0, 2.0, 3.0], [4.0, 5.0], None]),
        ], ['fa', 'fb']),
        pa.Table.from_arrays([
            pa.array([['a', 'a', 'b', 'c', 'd']]),
            pa.array([None], type=pa.list_(pa.float32())),
        ], ['fa', 'fb']),
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: STRING
                string_stats {
                  unique: 5
                  top_values {
                    value: 'a'
                    frequency: 4
                  }
                  top_values {
                    value: 'c'
                    frequency: 3
                  }
                  top_values {
                    value: 'd'
                    frequency: 2
                  }
                  top_values {
                    value: 'b'
                    frequency: 2
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "a"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "d"
                      sample_count: 2.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_categorical_feature(self):
    # fa: 4 12, 2 23, 2 34, 2 45
    batches = [
        pa.Table.from_arrays([pa.array([[12, 23, 34, 12], [45, 23]])], ['fa']),
        pa.Table.from_arrays([pa.array([[12, 12, 34, 45]])], ['fa']),
        pa.Table.from_arrays(
            [pa.array([None, None, None, None], type=pa.null())], ['fa']),
    ]
    expected_result = {
        types.FeaturePath(['fa']):
            text_format.Parse(
                """
                path {
                  step: 'fa'
                }
                type: INT
                string_stats {
                  unique: 4
                  top_values {
                    value: '12'
                    frequency: 4
                  }
                  top_values {
                    value: '45'
                    frequency: 2
                  }
                  top_values {
                    value: '34'
                    frequency: 2
                  }
                  top_values {
                    value: '23'
                    frequency: 2
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "12"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "45"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "34"
                      sample_count: 2.0
                    }
                  }
              }""", statistics_pb2.FeatureNameStatistics())
    }
    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            schema=schema, num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_with_frequency_threshold(self):
    batches = [
        pa.Table.from_arrays([
            pa.array([['a', 'b', 'y', 'b']]),
            pa.array([[5.0]]),
        ], ['fa', 'w']),
        pa.Table.from_arrays([
            pa.array([['a', 'x', 'a', 'z']]),
            pa.array([[15.0]]),
        ], ['fa', 'w'])
    ]
    expected_result = {
        types.FeaturePath(['fa']): text_format.Parse("""
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 5
          top_values {
            value: 'a'
            frequency: 3
          }
          top_values {
            value: 'b'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "b"
              sample_count: 2.0
            }
          }
          weighted_string_stats {
            top_values {
              value: 'a'
              frequency: 35.0
            }
            top_values {
              value: 'z'
              frequency: 15.0
            }
            top_values {
              value: 'x'
              frequency: 15.0
            }
            rank_histogram {
              buckets {
                low_rank: 0
                high_rank: 0
                label: "a"
                sample_count: 35.0
              }
              buckets {
                low_rank: 1
                high_rank: 1
                label: "z"
                sample_count: 15.0
              }
              buckets {
                low_rank: 2
                high_rank: 2
                label: "x"
                sample_count: 15.0
              }
            }
          }
        }""", statistics_pb2.FeatureNameStatistics())
    }

    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            weight_feature='w',
            num_top_values=5, frequency_threshold=2,
            weighted_frequency_threshold=15, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_struct_leaves(self):
    batches = [
        pa.Table.from_arrays([
            pa.array([[1.0], [2.0]]),
            pa.array([[{
                'f1': ['a', 'b'],
                'f2': [1, 2]
            }, {
                'f1': ['b'],
            }], [{
                'f1': ['c', 'd'],
                'f2': [2, 3]
            }, {
                'f2': [3]
            }]]),
        ], ['w', 'c']),
        pa.Table.from_arrays([
            pa.array([[3.0]]),
            pa.array([[{
                'f1': ['d'],
                'f2': [4]
            }]]),
        ], ['w', 'c']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: "c"
          type: STRUCT
          struct_domain {
            feature {
              name: "f2"
              type: INT
              int_domain {
                is_categorical: true
              }
            }
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['c', 'f1']):
            text_format.Parse("""
              type: STRING
              string_stats {
                unique: 4
                top_values {
                  value: "d"
                  frequency: 2.0
                }
                top_values {
                  value: "b"
                  frequency: 2.0
                }
                top_values {
                  value: "c"
                  frequency: 1.0
                }
                rank_histogram {
                  buckets {
                    label: "d"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "c"
                    sample_count: 1.0
                  }
                }
                weighted_string_stats {
                  top_values {
                    value: "d"
                    frequency: 5.0
                  }
                  top_values {
                    value: "c"
                    frequency: 2.0
                  }
                  top_values {
                    value: "b"
                    frequency: 2.0
                  }
                  rank_histogram {
                    buckets {
                      label: "d"
                      sample_count: 5.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "b"
                      sample_count: 2.0
                    }
                  }
                }
              }
              path {
                step: "c"
                step: "f1"
              }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['c', 'f2']):
            text_format.Parse("""
              string_stats {
                unique: 4
                top_values {
                  value: "3"
                  frequency: 2.0
                }
                top_values {
                  value: "2"
                  frequency: 2.0
                }
                top_values {
                  value: "4"
                  frequency: 1.0
                }
                rank_histogram {
                  buckets {
                    label: "3"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "2"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "4"
                    sample_count: 1.0
                  }
                }
                weighted_string_stats {
                  top_values {
                    value: "3"
                    frequency: 4.0
                  }
                  top_values {
                    value: "4"
                    frequency: 3.0
                  }
                  top_values {
                    value: "2"
                    frequency: 3.0
                  }
                  rank_histogram {
                    buckets {
                      label: "3"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "4"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "2"
                      sample_count: 3.0
                    }
                  }
                }
              }
              path {
                step: "c"
                step: "f2"
              }""", statistics_pb2.FeatureNameStatistics()),
    }
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            schema=schema,
            weight_feature='w',
            num_top_values=3,
            num_rank_histogram_buckets=3))

    self.assertCombinerOutputEqual(batches, generator, expected_result)


if __name__ == '__main__':
  absltest.main()
