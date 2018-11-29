"""Tests for TopK and Uniques statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import top_k_uniques_combiner_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class TopKUniquesCombinerStatsGeneratorTest(
    test_util.CombinerStatsGeneratorTest):
  """Tests for TopKUniquesCombinerStatsGenerator."""

  def test_topk_uniques_combiner_with_single_string_feature(self):
    # 'fa': 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']),
                np.array(['a', 'c', 'd', 'a'])
            ],
                     dtype=np.object)
    }, {
        'fa': np.array([np.array(['a', 'b', 'c', 'd'])], dtype=np.object)
    }]
    # Note that if two feature values have the same frequency, the one with the
    # lexicographically larger feature value will be higher in the order.
    expected_result = {
        'fa':
            text_format.Parse(
                """
        name: 'fa'
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
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']),
                np.array(['a', 'c', 'd', 'a'])
            ],
                     dtype=np.object),
        'w':
            np.array([np.array([5.0]), np.array([5.0])])
    },
               {
                   'fa': np.array([np.array(['d', 'e'])], dtype=np.object),
                   'w': np.array([np.array([15.0])])
               }]
    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']),
                np.array(['a', 'c', 'd', 'a'])
            ],
                     dtype=np.unicode_)
    }, {
        'fa': np.array([np.array(['a', 'b', 'c', 'd'])], dtype=np.unicode_)
    }]
    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']), None,
                np.array(['a', 'c', 'd'])
            ],
                     dtype=np.object),
        'fb':
            np.array([np.array(['a', 'c', 'c']),
                      np.array(['b']), None],
                     dtype=np.object)
    },
               {
                   'fa':
                       np.array([np.array(['a', 'a', 'b', 'c', 'd']), None],
                                dtype=np.object),
                   'fb':
                       np.array([None, np.array(['b', 'c'])], dtype=np.object)
               }]

    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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
        'fb':
            text_format.Parse(
                """
                name: 'fb'
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

  def test_topk_uniques_combiner_with_empty_batch(self):
    batches = [{'a': np.array([])}]
    expected_result = {}
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_empty_dict(self):
    batches = [{}]
    expected_result = {}
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_empty_list(self):
    batches = []
    expected_result = {}
    generator = (
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            num_top_values=4, num_rank_histogram_buckets=3))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_topk_uniques_combiner_with_missing_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 1 'b', 2 'c'
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']), None,
                np.array(['a', 'c', 'd'])
            ],
                     dtype=np.object),
        'fb':
            np.array([np.array(['a', 'c', 'c']),
                      np.array(['b']), None],
                     dtype=np.object)
    },
               {
                   'fa':
                       np.array([np.array(['a', 'a', 'b', 'c', 'd']), None],
                                dtype=np.object)
               }]
    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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
        'fb':
            text_format.Parse(
                """
                name: 'fb'
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
    batches = [{
        'fa':
            np.array([
                np.array(['a', 'b', 'c', 'e']), None,
                np.array(['a', 'c', 'd'])
            ],
                     dtype=np.object),
        'fb':
            np.array([np.array([1.0, 2.0, 3.0]),
                      np.array([4.0, 5.0]), None])
    },
               {
                   'fa':
                       np.array([np.array(['a', 'a', 'b', 'c', 'd'])],
                                dtype=np.object),
                   'fb':
                       np.array([None])
               }]
    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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
    batches = [{
        'fa': np.array([np.array([12, 23, 34, 12]),
                        np.array([45, 23])])
    }, {
        'fa': np.array([np.array([12, 12, 34, 45])])
    }]
    expected_result = {
        'fa':
            text_format.Parse(
                """
                name: 'fa'
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


if __name__ == '__main__':
  absltest.main()
