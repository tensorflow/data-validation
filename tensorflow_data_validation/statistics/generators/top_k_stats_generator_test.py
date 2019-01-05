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

"""Tests for TopK statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import top_k_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class MakeFeatureStatsProtoTest(absltest.TestCase):
  """Tests for the make_feature_stats_proto_with_topk_stats function."""

  def test_make_feature_stats_proto_with_topk_stats(self):
    expected_result = text_format.Parse(
        """
        name: 'fa'
        type: STRING
        string_stats {
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
          }
    }""", statistics_pb2.FeatureNameStatistics())
    value_counts = [('a', 4), ('c', 3), ('d', 2), ('b', 2)]
    top_k_value_count_list = [
        top_k_stats_generator.FeatureValueCount(value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
        'fa', top_k_value_count_list, False, False, 3, 2)
    compare.assertProtoEqual(self, result, expected_result)

  def test_make_feature_stats_proto_with_topk_stats_unsorted_value_counts(self):
    expected_result = text_format.Parse(
        """
        name: 'fa'
        type: STRING
        string_stats {
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
          }
    }""", statistics_pb2.FeatureNameStatistics())
    # 'b' has a lower count than 'c'.
    value_counts = [('a', 4), ('b', 2), ('c', 3), ('d', 2)]
    top_k_value_count_list = [
        top_k_stats_generator.FeatureValueCount(value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
        'fa', top_k_value_count_list, False, False, 3, 2)
    compare.assertProtoEqual(self, result, expected_result)

  def test_make_feature_stats_proto_with_topk_stats_categorical_feature(self):
    expected_result = text_format.Parse(
        """
        name: 'fa'
        type: INT
        string_stats {
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
          }
    }""", statistics_pb2.FeatureNameStatistics())
    value_counts = [('a', 4), ('c', 3), ('d', 2), ('b', 2)]
    top_k_value_count_list = [
        top_k_stats_generator.FeatureValueCount(value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
        'fa', top_k_value_count_list, True, False, 3, 2)
    compare.assertProtoEqual(self, result, expected_result)

  def test_make_feature_stats_proto_with_topk_stats_weighted(self):
    expected_result = text_format.Parse(
        """
        name: 'fa'
        type: STRING
        string_stats {
          weighted_string_stats {
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
            }
          }
    }""", statistics_pb2.FeatureNameStatistics())
    value_counts = [('a', 4), ('c', 3), ('d', 2), ('b', 2)]
    top_k_value_count_list = [
        top_k_stats_generator.FeatureValueCount(value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = top_k_stats_generator.make_feature_stats_proto_with_topk_stats(
        'fa', top_k_value_count_list, False, True, 3, 2)
    compare.assertProtoEqual(self, result, expected_result)


class TopKStatsGeneratorTest(test_util.TransformStatsGeneratorTest):
  """Tests for TopkStatsGenerator."""

  def test_topk_with_single_string_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e'])},
                {'fa': np.array(['a', 'c', 'd', 'a'])},
                {'fa': np.array(['a', 'b', 'c', 'd'])}]

    # Note that if two feature values have the same frequency, the one with the
    # lexicographically larger feature value will be higher in the order.
    expected_result = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, [expected_result])

  def test_topk_with_weights(self):
    # non-weighted ordering
    # 3 'a', 2 'e', 2 'd', 2 'c', 1 'b'
    # weighted ordering
    # fa: 20 'e', 20 'd', 15 'a', 10 'c', 5 'b'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e']),
                 'w': np.array([5.0])},
                {'fa': np.array(['a', 'c', 'd', 'a']),
                 'w': np.array([5.0])},
                {'fa': np.array(['d', 'e']),
                 'w': np.array([15.0])}]

    expected_result = [
        text_format.Parse(
            """
            features {
              name: 'fa'
              type: STRING
              string_stats {
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
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              name: 'fa'
              type: STRING
              string_stats {
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
              }
        }""", statistics_pb2.DatasetFeatureStatistics())]
    generator = top_k_stats_generator.TopKStatsGenerator(
        weight_feature='w',
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, expected_result)

  def test_topk_with_single_unicode_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e'], dtype=np.unicode_)},
                {'fa': np.array(['a', 'c', 'd', 'a'], dtype=np.unicode_)},
                {'fa': np.array(['a', 'b', 'c', 'd'], dtype=np.unicode_)}]

    expected_result = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, [expected_result])

  def test_topk_with_multiple_features(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 2 'b', 3 'c'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e']),
                 'fb': np.array(['a', 'c', 'c'])},
                {'fa': None,
                 'fb': np.array(['b'])},
                {'fa': np.array(['a', 'c', 'd']),
                 'fb': None},
                {'fa': np.array(['a', 'a', 'b', 'c', 'd']),
                 'fb': None},
                {'fa': None,
                 'fb': np.array(['b', 'c'])}]

    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    expected_result_fb = text_format.Parse(
        """
      features {
        name: 'fb'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator,
                                    [expected_result_fa, expected_result_fb])

  def test_topk_with_empty_dict(self):
    examples = [{}]
    expected_result = []
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, expected_result)

  def test_topk_with_empty_list(self):
    examples = []
    expected_result = []
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, expected_result)

  def test_topk_with_missing_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 1 'b', 2 'c'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e']),
                 'fb': np.array(['a', 'c', 'c'])},
                {'fa': None,
                 'fb': np.array(['b'])},
                {'fa': np.array(['a', 'c', 'd']),
                 'fb': None},
                {'fa': np.array(['a', 'a', 'b', 'c', 'd'])},
                {'fa': None}]
    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    expected_result_fb = text_format.Parse(
        """
      features {
        name: 'fb'
        type: STRING
        string_stats {
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator,
                                    [expected_result_fa, expected_result_fb])

  def test_topk_with_numeric_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e']),
                 'fb': np.array([1.0, 2.0, 3.0])},
                {'fa': None,
                 'fb': np.array([4.0, 5.0])},
                {'fa': np.array(['a', 'c', 'd']),
                 'fb': None},
                {'fa': np.array(['a', 'a', 'b', 'c', 'd']),
                 'fb': None}]

    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=2, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, [expected_result_fa])

  def test_topk_with_categorical_feature(self):
    examples = [{'fa': np.array([12, 23, 34, 12])},
                {'fa': np.array([45, 23])},
                {'fa': np.array([12, 12, 34, 45])}]
    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: INT
        string_stats {
          top_values {
            value: '12'
            frequency: 4
          }
          top_values {
            value: '45'
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
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
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
    generator = top_k_stats_generator.TopKStatsGenerator(
        schema=schema,
        num_top_values=2, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, [expected_result_fa])

  def test_topk_with_invalid_utf8_value(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [{'fa': np.array(['a', b'\x80abc', 'a', b'\x80abc', 'a'],
                                dtype=np.object)}]

    expected_result = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
          top_values {
            value: 'a'
            frequency: 3
          }
          top_values {
            value: '__BYTES_VALUE__'
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
              label: "__BYTES_VALUE__"
              sample_count: 2.0
            }
          }
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = top_k_stats_generator.TopKStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertTransformOutputEqual(examples, generator, [expected_result])

if __name__ == '__main__':
  absltest.main()
