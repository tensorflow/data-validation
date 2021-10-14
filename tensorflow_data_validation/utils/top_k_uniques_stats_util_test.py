# Copyright 2020 Google LLC
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
"""Tests for top_k_uniques_stats_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils import top_k_uniques_stats_util

from google.protobuf import text_format

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class TopKUniquesStatsUtilTest(absltest.TestCase):

  def test_make_feature_stats_proto_topk_uniques(self):
    expected_result = text_format.Parse(
        """
        path {
            step: "fa"
          }
        string_stats {
          unique: 5
          top_values {
            value: "a"
            frequency: 3.0
          }
          top_values {
            value: "e"
            frequency: 2.0
          }
          top_values {
            value: "d"
            frequency: 2.0
          }
          rank_histogram {
            buckets {
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "e"
              sample_count: 2.0
            }
          }
          weighted_string_stats {
            top_values {
              value: "e"
              frequency: 20.0
            }
            top_values {
              value: "d"
              frequency: 20.0
            }
            top_values {
              value: "a"
              frequency: 15.0
            }
            rank_histogram {
              buckets {
                label: "e"
                sample_count: 20.0
              }
              buckets {
                low_rank: 1
                high_rank: 1
                label: "d"
                sample_count: 20.0
              }
            }
          }
        }
        """, statistics_pb2.FeatureNameStatistics())

    unweighted_value_counts = [('a', 3), ('e', 2), ('d', 2), ('c', 2), ('b', 1)]
    weighted_value_counts = [
        ('e', 20), ('d', 20), ('a', 15), ('c', 10), ('b', 5)]
    top_k_value_count_list = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in unweighted_value_counts
    ]
    top_k_value_count_list_weighted = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in weighted_value_counts
    ]
    result = (
        top_k_uniques_stats_util.make_feature_stats_proto_topk_uniques(
            types.FeaturePath(['fa']),
            num_top_values=3,
            frequency_threshold=1,
            weighted_frequency_threshold=1.,
            num_rank_histogram_buckets=2,
            num_unique=5,
            value_count_list=top_k_value_count_list,
            weighted_value_count_list=top_k_value_count_list_weighted))
    test_util.assert_feature_proto_equal(self, result, expected_result)

  def test_make_feature_stats_proto_topk_uniques_custom_stats(self):
    expected_result = text_format.Parse(
        """
        path {
            step: "fa"
          }
        custom_stats {
          name: "topk_sketch_rank_histogram"
          rank_histogram {
            buckets {
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "e"
              sample_count: 2.0
            }
          }
        }
        custom_stats {
          name: "weighted_topk_sketch_rank_histogram"
          rank_histogram {
              buckets {
                label: "e"
                sample_count: 20.0
              }
              buckets {
                low_rank: 1
                high_rank: 1
                label: "d"
                sample_count: 20.0
              }
          }
        }
        custom_stats {
          name: "uniques_sketch_num_uniques"
          num: 5
        }
        """, statistics_pb2.FeatureNameStatistics())

    unweighted_value_counts = [('a', 3), ('e', 2), ('d', 2), ('c', 2), ('b', 1)]
    weighted_value_counts = [
        ('e', 20), ('d', 20), ('a', 15), ('c', 10), ('b', 5)]
    top_k_value_count_list = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in unweighted_value_counts
    ]
    top_k_value_count_list_weighted = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in weighted_value_counts
    ]
    result = (
        top_k_uniques_stats_util
        .make_feature_stats_proto_topk_uniques_custom_stats(
            types.FeaturePath(['fa']),
            num_top_values=3,
            frequency_threshold=1,
            weighted_frequency_threshold=1.,
            num_rank_histogram_buckets=2,
            num_unique=5,
            value_count_list=top_k_value_count_list,
            weighted_value_count_list=top_k_value_count_list_weighted))
    test_util.assert_feature_proto_equal(self, result, expected_result)

  def test_make_feature_stats_proto_topk_uniques_categorical(self):
    expected_result = text_format.Parse(
        """
        path {
          step: 'fa'
        }
        string_stats {
          unique: 4
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

    value_counts = [('d', 2), ('c', 3), ('a', 4), ('b', 2)]
    top_k_value_count_list = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = (
        top_k_uniques_stats_util.make_feature_stats_proto_topk_uniques(
            types.FeaturePath(['fa']),
            num_top_values=3,
            frequency_threshold=1,
            num_rank_histogram_buckets=2,
            num_unique=4,
            value_count_list=top_k_value_count_list))
    test_util.assert_feature_proto_equal(self, result, expected_result)

  def test_make_feature_stats_proto_topk_uniques_unordered(self):
    expected_result = text_format.Parse(
        """
        path {
          step: 'fa'
        }
        string_stats {
          unique: 4
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
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = (
        top_k_uniques_stats_util.make_feature_stats_proto_topk_uniques(
            types.FeaturePath(['fa']),
            num_top_values=3,
            frequency_threshold=1,
            num_rank_histogram_buckets=2,
            num_unique=4,
            value_count_list=top_k_value_count_list))
    test_util.assert_feature_proto_equal(self, result, expected_result)

  def test_make_dataset_feature_stats_proto_topk_single(self):
    expected_result = text_format.Parse(
        """
        features {
          string_stats {
            top_values {
              value: "e"
              frequency: 20.0
            }
            top_values {
              value: "d"
              frequency: 20.0
            }
            top_values {
              value: "a"
              frequency: 15.0
            }
            rank_histogram {
              buckets {
                label: "e"
                sample_count: 20.0
              }
              buckets {
                low_rank: 1
                high_rank: 1
                label: "d"
                sample_count: 20.0
              }
            }
          }
          path {
            step: "fa"
          }
    }""", statistics_pb2.DatasetFeatureStatistics())

    value_counts = [('e', 20), ('d', 20), ('a', 15), ('c', 10), ('b', 5)]
    value_count_list = [
        top_k_uniques_stats_util.FeatureValueCount(
            value_count[0], value_count[1])
        for value_count in value_counts
    ]
    result = (
        top_k_uniques_stats_util.make_dataset_feature_stats_proto_topk_single(
            types.FeaturePath(['fa']).steps(),
            value_count_list=value_count_list,
            is_weighted_stats=False,
            num_top_values=3,
            frequency_threshold=1,
            num_rank_histogram_buckets=2))
    test_util.assert_dataset_feature_stats_proto_equal(
        self, result, expected_result)

  def test_output_categorical_numeric(self):
    type_mapping = {
        types.FeaturePath(['fa']): schema_pb2.INT,
        types.FeaturePath(['fb']): schema_pb2.FLOAT,
    }
    self.assertTrue(
        top_k_uniques_stats_util.output_categorical_numeric(
            type_mapping, types.FeaturePath(['fa']),
            statistics_pb2.FeatureNameStatistics.INT))
    self.assertTrue(
        top_k_uniques_stats_util.output_categorical_numeric(
            type_mapping, types.FeaturePath(['fb']),
            statistics_pb2.FeatureNameStatistics.FLOAT))
    self.assertFalse(
        top_k_uniques_stats_util.output_categorical_numeric(
            type_mapping, types.FeaturePath(['fc']),
            statistics_pb2.FeatureNameStatistics.INT))
    self.assertFalse(
        top_k_uniques_stats_util.output_categorical_numeric(
            type_mapping, types.FeaturePath(['fb']),
            statistics_pb2.FeatureNameStatistics.INT))
    self.assertFalse(
        top_k_uniques_stats_util.output_categorical_numeric(
            type_mapping, types.FeaturePath(['fa']),
            statistics_pb2.FeatureNameStatistics.FLOAT))


if __name__ == '__main__':
  absltest.main()
