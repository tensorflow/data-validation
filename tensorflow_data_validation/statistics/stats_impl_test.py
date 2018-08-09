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

"""Tests for the statistics generation implementation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics.generators import string_stats_generator
from tensorflow_data_validation.statistics.generators import uniques_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsImplTest(absltest.TestCase):

  def test_generate_stats_impl(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array(['xyz']), np.array(['qwe'])])},
               {'a': np.array([np.array(['ab'])])}]

    generator1 = string_stats_generator.StringStatsGenerator()
    generator2 = uniques_stats_generator.UniquesStatsGenerator()

    expected_result = text_format.Parse(
        """
        datasets {
          features {
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
              unique: 3
            }

          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    with beam.Pipeline() as p:
      result = (p | beam.Create(batches) |
                stats_impl.GenerateStatisticsImpl(
                    generators=[generator1, generator2]))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_merge_dataset_feature_stats_protos(self):
    proto1 = text_format.Parse(
        """
        num_examples: 7
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    proto2 = text_format.Parse(
        """
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            unique: 3
          }
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    expected = text_format.Parse(
        """
        num_examples: 7
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
            unique: 3
          }
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    actual = stats_impl._merge_dataset_feature_stats_protos([proto1, proto2])
    self.assertEqual(actual, expected)

  def test_merge_dataset_feature_stats_protos_single_proto(self):
    proto1 = text_format.Parse(
        """
        num_examples: 7
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    expected = text_format.Parse(
        """
        num_examples: 7
        features: {
          name: 'feature1'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 4
              min_num_values: 1
              max_num_values: 1
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    actual = stats_impl._merge_dataset_feature_stats_protos([proto1])
    self.assertEqual(actual, expected)

  def test_merge_dataset_feature_stats_protos_empty(self):
    self.assertEqual(stats_impl._merge_dataset_feature_stats_protos([]),
                     statistics_pb2.DatasetFeatureStatistics())

  def test_make_dataset_feature_statistics_list_proto(self):
    input_proto = text_format.Parse(
        """
        num_examples: 7
        features: {
          name: 'feature1'
          type: STRING
        }
        """, statistics_pb2.DatasetFeatureStatistics())

    expected = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            name: 'feature1'
            type: STRING
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    self.assertEqual(
        stats_impl._make_dataset_feature_statistics_list_proto(input_proto),
        expected)


if __name__ == '__main__':
  absltest.main()
