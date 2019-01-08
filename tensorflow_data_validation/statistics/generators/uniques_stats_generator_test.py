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

"""Tests for Unique statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import uniques_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class UniquesStatsGeneratorTest(test_util.TransformStatsGeneratorTest):
  """Tests for UniquesStatsGenerator."""

  def test_with_empty_dict(self):
    examples = [{}]
    expected_result = []
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator, expected_result)

  def test_with_empty_list(self):
    examples = []
    expected_result = []
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator, expected_result)

  def test_all_string_features(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 2 'b', 3 'c'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e']),
                 'fb': np.array(['a', 'c', 'c'])},
                {'fa': None,
                 'fb': np.array(['a', 'c', 'c'])},
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
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    expected_result_fb = text_format.Parse(
        """
      features {
        name: 'fb'
        type: STRING
        string_stats {
          unique: 3
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator,
                                    [expected_result_fa, expected_result_fb])

  def test_single_unicode_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [{'fa': np.array(['a', 'b', 'c', 'e'], dtype=np.unicode_)},
                {'fa': np.array(['a', 'c', 'd', 'a'], dtype=np.unicode_)},
                {'fa': np.array(['a', 'b', 'c', 'd'], dtype=np.unicode_)}]
    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: STRING
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator, [expected_result_fa])

  def test_with_missing_feature(self):
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
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    expected_result_fb = text_format.Parse(
        """
      features {
        name: 'fb'
        type: STRING
        string_stats {
          unique: 3
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator,
                                    [expected_result_fa, expected_result_fb])

  def test_one_numeric_feature(self):
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
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
    generator = uniques_stats_generator.UniquesStatsGenerator()
    self.assertTransformOutputEqual(examples, generator, [expected_result_fa])

  def test_with_categorical_feature(self):
    examples = [{'fa': np.array([12, 23, 34, 12])},
                {'fa': np.array([45, 23])},
                {'fa': np.array([12, 12, 34, 45])}]
    expected_result_fa = text_format.Parse(
        """
      features {
        name: 'fa'
        type: INT
        string_stats {
          unique: 4
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
    generator = uniques_stats_generator.UniquesStatsGenerator(schema=schema)
    self.assertTransformOutputEqual(examples, generator, [expected_result_fa])

if __name__ == '__main__':
  absltest.main()
