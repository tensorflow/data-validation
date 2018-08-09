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

"""Tests for string statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import string_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StringStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_string_stats_generator_single_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array(['xyz']), np.array(['qwe'])])},
               {'a': np.array([np.array(['ab'])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_with_missing_values(self):
    # input with two batches: first batch has three examples and second batch
    # has two examples.
    batches = [{'a': np.array([np.array(['xyz']), None,
                               np.array(['qwe'])], dtype=np.object)},
               {'a': np.array([np.array(['ab']), None], dtype=np.object)}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_with_multiple_features(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array(['xyz']), np.array(['qwe'])]),
                'b': np.array([np.array(['hello', 'world']),
                               np.array(['foo', 'bar'])])},
               {'a': np.array([np.array(['ab'])]),
                'b': np.array([np.array(['zzz', 'aaa', 'ddd'])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: STRING
            string_stats {
              avg_length: 3.57142857
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_with_missing_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'b': np.array([np.array(['hello', 'world']),
                               np.array(['foo', 'bar'])])},
               {'a': np.array([np.array(['ab', 'xyz', 'qwe'])]),
                'b': np.array([np.array(['zzz', 'aaa', 'ddd'])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: STRING
            string_stats {
              avg_length: 3.57142857
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_with_one_numeric_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array(['xyz']), np.array(['qwe'])]),
                'b': np.array([np.array([1.0, 2.0, 3.0]),
                               np.array([4.0, 5.0])])},
               {'a': np.array([np.array(['ab'])]),
                'b': np.array([np.array([5.0, 6.0])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_unicode_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array(['xyz']), np.array(['qwe'])],
                              dtype=np.unicode_)},
               {'a': np.array([np.array(['ab'])], dtype=np.unicode_)}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_categorical_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([123]),
                               np.array([45])])},
               {'a': np.array([np.array([456])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: INT
            string_stats {
              avg_length: 2.66666666
            }
            """, statistics_pb2.FeatureNameStatistics())}
    schema = text_format.Parse(
        """
        feature {
          name: "a"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    generator = string_stats_generator.StringStatsGenerator(schema=schema)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_empty_batch(self):
    batches = [{'a': np.array([])}]
    expected_result = {}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_empty_dict(self):
    batches = [{}]
    expected_result = {}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_string_stats_generator_empty_list(self):
    batches = []
    expected_result = {}
    generator = string_stats_generator.StringStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

if __name__ == '__main__':
  absltest.main()
