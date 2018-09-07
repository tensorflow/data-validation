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

"""Tests for numeric statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import numeric_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class NumericStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_numeric_stats_generator_single_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1.0, 2.0]),
                               np.array([3.0, 4.0, 5.0])])},
               {'a': np.array([np.array([1.0])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 2.9866667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0066667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = numeric_stats_generator.NumericStatsGenerator(
        num_histogram_buckets=3, num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_with_entire_feature_value_list_missing(self):
    # input with two batches: first batch has three examples and second batch
    # has two examples.
    batches = [{'a': np.array([np.array([1.0, 2.0]), None,
                               np.array([3.0, 4.0, 5.0])])},
               {'a': np.array([np.array([1.0]), None])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 2.9866667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0066667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = numeric_stats_generator.NumericStatsGenerator(
        num_histogram_buckets=3, num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_with_individual_feature_value_missing(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1.0, 2.0]),
                               np.array([3.0, 4.0, np.NaN, 5.0])])},
               {'a': np.array([np.array([np.NaN, 1.0])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                num_nan: 2
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 2.9866667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0066667
                }
                type: STANDARD
              }
              histograms {
                num_nan: 2
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = numeric_stats_generator.NumericStatsGenerator(
        num_histogram_buckets=3, num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_with_multiple_features(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1.0, 2.0]),
                               np.array([3.0, 4.0, 5.0])]),
                'b': np.array([np.linspace(1, 1000, 1000, dtype=np.int32),
                               np.linspace(1001, 2000, 1000, dtype=np.int32)])},
               {'a': np.array([np.array([1.0])]),
                'b': np.array([np.linspace(2001, 3000, 1000, dtype=np.int32)])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 2.9866667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0066667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: INT
            num_stats {
              mean: 1500.5
              std_dev: 866.025355672
              min: 1.0
              max: 3000.0
              median: 1501.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1000.66666667
                  sample_count: 999.666666667
                }
                buckets {
                  low_value: 1000.66666667
                  high_value: 2000.33333333
                  sample_count: 999.666666667
                }
                buckets {
                  low_value: 2000.33333333
                  high_value: 3000.0
                  sample_count: 1000.66666667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 751.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 751.0
                  high_value: 1501.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 1501.0
                  high_value: 2251.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 2251.0
                  high_value: 3000.0
                  sample_count: 750.0
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = numeric_stats_generator.NumericStatsGenerator(
        num_histogram_buckets=3, num_quantiles_histogram_buckets=4,
        epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_with_missing_feature(self):
    # Input with two batches: first batch has two examples and second batch
    # has a single example. The first batch is missing feature 'b'.
    batches = [{'a': np.array([np.array([1.0, 2.0]),
                               np.array([3.0, 4.0, 5.0])])},
               {'a': np.array([np.array([1.0])]),
                'b': np.array([np.linspace(1, 3000, 3000, dtype=np.int32)])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 2.9866667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0066667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: INT
            num_stats {
              mean: 1500.5
              std_dev: 866.025355672
              min: 1.0
              max: 3000.0
              median: 1501.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1000.66666667
                  sample_count: 999.666666667
                }
                buckets {
                  low_value: 1000.66666667
                  high_value: 2000.33333333
                  sample_count: 999.666666667
                }
                buckets {
                  low_value: 2000.33333333
                  high_value: 3000.0
                  sample_count: 1000.66666667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 751.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 751.0
                  high_value: 1501.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 1501.0
                  high_value: 2250.0
                  sample_count: 750.0
                }
                buckets {
                  low_value: 2250.0
                  high_value: 3000.0
                  sample_count: 750.0
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = numeric_stats_generator.NumericStatsGenerator(
        num_histogram_buckets=3, num_quantiles_histogram_buckets=4,
        epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_categorical_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1, 0]),
                               np.array([0, 1, 0])])},
               {'a': np.array([np.array([1])])}]
    expected_result = {}
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
    generator = numeric_stats_generator.NumericStatsGenerator(schema=schema)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_empty_batch(self):
    batches = [{'a': np.array([])}]
    expected_result = {}
    generator = numeric_stats_generator.NumericStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_empty_dict(self):
    batches = [{}]
    expected_result = {}
    generator = numeric_stats_generator.NumericStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_empty_list(self):
    batches = []
    expected_result = {}
    generator = numeric_stats_generator.NumericStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_numeric_stats_generator_invalid_value_type(self):
    batches = [{'a': np.array([np.array([1.34]), np.array([12])])}]
    generator = numeric_stats_generator.NumericStatsGenerator()
    with self.assertRaises(TypeError):
      self.assertCombinerOutputEqual(batches, generator, None)

if __name__ == '__main__':
  absltest.main()
