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

"""Tests for common statistics generator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import common_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class CommonStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_common_stats_generator_single_feature(self):
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
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.75
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = common_stats_generator.CommonStatsGenerator(
        num_values_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_with_entire_feature_value_list_missing(self):
    # input with two batches: first batch has three examples and second batch
    # has two examples.
    batches = [{'a': np.array([np.array([1.0, 2.0]), None,
                               np.array([3.0, 4.0, 5.0])], dtype=np.object),
                'b': np.array([np.array(['x', 'y', 'z', 'w']), None,
                               np.array(['qwe', 'abc'])], dtype=np.object)},
               {'a': np.array([np.array([1.0]), None], dtype=np.object),
                'b': np.array([None, np.array(['qwe'])], dtype=np.object)}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 2
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 2
                min_num_values: 1
                max_num_values: 4
                avg_num_values: 2.33333333
                tot_num_values: 7
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = common_stats_generator.CommonStatsGenerator(
        num_values_histogram_buckets=3)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_with_individual_feature_value_missing(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1.0, np.NaN]),
                               np.array([3.0, np.NaN, 5.0])])},
               {'a': np.array([np.array([np.NaN])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.75
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = common_stats_generator.CommonStatsGenerator(
        num_values_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_with_multiple_features(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{'a': np.array([np.array([1.0, 2.0]),
                               np.array([3.0, 4.0, 5.0])]),
                'b': np.array([np.array(['x', 'y', 'z', 'w']),
                               np.array(['qwe', 'abc'])]),
                'c': np.array([np.array([1, 5, 10]), np.array([0])])},
               {'a': np.array([np.array([1.0])]),
                'b': np.array([np.array(['ab'])]),
                'c': np.array([np.array([1, 1, 1, 5, 15])])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b': text_format.Parse(
            """
            name: 'b'
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 4
                avg_num_values: 2.33333333
                tot_num_values: 7
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'c': text_format.Parse(
            """
            name: 'c'
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 5
                avg_num_values: 3.0
                tot_num_values: 9
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 5.0
                    high_value: 5.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = common_stats_generator.CommonStatsGenerator(
        num_values_histogram_buckets=3, epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_categorical_feature(self):
    batches = [{'c': np.array([np.array([1, 5, 10]), np.array([0])])},
               {'c': np.array([np.array([1, 1, 1, 5, 15])])}]
    expected_result = {
        'c': text_format.Parse(
            """
            name: 'c'
            type: INT
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                min_num_values: 1
                max_num_values: 5
                avg_num_values: 3.0
                tot_num_values: 9
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 5.0
                    high_value: 5.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    schema = text_format.Parse(
        """
        feature {
          name: "c"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    generator = common_stats_generator.CommonStatsGenerator(
        schema=schema, num_values_histogram_buckets=3)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_empty_batch(self):
    batches = [{'a': np.array([])}]
    expected_result = {
        'a': text_format.Parse(
            """
            name: 'a'
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 0
                num_missing: 0
                tot_num_values: 0
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = common_stats_generator.CommonStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_empty_dict(self):
    batches = [{}]
    expected_result = {}
    generator = common_stats_generator.CommonStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_empty_list(self):
    batches = []
    expected_result = {}
    generator = common_stats_generator.CommonStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_common_stats_generator_invalid_value_type(self):
    batches = [{'a': np.array([{}])}]
    generator = common_stats_generator.CommonStatsGenerator()
    with self.assertRaises(TypeError):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_common_stats_generator_invalid_value_numpy_dtype(self):
    batches = [{'a': np.array([np.array([1+2j])])}]
    generator = common_stats_generator.CommonStatsGenerator()
    with self.assertRaises(TypeError):
      self.assertCombinerOutputEqual(batches, generator, None)

if __name__ == '__main__':
  absltest.main()
