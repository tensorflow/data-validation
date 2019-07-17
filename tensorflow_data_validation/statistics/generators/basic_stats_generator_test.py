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

"""Tests for basic statistics generator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class BasicStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_basic_stats_generator_single_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.Table.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]])], ['a'])
    b2 = pa.Table.from_arrays([pa.array([[1.0]])], ['a'])
    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
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
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_no_runtime_warnings_close_to_max_int(self):
    # input has batches with values that are slightly smaller than the maximum
    # integer value.
    less_than_max_int_value = np.iinfo(np.int64).max - 1
    batches = (
        [pa.Table.from_arrays([pa.array([[less_than_max_int_value]])], ['a'])] *
        2)
    generator = basic_stats_generator.BasicStatsGenerator()
    with np.testing.assert_no_warnings():
      accumulators = [
          generator.add_input(generator.create_accumulator(), batch)
          for batch in batches
      ]
      generator.merge_accumulators(accumulators)

  def test_basic_stats_generator_handle_null_column(self):
    # Feature 'a' covers null coming before non-null.
    # Feature 'b' covers null coming after non-null.
    b1 = pa.Table.from_arrays([
        pa.array([None, None, None], type=pa.null()),
        pa.array([[1.0, 2.0, 3.0], [4.0], [5.0]]),
    ], ['a', 'b'])
    b2 = pa.Table.from_arrays([
        pa.array([[1, 2], None], type=pa.list_(pa.int64())),
        pa.array([None, None], type=pa.null()),
    ], ['a', 'b'])
    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: "a"
            }
            num_stats {
              common_stats {
                num_non_missing: 1
                min_num_values: 2
                max_num_values: 2
                avg_num_values: 2.0
                num_values_histogram {
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 0.25
                  }
                  type: QUANTILES
                }
                tot_num_values: 2
              }
              mean: 1.5
              std_dev: 0.5
              min: 1.0
              median: 2.0
              max: 2.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.3333333
                  sample_count: 0.9955556
                }
                buckets {
                  low_value: 1.3333333
                  high_value: 1.6666667
                  sample_count: 0.0022222
                }
                buckets {
                  low_value: 1.6666667
                  high_value: 2.0
                  sample_count: 1.0022222
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 0.5
                }
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 0.5
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.5
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['b']): text_format.Parse(
            """
            path {
              step: 'b'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 1.66666698456
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 1.0
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
                tot_num_values: 5
              }
              mean: 3.0
              std_dev: 1.4142136
              min: 1.0
              median: 3.0
              max: 5.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 1.9888889
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0055556
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 2.0055556
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 1.25
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 1.25
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.25
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.25
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_with_weight_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.Table.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
                               pa.array([[1, 2], [3, 4, 5]]),
                               pa.array([[1.0], [2.0]])], ['a', 'b', 'w'])
    b2 = pa.Table.from_arrays([pa.array([[1.0, np.NaN, np.NaN, np.NaN], None]),
                               pa.array([[1], None]),
                               pa.array([[3.0], [2.0]])], ['a', 'b', 'w'])

    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                min_num_values: 2
                max_num_values: 4
                avg_num_values: 3.0
                tot_num_values: 9
                num_values_histogram {
                  buckets {
                    low_value: 2.0
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
                    high_value: 4.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.75
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 6.0
                  avg_num_values: 3.33333333
                  tot_num_values: 20.0
                }
              }
              mean: 2.66666666
              std_dev: 1.49071198
              num_zeros: 0
              min: 1.0
              max: 5.0
              median: 3.0
              histograms {
                num_nan: 3
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
                num_nan: 3
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
              weighted_numeric_stats {
                mean: 2.7272727
                std_dev: 1.5427784
                median: 3.0
                histograms {
                  num_nan: 3
                  buckets {
                    low_value: 1.0
                    high_value: 2.3333333
                    sample_count: 4.9988889
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 1.9922222
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 4.0088889
                  }
                }
                histograms {
                  num_nan: 3
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 2.75
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['b']): text_format.Parse(
            """
            path {
              step: 'b'
            }
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 3
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
                weighted_common_stats {
                  num_non_missing: 6.0
                  avg_num_values: 1.83333333
                  tot_num_values: 11.0
                }
              }
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
              weighted_numeric_stats {
                mean: 2.7272727
                std_dev: 1.5427784
                median: 3.0
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 2.3333333
                    sample_count: 4.9988889
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 1.9922222
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 4.0088889
                  }
                }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 2.75
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 2.75
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        weight_feature='w',
        num_values_histogram_buckets=4, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_with_entire_feature_value_list_missing(self):
    # input with two batches: first batch has three examples and second batch
    # has two examples.
    b1 = pa.Table.from_arrays([
        pa.array([[1.0, 2.0], None, [3.0, 4.0, 5.0]]),
        pa.array([['x', 'y', 'z', 'w'], None, ['qwe', 'abc']]),
    ], ['a', 'b'])
    b2 = pa.Table.from_arrays(
        [pa.array([[1.0], None]),
         pa.array([None, ['qwe']])], ['a', 'b'])
    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
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
        types.FeaturePath(['b']): text_format.Parse(
            """
            path {
              step: 'b'
            }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
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
              avg_length: 1.85714285
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_with_individual_feature_value_missing(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.Table.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, np.NaN, 5.0]])],
                              ['a'])
    b2 = pa.Table.from_arrays([pa.array([[np.NaN, 1.0]])], ['a'])
    batches = [b1, b2]

    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                min_num_values: 2
                max_num_values: 4
                avg_num_values: 2.66666666
                tot_num_values: 8
                num_values_histogram {
                  buckets {
                    low_value: 2.0
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
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_with_multiple_features(self):

    b1 = pa.Table.from_arrays([
        pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
        pa.array([[b'x', b'y', b'z', b'w'], [b'qwe', b'abc']]),
        pa.array([
            np.linspace(1, 1000, 1000, dtype=np.int32),
            np.linspace(1001, 2000, 1000, dtype=np.int32)
        ]),
    ], ['a', 'b', 'c'])
    b2 = pa.Table.from_arrays([
        pa.array([[1.0]]),
        pa.array([[b'ab']]),
        pa.array([np.linspace(2001, 3000, 1000, dtype=np.int32)]),
    ], ['a', 'b', 'c'])

    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
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
        types.FeaturePath(['b']): text_format.Parse(
            """
            path {
              step: 'b'
            }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
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
              avg_length: 1.71428571
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['c']): text_format.Parse(
            """
            path {
              step: 'c'
            }
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 3
                min_num_values: 1000
                max_num_values: 1000
                avg_num_values: 1000.0
                tot_num_values: 3000
                num_values_histogram {
                  buckets {
                    low_value: 1000.0
                    high_value: 1000.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1000.0
                    high_value: 1000.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1000.0
                    high_value: 1000.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
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
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4, epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_categorical_feature(self):
    batches = [
        pa.Table.from_arrays([pa.array([[1, 5, 10], [0]])], ['c']),
        pa.Table.from_arrays([pa.array([[1, 1, 1, 5, 15], [-1]])], ['c']),
    ]
    expected_result = {
        types.FeaturePath(['c']): text_format.Parse(
            """
            path {
              step: 'c'
            }
            string_stats {
              common_stats {
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 5
                avg_num_values: 2.5
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.3333333
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 1.3333333
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 1.3333333
                  }
                  type: QUANTILES
                }
                tot_num_values: 10
              }
              avg_length: 1.29999995232
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
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_empty_batch(self):
    batches = [
        pa.Table.from_arrays([pa.array([], type=pa.list_(pa.binary()))], ['a'])]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 0
                tot_num_values: 0
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_no_value_in_batch(self):
    batches = [
        pa.Table.from_arrays([
            pa.array([[], [], []], type=pa.list_(pa.int64()))], ['a'])]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            num_stats {
              common_stats {
                num_non_missing: 3
                num_values_histogram {
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  buckets {
                    sample_count: 0.3
                  }
                  type: QUANTILES
                }
              }
            }""", statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator()
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_only_nan(self):
    b1 = pa.Table.from_arrays([pa.array([[np.NaN]],
                                        type=pa.list_(pa.float32()))], ['a'])
    batches = [b1]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 1
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
                tot_num_values: 1
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
              }
              histograms {
                num_nan: 1
                type: STANDARD
              }
              histograms {
                num_nan: 1
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_basic_stats_generator_column_not_list(self):
    batches = [pa.Table.from_arrays([pa.array([1, 2, 3])], ['a'])]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegexp(TypeError,
                                 'Expected feature column to be a List'):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_basic_stats_generator_invalid_value_numpy_dtype(self):
    batches = [pa.Table.from_arrays(
        [pa.array([[]], type=pa.list_(pa.date32()))], ['a'])]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegexp(
        TypeError, 'Feature a has unsupported arrow type'):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_basic_stats_generator_feature_with_different_types(self):
    batches = [
        pa.Table.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]])], ['a']),
        pa.Table.from_arrays([pa.array([[1]])], ['a']),
    ]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegexp(TypeError, 'Cannot determine the type'):
      self.assertCombinerOutputEqual(batches, generator, None)


if __name__ == '__main__':
  absltest.main()
