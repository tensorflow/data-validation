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

"""Tests for basic statistics generator."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class BasicStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_single_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.RecordBatch.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]])],
                                    ['a'])
    b2 = pa.RecordBatch.from_arrays([pa.array([[1.0]])], ['a'])
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
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_two_feature_partitions(self):
    # Note: default partitioner assigns a->1, b->0
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
         pa.array([['abc'], ['xyz']])], ['a', 'b'])
    batches = [b1]
    expected_result = {
        types.FeaturePath(['b']):
            text_format.Parse(
                """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
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
                tot_num_values: 2
              }
              avg_length: 3.0
            }
            path {
              step: "b"
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    # Note: partition 0 contains feature "b"
    generator = generator._copy_for_partition_index(0, 2)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_with_feature_config(self):
    config = types.PerFeatureStatsConfig(
        [types.FeaturePath(['a']), types.FeaturePath(['b'])],
        types.PerFeatureStatsConfig.INCLUDE,
    )
    b1 = pa.RecordBatch.from_arrays(
        [
            pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
            pa.array([['abc'], ['xyz']]),
            pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
            pa.array([['abc'], ['xyz']]),
        ],
        ['a', 'b', 'c', 'd'],
    )
    batches = [b1]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse(
            """
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 2
                max_num_values: 3
                avg_num_values: 2.5
                num_values_histogram {
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.3333333333333333
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.3333333333333333
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.3333333333333333
                  }
                  type: QUANTILES
                }
                tot_num_values: 5
              }
              mean: 3.0
              std_dev: 1.4142135623730951
              min: 1.0
              median: 3.0
              max: 5.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.333333333333333
                  sample_count: 2.004166666666662
                }
                buckets {
                  low_value: 2.333333333333333
                  high_value: 3.6666666666666665
                  sample_count: 1.0041666666666653
                }
                buckets {
                  low_value: 3.6666666666666665
                  high_value: 5.0
                  sample_count: 1.9916666666666696
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            path {
              step: "a"
            }
            """,
            statistics_pb2.FeatureNameStatistics(),
        ),
        types.FeaturePath(['b']): text_format.Parse(
            """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
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
                tot_num_values: 2
              }
              avg_length: 3.0
            }
            path {
              step: "b"
            }
            """,
            statistics_pb2.FeatureNameStatistics(),
        ),
        types.FeaturePath(['c']): text_format.Parse(
            """
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 2
                max_num_values: 3
                avg_num_values: 2.5
                tot_num_values: 5
              }
              mean: 3.0
              std_dev: 1.4142135623730951
              min: 1.0
              median: nan
              max: 5.0
            }
            path {
              step: "c"
            }
            """,
            statistics_pb2.FeatureNameStatistics(),
        ),
        types.FeaturePath(['d']): text_format.Parse(
            """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
                tot_num_values: 2
              }
              avg_length: 3.0
            }
            path {
              step: "d"
            }
            """,
            statistics_pb2.FeatureNameStatistics(),
        ),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4,
        feature_config=config,
    )
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_two_feature_partitions_with_weights(self):
    # Note: default partitioner assigns a->1, b->0
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0], [10.0]]),
         pa.array([['a'], ['xyz']])], ['a', 'b'])
    batches = [b1]
    expected_result = {
        types.FeaturePath(['b']):
            text_format.Parse(
                """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
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
                weighted_common_stats {
                    num_non_missing: 11.0
                    avg_num_values: 1.0
                    tot_num_values: 11.0
                }
                tot_num_values: 2
              }
              avg_length: 2.0
            }
            path {
              step: "b"
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4,
        example_weight_map=ExampleWeightMap(weight_feature='a'),
        )
    # Note: partition 0 contains feature "b"
    generator = generator._copy_for_partition_index(0, 2)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_no_feature_falls_in_partition(self):
    # Note: default partitioner assigns a->0, b->1
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
         pa.array([['abc'], ['xyz']])], ['a', 'b'])
    batches = [b1]
    expected_result = {}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    generator = generator._copy_for_partition_index(0, 3)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_infinity(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0, np.inf, np.inf, -np.inf], [3.0, 4.0, 5.0, -np.inf]
                 ])
    ], ['a'])
    b2 = pa.RecordBatch.from_arrays([pa.array([[1.0, np.inf, -np.inf]])], ['a'])
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
                min_num_values: 3
                max_num_values: 5
                avg_num_values: 4.0
                tot_num_values: 12
                num_values_histogram {
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 5.0
                    high_value: 5.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
              }
              mean: nan
              num_zeros: 0
              min: -inf
              max: inf
              median: 3.0
              histograms {
                buckets {
                  low_value: -inf
                  high_value: -inf
                  sample_count: 3.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 3.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: inf
                  high_value: inf
                  sample_count: 3.0
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: -inf
                  high_value: 1.0
                  sample_count: 5.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: inf
                  sample_count: 2.5
                }
                buckets {
                  low_value: inf
                  high_value: inf
                  sample_count: 2.5
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4, num_histogram_buckets=4,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_no_runtime_warnings_close_to_max_int(self):
    # input has batches with values that are slightly smaller than the maximum
    # integer value.
    less_than_max_int_value = np.iinfo(np.int64).max - 1
    batches = ([
        pa.RecordBatch.from_arrays([pa.array([[less_than_max_int_value]])],
                                   ['a'])
    ] * 2)
    generator = basic_stats_generator.BasicStatsGenerator()
    old_nperr = np.geterr()
    np.seterr(over='raise')
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    generator.merge_accumulators(accumulators)
    np.seterr(**old_nperr)

  def test_handle_null_column(self):
    # Feature 'a' covers null coming before non-null.
    # Feature 'b' covers null coming after non-null.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([None, None, None], type=pa.null()),
        pa.array([[1.0, 2.0, 3.0], [4.0], [5.0]]),
    ], ['a', 'b'])
    b2 = pa.RecordBatch.from_arrays([
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
                num_missing: 4
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
                  sample_count: 1.0016584
                }
                buckets {
                  low_value: 1.3333333
                  high_value: 1.6666667
                  sample_count: 0.0016584
                }
                buckets {
                  low_value: 1.6666667
                  high_value: 2.0
                  sample_count: 0.9966833
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 0.3333333
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.3333333
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.3333333
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
                num_missing: 2
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 1.66666698456
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
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
                  sample_count: 2.0041667
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0041667
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9916667
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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

  def test_pure_null_column(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([None, None], type=pa.null()),
            pa.array([[1.0], [1.0]]),
        ], ['a', 'w']),
        pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.null()),
            pa.array([[1.0]]),
        ], ['a', 'w']),
    ]
    expected_result = {
        types.FeaturePath(['a']):
            text_format.Parse("""
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                weighted_common_stats {
                  num_missing: 3.0
                }
              }
            }
            path {
              step: "a"
            }
            """, statistics_pb2.FeatureNameStatistics()),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        example_weight_map=ExampleWeightMap(weight_feature='w'),
        num_values_histogram_buckets=4, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(
        batches, generator, expected_result,
        only_match_expected_feature_stats=True)

  def test_with_weight_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
        pa.array([[1, 2], [3, 4, 5]]),
        pa.array([[1.0], [2.0]])
    ], ['a', 'b', 'w'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, np.NaN, np.NaN, np.NaN], None]),
        pa.array([[1], None]),
        pa.array([[3.0], [2.0]])
    ], ['a', 'b', 'w'])

    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']):
            text_format.Parse(
                """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 1
                min_num_values: 2
                max_num_values: 4
                avg_num_values: 3.0
                tot_num_values: 9
                num_values_histogram {
                  buckets {
                    low_value: 2.0
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
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 6.0
                  num_missing: 2.0
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                num_nan: 3
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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
                    sample_count: 5.0091324
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 2.0091324
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 3.9817352
                  }
                }
                histograms {
                  num_nan: 3
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 4.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 3.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 2.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['b']):
            text_format.Parse(
                """
            path {
              step: 'b'
            }
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 1
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 6.0
                  num_missing: 2.0
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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
                    sample_count: 5.0091324
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 2.0091324
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 3.9817352
                  }
                }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 4.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 3.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 2.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['w']):
            text_format.Parse(
                """
            path {
              step: 'w'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
                tot_num_values: 4
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 8.0
                  avg_num_values: 1.0
                  tot_num_values: 8.0
                }
              }
              mean: 2.0
              std_dev: 0.7071068
              num_zeros: 0
              min: 1.0
              max: 3.0
              median: 2.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.6666667
                  sample_count: 1.0066667
                }
                buckets {
                  low_value: 1.6666667
                  high_value: 2.3333333
                  sample_count: 1.9966337
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.0
                  sample_count: 0.9966997
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 0.5
                }
                buckets {
                  low_value: 3.0
                  high_value: 3.0
                  sample_count: 0.5
                }
                type: QUANTILES
              }
              weighted_numeric_stats {
                mean: 2.25
                std_dev: 0.6614378
                median: 2.0
                histograms {
                  buckets {
                     low_value: 1.0
                      high_value: 1.6666667
                      sample_count: 1.0133333
                    }
                    buckets {
                      low_value: 1.6666667
                      high_value: 2.3333333
                      sample_count: 3.9932892
                    }
                    buckets {
                      low_value: 2.3333333
                      high_value: 3.0
                      sample_count: 2.9933775
                    }
                  }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 3.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 1.5
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        example_weight_map=ExampleWeightMap(weight_feature='w'),
        num_values_histogram_buckets=4, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_with_per_feature_weight(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]]),
        pa.array([[1, 2], [3, 4, 5]]),
        pa.array([[1.0], [2.0]]),
        pa.array([[2.0], [1.0]]),
    ], ['a', 'b', 'w_a', 'w_b'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, np.NaN, np.NaN, np.NaN], None]),
        pa.array([[1], None]),
        pa.array([[3.0], [2.0]]),
        pa.array([[2.0], [3.0]]),
    ], ['a', 'b', 'w_a', 'w_b'])

    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']):
            text_format.Parse(
                """
            path {
              step: 'a'
            }
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 1
                min_num_values: 2
                max_num_values: 4
                avg_num_values: 3.0
                tot_num_values: 9
                num_values_histogram {
                  buckets {
                    low_value: 2.0
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
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 6.0
                  num_missing: 2.0
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                num_nan: 3
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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
                    sample_count: 5.0091324
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 2.0091324
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 3.9817352
                  }
                }
                histograms {
                  num_nan: 3
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 4.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 3.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 2.0
                  }
                  type: QUANTILES
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['b']):
            text_format.Parse(
                """
            num_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 1
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 5.0
                  num_missing: 3.0
                  avg_num_values: 1.8
                  tot_num_values: 9.0
                }
                tot_num_values: 6
              }
              mean: 2.6666667
              std_dev: 1.490712
              min: 1.0
              median: 3.0
              max: 5.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 2.3333333
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
              weighted_numeric_stats {
                mean: 2.2222222
                std_dev: 1.396645
                median: 2.0
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 2.3333333
                    sample_count: 6.0074074
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 1.0077441
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 1.9848485
                  }
                }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 4.0
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 2.0
                  }
                  type: QUANTILES
                }
              }
            }
            path {
              step: "b"
            }
            """, statistics_pb2.FeatureNameStatistics()),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        example_weight_map=ExampleWeightMap(
            weight_feature='w_a',
            per_feature_override={types.FeaturePath(['b']): 'w_b'}),
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result,
                                   only_match_expected_feature_stats=True)

  def test_with_entire_feature_value_list_missing(self):
    # input with two batches: first batch has three examples and second batch
    # has two examples.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0], None, [3.0, 4.0, 5.0]]),
        pa.array([['x', 'y', 'z', 'w'], None, ['qwe', 'abc']]),
    ], ['a', 'b'])
    b2 = pa.RecordBatch.from_arrays(
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
                num_missing: 2
                min_num_values: 1
                max_num_values: 3
                avg_num_values: 2.0
                tot_num_values: 6
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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
                num_missing: 2
                min_num_values: 1
                max_num_values: 4
                avg_num_values: 2.33333333
                tot_num_values: 7
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
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

  def test_with_individual_feature_value_missing(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0, 2.0], [3.0, 4.0, np.NaN, 5.0]])], ['a'])
    b2 = pa.RecordBatch.from_arrays([pa.array([[np.NaN, 1.0]])], ['a'])
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
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                num_nan: 2
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_with_multiple_features(self):

    # Test that columns of ListArray, LargeListArray can be handled. Also test
    # that columns whose values are LargeBinaryArray can be handled.
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]],
                 type=pa.large_list(pa.float32())),
        pa.array([[b'x', b'y', b'z', b'w'], [b'qwe', b'abc']],
                 type=pa.list_(pa.large_binary())),
        pa.array([
            np.linspace(1, 1000, 1000, dtype=np.int32),
            np.linspace(1001, 2000, 1000, dtype=np.int32)
        ],
                 type=pa.list_(pa.int32())),
    ], ['a', 'b', 'c'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[1.0]], type=pa.large_list(pa.float32())),
        pa.array([[b'ab']], type=pa.list_(pa.large_binary())),
        pa.array([np.linspace(2001, 3000, 1000, dtype=np.int32)],
                 type=pa.list_(pa.int32())),
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
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
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
                  sample_count: 3.0049751
                }
                buckets {
                  low_value: 2.3333333
                  high_value: 3.6666667
                  sample_count: 1.0049751
                }
                buckets {
                  low_value: 3.6666667
                  high_value: 5.0
                  sample_count: 1.9900498
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 3.0
                  high_value: 4.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 4.0
                  high_value: 5.0
                  sample_count: 1.0
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
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
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
                  sample_count: 1000.6666667
                }
                buckets {
                  low_value: 1000.66666667
                  high_value: 2000.33333333
                  sample_count: 999.6666667
                }
                buckets {
                  low_value: 2000.33333333
                  high_value: 3000.0
                  sample_count: 999.6666667
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 751.0
                  sample_count: 751.0
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
                  sample_count: 749.0
                }
                type: QUANTILES
              }
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4, epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_with_bytes_features(self):

    b1 = pa.RecordBatch.from_arrays([
        pa.array([[b'x', b'y', b'z', b'w'], [b'qwe', b'abc']]),], ['b'])
    b2 = pa.RecordBatch.from_arrays([pa.array([[b'ab']]),], ['b'])
    batches = [b1, b2]
    schema = text_format.Parse(
        """
        feature {
          name: "b"
          type: BYTES
          image_domain { }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['b']): text_format.Parse(
            """
            path {
              step: 'b'
            }
            type: BYTES
            bytes_stats {
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
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 4.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
              }
              avg_num_bytes: 1.71428571
              min_num_bytes: 1
              max_num_bytes: 3
              max_num_bytes_int: 3
            }
            """, statistics_pb2.FeatureNameStatistics()),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4, epsilon=0.001)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_categorical_int_feature(self):
    batches = [
        pa.RecordBatch.from_arrays([pa.array([[1, 5, 10], [0]])], ['c']),
        pa.RecordBatch.from_arrays([pa.array([[1, 1, 1, 5, 15], [-1]])], ['c']),
        pa.RecordBatch.from_arrays([pa.array([None, None], type=pa.null())],
                                   ['c'])
    ]
    expected_result = {
        types.FeaturePath(['c']):
            text_format.Parse(
                """
            path {
              step: 'c'
            }
            type: INT
            string_stats {
              common_stats {
                num_non_missing: 4
                num_missing: 2
                min_num_values: 1
                max_num_values: 5
                avg_num_values: 2.5
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 2
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 1
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 1
                  }
                  type: QUANTILES
                }
                tot_num_values: 10
              }
              avg_length: 1.29999995232
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
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
        num_values_histogram_buckets=3,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_categorical_float_feature(self):
    batches = [
        pa.RecordBatch.from_arrays([pa.array([[1.0, 5.0, 10.0], [0.0]])],
                                   ['c']),
        pa.RecordBatch.from_arrays(
            [pa.array([[1.0, 1.0, 1.0, 5.0, 15.0], [-1.0]])], ['c']),
        pa.RecordBatch.from_arrays([pa.array([None, None], type=pa.null())],
                                   ['c'])
    ]
    expected_result = {
        types.FeaturePath(['c']):
            text_format.Parse(
                """
            path {
              step: 'c'
            }
            type: FLOAT
            string_stats {
              common_stats {
                num_non_missing: 4
                num_missing: 2
                min_num_values: 1
                max_num_values: 5
                avg_num_values: 2.5
                num_values_histogram {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 2
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 3.0
                    sample_count: 1
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 5.0
                    sample_count: 1
                  }
                  type: QUANTILES
                }
                tot_num_values: 10
              }
              avg_length: 3.3
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    schema = text_format.Parse(
        """
        feature {
          name: "c"
          type: FLOAT
          float_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=3, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_empty_batch(self):
    batches = [
        pa.RecordBatch.from_arrays([pa.array([], type=pa.list_(pa.binary()))],
                                   ['a'])
    ]
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

  def test_no_value_in_batch(self):
    batches = [
        pa.RecordBatch.from_arrays([
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

  def test_only_nan(self):
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[np.NaN]], type=pa.list_(pa.float32()))], ['a'])
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

  def test_schema_claims_bytes_but_actually_int(self):
    schema = text_format.Parse("""
        feature {
          name: "a"
          type: BYTES
          image_domain { }
        }""", schema_pb2.Schema())
    batches = [pa.RecordBatch.from_arrays([
        pa.array([], type=pa.list_(pa.int64()))], ['a'])]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse("""
            type: INT
            num_stats {
              common_stats {
              }
            }
            path {
              step: "a"
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_schema_claims_categorical_int_but_actually_float(self):
    # Categorical generators do not run for mismatched declared vs. actual
    # numeric types.
    schema = text_format.Parse("""
    feature {
      name: "a"
      type: INT
      int_domain { is_categorical: true }
    }""", schema_pb2.Schema())
    batches = [pa.RecordBatch.from_arrays([
        pa.array([], type=pa.list_(pa.float32()))], ['a'])]
    expected_result = {
        types.FeaturePath(['a']): text_format.Parse("""
            type: FLOAT
            num_stats {
              common_stats {
              }
            }
            path {
              step: "a"
            }
            """, statistics_pb2.FeatureNameStatistics())}
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_schema_claims_categorical_int_but_type_missing(self):
    # Categorical generators will run for a declared numeric with actual string
    # type, but output will be correctly string typed.
    schema = text_format.Parse(
        """
        feature {
          name: "a"
          type: INT
          int_domain { is_categorical: true }
        }""", schema_pb2.Schema())
    batches = [pa.RecordBatch.from_arrays([pa.array([[]])], ['a'])]
    expected_result = {
        types.FeaturePath(['a']):
            text_format.Parse(
                """
                type: STRING
                string_stats {
                  common_stats {
                    num_non_missing: 1
                    num_missing: 0
                    num_values_histogram {
                      buckets {
                        sample_count: 0.5
                      }
                      buckets {
                        sample_count: 0.5
                      }
                      type: QUANTILES
                    }
                    presence_and_valency_stats {
                      num_non_missing: 1
                    }
                    presence_and_valency_stats {}
                  }
                }
                path {
                  step: "a"
                }
                """, statistics_pb2.FeatureNameStatistics())
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=2,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_column_not_list(self):
    batches = [pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ['a'])]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError, r'Expected feature column to be a \(Large\)List'):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_invalid_value_numpy_dtype(self):
    batches = [pa.RecordBatch.from_arrays(
        [pa.array([[]], type=pa.list_(pa.date32()))], ['a'])]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError, 'Feature a has unsupported arrow type'):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_feature_with_inconsistent_types(self):
    batches = [
        pa.RecordBatch.from_arrays([pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]])],
                                   ['a']),
        pa.RecordBatch.from_arrays([pa.array([[1]])], ['a']),
    ]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError, 'Cannot determine the type'):
      self.assertCombinerOutputEqual(batches, generator, None)

  def test_with_invalid_utf8(self):
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[b'a'], [b'\xfc\xa1\xa1\xa1\xa1\xa1'], None])], ['a'])
    b2 = pa.RecordBatch.from_arrays([pa.array([[b'\xfc\xa1\xa1\xa1\xa1\xa1']])],
                                    ['a'])
    batches = [b1, b2]
    expected_result = {
        types.FeaturePath(['a']):
            text_format.Parse("""
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 1
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
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
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.75
                  }
                  type: QUANTILES
                }
                tot_num_values: 3
              }
              avg_length: 4.333333
              invalid_utf8_count: 2
            }
            path {
              step: "a"
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=4,
        num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)


_STRUCT_TEST_CASES = [
    dict(
        testcase_name='deep_struct',
        struct_column_as_list_dicts=[[{
            'l2': [
                {
                    'l3': [1, 2, 3]
                },
                {
                    'l3': [4, 5]
                },
            ],
        }, {
            'l2': [{}],
        }, {
            'l2': [{
                'l3': None
            }],
        }], None],
        expected_result_text_protos={
            ('c',):
                """
              type: STRUCT
              struct_stats {
                common_stats {
                  num_non_missing: 1
                  num_missing: 1
                  min_num_values: 3
                  max_num_values: 3
                  avg_num_values: 3.0
                  num_values_histogram {
                    buckets {
                      low_value: 3.0
                      high_value: 3.0
                      sample_count: 0.5
                    }
                    buckets {
                      low_value: 3.0
                      high_value: 3.0
                      sample_count: 0.5
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 3
                }
              }""",
            ('c', 'l2'):
                """
              type: STRUCT
              struct_stats {
                common_stats {
                  num_non_missing: 3
                  min_num_values: 1
                  max_num_values: 2
                  avg_num_values: 1.333333
                  num_values_histogram {
                    buckets {
                      low_value: 1.0
                      high_value: 1.0
                      sample_count: 2.0
                    }
                    buckets {
                      low_value: 1.0
                      high_value: 2.0
                      sample_count: 1.0
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 4
                }
              }""",
            ('c', 'l2', 'l3'):
                """
              type: INT
              num_stats {
                common_stats {
                  num_non_missing: 2
                  num_missing: 2
                  min_num_values: 2
                  max_num_values: 3
                  avg_num_values: 2.5
                  num_values_histogram {
                    buckets {
                      low_value: 2.0
                      high_value: 3.0
                      sample_count: 1.5
                    }
                    buckets {
                      low_value: 3.0
                      high_value: 3.0
                      sample_count: 0.5
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
                    sample_count: 2.0041667
                  }
                  buckets {
                    low_value: 2.3333333
                    high_value: 3.6666667
                    sample_count: 1.0041667
                  }
                  buckets {
                    low_value: 3.6666667
                    high_value: 5.0
                    sample_count: 1.9916667
                  }
                }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 2.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 4.0
                    sample_count: 1.0
                  }
                  buckets {
                    low_value: 4.0
                    high_value: 5.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }""",
        }),
    dict(
        testcase_name='leaf_is_categorical',
        struct_column_as_list_dicts=[
            [{
                'f1': [1, 2, 3],
                'f2': ['b']
            }],
            [{
                'f1': [3, 1],
                'f2': ['a']
            }, {
                'f1': [2]
            }],
        ],
        struct_column_schema="""
        name: "f1"
        type: INT
        int_domain {
          is_categorical: true
        }
        """,
        expected_result_text_protos={
            ('c',):
                """
              type: STRUCT
              struct_stats {
                common_stats {
                  num_non_missing: 2
                  min_num_values: 1
                  max_num_values: 2
                  avg_num_values: 1.5
                  num_values_histogram {
                    buckets {
                      low_value: 1.0
                      high_value: 2.0
                      sample_count: 1.5
                    }
                    buckets {
                      low_value: 2.0
                      high_value: 2.0
                      sample_count: 0.5
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 3
                }
              }""",
            ('c', 'f1'):
                """
              string_stats {
                common_stats {
                  num_non_missing: 3
                  min_num_values: 1
                  max_num_values: 3
                  avg_num_values: 2.0
                  num_values_histogram {
                    buckets {
                      low_value: 1.0
                      high_value: 2.0
                      sample_count: 2.0
                    }
                    buckets {
                      low_value: 2.0
                      high_value: 3.0
                      sample_count: 1.0
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 6
                }
                avg_length: 1.0
              }""",
            ('c', 'f2'):
                """
              type: STRING
              string_stats {
                common_stats {
                  num_non_missing: 2
                  num_missing: 1
                  min_num_values: 1
                  max_num_values: 1
                  avg_num_values: 1.0
                  num_values_histogram {
                    buckets {
                      low_value: 1.0
                      high_value: 1.0
                      sample_count: 1.0
                    }
                    buckets {
                      low_value: 1.0
                      high_value: 1.0
                      sample_count: 1.0
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 2
                }
                avg_length: 1.0
              }""",
        }),
    dict(
        testcase_name='nulls',
        struct_column_as_list_dicts=[
            [  # first element of 'c'
                {
                    'f1': [1.0],
                    # f2 is missing.
                },
                {
                    # f1, f2 are missing.
                }
            ],
            None,  # second element of 'c' -- missing/null.
            [  # third element of 'c' -- a list<struct> of length 2.
                {
                    'f2': [2.0],
                    # f1 is missing
                },
                None,  # f1, f2 are missing
            ],
            [  # fourth element of 'c'
                None,  # f1, f2 are missing
            ],
            [],  # fifth element of 'c'; note this is not counted as missing.
        ],
        expected_result_text_protos={
            ('c',): """
              type: STRUCT
              struct_stats {
                common_stats {
                  num_non_missing: 4
                  num_missing: 1
                  max_num_values: 2
                  avg_num_values: 1.25
                  num_values_histogram {
                    buckets {
                      high_value: 2.0
                      sample_count: 2.5
                    }
                    buckets {
                      low_value: 2.0
                      high_value: 2.0
                      sample_count: 1.5
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 5
                }
              }
            """,
            ('c', 'f1'): """
              type: FLOAT
              num_stats {
                common_stats {
                  num_non_missing: 1
                  num_missing: 4
                  min_num_values: 1
                  max_num_values: 1
                  avg_num_values: 1.0
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
                  tot_num_values: 1
                }
                mean: 1.0
                min: 1.0
                median: 1.0
                max: 1.0
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 1.0
                  }
                }
                histograms {
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.25
                  }
                  buckets {
                    low_value: 1.0
                    high_value: 1.0
                    sample_count: 0.25
                  }
                  type: QUANTILES
                }
              }""",
            ('c', 'f2'): """
              type: FLOAT
              num_stats {
                common_stats {
                  num_non_missing: 1
                  num_missing: 4
                  min_num_values: 1
                  max_num_values: 1
                  avg_num_values: 1.0
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
                  tot_num_values: 1
                }
                mean: 2.0
                min: 2.0
                median: 2.0
                max: 2.0
                histograms {
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 1.0
                  }
                }
                histograms {
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
              }""",
        }),
    dict(
        testcase_name='struct_not_nested_in_list',
        struct_column_as_list_dicts=[
            {'a': [b'meow', b'nyan']},
            {'b': [b'foo']},
        ],
        expected_result_text_protos={
            ('c',): """
              type: STRUCT
              struct_stats {
                common_stats {
                  num_non_missing: 2
                  min_num_values: 1
                  max_num_values: 1
                  avg_num_values: 1.0
                  num_values_histogram {
                    buckets {
                      low_value: 1.0
                      high_value: 1.0
                      sample_count: 1.0
                    }
                    buckets {
                      low_value: 1.0
                      high_value: 1.0
                      sample_count: 1.0
                    }
                    type: QUANTILES
                  }
                  tot_num_values: 2
                }
              }""",
            ('c', 'a'): """
              type: STRING
              string_stats {
                common_stats {
                  num_non_missing: 1
                  num_missing: 1
                  min_num_values: 2
                  max_num_values: 2
                  avg_num_values: 2.0
                  num_values_histogram {
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
                  tot_num_values: 2
                }
                avg_length: 4.0
              }""",
            ('c', 'b'): """
              type: STRING
              string_stats {
                common_stats {
                  num_non_missing: 1
                  num_missing: 1
                  min_num_values: 1
                  max_num_values: 1
                  avg_num_values: 1.0
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
                  tot_num_values: 1
                }
                avg_length: 3.0
              }""",
        }
    ),
]


class BasicStatsGeneratorStructStatsTest(test_util.CombinerStatsGeneratorTest,
                                         parameterized.TestCase):

  @parameterized.named_parameters(*_STRUCT_TEST_CASES)
  def test_struct(self, struct_column_as_list_dicts,
                  expected_result_text_protos, struct_column_schema=None):
    mid = len(struct_column_as_list_dicts) // 2

    # Also test merging multiple batches.
    batches = [
        pa.RecordBatch.from_arrays(
            [pa.array(struct_column_as_list_dicts[:mid])], ['c']),
        pa.RecordBatch.from_arrays(
            [pa.array(struct_column_as_list_dicts[mid:])], ['c']),
    ]

    expected_result = {}
    for k, v in expected_result_text_protos.items():
      feature_stats = text_format.Parse(
          v, statistics_pb2.FeatureNameStatistics())
      feature_path = types.FeaturePath(k)
      feature_stats.path.CopyFrom(feature_path.to_proto())
      expected_result[types.FeaturePath(k)] = feature_stats

    schema = None
    if struct_column_schema is not None:
      schema = text_format.Parse("""
        feature {
          name: "c"
          type: STRUCT
          struct_domain {
          }
        }""", schema_pb2.Schema())
      schema.feature[0].struct_domain.feature.add().CopyFrom(text_format.Parse(
          struct_column_schema, schema_pb2.Feature()))
    generator = basic_stats_generator.BasicStatsGenerator(
        schema=schema,
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_with_weights(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[1.0], [2.0]]),
            pa.array([[{
                'f1': [{
                    'f2': [1, 2]
                }, {
                    'f2': [0]
                }]
            }], [{
                'f1': [{
                    'f2': [3, 3]
                }]
            }]])
        ], ['w', 'c'])
    ]

    expected_result = {
        types.FeaturePath(['c']):
            text_format.Parse(
                """
          type: STRUCT
          struct_stats {
            common_stats {
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1.0
              num_values_histogram {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 3.0
                avg_num_values: 1.0
                tot_num_values: 3.0
              }
              tot_num_values: 2
            }
          }
          path {
            step: "c"
          }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['c', 'f1']):
            text_format.Parse(
                """
          type: STRUCT
          struct_stats {
            common_stats {
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 2
              avg_num_values: 1.5
              num_values_histogram {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 1.5
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.5
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 3.0
                avg_num_values: 1.3333333
                tot_num_values: 4.0
              }
              tot_num_values: 3
            }
          }
          path {
            step: "c"
            step: "f1"
          }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['c', 'f1', 'f2']):
            text_format.Parse(
                """
          num_stats {
            common_stats {
              num_non_missing: 3
              min_num_values: 1
              max_num_values: 2
              avg_num_values: 1.666667
              num_values_histogram {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 4.0
                avg_num_values: 1.75
                tot_num_values: 7.0
              }
              tot_num_values: 5
            }
            mean: 1.8
            std_dev: 1.1661904
            num_zeros: 1
            median: 2.0
            max: 3.0
            histograms {
              buckets {
                high_value: 1.0
                sample_count: 2.0
              }
              buckets {
                low_value: 1.0
                high_value: 2.0
                sample_count: 1.0
              }
              buckets {
                low_value: 2.0
                high_value: 3.0
                sample_count: 2.0
              }
            }
            histograms {
              buckets {
                high_value: 1.0
                sample_count: 2.0
              }
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
            weighted_numeric_stats {
              mean: 2.1428571
              std_dev: 1.1248583
              median: 3.0
              histograms {
                buckets {
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 3.0
                  sample_count: 4.0
                }
              }
              histograms {
                buckets {
                  high_value: 1.0
                  sample_count: 2.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.6666667
                }
                buckets {
                  low_value: 3.0
                  high_value: 3.0
                  sample_count: 1.6666667
                }
                buckets {
                  low_value: 3.0
                  high_value: 3.0
                  sample_count: 1.6666667
                }
                type: QUANTILES
              }
            }
          }
          path {
            step: "c"
            step: "f1"
            step: "f2"
          }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['w']):
            text_format.Parse(
                """
          type: FLOAT
          num_stats {
            common_stats {
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
              avg_num_values: 1.0
              num_values_histogram {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 3.0
                avg_num_values: 1.0
                tot_num_values: 3.0
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
                high_value: 1.33333333333
                sample_count: 1.0016584
              }
              buckets {
                low_value: 1.33333333333
                high_value: 1.66666666667
                sample_count: 0.0016584
              }
              buckets {
                low_value: 1.66666666667
                high_value: 2.0
                sample_count: 0.9966833
              }
            }
            histograms {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 1.0
              }
              buckets {
                low_value: 1.0
                high_value: 2.0
                sample_count: 0.3333333
              }
              buckets {
                low_value: 2.0
                high_value: 2.0
                sample_count: 0.3333333
              }
              buckets {
                low_value: 2.0
                high_value: 2.0
                sample_count: 0.3333333
              }
              type: QUANTILES
            }
            weighted_numeric_stats {
              mean: 1.66666666667
              std_dev: 0.471404520791
              median: 2.0
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.33333333333
                  sample_count: 1.0024969
                }
                buckets {
                  low_value: 1.33333333333
                  high_value: 1.66666666667
                  sample_count: 0.0024969
                }
                buckets {
                  low_value: 1.66666666667
                  high_value: 2.0
                  sample_count: 1.9950062
                }
              }
              histograms {
                buckets {
                  low_value: 1.0
                  high_value: 1.0
                  sample_count: 1.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 0.6666667
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.6666667
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 0.6666667
                }
                type: QUANTILES
              }
            }
          }
      path {
        step: "w"
      }
    """, statistics_pb2.FeatureNameStatistics()),
    }
    generator = basic_stats_generator.BasicStatsGenerator(
        example_weight_map=ExampleWeightMap(weight_feature='w'),
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4)
    self.assertCombinerOutputEqual(batches, generator, expected_result)


_NESTED_TEST_CASES = [
    dict(
        testcase_name='nested',
        batches=[
            pa.RecordBatch.from_arrays([
                pa.array([None, None],
                         type=pa.large_list(
                             pa.large_list(pa.list_(pa.large_binary())))),
                pa.array([[1.0], [1.0]]),
            ], ['a', 'w']),
            pa.RecordBatch.from_arrays([
                pa.array([
                    [[[b'a', b'a'], [b'a'], None], None, []],
                    [[[b'a', b'a']], [[b'a']]],
                ]),
                pa.array([[1.0], [1.0]]),
            ], ['a', 'w']),
            # in this batch, 'a' has the same nestedness, but its type is
            # unknown. Note that here pa.null() means pa.list_(<unknown_type>).
            pa.RecordBatch.from_arrays([
                pa.array([
                    [[None, None], None, []],
                ],
                         type=pa.list_(pa.list_(pa.null()))),
                pa.array([[1.0]])
            ], ['a', 'w'])
        ],
        weight_column='w',
        expected_result={
            types.FeaturePath(['a']):
                """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 2
                min_num_values: 2
                max_num_values: 3
                avg_num_values: 2.666667
                num_values_histogram {
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 2.0
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
                weighted_common_stats {
                  num_non_missing: 3.0
                  num_missing: 2.0
                  avg_num_values: 2.6666667
                  tot_num_values: 8.0
                }
                tot_num_values: 8
                presence_and_valency_stats {
                  num_non_missing: 3
                  num_missing: 2
                  min_num_values: 2
                  max_num_values: 3
                  tot_num_values: 8
                }
                presence_and_valency_stats {
                  num_non_missing: 6
                  num_missing: 2
                  max_num_values: 3
                  tot_num_values: 7
                }
                presence_and_valency_stats {
                  num_non_missing: 4
                  num_missing: 3
                  min_num_values: 1
                  max_num_values: 2
                  tot_num_values: 6
                }
                weighted_presence_and_valency_stats {
                  num_non_missing: 3.0
                  num_missing: 2.0
                  avg_num_values: 2.6666667
                  tot_num_values: 8.0
                }
                weighted_presence_and_valency_stats {
                  num_non_missing: 6.0
                  num_missing: 2.0
                  avg_num_values: 1.1666667
                  tot_num_values: 7.0
                }
                weighted_presence_and_valency_stats {
                  num_non_missing: 4.0
                  num_missing: 3.0
                  avg_num_values: 1.5
                  tot_num_values: 6.0
                }
              }
              avg_length: 1.0
            }
            custom_stats {
              name: "level_2_value_list_length_quantiles"
              histogram {
                buckets {
                  high_value: 1.0
                  sample_count: 4.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 2.0
                }
                type: QUANTILES
              }
            }
            custom_stats {
              name: "level_2_value_list_length_standard"
              histogram {
                buckets {
                  high_value: 1.5
                  sample_count: 4.5
                }
                buckets {
                  low_value: 1.5
                  high_value: 3.0
                  sample_count: 1.5
                }
              }
            }
            custom_stats {
              name: "level_3_value_list_length_quantiles"
              histogram {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 3.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            custom_stats {
              name: "level_3_value_list_length_standard"
              histogram {
                buckets {
                  low_value: 1.0
                  high_value: 1.5
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.5
                  high_value: 2.0
                  sample_count: 2.5
                }
              }
            }
            path {
              step: "a"
            }"""
        }),
    dict(
        testcase_name='nested_null',
        batches=[
            pa.RecordBatch.from_arrays([
                pa.array([[None, None], None, []],
                         type=pa.large_list(pa.null()))
            ], ['a']),
        ],
        expected_result={
            types.FeaturePath(['a']):
                """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                num_missing: 1
                max_num_values: 2
                avg_num_values: 1.0
                num_values_histogram {
                  buckets {
                    high_value: 2.0
                    sample_count: 1.5
                  }
                  buckets {
                    low_value: 2.0
                    high_value: 2.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                tot_num_values: 2
                presence_and_valency_stats {
                  num_non_missing: 2
                  num_missing: 1
                  max_num_values: 2
                  tot_num_values: 2
                }
                presence_and_valency_stats {
                  num_missing: 2
                }
              }
            }
            path {
            step: "a"
            }"""
        }),
    dict(
        testcase_name='nested_with_non_utf8',
        batches=[
            pa.RecordBatch.from_arrays([
                pa.array([
                    [[[b'a', b'a'], [b'a'], None], None, []],
                    [[[b'a', b'\xfc\xa1\xa1\xa1\xa1\xa1']], [[b'a']]],
                ])
            ], ['a']),
        ],
        expected_result={
            types.FeaturePath(['a']):
                """
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 2
                min_num_values: 2
                max_num_values: 3
                avg_num_values: 2.5
                num_values_histogram {
                  buckets {
                    low_value: 2.0
                    high_value: 3.0
                    sample_count: 1.5
                  }
                  buckets {
                    low_value: 3.0
                    high_value: 3.0
                    sample_count: 0.5
                  }
                  type: QUANTILES
                }
                tot_num_values: 5
                presence_and_valency_stats {
                  num_non_missing: 2
                  min_num_values: 2
                  max_num_values: 3
                  tot_num_values: 5
                }
                presence_and_valency_stats {
                  num_non_missing: 4
                  num_missing: 1
                  max_num_values: 3
                  tot_num_values: 5
                }
                presence_and_valency_stats {
                  num_non_missing: 4
                  num_missing: 1
                  min_num_values: 1
                  max_num_values: 2
                  tot_num_values: 6
                }
              }
              avg_length: 1.833333
              invalid_utf8_count: 1
            }
            custom_stats {
              name: "level_2_value_list_length_quantiles"
              histogram {
                buckets {
                  high_value: 1.0
                  sample_count: 3.0
                }
                buckets {
                  low_value: 1.0
                  high_value: 3.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            custom_stats {
              name: "level_2_value_list_length_standard"
              histogram {
                buckets {
                  high_value: 1.5
                  sample_count: 3.25
                }
                buckets {
                  low_value: 1.5
                  high_value: 3.0
                  sample_count: 0.75
                }
              }
            }
            custom_stats {
              name: "level_3_value_list_length_quantiles"
              histogram {
                buckets {
                  low_value: 1.0
                  high_value: 2.0
                  sample_count: 3.0
                }
                buckets {
                  low_value: 2.0
                  high_value: 2.0
                  sample_count: 1.0
                }
                type: QUANTILES
              }
            }
            custom_stats {
              name: "level_3_value_list_length_standard"
              histogram {
                buckets {
                  low_value: 1.0
                  high_value: 1.5
                  sample_count: 1.5
                }
                buckets {
                  low_value: 1.5
                  high_value: 2.0
                  sample_count: 2.5
                }
              }
            }
            path {
              step: "a"
            }"""
        }),
]


class BasicStatsGeneratorNestedListTest(
    test_util.CombinerStatsGeneratorTest, parameterized.TestCase):
  # pylint: disable=g-error-prone-assert-raises

  @parameterized.named_parameters(*_NESTED_TEST_CASES)
  def test_nested_list(self, batches, expected_result, weight_column=None):
    generator = basic_stats_generator.BasicStatsGenerator(
        num_values_histogram_buckets=2, num_histogram_buckets=3,
        num_quantiles_histogram_buckets=4,
        example_weight_map=ExampleWeightMap(weight_feature=weight_column))
    expected_result = {
        path: text_format.Parse(pbtxt, statistics_pb2.FeatureNameStatistics())
        for path, pbtxt in expected_result.items()
    }
    self.assertCombinerOutputEqual(batches, generator, expected_result,
                                   only_match_expected_feature_stats=True)

  def test_basic_stats_generator_different_nest_levels(self):
    batches = [
        pa.RecordBatch.from_arrays([pa.array([[1]])], ['a']),
        pa.RecordBatch.from_arrays([pa.array([[[1]]])], ['a']),
    ]
    generator = basic_stats_generator.BasicStatsGenerator()
    with self.assertRaisesRegex(
        ValueError, 'Unable to merge common stats with different nest levels'):
      self.assertCombinerOutputEqual(batches, generator, None)

if __name__ == '__main__':
  absltest.main()
