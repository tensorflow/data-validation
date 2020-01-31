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

"""Tests for the overall statistics pipeline using Beam."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pyarrow as pa
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsAPITest(absltest.TestCase):

  def test_stats_pipeline(self):
    # input with three tables.
    tables = [
        pa.Table.from_arrays([
            pa.array([[1.0, 2.0]]),
            pa.array([['a', 'b', 'c', 'e']]),
            pa.array([np.linspace(1, 500, 500, dtype=np.int32)]),
        ], ['a', 'b', 'c']),
        pa.Table.from_arrays([
            pa.array([[3.0, 4.0, np.NaN, 5.0]]),
            pa.array([['a', 'c', 'd', 'a']]),
            pa.array([np.linspace(501, 1250, 750, dtype=np.int32)]),
        ], ['a', 'b', 'c']),
        pa.Table.from_arrays([
            pa.array([[1.0]]),
            pa.array([['a', 'b', 'c', 'd']]),
            pa.array([np.linspace(1251, 3000, 1750, dtype=np.int32)]),
        ], ['a', 'b', 'c'])
    ]

    expected_result = text_format.Parse("""
    datasets {
      num_examples: 3
      features {
        path {
          step: 'a'
        }
        type: FLOAT
        num_stats {
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
                high_value: 1.0
                sample_count: 1.0
              }
              buckets {
                low_value: 1.0
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
            num_nan: 1
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
            num_nan: 1
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
      }
      features {
        path {
          step: 'c'
        }
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 0
            min_num_values: 500
            max_num_values: 1750
            avg_num_values: 1000.0
            tot_num_values: 3000
            num_values_histogram {
              buckets {
                low_value: 500.0
                high_value: 500.0
                sample_count: 1.0
              }
              buckets {
                low_value: 500.0
                high_value: 1750.0
                sample_count: 1.0
              }
              buckets {
                low_value: 1750.0
                high_value: 1750.0
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
      }
      features {
        path {
          step: 'b'
        }
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 3
            min_num_values: 4
            max_num_values: 4
            avg_num_values: 4.0
            tot_num_values: 12
            num_values_histogram {
              buckets {
                low_value: 4.0
                high_value: 4.0
                sample_count: 1.0
              }
              buckets {
                low_value: 4.0
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
          unique: 5
          top_values {
            value: "a"
            frequency: 4.0
          }
          top_values {
            value: "c"
            frequency: 3.0
          }
          avg_length: 1.0
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
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          num_top_values=2,
          num_rank_histogram_buckets=3,
          num_values_histogram_buckets=3,
          num_histogram_buckets=3,
          num_quantiles_histogram_buckets=4,
          epsilon=0.001)
      result = (
          p | beam.Create(tables) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  _sampling_test_expected_result = text_format.Parse("""
    datasets {
      num_examples: 1
      features {
        path {
          step: 'c'
        }
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            min_num_values: 3000
            max_num_values: 3000
            avg_num_values: 3000.0
            tot_num_values: 3000
            num_values_histogram {
              buckets {
                low_value: 3000.0
                high_value: 3000.0
                sample_count: 0.5
              }
              buckets {
                low_value: 3000.0
                high_value: 3000.0
                sample_count: 0.5
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
              high_value: 1500.5
              sample_count: 1499.5
            }
            buckets {
              low_value: 1500.5
              high_value: 3000.0
              sample_count: 1500.5
            }
            type: STANDARD
          }
          histograms {
            buckets {
              low_value: 1.0
              high_value: 1501.0
              sample_count: 1500.0
            }
            buckets {
              low_value: 1501.0
              high_value: 3000.0
              sample_count: 1500.0
            }
            type: QUANTILES
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

  def test_stats_pipeline_with_examples_with_no_values(self):
    tables = [
        pa.Table.from_arrays([
            pa.array([[]], type=pa.list_(pa.float32())),
            pa.array([[]], type=pa.list_(pa.binary())),
            pa.array([[]], type=pa.list_(pa.int32())),
            pa.array([[2]]),
        ], ['a', 'b', 'c', 'w']),
        pa.Table.from_arrays([
            pa.array([[]], type=pa.list_(pa.float32())),
            pa.array([[]], type=pa.list_(pa.binary())),
            pa.array([[]], type=pa.list_(pa.int32())),
            pa.array([[2]]),
        ], ['a', 'b', 'c', 'w']),
        pa.Table.from_arrays([
            pa.array([[]], type=pa.list_(pa.float32())),
            pa.array([[]], type=pa.list_(pa.binary())),
            pa.array([[]], type=pa.list_(pa.int32())),
            pa.array([[2]]),
        ], ['a', 'b', 'c', 'w'])
    ]

    expected_result = text_format.Parse(
        """
      datasets{
        num_examples: 3
        features {
          path {
            step: 'a'
          }
          type: FLOAT
          num_stats {
            common_stats {
              num_non_missing: 3
              num_values_histogram {
                buckets {
                  sample_count: 1.5
                }
                buckets {
                  sample_count: 1.5
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 6
              }
            }
          }
        }
        features {
          path {
            step: 'b'
          }
          type: STRING
          string_stats {
            common_stats {
              num_non_missing: 3
              num_values_histogram {
                buckets {
                  sample_count: 1.5
                }
                buckets {
                  sample_count: 1.5
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 6
              }
            }
          }
        }
        features {
          path {
            step: 'c'
          }
          type: INT
          num_stats {
            common_stats {
              num_non_missing: 3
              num_values_histogram {
                buckets {
                  sample_count: 1.5
                }
                buckets {
                  sample_count: 1.5
                }
                type: QUANTILES
              }
              weighted_common_stats {
                num_non_missing: 6
              }
            }
          }
        }
        features {
          path {
          step: 'w'
        }
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 0
            min_num_values: 1
            max_num_values: 1
            avg_num_values: 1.0
            tot_num_values: 3
            num_values_histogram {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 1.5
              }
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 1.5
              }
              type: QUANTILES
            }
            weighted_common_stats {
                num_non_missing: 6.0
                avg_num_values: 1.0
                tot_num_values: 6.0
            }
          }
          mean: 2.0
          std_dev: 0.0
          min: 2.0
          max: 2.0
          median: 2.0
          histograms {
            buckets {
              low_value: 2.0
              high_value: 2.0
              sample_count: 3.0
            }
            type: STANDARD
          }
          histograms {
            buckets {
              low_value: 2.0
              high_value: 2.0
              sample_count: 3.0
            }
            type: QUANTILES
          }
          weighted_numeric_stats {
            mean: 2.0
            median: 2.0
            histograms {
              buckets {
                low_value: 2.0
                high_value: 2.0
                sample_count: 6.0
              }
              type: STANDARD
            }
            histograms {
              buckets {
                low_value: 2.0
                high_value: 2.0
                sample_count: 6.0
              }
              type: QUANTILES
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          weight_feature='w',
          num_top_values=1,
          num_rank_histogram_buckets=1,
          num_values_histogram_buckets=2,
          num_histogram_buckets=1,
          num_quantiles_histogram_buckets=1,
          epsilon=0.001)
      result = (
          p | beam.Create(tables) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_stats_pipeline_with_zero_examples(self):
    expected_result = text_format.Parse(
        """
        datasets {
          num_examples: 0
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          num_top_values=1,
          num_rank_histogram_buckets=1,
          num_values_histogram_buckets=2,
          num_histogram_buckets=1,
          num_quantiles_histogram_buckets=1,
          epsilon=0.001)
      result = (p | beam.Create([]) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_stats_pipeline_with_sample_count(self):
    # input with three tables.
    tables = [
        pa.Table.from_arrays([
            pa.array([np.linspace(1, 3000, 3000, dtype=np.int32)])], ['c']),
        pa.Table.from_arrays([
            pa.array([np.linspace(1, 3000, 3000, dtype=np.int32)])], ['c']),
        pa.Table.from_arrays([
            pa.array([np.linspace(1, 3000, 3000, dtype=np.int32)])], ['c']),
    ]

    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          sample_count=3000,
          num_top_values=2,
          num_rank_histogram_buckets=2,
          num_values_histogram_buckets=2,
          num_histogram_buckets=2,
          num_quantiles_histogram_buckets=2,
          epsilon=0.001,
          desired_batch_size=3000)
      result = (
          p | beam.Create(tables) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, self._sampling_test_expected_result))

  def test_stats_pipeline_with_sample_rate(self):
    tables = [
        pa.Table.from_arrays([
            pa.array([np.linspace(1, 3000, 3000, dtype=np.int32)])], ['c']),
    ]

    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          sample_rate=1.0,
          num_top_values=2,
          num_rank_histogram_buckets=2,
          num_values_histogram_buckets=2,
          num_histogram_buckets=2,
          num_quantiles_histogram_buckets=2,
          epsilon=0.001)
      result = (
          p | beam.Create(tables) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, self._sampling_test_expected_result))

  def test_invalid_stats_options(self):
    tables = [pa.Table.from_arrays([])]
    with self.assertRaisesRegexp(TypeError, '.*should be a StatsOptions.'):
      with beam.Pipeline() as p:
        _ = (p | beam.Create(tables)
             | stats_api.GenerateStatistics(options={}))


if __name__ == '__main__':
  absltest.main()
