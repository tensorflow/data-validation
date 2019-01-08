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
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsAPITest(absltest.TestCase):

  def test_stats_pipeline(self):
    # input with three examples.
    examples = [{'a': np.array([1.0, 2.0]),
                 'b': np.array(['a', 'b', 'c', 'e']),
                 'c': np.linspace(1, 500, 500, dtype=np.int32)},
                {'a': np.array([3.0, 4.0, np.NaN, 5.0]),
                 'b': np.array(['a', 'c', 'd', 'a']),
                 'c': np.linspace(501, 1250, 750, dtype=np.int32)},
                {'a': np.array([1.0]),
                 'b': np.array(['a', 'b', 'c', 'd']),
                 'c': np.linspace(1251, 3000, 1750, dtype=np.int32)}]

    expected_result = text_format.Parse("""
    datasets {
      num_examples: 3
      features {
        name: 'a'
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
        name: 'c'
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
        name: "b"
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
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_empty_input(self):
    examples = []
    expected_result = text_format.Parse("""
    datasets {
      num_examples: 0
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    with beam.Pipeline() as p:
      result = p | beam.Create(examples) | stats_api.GenerateStatistics(
          stats_options.StatsOptions())
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  _sampling_test_expected_result = text_format.Parse("""
    datasets {
      num_examples: 1
      features {
        name: 'c'
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

  def test_stats_pipeline_with_sample_count(self):
    # input with three examples.
    examples = [{'c': np.linspace(1, 3000, 3000, dtype=np.int32)},
                {'c': np.linspace(1, 3000, 3000, dtype=np.int32)},
                {'c': np.linspace(1, 3000, 3000, dtype=np.int32)}]

    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          sample_count=1,
          num_top_values=2,
          num_rank_histogram_buckets=2,
          num_values_histogram_buckets=2,
          num_histogram_buckets=2,
          num_quantiles_histogram_buckets=2,
          epsilon=0.001)
      result = (
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, self._sampling_test_expected_result))

  def test_stats_pipeline_with_sample_rate(self):
    # input with three examples.
    examples = [{'c': np.linspace(1, 3000, 3000, dtype=np.int32)}]

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
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, self._sampling_test_expected_result))

  def test_invalid_stats_options(self):
    examples = [{'a': np.array([1.0, 2.0])}]
    with self.assertRaisesRegexp(TypeError, '.*should be a StatsOptions.'):
      with beam.Pipeline() as p:
        _ = (p | beam.Create(examples)
             | stats_api.GenerateStatistics(options={}))


if __name__ == '__main__':
  absltest.main()
