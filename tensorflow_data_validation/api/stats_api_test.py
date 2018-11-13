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
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
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
                high_value: 750.0
                sample_count: 1.0
              }
              buckets {
                low_value: 750.0
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

  def test_stats_pipeline_with_feature_whitelist(self):
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
          feature_whitelist=['b'],
          num_top_values=2,
          num_rank_histogram_buckets=3,
          num_values_histogram_buckets=3,
          num_histogram_buckets=3,
          num_quantiles_histogram_buckets=4)
      result = (
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_invalid_feature_whitelist(self):
    examples = [{'a': np.array([1.0, 2.0])}]
    with self.assertRaises(TypeError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(feature_whitelist={})
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

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

  def test_stats_pipeline_with_schema(self):
    # input with three examples.
    examples = [{'a': np.array([1, 3, 5, 7])},
                {'a': np.array([2, 4, 6, 8])},
                {'a': np.array([0, 3, 6, 9])}]
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
    expected_result = text_format.Parse("""
    datasets {
      num_examples: 3
      features {
        name: "a"
        type: INT
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
          unique: 10
          top_values {
            value: "6"
            frequency: 2.0
          }
          top_values {
            value: "3"
            frequency: 2.0
          }
          avg_length: 1.0
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "6"
              sample_count: 2.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "3"
              sample_count: 2.0
            }
            buckets {
              low_rank: 2
              high_rank: 2
              label: "9"
              sample_count: 1.0
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          schema=schema,
          num_top_values=2,
          num_rank_histogram_buckets=3,
          num_values_histogram_buckets=3)
      result = (
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result, test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_stats_pipeline_with_weight_feature(self):
    # input with four examples.
    examples = [
        {'a': np.array([1.0, 2.0]),
         'b': np.array(['a', 'b', 'c', 'e']),
         'w': np.array([1.0])},
        {'a': np.array([3.0, 4.0, 5.0]),
         'b': None,
         'w': np.array([2.0])},
        {'a': np.array([1.0,]),
         'b': np.array(['d', 'e']),
         'w': np.array([3.0,])},
        {'a': None,
         'b': np.array(['a', 'c', 'd', 'a']),
         'w': np.array([1.0])}
    ]

    expected_result = text_format.Parse("""
    datasets {
      num_examples: 4
      features {
        name: 'a'
        type: FLOAT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 1
            min_num_values: 1
            max_num_values: 3
            avg_num_values: 2.0
            tot_num_values: 6
            num_values_histogram {
              buckets { low_value: 1.0 high_value: 2.0 sample_count: 1 }
              buckets { low_value: 2.0 high_value: 3.0 sample_count: 1 }
              buckets { low_value: 3.0 high_value: 3.0 sample_count: 1 }
              type: QUANTILES
            }
            weighted_common_stats {
              num_non_missing: 6.0
              num_missing: 1.0
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
            buckets { low_value: 1.0 high_value: 1.0 sample_count: 1.5 }
            buckets { low_value: 1.0 high_value: 3.0 sample_count: 1.5 }
            buckets { low_value: 3.0 high_value: 4.0 sample_count: 1.5 }
            buckets { low_value: 4.0 high_value: 5.0 sample_count: 1.5 }
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
              buckets { low_value: 1.0 high_value: 1.0 sample_count: 2.75 }
              buckets { low_value: 1.0 high_value: 3.0 sample_count: 2.75 }
              buckets { low_value: 3.0 high_value: 4.0 sample_count: 2.75 }
              buckets { low_value: 4.0 high_value: 5.0 sample_count: 2.75 }
              type: QUANTILES
            }
          }
        }
      }
      features {
        name: 'b'
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 1
            min_num_values: 2
            max_num_values: 4
            avg_num_values: 3.33333301544
            num_values_histogram {
              buckets { low_value: 2.0 high_value: 4.0 sample_count: 1.0 }
              buckets { low_value: 4.0 high_value: 4.0 sample_count: 1.0 }
              buckets { low_value: 4.0 high_value: 4.0 sample_count: 1.0 }
              type: QUANTILES
            }
            weighted_common_stats {
              num_non_missing: 5.0
              num_missing: 2.0
              avg_num_values: 2.8
              tot_num_values: 14.0
            }
            tot_num_values: 10
          }
          avg_length: 1.0
          unique: 5
          top_values { value: 'a' frequency: 3.0 }
          top_values { value: 'e' frequency: 2.0 }
          rank_histogram {
            buckets { low_rank: 0 high_rank: 0 label: "a" sample_count: 3.0 }
            buckets { low_rank: 1 high_rank: 1 label: "e" sample_count: 2.0 }
            buckets { low_rank: 2 high_rank: 2 label: "d" sample_count: 2.0 }
          }
          weighted_string_stats {
            top_values { value: 'e' frequency: 4.0 }
            top_values { value: 'd' frequency: 4.0 }
            rank_histogram {
              buckets { low_rank: 0 high_rank: 0 label: "e" sample_count: 4.0 }
              buckets { low_rank: 1 high_rank: 1 label: "d" sample_count: 4.0 }
              buckets { low_rank: 2 high_rank: 2 label: "a" sample_count: 3.0 }
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          weight_feature='w',
          num_top_values=2,
          num_rank_histogram_buckets=3,
          num_values_histogram_buckets=3,
          num_histogram_buckets=3,
          num_quantiles_histogram_buckets=4)
      result = (
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
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

  def test_invalid_sample_count_zero(self):
    examples = [{}]
    with self.assertRaises(ValueError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(sample_count=0)
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

  def test_invalid_sample_count_negative(self):
    examples = [{}]
    with self.assertRaises(ValueError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(sample_count=-1)
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

  def test_invalid_both_sample_count_and_sample_rate(self):
    examples = [{}]
    with self.assertRaises(ValueError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(sample_count=100, sample_rate=0.5)
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

  def test_invalid_sample_rate_zero(self):
    examples = [{}]
    with self.assertRaises(ValueError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(sample_rate=0)
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

  def test_invalid_sample_rate_negative(self):
    examples = [{}]
    with self.assertRaises(ValueError):
      with beam.Pipeline() as p:
        options = stats_options.StatsOptions(sample_rate=-1)
        _ = (p | beam.Create(examples) | stats_api.GenerateStatistics(options))

  def test_custom_generators(self):

    # Dummy PTransform that returns two DatasetFeatureStatistics protos.
    class CustomPTransform(beam.PTransform):

      def expand(self, pcoll):
        stats_proto1 = statistics_pb2.DatasetFeatureStatistics()
        proto1_feat = stats_proto1.features.add()
        proto1_feat.name = 'a'
        custom_stat1 = proto1_feat.custom_stats.add()
        custom_stat1.name = 'my_stat_a'
        custom_stat1.str = 'my_val_a'

        stats_proto2 = statistics_pb2.DatasetFeatureStatistics()
        proto2_feat = stats_proto2.features.add()
        proto2_feat.name = 'b'
        custom_stat2 = proto2_feat.custom_stats.add()
        custom_stat2.name = 'my_stat_b'
        custom_stat2.str = 'my_val_b'
        return [stats_proto1, stats_proto2]

    examples = [{'a': np.array([], dtype=np.int32),
                 'b': np.array([], dtype=np.int32)}]
    expected_result = text_format.Parse("""
    datasets {
      num_examples: 1
      features {
        name: 'a'
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            tot_num_values: 0
            num_values_histogram {
              buckets {
                low_value: 0
                high_value: 0
                sample_count: 0.5
              }
              buckets {
                low_value: 0
                high_value: 0
                sample_count: 0.5
              }
              type: QUANTILES
            }
          }
        }
        custom_stats {
          name: 'my_stat_a'
          str: 'my_val_a'
        }
      }
      features {
        name: 'b'
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            tot_num_values: 0
            num_values_histogram {
              buckets {
                low_value: 0
                high_value: 0
                sample_count: 0.5
              }
              buckets {
                low_value: 0
                high_value: 0
                sample_count: 0.5
              }
              type: QUANTILES
            }
          }
        }
        custom_stats {
          name: 'my_stat_b'
          str: 'my_val_b'
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    # Create a transform stats generator.
    transform_stats_gen = stats_generator.TransformStatsGenerator(
        name='CustomStatsGenerator',
        ptransform=CustomPTransform())
    with beam.Pipeline() as p:
      options = stats_options.StatsOptions(
          generators=[transform_stats_gen], num_values_histogram_buckets=2)
      result = (
          p | beam.Create(examples) | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_invalid_custom_generators(self):
    examples = [{'a': np.array([1.0, 2.0])}]
    with self.assertRaises(TypeError):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(examples) | stats_api.GenerateStatistics(
                stats_options.StatsOptions(generators={})))

  def test_filter_features(self):
    input_batch = {'a': np.array([]), 'b': np.array([]), 'c': np.array([])}
    actual = stats_api._filter_features(input_batch, ['a', 'c'])
    expected = {'a': np.array([]), 'c': np.array([])}
    self.assertEqual(sorted(actual), sorted(expected))

  def test_filter_features_empty(self):
    input_batch = {'a': np.array([])}
    actual = stats_api._filter_features(input_batch, [])
    expected = {}
    self.assertEqual(sorted(actual), sorted(expected))

  def test_invalid_stats_options(self):
    examples = [{'a': np.array([1.0, 2.0])}]
    with self.assertRaisesRegexp(TypeError, '.*should be a StatsOptions.'):
      with beam.Pipeline() as p:
        _ = (p | beam.Create(examples)
             | stats_api.GenerateStatistics(options={}))


if __name__ == '__main__':
  absltest.main()
