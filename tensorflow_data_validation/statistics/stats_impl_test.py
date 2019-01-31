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
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.types_compat import List

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

GENERATE_STATS_TESTS = [
    {
        'testcase_name':
            'feature_whitelist',
        'examples': [
            {
                'a': np.array([1.0, 2.0]),
                'b': np.array(['a', 'b', 'c', 'e']),
                'c': np.linspace(1, 500, 500, dtype=np.int32)
            },
            {
                'a': np.array([3.0, 4.0, np.NaN, 5.0]),
                'b': np.array(['a', 'c', 'd', 'a']),
                'c': np.linspace(501, 1250, 750, dtype=np.int32)
            },
            {
                'a': np.array([1.0]),
                'b': np.array(['a', 'b', 'c', 'd']),
                'c': np.linspace(1251, 3000, 1750, dtype=np.int32)
            }
        ],
        'options':
            stats_options.StatsOptions(
                feature_whitelist=['b'],
                num_top_values=2,
                num_rank_histogram_buckets=3,
                num_values_histogram_buckets=3,
                num_histogram_buckets=3,
                num_quantiles_histogram_buckets=4),
        'expected_result_proto_text':
            """
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
                    """,
    },
    {
        'testcase_name':
            'schema',
        'examples': [
            {'a': np.array([1, 3, 5, 7])},
            {'a': np.array([2, 4, 6, 8])},
            {'a': np.array([0, 3, 6, 9])}
        ],
        'options':
            stats_options.StatsOptions(
                num_top_values=2,
                num_rank_histogram_buckets=3,
                num_values_histogram_buckets=3),
        'expected_result_proto_text':
            """
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
              """,
        'schema':
            text_format.Parse(
                """
              feature {
                name: "a"
                type: INT
                int_domain {
                  is_categorical: true
                }
              }
              """, schema_pb2.Schema())
    },
    {
        'testcase_name':
            'weight_feature',
        'examples': [
            {
                'a': np.array([1.0, 2.0]),
                'b': np.array(['a', 'b', 'c', 'e']),
                'w': np.array([1.0])
            }, {
                'a': np.array([3.0, 4.0, 5.0, 6.0]),
                'b': np.array(['d', 'e']),
                'w': np.array([2.0])
            },
        ],
        'options':
            stats_options.StatsOptions(
                weight_feature='w',
                num_top_values=2,
                num_rank_histogram_buckets=2,
                num_values_histogram_buckets=2,
                num_histogram_buckets=2,
                num_quantiles_histogram_buckets=2),
        'expected_result_proto_text':
            """
            datasets {
              num_examples: 2
              features {
                name: 'a'
                type: FLOAT
                num_stats {
                  common_stats {
                    num_non_missing: 2
                    num_missing: 0
                    min_num_values: 2
                    max_num_values: 4
                    avg_num_values: 3.0
                    tot_num_values: 6
                    num_values_histogram {
                      buckets {
                        low_value: 2.0
                        high_value: 4.0
                        sample_count: 1
                      }
                      buckets {
                        low_value: 4.0
                        high_value: 4.0
                        sample_count: 1
                      }
                      type: QUANTILES
                    }
                    weighted_common_stats {
                      num_non_missing: 3.0
                      num_missing: 0.0
                      avg_num_values: 3.3333333
                      tot_num_values: 10.0
                    }
                  }
                  mean: 3.5
                  std_dev: 1.7078251
                  num_zeros: 0
                  min: 1.0
                  max: 6.0
                  median: 4.0
                  histograms {
                    buckets {
                      low_value: 1.0
                      high_value: 3.5
                      sample_count: 2.985
                    }
                    buckets {
                      low_value: 3.5
                      high_value: 6.0
                      sample_count: 3.015
                    }
                    type: STANDARD
                  }
                  histograms {
                    buckets {
                      low_value: 1.0
                      high_value: 4.0
                      sample_count: 3.0
                    }
                    buckets {
                      low_value: 4.0
                      high_value: 6.0
                      sample_count: 3.0
                    }
                    type: QUANTILES
                  }
                  weighted_numeric_stats {
                    mean: 3.9
                    std_dev: 1.5779734
                    median: 4.0
                    histograms {
                      buckets {
                        low_value: 1.0
                        high_value: 3.5
                        sample_count: 3.975
                      }
                      buckets {
                        low_value: 3.5
                        high_value: 6.0
                        sample_count: 6.025
                      }
                    }
                    histograms {
                      buckets {
                        low_value: 1.0
                        high_value: 4.0
                        sample_count: 5.0
                      }
                      buckets {
                        low_value: 4.0
                        high_value: 6.0
                        sample_count: 5.0
                      }
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
                    num_non_missing: 2
                    num_missing: 0
                    min_num_values: 2
                    max_num_values: 4
                    avg_num_values: 3.0
                    num_values_histogram {
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
                    weighted_common_stats {
                      num_non_missing: 3.0
                      num_missing: 0.0
                      avg_num_values: 2.6666667
                      tot_num_values: 8.0
                    }
                    tot_num_values: 6
                  }
                  avg_length: 1.0
                  unique: 5
                  top_values { value: 'e' frequency: 2.0 }
                  top_values { value: 'd' frequency: 1.0 }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "e"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "d"
                      sample_count: 1.0
                    }
                  }
                  weighted_string_stats {
                    top_values { value: 'e' frequency: 3.0 }
                    top_values { value: 'd' frequency: 2.0 }
                    rank_histogram {
                      buckets {
                        low_rank: 0
                        high_rank: 0
                        label: "e"
                        sample_count: 3.0
                      }
                      buckets {
                        low_rank: 1
                        high_rank: 1
                        label: "d"
                        sample_count: 2.0
                      }
                    }
                  }
                }
              }
            }
            """,
    },
]


class StatsImplTest(parameterized.TestCase):

  @parameterized.named_parameters(*GENERATE_STATS_TESTS)
  def test_stats_impl(self,
                      examples,
                      options,
                      expected_result_proto_text,
                      schema=None):
    expected_result = text_format.Parse(
        expected_result_proto_text,
        statistics_pb2.DatasetFeatureStatisticsList())
    if schema is not None:
      options.schema = schema
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(examples)
          | stats_impl.GenerateStatisticsImpl(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  @parameterized.named_parameters(*GENERATE_STATS_TESTS)
  def test_generate_statistics_in_memory(
      self, examples, options, expected_result_proto_text, schema=None):
    expected_result = text_format.Parse(
        expected_result_proto_text,
        statistics_pb2.DatasetFeatureStatisticsList())
    if schema is not None:
      options.schema = schema
    result = stats_impl.generate_statistics_in_memory(
        examples, options)
    # generate_statistics_in_memory does not deterministically
    # order multiple features within a DatasetFeatureStatistics proto. So, we
    # cannot use compare.assertProtoEqual (which requires the same ordering of
    # repeated fields) here.
    test_util.assert_dataset_feature_stats_proto_equal(
        self, result.datasets[0], expected_result.datasets[0])

  def test_stats_impl_custom_generators(self):

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
      result = (p | beam.Create(examples) |
                stats_impl.GenerateStatisticsImpl(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result))

  def test_generate_statistics_in_memory_empty_examples(self):
    examples = []
    expected_result = text_format.Parse(
        """
        datasets {
          num_examples: 0
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    result = stats_impl.generate_statistics_in_memory(examples)
    compare.assertProtoEqual(
        self, result, expected_result, normalize_numbers=True)

  def test_generate_statistics_in_memory_valid_custom_generator(
      self):

    # CombinerStatsGenerator that returns a DatasetFeatureStatistic proto with
    # custom stat.
    class CustomCombinerStatsGenerator(stats_generator.CombinerStatsGenerator):

      def create_accumulator(self):
        return 0

      def add_input(self, accumulator,
                    input_batch):
        return 0

      def merge_accumulators(self, accumulators):
        return 0

      def extract_output(
          self, accumulator):
        stats_proto = statistics_pb2.DatasetFeatureStatistics()
        proto_feature = stats_proto.features.add()
        proto_feature.name = 'a'
        custom_stat = proto_feature.custom_stats.add()
        custom_stat.name = 'custom_stat'
        custom_stat.str = 'custom_stat_value'
        return stats_proto

    examples = [
        {'a': np.array(['xyz', 'qwe'])},
        {'a': np.array(['qwe'])},
        {'a': np.array(['qwe'])},
    ]

    expected_result = text_format.Parse(
        """
        datasets {
          num_examples: 3
          features {
            name: 'a'
            type: STRING
            custom_stats {
              name: 'custom_stat'
              str: 'custom_stat_value'
            }
            string_stats {
              avg_length: 3
              unique: 2
              common_stats {
                num_non_missing: 3
                min_num_values: 1
                max_num_values: 2
                avg_num_values: 1.333333
                tot_num_values: 4
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
                    high_value: 2.0
                    sample_count: 1.0
                  }
                  type: QUANTILES
                }
              }
              top_values {
                value: 'qwe'
                frequency: 3
              }
              top_values {
                value: 'xyz'
                frequency: 1
              }
              rank_histogram {
                buckets {
                  low_rank: 0
                  high_rank: 0
                  label: "qwe"
                  sample_count: 3.0
                }
                buckets {
                  low_rank: 1
                  high_rank: 1
                  label: "xyz"
                  sample_count: 1.0
                }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    options = stats_options.StatsOptions(
        generators=[CustomCombinerStatsGenerator('CustomStatsGenerator')],
        num_top_values=4,
        num_rank_histogram_buckets=3,
        num_values_histogram_buckets=3)
    result = stats_impl.generate_statistics_in_memory(
        examples, options)
    compare.assertProtoEqual(
        self, result, expected_result, normalize_numbers=True)

  def test_generate_statistics_in_memory_invalid_custom_generator(
      self):

    # Dummy PTransform that does nothing.
    class CustomPTransform(beam.PTransform):

      def expand(self, pcoll):
        pass

    examples = [{'a': np.array([1.0])}]
    custom_generator = stats_generator.TransformStatsGenerator(
        name='CustomStatsGenerator', ptransform=CustomPTransform())
    options = stats_options.StatsOptions(generators=[custom_generator])
    with self.assertRaisesRegexp(
        TypeError, 'Statistics generator.* found object of type '
        'TransformStatsGenerator.'):
      stats_impl.generate_statistics_in_memory(examples, options)

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

  def test_tfdv_telemetry(self):
    examples = [
        {
            'a': np.array([1.0, 2.0], dtype=np.floating),
            'b': np.array(['a', 'b', 'c', 'e'], dtype=np.object),
            'c': None
        },
        {
            'a': np.array([3.0, 4.0, np.NaN, 5.0], dtype=np.floating),
            'b': np.array(['d', 'e', 'f'], dtype=np.object),
            'c': None
        },
        {
            'a': None,
            'b': np.array(['a', 'b', 'c'], dtype=np.object),
            'c': np.array([10, 20, 30], dtype=np.integer)
        },
        {
            'a': np.array([5.0], dtype=np.floating),
            'b': np.array(['d', 'e', 'f'], dtype=np.object),
            'c': np.array([1], dtype=np.integer)
        }
    ]

    p = beam.Pipeline()
    _ = (p
         | 'CreateBatches' >> beam.Create(examples)
         | 'BasicStatsCombiner' >> beam.CombineGlobally(
             stats_impl._BatchedCombineFnWrapper(
                 basic_stats_generator.BasicStatsGenerator())))

    runner = p.run()
    runner.wait_until_finish()
    result_metrics = runner.metrics()

    num_metrics = len(
        result_metrics.query(beam.metrics.metric.MetricsFilter().with_namespace(
            constants.METRICS_NAMESPACE))['counters'])
    self.assertEqual(num_metrics, 14)

    expected_result = {
        'num_instances': 4,
        'num_missing_feature_values': 3,
        'num_int_feature_values': 2,
        'int_feature_values_min_count': 1,
        'int_feature_values_max_count': 3,
        'int_feature_values_mean_count': 2,
        'num_float_feature_values': 3,
        'float_feature_values_min_count': 1,
        'float_feature_values_max_count': 4,
        'float_feature_values_mean_count': 2,
        'num_string_feature_values': 4,
        'string_feature_values_min_count': 3,
        'string_feature_values_max_count': 4,
        'string_feature_values_mean_count': 3,
    }
    # Check number of counters.
    actual_metrics = result_metrics.query(
        beam.metrics.metric.MetricsFilter().with_namespace(
            constants.METRICS_NAMESPACE))['counters']
    self.assertLen(actual_metrics, len(expected_result))

    # Check each counter.
    for counter_name in expected_result:
      actual_counter = result_metrics.query(
          beam.metrics.metric.MetricsFilter().with_name(counter_name)
          )['counters']
      self.assertLen(actual_counter, 1)
      self.assertEqual(actual_counter[0].committed,
                       expected_result[counter_name])

  def test_filter_features(self):
    input_batch = {'a': np.array([]), 'b': np.array([]), 'c': np.array([])}
    actual = stats_impl._filter_features(input_batch, ['a', 'c'])
    expected = {'a': np.array([]), 'c': np.array([])}
    self.assertEqual(set(actual.keys()), set(expected.keys()))

  def test_filter_features_empty(self):
    input_batch = {'a': np.array([])}
    actual = stats_impl._filter_features(input_batch, [])
    expected = {}
    self.assertEqual(set(actual.keys()), set(expected.keys()))


if __name__ == '__main__':
  absltest.main()
