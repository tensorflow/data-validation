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
"""Tests for partitioned_stats_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.statistics.generators import sklearn_mutual_information
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

TEST_SEED = 12345

_SAMPLE_PARTITION_TESTS = [{
    'testcase_name': 'sample_2_from_4',
    'partitioned_record_batches': [
        (1,
         pa.RecordBatch.from_arrays([
             pa.array([['Green'], ['Red'], ['Blue'], ['Green']]),
             pa.array([['Label'], ['Label'], ['Label'], ['Label']]),
         ], ['fa', 'label_key']))
    ],
    'expected': [(1,
                  pa.RecordBatch.from_arrays([
                      pa.array([['Blue'], ['Green']]),
                      pa.array([['Label'], ['Label']]),
                  ], ['fa', 'label_key']))],
    'sample_size': 2,
    'num_compacts': 1
}, {
    'testcase_name': 'num_records_smaller_than_max',
    'partitioned_record_batches': [
        (1,
         pa.RecordBatch.from_arrays([
             pa.array([['Green'], ['Blue'], ['Red']]),
             pa.array([['Label'], ['Label'], ['Label']]),
         ], ['fa', 'label_key'])),
        (1,
         pa.RecordBatch.from_arrays([
             pa.array([['Green'], ['Blue'], ['Red']]),
             pa.array([['Label'], ['Label'], ['Label']]),
         ], ['fa', 'label_key']))
    ],
    'expected': [(1,
                  pa.RecordBatch.from_arrays([
                      pa.array([['Blue'], ['Red'], ['Green'], ['Green'],
                                ['Blue'], ['Red']]),
                      pa.array([['Label'], ['Label'], ['Label'], ['Label'],
                                ['Label'], ['Label']]),
                  ], ['fa', 'label_key']))],
    'sample_size': 10,
    'num_compacts': 1
}, {
    'testcase_name':
        'combine_many_to_one',
    'partitioned_record_batches': [(1,
                                    pa.RecordBatch.from_arrays([
                                        pa.array([['Green']]),
                                        pa.array([['Label']]),
                                    ], ['fa', 'label_key']))] * 11,
    'expected': [(1,
                  pa.RecordBatch.from_arrays([
                      pa.array([['Green']] * 10),
                      pa.array([['Label']] * 10),
                  ], ['fa', 'label_key']))],
    'sample_size':
        10,
    'num_compacts':
        1
}, {
    'testcase_name': 'partition_of_empty_rb',
    'partitioned_record_batches': [(1,
                                    pa.RecordBatch.from_arrays([
                                        pa.array([]),
                                        pa.array([]),
                                    ], ['fa', 'label_key'])),
                                   (1,
                                    pa.RecordBatch.from_arrays([
                                        pa.array([['Green']] * 10),
                                        pa.array([['Label']] * 10),
                                    ], ['fa', 'label_key'])),
                                   (1,
                                    pa.RecordBatch.from_arrays([
                                        pa.array([]),
                                        pa.array([]),
                                    ], ['fa', 'label_key']))],
    'expected': [(1,
                  pa.RecordBatch.from_arrays([
                      pa.array([['Green']] * 10),
                      pa.array([['Label']] * 10),
                  ], ['fa', 'label_key']))],
    'sample_size': 10,
    'num_compacts': 1
}, {
    'testcase_name': 'empty_partition',
    'partitioned_record_batches': [],
    'expected': [],
    'sample_size': 10,
    'num_compacts': 1
}, {
    'testcase_name':
        'many_compacts',
    'partitioned_record_batches': [(1,
                                    pa.RecordBatch.from_arrays([
                                        pa.array([['Green']]),
                                        pa.array([['Label']]),
                                    ], ['fa', 'label_key']))] * 18,
    'expected': [(1,
                  pa.RecordBatch.from_arrays([
                      pa.array([['Green']] * 2),
                      pa.array([['Label']] * 2),
                  ], ['fa', 'label_key']))],
    'sample_size':
        2,
    'num_compacts':
        2
}]


class AssignToPartitionTest(absltest.TestCase):
  """Tests for _asssign_to_partition."""

  def test_partitioner(self):
    """Tests that batches are randomly partitioned.

    Tests an input batch with one univalent feature taking on values in {0,1,2}.
    The input batch has 4500 examples. The partitioner is configured to have 3
    partitions. So, we expect there to be around 4500/3/3 = 500 of each.
    {0,1,2} in each partition.
    """

    np.random.seed(TEST_SEED)
    record_batches = [
        pa.RecordBatch.from_arrays([pa.array([x])], ['a'])
        for x in np.random.randint(0, 3, (4500, 1))
    ]
    record_batches = [(constants.DEFAULT_SLICE_KEY, record_batch)
                      for record_batch in record_batches]
    num_partitions = 3

    # The i,jth value of result represents the number of examples with value j
    # assigned to partition i.
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    partitioned_record_batches = [
        partitioned_stats_generator._default_assign_to_partition(
            record_batch, num_partitions) for record_batch in record_batches
    ]
    for (unused_slice_key,
         partition_key), record_batch in partitioned_record_batches:
      result[partition_key][record_batch.column(0).to_pylist()[0][0]] += 1

    for partition in result:
      for count in partition:
        self.assertBetween(count, 400, 600)


class PartitionedStatisticsAnalyzer(absltest.TestCase):
  """Tests PartitionedStatisticsAnalyzer."""

  def _assert_combiner_output_equal(self, statistics, combiner, expected):
    accumulators = [
        combiner.add_input(combiner.create_accumulator(), statistic)
        for statistic in statistics
    ]
    actual = combiner.extract_output(combiner.merge_accumulators(accumulators))
    compare.assertProtoEqual(self, actual, expected, normalize_numbers=True)

  def test_statistic_analyzer_with_invalid_feature(self):
    statistics = [
        text_format.Parse(
            """
      features {
        path {
          step: 'valid_feature'
        }
        custom_stats {
              name: 'MI'
              num: 0.5
            }
        custom_stats {
              name: 'Cov'
              num: 0.3
            }
      }
      features {
        path {
          step: 'invalid_feature'
        }
        custom_stats {
          name: 'MI'
          num: 0.5
        }
        custom_stats {
          name: 'Cov'
          num: 0.3
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'valid_feature'
        }
        custom_stats {
              name: 'MI'
              num: 1.5
            }
        custom_stats {
              name: 'Cov'
              num: 0.7
            }
      }""", statistics_pb2.DatasetFeatureStatistics())
    ]
    expected = text_format.Parse(
        """
      features {
        path {
          step: 'valid_feature'
        }
        custom_stats {
              name: 'max_Cov'
              num: 0.7
            }
        custom_stats {
              name: 'max_MI'
              num: 1.5
            }
        custom_stats {
              name: 'mean_Cov'
              num: 0.5
            }
        custom_stats {
              name: 'mean_MI'
              num: 1
            }
        custom_stats {
              name: 'median_Cov'
              num: 0.5
            }
        custom_stats {
              name: 'median_MI'
              num: 1
            }
        custom_stats {
              name: 'min_Cov'
              num: 0.3
            }
        custom_stats {
              name: 'min_MI'
              num: 0.5
            }
        custom_stats {
              name: 'num_partitions_Cov'
              num: 2
            }
        custom_stats {
              name: 'num_partitions_MI'
              num: 2
            }
        custom_stats {
              name: 'std_dev_Cov'
              num: 0.2
            }
        custom_stats {
              name: 'std_dev_MI'
              num: 0.5
            }
      }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_combiner_output_equal(
        statistics,
        partitioned_stats_generator.PartitionedStatisticsAnalyzer(
            min_partitions_stat_presence=2), expected)


class SampleRecordBatchRows(parameterized.TestCase):
  """Tests SamplePartition."""

  def _partition_matcher(self, expected):
    """This matches partitions for equality.

    A partition is Tuple(int, record_batch).

    Args:
      expected: Tuple(int, record_batch). The expected partition.

    Returns:
      A callable that can be used with `beam_test_util.assert_that`.
    """

    def _matcher(actual):
      for expected_tuple, actual_tuple in zip(expected, actual):
        self.assertEqual(expected_tuple[0], actual_tuple[0])
        expected_rb = expected_tuple[1]
        actual_rb = actual_tuple[1]
        self.assertIsInstance(expected_rb, pa.RecordBatch)
        self.assertTrue(
            actual_rb.equals(expected_rb), f'Record Batches not equal. '
            f'actual: {actual_rb.to_pydict()}\nexpected: {expected_rb.to_pydict()}'
        )

    return _matcher

  @parameterized.named_parameters(*(_SAMPLE_PARTITION_TESTS))
  def test_sample_partition_combine(self, partitioned_record_batches, expected,
                                    sample_size, num_compacts):
    np.random.seed(TEST_SEED)
    p = beam.Pipeline()
    result = (
        p | beam.Create(partitioned_record_batches, reshuffle=False)
        | beam.CombinePerKey(
            partitioned_stats_generator._SampleRecordBatchRows(sample_size)))

    beam_test_util.assert_that(result, self._partition_matcher(expected))

    # Validate metrics.
    np.random.seed(TEST_SEED)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    metrics = pipeline_result.metrics()
    num_compacts_metric = metrics.query(
        beam.metrics.metric.MetricsFilter().with_name(
            'sample_record_batch_rows_num_compacts'))['counters']
    metric_num_compacts = 0
    for metric in num_compacts_metric:
      metric_num_compacts += metric.committed
    if num_compacts_metric:
      self.assertEqual(metric_num_compacts, num_compacts)

  def test_sample_metrics(self):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([['Green']]),
        pa.array([['Label']]),
    ], ['fa', 'label_key'])
    partitioned_rbs = [(1, record_batch)] * 2
    with beam.Pipeline() as p:
      _ = (
          p | beam.Create(partitioned_rbs, reshuffle=False)
          | beam.CombinePerKey(
              partitioned_stats_generator._SampleRecordBatchRows(1)))

    runner = p.run()
    runner.wait_until_finish()
    actual_metrics = runner.metrics()
    expected_counters = {
        'sample_record_batch_rows_num_instances': 2,
        'sample_record_batch_rows_num_compacts': 1,
    }
    expected_distributions = {
        'sample_record_batch_rows_combine_num_record_batches': {
            'count': 1,
            'sum': 2,
            'max': 2,
            'min': 2,
        },
        'sample_record_batch_rows_combine_byte_size': {
            'count': 1,
            'sum': 42,
            'max': 42,
            'min': 42,
        },
    }
    for counter_name in expected_counters:
      actual_counter = actual_metrics.query(
          beam.metrics.metric.MetricsFilter().with_name(
              counter_name))['counters']
      self.assertEqual(actual_counter[0].committed,
                       expected_counters[counter_name])
    for counter_name in expected_distributions:
      actual_counter = actual_metrics.query(
          beam.metrics.metric.MetricsFilter().with_name(
              counter_name))['distributions']
      self.assertEqual(actual_counter[0].committed.count,
                       expected_distributions[counter_name]['count'])
      # The size estimation used delegate to Arrow, which empirically
      # sometimes changes how size of batches is calculated.
      self.assertAlmostEqual(
          actual_counter[0].committed.sum,
          expected_distributions[counter_name]['sum'],
          delta=20)
      self.assertAlmostEqual(
          actual_counter[0].committed.max,
          expected_distributions[counter_name]['max'],
          delta=20)
      self.assertAlmostEqual(
          actual_counter[0].committed.min,
          expected_distributions[counter_name]['min'],
          delta=20)


def _get_test_stats_with_mi(feature_paths):
  """Get stats proto for MI test."""
  result = statistics_pb2.DatasetFeatureStatistics()
  for feature_path in feature_paths:
    feature_proto = text_format.Parse(
        """
                custom_stats {
                  name: "max_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "max_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "max_sklearn_normalized_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_normalized_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_normalized_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_normalized_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_adjusted_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_normalized_adjusted_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "std_dev_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "std_dev_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "std_dev_sklearn_normalized_adjusted_mutual_information"
                  num: 0.0
                }
        """, statistics_pb2.FeatureNameStatistics())
    feature_proto.path.CopyFrom(feature_path.to_proto())
    result.features.add().CopyFrom(feature_proto)
  return result


class NonStreamingCustomStatsGeneratorTest(test_util.TransformStatsGeneratorTest
                                          ):
  """Tests for NonStreamingCustomStatsGenerator."""

  def setUp(self):
    super(NonStreamingCustomStatsGeneratorTest, self).setUp()
    # Integration tests involving Beam and AMI are challenging to write
    # because Beam PCollections are unordered while the results of adjusted MI
    # depend on the order of the data for small datasets. This test case tests
    # MI with one label which will give a value of 0 regardless of
    # the ordering of elements in the PCollection. The purpose of this test is
    # to ensure that the Mutual Information pipeline is able to handle a
    # variety of input types. Unit tests ensuring correctness of the MI value
    # itself are included in sklearn_mutual_information_test.

    # fa is categorical, fb is numeric, fc is multivalent and fd has null values
    self.record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['Red']]),
            pa.array([[1.0]]),
            pa.array([[1, 3, 1]]),
            pa.array([[0.4]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Green']]),
            pa.array([[2.2]]),
            pa.array([[2, 6]]),
            pa.array([[0.4]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Blue']]),
            pa.array([[3.3]]),
            pa.array([[4, 6]]),
            pa.array([[0.3]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Green']]),
            pa.array([[1.3]]),
            pa.array([None]),
            pa.array([[0.2]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Red']]),
            pa.array([[1.2]]),
            pa.array([[1]]),
            pa.array([[0.3]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Blue']]),
            pa.array([[0.5]]),
            pa.array([[3, 2]]),
            pa.array([[0.4]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Blue']]),
            pa.array([[1.3]]),
            pa.array([[1, 4]]),
            pa.array([[1.7]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Green']]),
            pa.array([[2.3]]),
            pa.array([[0]]),
            pa.array([[np.NaN]], type=pa.list_(pa.float64())),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
        pa.RecordBatch.from_arrays([
            pa.array([['Green']]),
            pa.array([[0.3]]),
            pa.array([[3]]),
            pa.array([[4.4]]),
            pa.array([['Label']]),
        ], ['fa', 'fb', 'fc', 'fd', 'label_key']),
    ]

    self.schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "fb"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "fc"
          type: INT
          value_count: {
            min: 0
            max: 2
          }
        }
        feature {
          name: "fd"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }""", schema_pb2.Schema())

  def test_sklearn_mi(self):
    expected_result = [
        _get_test_stats_with_mi([
            types.FeaturePath(['fa']),
            types.FeaturePath(['fb']),
            types.FeaturePath(['fd'])
        ])
    ]
    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        sklearn_mutual_information.SkLearnMutualInformation(
            label_feature=types.FeaturePath(['label_key']),
            schema=self.schema,
            seed=TEST_SEED),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        batch_size=1,
        name='NonStreaming Mutual Information')
    self.assertSlicingAwareTransformOutputEqual(
        self.record_batches,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_sklearn_mi_with_slicing(self):
    sliced_record_batches = []
    for slice_key in ['slice1', 'slice2']:
      for record_batch in self.record_batches:
        sliced_record_batches.append((slice_key, record_batch))

    expected_result = [
        ('slice1',
         _get_test_stats_with_mi([
             types.FeaturePath(['fa']),
             types.FeaturePath(['fb']),
             types.FeaturePath(['fd'])
         ])),
        ('slice2',
         _get_test_stats_with_mi([
             types.FeaturePath(['fa']),
             types.FeaturePath(['fb']),
             types.FeaturePath(['fd'])
         ])),
    ]
    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        sklearn_mutual_information.SkLearnMutualInformation(
            label_feature=types.FeaturePath(['label_key']),
            schema=self.schema,
            seed=TEST_SEED),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        batch_size=1,
        name='NonStreaming Mutual Information')
    self.assertSlicingAwareTransformOutputEqual(sliced_record_batches,
                                                generator, expected_result)


if __name__ == '__main__':
  absltest.main()
