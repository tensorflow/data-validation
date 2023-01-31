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

import os
import tempfile
from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pyarrow as pa
import tensorflow as tf

from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.utils import artifacts_io_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import io_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsAPITest(absltest.TestCase):

  def _get_temp_dir(self):
    return tempfile.mkdtemp()

  def test_stats_pipeline(self):
    record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[1.0, 2.0]]),
            pa.array([['a', 'b', 'c', 'd']]),
            pa.array([np.linspace(1, 500, 500, dtype=np.int32)]),
        ], ['a', 'b', 'c']),
        pa.RecordBatch.from_arrays([
            pa.array([[3.0, 4.0, np.NaN, 5.0]]),
            pa.array([['a', 'c', '∞', 'a']]),
            pa.array([np.linspace(501, 1250, 750, dtype=np.int32)]),
        ], ['a', 'b', 'c']),
        pa.RecordBatch.from_arrays([
            pa.array([[1.0]]),
            pa.array([['a', 'b', 'c', '∞']]),
            pa.array([np.linspace(1251, 3000, 1750, dtype=np.int32)]),
        ], ['a', 'b', 'c'])
    ]

    expected_result = text_format.Parse(
        """
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
          }
          mean: 2.66666666
          std_dev: 1.49071198
          num_zeros: 0
          min: 1.0
          max: 5.0
          median: 3.0
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
          }
          mean: 1500.5
          std_dev: 866.025355672
          min: 1.0
          max: 3000.0
          median: 1501.0
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
          avg_length: 1.333333
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
              label: "∞"
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
          p | beam.Create(record_batches)
          | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result, check_histograms=False))

  _sampling_test_expected_result = text_format.Parse(
      """
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
          }
          mean: 1500.5
          std_dev: 866.025355672
          min: 1.0
          max: 3000.0
          median: 1501.0
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

  def test_stats_pipeline_with_examples_with_no_values(self):
    record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[]], type=pa.list_(pa.float32())),
            pa.array([[]], type=pa.list_(pa.binary())),
            pa.array([[]], type=pa.list_(pa.int32())),
            pa.array([[2]]),
        ], ['a', 'b', 'c', 'w']),
        pa.RecordBatch.from_arrays([
            pa.array([[]], type=pa.list_(pa.float32())),
            pa.array([[]], type=pa.list_(pa.binary())),
            pa.array([[]], type=pa.list_(pa.int32())),
            pa.array([[2]]),
        ], ['a', 'b', 'c', 'w']),
        pa.RecordBatch.from_arrays([
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
          weighted_numeric_stats {
            mean: 2.0
            median: 2.0
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
          p | beam.Create(record_batches)
          | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, expected_result, check_histograms=False))

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
              self, expected_result, check_histograms=False))

  def test_stats_pipeline_with_sample_rate(self):
    record_batches = [
        pa.RecordBatch.from_arrays(
            [pa.array([np.linspace(1, 3000, 3000, dtype=np.int32)])], ['c']),
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
          p | beam.Create(record_batches)
          | stats_api.GenerateStatistics(options))
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, self._sampling_test_expected_result,
              check_histograms=False))

  def test_invalid_stats_options(self):
    record_batches = [pa.RecordBatch.from_arrays([])]
    with self.assertRaisesRegexp(TypeError, '.*should be a StatsOptions.'):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(record_batches)
            | stats_api.GenerateStatistics(options={}))

  def test_write_stats_to_binary_file(self):
    stats = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    output_path = os.path.join(self._get_temp_dir(), 'stats')
    with beam.Pipeline() as p:
      _ = (p | beam.Create([stats]) | stats_api.WriteStatisticsToBinaryFile(
          output_path))
    stats_from_file = statistics_pb2.DatasetFeatureStatisticsList()
    serialized_stats = io_util.read_file_to_string(
        output_path, binary_mode=True)
    stats_from_file.ParseFromString(serialized_stats)
    self.assertLen(stats_from_file.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self,
        stats_from_file.datasets[0],
        stats.datasets[0])

  def test_write_stats_to_tfrecrod(self):
    stats = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    output_path = os.path.join(self._get_temp_dir(), 'stats')
    with beam.Pipeline() as p:
      _ = (p | beam.Create([stats]) | stats_api.WriteStatisticsToTFRecord(
          output_path))
    stats_from_file = stats_util.load_statistics(output_path)
    self.assertLen(stats_from_file.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self,
        stats_from_file.datasets[0],
        stats.datasets[0])

  def test_write_stats_to_tfrecord_and_binary(self):
    stats1 = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
          features: {
             path: {
                step: "f1"
             }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    stats2 = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
          features: {
             path: {
                step: "f2"
             }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    stats_combined = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
          features: {
             path: {
                step: "f1"
             }
          }
          features: {
             path: {
                step: "f2"
             }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    output_path_binary = os.path.join(self._get_temp_dir(), 'stats.pb')
    output_path_prefix = os.path.join(self._get_temp_dir(), 'stats_shards')
    columnar_path_prefix = os.path.join(self._get_temp_dir(),
                                        'columnar_outputs')
    with beam.Pipeline() as p:
      _ = (
          p | beam.Create([stats1, stats2])
          | stats_api.WriteStatisticsToRecordsAndBinaryFile(
              output_path_binary, output_path_prefix, columnar_path_prefix))

    stats_from_pb = statistics_pb2.DatasetFeatureStatisticsList()
    serialized_stats = io_util.read_file_to_string(
        output_path_binary, binary_mode=True)
    stats_from_pb.ParseFromString(serialized_stats)
    self.assertLen(stats_from_pb.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self, stats_from_pb.datasets[0], stats_combined.datasets[0])

    stats_from_shards = stats_util.load_sharded_statistics(output_path_prefix +
                                                           '*').proto()
    self.assertLen(stats_from_shards.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self,
        stats_from_shards.datasets[0],
        stats_combined.datasets[0])

    if artifacts_io_impl.get_default_columnar_provider():
      self.assertNotEmpty(tf.io.gfile.glob(columnar_path_prefix + '-*-of-*'))


class MergeDatasetFeatureStatisticsListTest(absltest.TestCase):

  def test_merges_two_shards(self):
    stats1 = text_format.Parse(
        """
      datasets {
        name: 'x'
        num_examples: 100
        features: {
           path: {
              step: "f1"
           }
        }
      }
      """, statistics_pb2.DatasetFeatureStatisticsList())
    stats2 = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
          features: {
             path: {
                step: "f2"
             }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    stats_combined = text_format.Parse(
        """
        datasets {
          name: 'x'
          num_examples: 100
          features: {
             path: {
                step: "f1"
             }
          }
          features: {
             path: {
                step: "f2"
             }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    with beam.Pipeline() as p:
      result = (
          p | beam.Create([stats1, stats2])
          | stats_api.MergeDatasetFeatureStatisticsList())
      util.assert_that(
          result,
          test_util.make_dataset_feature_stats_list_proto_equal_fn(
              self, stats_combined, check_histograms=False))

if __name__ == '__main__':
  absltest.main()
