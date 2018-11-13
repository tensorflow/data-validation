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

"""Tests for utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.utils import stats_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsUtilTest(absltest.TestCase):

  def test_get_feature_type_get_int(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int8')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int16')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int32')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int64')),
        statistics_pb2.FeatureNameStatistics.INT)

  def test_get_feature_type_get_float(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float16')),
        statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float32')),
        statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float64')),
        statistics_pb2.FeatureNameStatistics.FLOAT)

  def test_get_feature_type_get_string(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('S')),
        statistics_pb2.FeatureNameStatistics.STRING)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('U')),
        statistics_pb2.FeatureNameStatistics.STRING)

  def test_get_feature_type_get_none(self):
    self.assertIsNone(stats_util.get_feature_type(np.dtype('complex64')))

  def test_make_dataset_feature_stats_proto(self):
    stats = {
        'feature_1': {
            'Mutual Information': 0.5,
            'Correlation': 0.1
        },
        'feature_2': {
            'Mutual Information': 0.8,
            'Correlation': 0.6
        }
    }
    expected = {
        'feature_1':
            text_format.Parse(
                """
            name: 'feature_1'
            custom_stats {
              name: 'Correlation'
              num: 0.1
            }
            custom_stats {
              name: 'Mutual Information'
              num: 0.5
            }
           """, statistics_pb2.FeatureNameStatistics()),
        'feature_2':
            text_format.Parse(
                """
            name: 'feature_2'
            custom_stats {
              name: 'Correlation'
              num: 0.6
            }
            custom_stats {
              name: 'Mutual Information'
              num: 0.8
            }
           """, statistics_pb2.FeatureNameStatistics())
    }
    actual = stats_util.make_dataset_feature_stats_proto(stats)
    self.assertEqual(len(actual.features), len(expected))
    for actual_feature_stats in actual.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected[actual_feature_stats.name],
          normalize_numbers=True)

  def test_get_weight_feature_with_valid_weight_feature(self):
    batch = {'a': np.array([np.array([1, 2]), np.array([3])]),
             'w': np.array([np.array([10]), np.array([20])])}
    actual = stats_util.get_weight_feature(batch, 'w')
    np.testing.assert_equal(actual, batch['w'])

  def test_get_weight_feature_invalid_weight_feature(self):
    batch = {'a': np.array([np.array([1])])}
    with self.assertRaisesRegexp(ValueError, 'Weight feature.*not present'):
      stats_util.get_weight_feature(batch, 'w')

  def test_get_weight_feature_with_weight_feature_missing(self):
    batch = {'a': np.array([np.array([1])]), 'w': np.array([None])}
    with self.assertRaisesRegexp(ValueError, 'Weight feature.*missing'):
      stats_util.get_weight_feature(batch, 'w')

  def test_get_weight_feature_with_weight_feature_string_type(self):
    batch = {'a': np.array([np.array([1])]), 'w': np.array([np.array(['a'])])}
    with self.assertRaisesRegexp(ValueError, 'Weight feature.*numeric type'):
      stats_util.get_weight_feature(batch, 'w')

  def test_get_weight_feature_with_weight_feature_multiple_values(self):
    batch = {'a': np.array([np.array([1])]),
             'w': np.array([np.array([2, 3])])}
    with self.assertRaisesRegexp(ValueError, 'Weight feature.*single value'):
      stats_util.get_weight_feature(batch, 'w')


if __name__ == '__main__':
  absltest.main()
