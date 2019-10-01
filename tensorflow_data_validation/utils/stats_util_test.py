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

import os
from absl import flags
from absl.testing import absltest
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import stats_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2

FLAGS = flags.FLAGS


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
        types.FeaturePath(['feature_1']): {
            'Mutual Information': 0.5,
            'Correlation': 0.1
        },
        types.FeaturePath(['feature_2']): {
            'Mutual Information': 0.8,
            'Correlation': 0.6
        }
    }
    expected = {
        types.FeaturePath(['feature_1']):
            text_format.Parse(
                """
            path {
              step: 'feature_1'
            }
            custom_stats {
              name: 'Correlation'
              num: 0.1
            }
            custom_stats {
              name: 'Mutual Information'
              num: 0.5
            }
           """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['feature_2']):
            text_format.Parse(
                """
            path {
              step: 'feature_2'
            }
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
          expected[types.FeaturePath.from_proto(actual_feature_stats.path)],
          normalize_numbers=True)

  def test_get_utf8(self):
    self.assertEqual(u'This is valid.',
                     stats_util.maybe_get_utf8(b'This is valid.'))
    self.assertIsNone(stats_util.maybe_get_utf8(b'\xF0'))

  def test_write_load_stats_text(self):
    stats = text_format.Parse("""
      datasets {}
    """, statistics_pb2.DatasetFeatureStatisticsList())
    stats_path = os.path.join(FLAGS.test_tmpdir, 'stats.pbtxt')
    stats_util.write_stats_text(stats=stats, output_path=stats_path)
    loaded_stats = stats_util.load_stats_text(input_path=stats_path)
    self.assertEqual(stats, loaded_stats)

  def test_write_stats_text_invalid_stats_input(self):
    with self.assertRaisesRegexp(
        TypeError, '.*should be a DatasetFeatureStatisticsList proto.'):
      _ = stats_util.write_stats_text({}, 'stats.pbtxt')

  def test_get_custom_stats_numeric(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              num: 100.0
            }
        """, statistics_pb2.FeatureNameStatistics())
    self.assertEqual(stats_util.get_custom_stats(stats, 'abc'), 100.0)

  def test_get_custom_stats_string(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              str: 'xyz'
            }
        """, statistics_pb2.FeatureNameStatistics())
    self.assertEqual(stats_util.get_custom_stats(stats, 'abc'), 'xyz')

  def test_get_custom_stats_not_found(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              num: 100.0
            }
        """, statistics_pb2.FeatureNameStatistics())
    with self.assertRaisesRegexp(ValueError, 'Custom statistics.*not found'):
      stats_util.get_custom_stats(stats, 'xyz')


if __name__ == '__main__':
  absltest.main()
