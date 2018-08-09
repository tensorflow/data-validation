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
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsUtilTest(absltest.TestCase):

  def test_make_feature_type_int(self):
    self.assertEqual(stats_util.make_feature_type(np.dtype('int8')),
                     statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(stats_util.make_feature_type(np.dtype('int16')),
                     statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(stats_util.make_feature_type(np.dtype('int32')),
                     statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(stats_util.make_feature_type(np.dtype('int64')),
                     statistics_pb2.FeatureNameStatistics.INT)

  def test_make_feature_type_float(self):
    self.assertEqual(stats_util.make_feature_type(np.dtype('float16')),
                     statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(stats_util.make_feature_type(np.dtype('float32')),
                     statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(stats_util.make_feature_type(np.dtype('float64')),
                     statistics_pb2.FeatureNameStatistics.FLOAT)

  def test_make_feature_type_string(self):
    self.assertEqual(stats_util.make_feature_type(np.dtype('S')),
                     statistics_pb2.FeatureNameStatistics.STRING)
    self.assertEqual(stats_util.make_feature_type(np.dtype('U')),
                     statistics_pb2.FeatureNameStatistics.STRING)

  def test_make_feature_type_none(self):
    self.assertIsNone(stats_util.make_feature_type(np.dtype('complex64')))

  def test_make_feature_type_invalid_dtype(self):
    with self.assertRaises(TypeError):
      stats_util.make_feature_type(int)

  def test_get_categorical_numeric_features(self):
    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        feature {
          name: "fb"
          type: BYTES
        }
        feature {
          name: "fc"
          type: FLOAT
        }
        """, schema_pb2.Schema())
    self.assertEqual(
        stats_util.get_categorical_numeric_features(schema), ['fa'])


if __name__ == '__main__':
  absltest.main()
