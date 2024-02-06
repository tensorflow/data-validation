# Copyright 2019 Google LLC
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

"""Tests for cross feature statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation.statistics.generators import cross_feature_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class CrossFeatureStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_cross_feature_stats_generator(self):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        sample_rate=1.0)
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0], [3.0], [5.0]]),
        pa.array([[2.0], [4.0], [6.0]]),
        pa.array([[5.0], [3.0], [7.0]]),
    ], ['a', 'b', 'c'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[6.0], [10.0]]),
        pa.array([[14.0], [16.0]]),
        pa.array([[-1.0], [0]]),
    ], ['a', 'b', 'c'])
    b3 = pa.RecordBatch.from_arrays([
        pa.array([None, None], type=pa.null()),
        pa.array([None, None], type=pa.null()),
        pa.array([None, None], type=pa.null()),
    ], ['a', 'b', 'c'])
    batches = [b1, b2, b3]
    expected_result = {
        ('a', 'b'): text_format.Parse(
            """
            path_x { step: "a" }
            path_y { step: "b" }
            count: 5
            num_cross_stats {
              correlation: 0.923145
              covariance: 15.6
            }
            """, statistics_pb2.CrossFeatureStatistics()),
        ('a', 'c'): text_format.Parse(
            """
            path_x { step: "a" }
            path_y { step: "c" }
            count: 5
            num_cross_stats {
              correlation: -0.59476602
              covariance: -5.4000001
            }
            """, statistics_pb2.CrossFeatureStatistics()),
        ('b', 'c'): text_format.Parse(
            """
            path_x { step: "b" }
            path_y { step: "c" }
            count: 5
            num_cross_stats {
              correlation: -0.81070298
              covariance: -13.52
            }
            """, statistics_pb2.CrossFeatureStatistics())}
    self.assertCombinerOutputEqual(batches, generator, {}, expected_result)

  def test_cross_feature_stats_generator_with_crosses_specified(self):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        feature_crosses=[('a', 'c'), ('b', 'c')], sample_rate=1.0)
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0], [3.0], [5.0]]),
        pa.array([[2.0], [4.0], [6.0]]),
        pa.array([[5.0], [3.0], [7.0]]),
    ], ['a', 'b', 'c'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[6.0], [10.0]]),
        pa.array([[14.0], [16.0]]),
        pa.array([[-1.0], [0]]),
    ], ['a', 'b', 'c'])
    batches = [b1, b2]
    expected_result = {
        ('a', 'c'): text_format.Parse(
            """
            path_x { step: "a" }
            path_y { step: "c" }
            count: 5
            num_cross_stats {
              correlation: -0.59476602
              covariance: -5.4000001
            }
            """, statistics_pb2.CrossFeatureStatistics()),
        ('b', 'c'): text_format.Parse(
            """
            path_x { step: "b" }
            path_y { step: "c" }
            count: 5
            num_cross_stats {
              correlation: -0.81070298
              covariance: -13.52
            }
            """, statistics_pb2.CrossFeatureStatistics())}
    self.assertCombinerOutputEqual(batches, generator, {}, expected_result)

  def test_cross_feature_stats_generator_with_string_crosses_configured(
      self,
  ):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        feature_crosses=[('a', 'b')], sample_rate=1.0
    )
    b1 = pa.RecordBatch.from_arrays(
        [
            pa.array([['x'], ['y'], ['z']]),
            pa.array([[2.0], [4.0], [6.0]]),
        ],
        ['a', 'b'],
    )
    b2 = pa.RecordBatch.from_arrays(
        [
            pa.array([['x'], ['y']]),
            pa.array([[14.0], [16.0]]),
        ],
        ['a', 'b'],
    )
    batches = [b1, b2]
    self.assertCombinerOutputEqual(batches, generator, {}, {})

  def test_cross_feature_stats_generator_with_multivalent_crosses_configured(
      self,
  ):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        feature_crosses=[('a', 'b')], sample_rate=1.0
    )
    b1 = pa.RecordBatch.from_arrays(
        [
            pa.array([[1.0, 1.0], [2.5, 2.5], [3.0, 3.0]]),
            pa.array([[2.0], [4.0], [6.0]]),
        ],
        ['a', 'b'],
    )
    b2 = pa.RecordBatch.from_arrays(
        [
            pa.array([[1.0, 1.0], [2.5, 2.5]]),
            pa.array([[14.0], [16.0]]),
        ],
        ['a', 'b'],
    )
    batches = [b1, b2]
    self.assertCombinerOutputEqual(batches, generator, {}, {})

  def test_cross_feature_stats_generator_multivalent_feature(self):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        sample_rate=1.0)
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0], [3.0], [5.0]]),
         pa.array([[2.0], [4.0], [6.0]])], ['a', 'b'])
    b2 = pa.RecordBatch.from_arrays([
        pa.array([[6.0], [10.0], [1.0, 2.0]]),
        pa.array([[14.0], [16.0], [3.9]])
    ], ['a', 'b'])
    batches = [b1, b2]
    expected_result = {
        ('a', 'b'): text_format.Parse(
            """
            path_x { step: "a" }
            path_y { step: "b" }
            count: 5
            num_cross_stats {
              correlation: 0.923145
              covariance: 15.6
            }
            """, statistics_pb2.CrossFeatureStatistics())}
    self.assertCombinerOutputEqual(batches, generator, {}, expected_result)

  def test_cross_feature_stats_generator_single_feature(self):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        sample_rate=1.0)
    b1 = pa.RecordBatch.from_arrays([pa.array([[1.0], [3.0]])], ['a'])
    self.assertCombinerOutputEqual([b1], generator, {}, {})

  def test_cross_feature_stats_generator_string_feature(self):
    generator = cross_feature_stats_generator.CrossFeatureStatsGenerator(
        sample_rate=1.0)
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([['x'], ['y']]),
         pa.array([[2.0], [4.0]])], ['a', 'b'])
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([['a'], ['b']]),
         pa.array([[14.0], [16.0]])], ['a', 'b'])
    batches = [b1, b2]
    self.assertCombinerOutputEqual(batches, generator, {}, {})

if __name__ == '__main__':
  absltest.main()
