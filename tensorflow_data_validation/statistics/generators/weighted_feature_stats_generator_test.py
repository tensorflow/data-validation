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
"""Tests for WeightedFeatureStatsGenerator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import weighted_feature_stats_generator
from tensorflow_data_validation.utils import test_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class WeightedFeatureStatsGeneratorTest(parameterized.TestCase,
                                        test_util.CombinerStatsGeneratorTest):

  @parameterized.named_parameters(
      {
          'testcase_name': 'AllMatching',
          'batches': [
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a', 'b']]),
                   pa.array([[2], [2, 4]])], ['value', 'weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': 0.0,
          'expected_max_weight_length_diff': 0.0
      }, {
          'testcase_name': 'AllMatchingMultiBatch',
          'batches': [
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a', 'b']]),
                   pa.array([[2], [2, 4]])], ['value', 'weight']),
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a', 'b']]),
                   pa.array([[2], [2, 4]])], ['value', 'weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': 0.0,
          'expected_max_weight_length_diff': 0.0
      }, {
          'testcase_name': 'LengthMismatchPositive',
          'batches': [
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a']]),
                   pa.array([[2], [2, 4]])], ['value', 'weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': 0.0,
          'expected_max_weight_length_diff': 1.0
      }, {
          'testcase_name': 'LengthMismatchNegative',
          'batches': [
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a', 'b']]),
                   pa.array([[2], [2]])], ['value', 'weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': -1.0,
          'expected_max_weight_length_diff': 0.0
      }, {
          'testcase_name': 'LengthMismatchMultiBatch',
          'batches': [
              pa.RecordBatch.from_arrays(
                  [pa.array([['a'], ['a', 'b']]),
                   pa.array([[], []])], ['value', 'weight']),
              pa.RecordBatch.from_arrays([pa.array([[1], [1, 1]])], ['other'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': -2.0,
          'expected_max_weight_length_diff': -1.0
      }, {
          'testcase_name': 'SomePairsMissing',
          'batches': [
              pa.RecordBatch.from_arrays([
                  pa.array([['a'], None, ['a', 'b']]),
                  pa.array([[1, 1], None, [1, 1, 1]])
              ], ['value', 'weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': 1.0,
          'expected_max_weight_length_diff': 1.0
      }, {
          'testcase_name': 'EmptyWeights',
          'batches': [
              pa.RecordBatch.from_arrays([pa.array([['a'], ['a', 'b']])],
                                         ['value'])
          ],
          'expected_missing_weight': 2.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': -2.0,
          'expected_max_weight_length_diff': -1.0
      }, {
          'testcase_name': 'EmptyValues',
          'batches': [
              pa.RecordBatch.from_arrays([pa.array([[1], [1, 2]])], ['weight'])
          ],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 2.0,
          'expected_min_weight_length_diff': 1.0,
          'expected_max_weight_length_diff': 2.0
      }, {
          'testcase_name': 'EmptyWeightsAndValues',
          'batches': [pa.RecordBatch.from_arrays([])],
          'expected_missing_weight': 0.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': 0.0,
          'expected_max_weight_length_diff': 0.0
      }, {
          'testcase_name': 'NullWeightArray',
          'batches': [
              pa.RecordBatch.from_arrays([
                  pa.array([['a'], ['a', 'b']]),
                  pa.array([None, None], type=pa.null())
              ], ['value', 'weight'])
          ],
          'expected_missing_weight': 2.0,
          'expected_missing_value': 0.0,
          'expected_min_weight_length_diff': -2.0,
          'expected_max_weight_length_diff': -1.0
      })
  def test_single_weighted_feature(self, batches, expected_missing_weight,
                                   expected_missing_value,
                                   expected_min_weight_length_diff,
                                   expected_max_weight_length_diff):
    schema = text_format.Parse(
        """
        weighted_feature {
          name: 'weighted_feature'
          feature {
            step: 'value'
          }
          weight_feature {
            step: 'weight'
          }
        }
        """, schema_pb2.Schema())
    generator = (
        weighted_feature_stats_generator.WeightedFeatureStatsGenerator(schema))

    expected_stats = statistics_pb2.FeatureNameStatistics()
    expected_stats.path.step.append('weighted_feature')
    expected_stats.custom_stats.add(
        name='missing_weight', num=expected_missing_weight)
    expected_stats.custom_stats.add(
        name='missing_value', num=expected_missing_value)
    expected_stats.custom_stats.add(
        name='min_weight_length_diff',
        num=expected_min_weight_length_diff)
    expected_stats.custom_stats.add(
        name='max_weight_length_diff',
        num=expected_max_weight_length_diff)
    expected_result = {types.FeaturePath(['weighted_feature']): expected_stats}

    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_shared_weight(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a', 'b'], ['a']]),
            pa.array([['x'], ['y'], ['x']]),
            pa.array([[2], [4], None])
        ], ['value1', 'value2', 'weight'])
    ]
    schema = text_format.Parse(
        """
        weighted_feature {
          name: 'weighted_feature1'
          feature {
            step: 'value1'
          }
          weight_feature {
            step: 'weight'
          }
        }
        weighted_feature {
          name: 'weighted_feature2'
          feature {
            step: 'value2'
          }
          weight_feature {
            step: 'weight'
          }
        }""", schema_pb2.Schema())
    generator = (
        weighted_feature_stats_generator.WeightedFeatureStatsGenerator(schema))

    expected_result = {
        types.FeaturePath(['weighted_feature1']):
            text_format.Parse(
                """
                path {
                  step: 'weighted_feature1'
                }
                custom_stats {
                  name: 'missing_weight'
                  num: 1.0
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0.0
                }
                custom_stats {
                  name: 'min_weight_length_diff'
                  num: -1.0
                }
                custom_stats {
                  name: 'max_weight_length_diff'
                  num: 0.0
                }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['weighted_feature2']):
            text_format.Parse(
                """
                path {
                  step: 'weighted_feature2'
                }
                custom_stats {
                  name: 'missing_weight'
                  num: 1.0
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0.0
                }
                custom_stats {
                  name: 'min_weight_length_diff'
                  num: -1.0
                }
                custom_stats {
                  name: 'max_weight_length_diff'
                  num: 0.0
                }""", statistics_pb2.FeatureNameStatistics())
    }

    self.assertCombinerOutputEqual(batches, generator, expected_result)


if __name__ == '__main__':
  absltest.main()
