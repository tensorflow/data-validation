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
"""Tests for the SparseFeature stats generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import sparse_feature_stats_generator
from tensorflow_data_validation.utils import test_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class SparseFeatureStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  def test_sparse_feature_generator_valid_input(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a', 'b']]),
            pa.array([[1], [1, 3]]),
            pa.array([[2], [2, 4]])
        ], ['value_feature', 'index_feature1', 'index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_missing_value_and_index(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([None, None, ['a', 'b'], ['a', 'b'], ['a', 'b']]),
            pa.array([[1], [1], None, None, None]),
            pa.array([[2], [2], [2, 4], [2, 4], [2, 4]])
        ], ['value_feature', 'index_feature1', 'index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 2
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 3
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 1
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 1
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: -2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_length_mismatch(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[], [], ['a', 'b'], ['a', 'b'], ['a', 'b']]),
            pa.array([[1], [1], [1, 3], [1, 3], [1, 3]]),
            pa.array([[2], [2], [2, 4, 6, 7, 9], [2, 4, 6, 7, 9],
                      [2, 4, 6, 7, 9]])
        ], ['value_feature', 'index_feature1', 'index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 1
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 3
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 1
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_with_struct_leaves(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[{
                'value_feature': ['a'],
                'index_feature1': [1],
                'index_feature2': [2]
            }]]),
        ], ['parent']),
        pa.RecordBatch.from_arrays([
            pa.array([[{
                'value_feature': ['a', 'b'],
                'index_feature1': [1, 3],
                'index_feature2': [2, 4]
            }]]),
        ], ['parent']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'parent'
          type: STRUCT
          struct_domain {
            feature {
              name: 'index_feature1'
            }
            feature {
              name: 'index_feature2'
            }
            feature {
              name: 'value_feature'
            }
            sparse_feature {
              name: 'sparse_feature'
              index_feature {
                name: 'index_feature1'
              }
              index_feature {
                name: 'index_feature2'
              }
              value_feature {
                name: 'value_feature'
              }
            }
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['parent', 'sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'parent'
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_value_feature_not_in_batch(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a', 'b']]),
            pa.array([[1], [1, 3]]),
            pa.array([[2], [2, 4]])
        ], ['not_value_feature', 'index_feature1', 'index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 2
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 2
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 1
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 1
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_index_feature_not_in_batch(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a', 'b']]),
            pa.array([[1], [1, 3]]),
            pa.array([[2], [2, 4]])
        ], ['value_feature', 'index_feature1', 'not_index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 2
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: -1
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: -2
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_component_feature_null_array(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a', 'b']]),
            pa.array([[1], [1, 3]]),
            pa.array([None, None], type=pa.null())
        ], ['value_feature', 'index_feature1', 'index_feature2']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 2
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: -1
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: -2
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_batch_missing_entire_sparse_feature(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array(
                [None, None, ['a', 'b'], ['a', 'b'], ['a', 'b'], None, None]),
            pa.array([[1, 2], [1, 2], None, None, None, None, None]),
            pa.array([[2, 4], [2, 4], [2, 4, 6], [2, 4, 6], [2, 4, 6], None,
                      None]),
            pa.array([None, None, None, None, None, ['a', 'b'], ['a', 'b']]),
            pa.array([None, None, None, None, None, [2, 4], [2, 4]]),
            pa.array([None, None, None, None, None, None, None],
                     type=pa.null()),
        ], [
            'value_feature', 'index_feature1', 'index_feature2',
            'other_feature1', 'other_feature2', 'other_feature3'
        ]),
        pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None, None, ['a', 'b'], ['a', 'b']]),
            pa.array([None, None, None, None, None, [2, 4], [2, 4]]),
            pa.array([None, None, None, None, None, None, None], type=pa.null())
        ], ['other_feature1', 'other_feature2', 'other_feature3']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 2
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 3
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 2
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: -2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 1
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_dataset_missing_entire_sparse_feature(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array([['a']]),
        ], ['other_feature']),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        """, schema_pb2.Schema())
    # This is a semantically empty result which should not raise any anomalies.
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                    }
                    buckets {
                      label: 'index_feature2'
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                    }
                    buckets {
                      label: 'index_feature2'
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                    }
                    buckets {
                      label: 'index_feature2'
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_sparse_feature_generator_multiple_sparse_features(self):
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array(
                [None, None, ['a', 'b'], ['a', 'b'], ['a', 'b'], None, None]),
            pa.array([[1, 2], [1, 2], None, None, None, None, None]),
            pa.array([[2, 4], [2, 4], [2, 4, 6], [2, 4, 6], [2, 4, 6], None,
                      None]),
            pa.array([None, None, None, None, None, ['a', 'b'], ['a', 'b']]),
            pa.array([None, None, None, None, None, [2, 4], [2, 4]]),
            pa.array([None, None, None, None, None, None, None],
                     type=pa.null()),
        ], [
            'value_feature', 'index_feature1', 'index_feature2',
            'other_value_feature', 'other_index_feature1',
            'other_index_feature2'
        ]),
        pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None, None, ['a', 'b'], ['a', 'b']]),
            pa.array([None, None, None, None, None, [2, 4], [2, 4]]),
            pa.array([None, None, None, None, None, None, None], type=pa.null())
        ], [
            'other_value_feature', 'other_index_feature1',
            'other_index_feature2'
        ]),
    ]
    schema = text_format.Parse(
        """
        sparse_feature {
          name: 'sparse_feature'
          index_feature {
            name: 'index_feature1'
          }
          index_feature {
            name: 'index_feature2'
          }
          value_feature {
            name: 'value_feature'
          }
        }
        sparse_feature {
          name: 'other_sparse_feature'
          index_feature {
            name: 'other_index_feature1'
          }
          index_feature {
            name: 'other_index_feature2'
          }
          value_feature {
            name: 'other_value_feature'
          }
        }
        """, schema_pb2.Schema())
    expected_result = {
        types.FeaturePath(['sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 2
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 3
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 0
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: 2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 2
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'index_feature1'
                      sample_count: -2
                    }
                    buckets {
                      label: 'index_feature2'
                      sample_count: 1
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['other_sparse_feature']):
            text_format.Parse(
                """
                path {
                  step: 'other_sparse_feature'
                }
                custom_stats {
                  name: 'missing_value'
                  num: 0
                }
                custom_stats {
                  name: 'missing_index'
                  rank_histogram {
                    buckets {
                      label: 'other_index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'other_index_feature2'
                      sample_count: 4
                    }
                  }
                }
                custom_stats {
                  name: 'max_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'other_index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'other_index_feature2'
                      sample_count: -2
                    }
                  }
                }
                custom_stats {
                  name: 'min_length_diff'
                  rank_histogram {
                    buckets {
                      label: 'other_index_feature1'
                      sample_count: 0
                    }
                    buckets {
                      label: 'other_index_feature2'
                      sample_count: -2
                    }
                  }
                }""", statistics_pb2.FeatureNameStatistics())
    }
    generator = (
        sparse_feature_stats_generator.SparseFeatureStatsGenerator(
            schema))
    self.assertCombinerOutputEqual(batches, generator, expected_result)


if __name__ == '__main__':
  absltest.main()
