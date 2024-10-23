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

"""Tests for LiftStatsGenerator."""
from typing import Optional, Sequence, Text

from absl.testing import absltest
import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import lift_stats_generator
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def _get_example_value_presence_as_dataframe(
    record_batch: pa.RecordBatch, path: types.FeaturePath,
    boundaries: Optional[Sequence[float]],
    weight_column_name: Optional[Text]) -> Optional[pd.DataFrame]:
  value_presence = lift_stats_generator._get_example_value_presence(
      record_batch, path, boundaries, weight_column_name)
  if not value_presence:
    return
  df = pd.DataFrame({
      'example_indices': value_presence.example_indices,
      'values': value_presence.values,
      'weights': value_presence.weights,
  })
  return df.set_index('example_indices')


class GetExampleValuePresenceTest(absltest.TestCase):
  """Tests for _get_example_value_presence."""

  def test_example_value_presence(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [1, 1], [1, 2], [2]]),
    ], ['x'])
    expected_df = pd.DataFrame(
        {
            'values': [1, 1, 1, 2, 2],
            'weights': [1, 1, 1, 1, 1],
        },
        index=pd.Index([0, 1, 2, 2, 3], name='example_indices'))
    pd.testing.assert_frame_equal(
        expected_df,
        _get_example_value_presence_as_dataframe(
            t,
            types.FeaturePath(['x']),
            boundaries=None,
            weight_column_name=None))

  def test_example_value_presence_weighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [1, 1], [1, 2], [2]]),
        pa.array([[.5], [1.0], [1.5], [2.0]]),
    ], ['x', 'w'])
    expected_df = pd.DataFrame(
        {
            'values': [1, 1, 1, 2, 2],
            'weights': [.5, 1.0, 1.5, 1.5, 2.0]
        },
        index=pd.Index([0, 1, 2, 2, 3], name='example_indices'))
    pd.testing.assert_frame_equal(
        expected_df,
        _get_example_value_presence_as_dataframe(
            t,
            types.FeaturePath(['x']),
            boundaries=None,
            weight_column_name='w'))

  def test_example_value_presence_none_value(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], None]),
    ], ['x'])
    expected_df = pd.DataFrame({
        'values': [1],
        'weights': [1],
    },
                               index=pd.Index([0], name='example_indices'))
    pd.testing.assert_frame_equal(
        expected_df,
        _get_example_value_presence_as_dataframe(
            t,
            types.FeaturePath(['x']),
            boundaries=None,
            weight_column_name=None))

  def test_example_value_presence_null_array(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([None, None], type=pa.null()),
    ], ['x'])
    self.assertIsNone(
        _get_example_value_presence_as_dataframe(
            t,
            types.FeaturePath(['x']),
            boundaries=None,
            weight_column_name=None))

  def test_example_value_presence_struct_leaf(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([
            [
                {'y': [1]},
                {'y': [1, 2]},
                {'y': [3]},
            ],
            [
                {'y': [1, 4]},
            ]
        ])], ['x'])
    expected_df = pd.DataFrame(
        {
            'values': [1, 2, 3, 1, 4],
            'weights': [1, 1, 1, 1, 1],
        },
        index=pd.Index([0, 0, 0, 1, 1], name='example_indices'))
    pd.testing.assert_frame_equal(
        expected_df,
        _get_example_value_presence_as_dataframe(
            t,
            types.FeaturePath(['x', 'y']),
            boundaries=None,
            weight_column_name=None))


class ToPartialCountsTest(absltest.TestCase):

  def test_text_to_partial_counts_unweighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([['a'], ['b'], ['a']]),
    ], ['x'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path, x='a', y=None), 2),
        (lift_stats_generator._SlicedXYKey('', x_path, x='b', y=None), 1),
    ]
    for (expected_key,
         expected_count), ((actual_key, actual_value), actual_count) in zip(
             expected_counts,
             lift_stats_generator._to_partial_counts(
                 ('', t),
                 path=types.FeaturePath(['x']),
                 boundaries=None,
                 weight_column_name=None)):
      self.assertEqual('', actual_key)
      self.assertEqual(expected_key.x, actual_value)
      self.assertEqual(expected_count, actual_count)

  def test_float_to_partial_counts_unweighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[100.], [200.], [100.]]),
    ], ['x'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path, x=100., y=None), 2),
        (lift_stats_generator._SlicedXYKey('', x_path, x=200., y=None), 1),
    ]
    for (expected_key,
         expected_count), ((actual_key, actual_value), actual_count) in zip(
             expected_counts,
             lift_stats_generator._to_partial_counts(
                 ('', t),
                 path=types.FeaturePath(['x']),
                 boundaries=None,
                 weight_column_name=None)):
      self.assertEqual('', actual_key)
      self.assertEqual(expected_key.x, actual_value)
      self.assertEqual(expected_count, actual_count)

  def test_to_partial_counts_weighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [2], [1]]),
        pa.array([[0.5], [0.5], [2.0]]),
    ], ['x', 'w'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path, x=1, y=None), 2.5),
        (lift_stats_generator._SlicedXYKey('', x_path, x=2, y=None), 0.5),
    ]
    for (expected_key,
         expected_count), ((actual_key, actual_value), actual_count) in zip(
             expected_counts,
             lift_stats_generator._to_partial_counts(
                 ('', t),
                 path=types.FeaturePath(['x']),
                 boundaries=None,
                 weight_column_name='w')):
      self.assertEqual('', actual_key)
      self.assertEqual(expected_key.x, actual_value)
      self.assertEqual(expected_count, actual_count)


class ToPartialXCountsTest(absltest.TestCase):

  def test_to_partial_x_counts_unweighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [2], [1]]),
    ], ['x'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1, y=None), 2),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=2, y=None), 1),
    ]
    for (expected_key, expected_count), (actual_key, actual_count) in zip(
        expected_counts,
        lift_stats_generator._to_partial_x_counts(
            ('', t),
            x_paths=[types.FeaturePath(['x'])],
            example_weight_map=ExampleWeightMap())):
      self.assertEqual(str(expected_key.x_path), str(actual_key.x_path))
      self.assertEqual(expected_key.x, actual_key.x)
      self.assertEqual(expected_count, actual_count)

  def test_to_partial_x_counts_weighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [2], [1]]),
        pa.array([[0.5], [0.5], [2.0]]),
    ], ['x', 'w'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1,
                                           y=None), 2.5),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=2,
                                           y=None), 0.5),
    ]
    for (expected_key, expected_count), (actual_key, actual_count) in zip(
        expected_counts,
        lift_stats_generator._to_partial_x_counts(
            ('', t), x_paths=[types.FeaturePath(['x'])],
            example_weight_map=ExampleWeightMap(weight_feature='w'))):
      self.assertEqual(str(expected_key.x_path), str(actual_key.x_path))
      self.assertEqual(expected_key.x, actual_key.x)
      self.assertEqual(expected_count, actual_count)


class ToPartialCopresenceCountsTest(absltest.TestCase):

  def test_to_partial_copresence_counts_unweighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [2], [1]]),
        pa.array([['a'], ['a'], ['b']]),
    ], ['x', 'y'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1, y='a'), 1),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1, y='b'), 1),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=2, y='a'), 1)
    ]
    actual_counts = list(
        lift_stats_generator._to_partial_copresence_counts(
            ('', t),
            y_path=types.FeaturePath(['y']),
            x_paths=[types.FeaturePath(['x'])],
            y_boundaries=None,
            example_weight_map=ExampleWeightMap()))
    self.assertSameElements(expected_counts, actual_counts)

  def test_to_partial_copresence_counts_weighted(self):
    t = pa.RecordBatch.from_arrays([
        pa.array([[1], [2], [1]]),
        pa.array([['a'], ['a'], ['b']]),
        pa.array([[0.5], [0.5], [2.0]]),
    ], ['x', 'y', 'w'])
    x_path = types.FeaturePath(['x'])
    expected_counts = [
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1,
                                           y='a'), 0.5),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=1,
                                           y='b'), 2.0),
        (lift_stats_generator._SlicedXYKey('', x_path.steps(), x=2, y='a'), 0.5)
    ]
    actual_counts = list(
        lift_stats_generator._to_partial_copresence_counts(
            ('', t),
            y_path=types.FeaturePath(['y']),
            x_paths=[types.FeaturePath(['x'])],
            y_boundaries=None,
            example_weight_map=ExampleWeightMap(weight_feature='w')))
    self.assertSameElements(expected_counts, actual_counts)


class LiftStatsGeneratorTest(test_util.TransformStatsGeneratorTest):
  """Tests for LiftStatsGenerator."""

  def test_lift_string_y_with_boundaries(self):
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    with self.assertRaisesRegex(ValueError,
                                r'Boundaries cannot be applied to a '
                                'categorical y_path.*'):
      lift_stats_generator.LiftStatsGenerator(
          schema=schema,
          y_path=types.FeaturePath(['string_y']),
          y_boundaries=[1, 2, 3])

  def test_lift_int_y_with_no_boundaries(self):
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'int_y'
          type: INT
        }
        """, schema_pb2.Schema())
    with self.assertRaisesRegex(ValueError,
                                r'Boundaries must be provided with a non-'
                                'categorical y_path.*'):
      lift_stats_generator.LiftStatsGenerator(
          schema=schema, y_path=types.FeaturePath(['int_y']))

  def test_lift_with_no_schema_or_x_path(self):
    with self.assertRaisesRegex(ValueError,
                                r'Either a schema or x_paths must be provided'):
      lift_stats_generator.LiftStatsGenerator(
          schema=None, y_path=types.FeaturePath(['int_y']))

  def test_lift_string_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            features {
              custom_stats {
                name: "Lift (Y=cat)"
                rank_histogram {
                  buckets {
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    label: "a"
                    sample_count: 0.6666667
                  }
                }
              }
              custom_stats {
                name: "Lift (Y=dog)"
                rank_histogram {
                  buckets {
                    label: "a"
                    sample_count: 1.3333333
                  }
                  buckets {
                    label: "b"
                  }
                }
              }
              path {
                step: "categorical_x"
              }
            }
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['string_y']),
        output_custom_stats=True)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_bytes_x_and_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([[b'a'], [b'a'], [b'\x80abc'], [b'a']]),
            pa.array([[b'cat'], [b'dog'], [b'cat'], [b'dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "__BYTES_VALUE__"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "__BYTES_VALUE__"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_int_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([[1], [0], [1], [0]]),
        ], ['categorical_x', 'int_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'int_y'
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            features {
              custom_stats {
                name: "Lift (Y=0)"
                rank_histogram {
                  buckets {
                    label: "a"
                    sample_count: 1.3333333
                  }
                  buckets {
                    label: "b"
                  }
                }
              }
              custom_stats {
                name: "Lift (Y=1)"
                rank_histogram {
                  buckets {
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    label: "a"
                    sample_count: 0.6666667
                  }
                }
              }
              path {
                step: "categorical_x"
              }
            }
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "int_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_int: 0
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_int: 1
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    def metrics_verify_fn(metric_results):
      num_xy_pairs_distinct_counters = metric_results.query(
          beam.metrics.metric.MetricsFilter().with_name('num_xy_pairs_distinct')
      )[beam.metrics.metric.MetricResults.COUNTERS]
      self.assertLen(num_xy_pairs_distinct_counters, 1)
      self.assertEqual(4, num_xy_pairs_distinct_counters[0].committed)

      num_xy_pairs_batch_copresent_dists = metric_results.query(
          beam.metrics.metric.MetricsFilter().with_name(
              'num_xy_pairs_batch_copresent'))[
                  beam.metrics.metric.MetricResults.DISTRIBUTIONS]
      self.assertLen(num_xy_pairs_batch_copresent_dists, 1)
      num_xy_pairs_batch_copresent_dist = num_xy_pairs_batch_copresent_dists[0]
      self.assertEqual(3, num_xy_pairs_batch_copresent_dist.committed.sum)
      self.assertEqual(1, num_xy_pairs_batch_copresent_dist.committed.count)
      self.assertEqual(3, num_xy_pairs_batch_copresent_dist.committed.min)
      self.assertEqual(3, num_xy_pairs_batch_copresent_dist.committed.max)

      placeholder_num_right_keys_dists = metric_results.query(
          beam.metrics.metric.MetricsFilter().with_name(
              'right_lookup_num_keys').with_step(
                  'JoinWithPlaceholderYRates'))[
                      beam.metrics.metric.MetricResults.DISTRIBUTIONS]
      self.assertLen(placeholder_num_right_keys_dists, 1)
      placeholder_num_right_keys_dist = placeholder_num_right_keys_dists[0]
      # min and max should always equal because this dist is really a gauge
      # The expected number of distinct keys is the number of slices, as the
      # lookup is built from a stream of slice_key / y-value pairs.
      self.assertEqual(1, placeholder_num_right_keys_dist.committed.min)
      self.assertEqual(1, placeholder_num_right_keys_dist.committed.max)

      placeholder_num_right_values_dists = metric_results.query(
          beam.metrics.metric.MetricsFilter().with_name(
              'right_lookup_num_values').with_step(
                  'JoinWithPlaceholderYRates'))[
                      beam.metrics.metric.MetricResults.DISTRIBUTIONS]
      self.assertLen(placeholder_num_right_values_dists, 1)
      placeholder_num_right_values_dist = placeholder_num_right_values_dists[0]
      # min and max should always equal because this dist is really a gauge
      # The expected number of values is the total number of y-values across
      # all slices, as the lookup is built from a stream of slice_key / y-value
      # pairs.
      self.assertEqual(2, placeholder_num_right_values_dist.committed.min)
      self.assertEqual(2, placeholder_num_right_values_dist.committed.max)

      placeholder_construction_dists = metric_results.query(
          beam.metrics.metric.MetricsFilter().with_name(
              'right_lookup_construction_seconds').with_step(
                  'JoinWithPlaceholderYRates'))[
                      beam.metrics.metric.MetricResults.DISTRIBUTIONS]
      self.assertLen(placeholder_construction_dists, 1)
      placeholder_construction_dist = placeholder_construction_dists[0]
      self.assertGreater(placeholder_construction_dist.committed.count, 0)

    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['int_y']),
        output_custom_stats=True)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        metrics_verify_fn=metrics_verify_fn,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_bool_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([[1], [0], [1], [0]]),
        ], ['categorical_x', 'bool_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'bool_y'
          type: INT
          bool_domain {}
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            features {
              custom_stats {
                name: "Lift (Y=0)"
                rank_histogram {
                  buckets {
                    label: "a"
                    sample_count: 1.3333333
                  }
                  buckets {
                    label: "b"
                  }
                }
              }
              custom_stats {
                name: "Lift (Y=1)"
                rank_histogram {
                  buckets {
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    label: "a"
                    sample_count: 0.6666667
                  }
                }
              }
              path {
                step: "categorical_x"
              }
            }
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "bool_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_int: 0
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_int: 1
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['bool_y']),
        output_custom_stats=True)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_float_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([[1.1], [2.2], [3.3], [4.4]]),
        ], ['categorical_x', 'float_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'float_y'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            features {
              custom_stats {
                name: "Lift (Y=[-inf,2))"
                rank_histogram {
                  buckets {
                    label: "a"
                    sample_count: 1.3333333
                  }
                  buckets {
                    label: "b"
                  }
                }
              }
              custom_stats {
                name: "Lift (Y=[2,4))"
                rank_histogram {
                  buckets {
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    label: "a"
                    sample_count: 0.6666667
                  }
                }
              }
              custom_stats {
                name: "Lift (Y=[4,inf])"
                rank_histogram {
                  buckets {
                    label: "a"
                    sample_count: 1.3333333
                  }
                  buckets {
                    label: "b"
                  }
                }
              }
              path {
                step: "categorical_x"
              }
            }
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "float_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_bucket {
                      low_value: -inf
                      high_value: 2
                    }
                    y_count: 1
                    lift_values {
                      x_string: 'a'
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_bucket {
                      low_value: 2
                      high_value: 4
                    }
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_bucket {
                      low_value: 4
                      high_value: inf
                    }
                    y_count: 1
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['float_y']),
        y_boundaries=[2, 4],
        output_custom_stats=True)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_weighted(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['x'], ['x'], ['y'], ['x']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
            pa.array([[.1], [.1], [.5], [.2]]),
            pa.array([[.5], [.5], [2], [1]]),
        ], [
            'categorical_x1', 'categorical_x2', 'string_y', 'weight_x2',
            'weight'
        ]),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x1'
          type: BYTES
        }
        feature {
          name: 'categorical_x2'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        feature {
          name: 'weight'
          type: FLOAT
        }
        feature {
          name: 'weight_x2'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    expected_results = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x1"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    weighted_y_count: 2.5
                    lift_values {
                      x_string: "b"
                      lift: 1.6
                      weighted_x_count: 2
                      weighted_x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.4
                      weighted_x_count: 2
                      weighted_x_and_y_count: .5
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    weighted_y_count: 1.5
                    lift_values {
                      x_string: "a"
                      lift: 2.0
                      weighted_x_count: 2
                      weighted_x_and_y_count: 1.5
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      weighted_x_count: 2
                      weighted_x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x1"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x2"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    weighted_y_count: 2.5
                    lift_values {
                      x_string: "y"
                      lift: 1.6
                      weighted_x_count: 0.5
                      weighted_x_and_y_count: 0.5
                    }
                    lift_values {
                      x_string: "x"
                      lift: 0.4
                      weighted_x_count: 0.4
                      weighted_x_and_y_count: 0.1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    weighted_y_count: 1.5
                    lift_values {
                      x_string: "x"
                      lift: 2.0
                      weighted_x_count: 0.4
                      weighted_x_and_y_count: 0.3
                    }
                    lift_values {
                      x_string: "y"
                      weighted_x_count: 0.5
                      weighted_x_and_y_count: 0.0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x2"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "y"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "x"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "x"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "y"
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['string_y']),
        example_weight_map=ExampleWeightMap(
            weight_feature='weight',
            per_feature_override={
                types.FeaturePath(['categorical_x2']): 'weight_x2'
            }))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_results,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_weighted_missing_weight(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a']]),
            pa.array([['cat'], ['dog']]),
            pa.array([[], [1]]),
        ], ['categorical_x', 'string_y', 'weight']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        feature {
          name: 'weight'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']),
        example_weight_map=ExampleWeightMap(weight_feature='weight'))
    examples = [(None, e) for e in examples]
    with self.assertRaisesRegex(ValueError,
                                r'Weight column "weight" must have exactly one '
                                'value in each example.*'):
      with beam.Pipeline() as p:
        _ = p | beam.Create(examples) | generator.ptransform

  def test_lift_weighted_weight_is_none(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a']]),
            pa.array([['cat']]),
            pa.array([None]),
        ], ['categorical_x', 'string_y', 'weight']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        feature {
          name: 'weight'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']),
        example_weight_map=ExampleWeightMap(weight_feature='weight'))
    examples = [(None, e) for e in examples]
    with self.assertRaisesRegex(ValueError,
                                r'Weight column "weight" cannot be null.*'):
      with beam.Pipeline() as p:
        _ = p | beam.Create(examples) | generator.ptransform

  def test_lift_no_categorical_features(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([[1.0], [2.0], [3.0], [4.0]]),
            pa.array([[1], [0], [1], [0]]),
        ], ['continous_x', 'int_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'continuous_x'
          type: FLOAT
        }
        feature {
          name: 'int_y'
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['int_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_x_is_none(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([None, None, ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_y_is_none(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([None, [.7], [.4], [.6]]),
        ], ['categorical_x', 'float_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'float_y'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "float_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_bucket {
                      low_value: -inf
                      high_value: 0.5
                    }
                    y_count: 1
                    lift_values {
                      x_string: "b"
                      lift: 4.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.0
                      x_count: 3
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_bucket {
                      low_value: 0.5
                      high_value: inf
                    }
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['float_y']),
        y_boundaries=[0.5])
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_null_x(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None], type=pa.null()),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_null_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([None, None, None, None], type=pa.null()),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_missing_x_and_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            # explicitly construct type to avoid treating as null type
            pa.array([], type=pa.list_(pa.binary())),
            pa.array([], type=pa.list_(pa.binary())),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_float_y_is_nan(self):
    # after calling bin_array, this is effectively an empty array.
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a']]),
            pa.array([[np.nan]]),
        ], ['categorical_x', 'float_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'float_y'
          type: FLOAT
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['float_y']), y_boundaries=[1])
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_min_x_count(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['string_y']),
        min_x_count=2)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_min_x_count_filters_all(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = []
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['string_y']),
        min_x_count=4)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_overlapping_top_bottom_k(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['b'], ['c'], ['a']]),
            pa.array([['cat'], ['cat'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 3
                    lift_values {
                      x_string: "b"
                      lift: 1.3333333
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "c"
                      lift: 1.3333333
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 2
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 1
                    lift_values {
                      x_string: "a"
                      lift: 2.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                    lift_values {
                      x_string: "c"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema,
        y_path=types.FeaturePath(['string_y']),
        top_k_per_y=3,
        bottom_k_per_y=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_flattened_x(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([
                [
                    {'docs': ['a', 'b']},
                    {'docs': ['a']},
                    {'docs': ['c']}
                ],
                [
                    {'docs': ['a', 'b']}
                ]
            ]),
            pa.array([['pos'], ['neg']]),
        ], ['doc_set', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'doc_set'
          struct_domain {
            feature {
              name: 'docs'
              type: BYTES
            }
          }
          type: STRUCT
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: 'doc_set'
                step: 'docs'
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "neg"
                    y_count: 1
                    lift_values {
                      x_string: "a"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "c"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_string: "pos"
                    y_count: 1
                    lift_values {
                      x_string: "c"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_flattened_x_leaf(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'a'], ['a'], ['b', 'b'], ['a', 'a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_multi_x(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['x'], ['x'], ['y'], ['x']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x1', 'categorical_x2', 'string_y']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x1'
          type: BYTES
        }
        feature {
          name: 'categorical_x2'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: "categorical_x2"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "y"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "x"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "x"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "y"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse("""
            cross_features {
              path_x {
                step: "categorical_x1"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_provided_x_no_schema(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['x'], ['x'], ['y'], ['x']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x1', 'categorical_x2', 'string_y']),
    ]
    expected_result = [
        text_format.Parse("""
            cross_features {
              path_x {
                step: "categorical_x1"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 2
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=None,
        y_path=types.FeaturePath(['string_y']),
        x_paths=[types.FeaturePath(['categorical_x1'])])
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_flattened_x_and_y(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([
                [
                    {'docs': ['a', 'b']},
                    {'docs': ['a']},
                    {'docs': ['c']}
                ],
                [
                    {'docs': ['a', 'b']}
                ]
            ]),
            pa.array([
                [
                    {'labels': ['y1', 'y2']},
                    {'labels': ['y1']}
                ],
                [
                    {'labels': ['y2']},
                ]
            ]),
        ], ['doc_set', 'evaluations']),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'doc_set'
          type: STRUCT
          struct_domain {
            feature {
              name: 'docs'
              type: BYTES
            }
          }
        }
        feature {
          name: 'evaluations'
          type: STRUCT
          struct_domain {
            feature {
              name: 'labels'
              type: BYTES
            }
          }
        }
        """, schema_pb2.Schema())
    expected_result = [
        text_format.Parse(
            """
            cross_features {
              path_x {
                step: 'doc_set'
                step: 'docs'
              }
              path_y {
                step: "evaluations"
                step: "labels"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "y1"
                    y_count: 1
                    lift_values {
                      x_string: "c"
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "a"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_string: "b"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "y2"
                    y_count: 2
                    lift_values {
                      x_string: "a"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "b"
                      lift: 1.0
                      x_count: 2
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "c"
                      lift: 1.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['evaluations', 'labels']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_slice_aware(self):
    examples = [
        ('slice1', pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice2', pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['a']]),
            pa.array([['cat'], ['dog'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice1', pa.RecordBatch.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice2', pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None], type=pa.null()),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
    ]
    schema = text_format.Parse(
        """
        feature {
          name: 'categorical_x'
          type: BYTES
        }
        feature {
          name: 'string_y'
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_result = [
        ('slice1',
         text_format.Parse(
             """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 4
                    lift_values {
                      x_string: "b"
                      lift: 2.0
                      x_count: 2
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_string: "a"
                      lift: 0.6666667
                      x_count: 6
                      x_and_y_count: 2
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 4
                    lift_values {
                      x_string: "a"
                      lift: 1.3333333
                      x_count: 6
                      x_and_y_count: 4
                    }
                    lift_values {
                      x_string: "b"
                      lift: 0.0
                      x_count: 2
                      x_and_y_count: 0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics())),
        ('slice2',
         text_format.Parse(
             """
            cross_features {
              path_x {
                step: "categorical_x"
              }
              path_y {
                step: "string_y"
              }
              categorical_cross_stats {
                lift {
                  lift_series {
                    y_string: "cat"
                    y_count: 3
                    lift_values {
                      x_string: "a"
                      lift: 0.7777778
                      x_count: 3
                      x_and_y_count: 1
                    }
                  }
                  lift_series {
                    y_string: "dog"
                    y_count: 4
                    lift_values {
                      x_string: "a"
                      lift: 1.1666667
                      x_count: 3
                      x_and_y_count: 2
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics())),
    ]
    generator = lift_stats_generator.LiftStatsGenerator(
        schema=schema, y_path=types.FeaturePath(['string_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result)


if __name__ == '__main__':
  absltest.main()
