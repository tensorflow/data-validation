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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pandas as pd
import pyarrow as pa

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import lift_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class GetExampleValuePresenceTest(absltest.TestCase):
  """Tests for _get_example_value_presence."""

  def test_example_value_presence(self):
    t = pa.Table.from_arrays([
        pa.array([[1], [1, 1], [1, 2], [2]]),
    ], ['x'])
    expected_series = pd.Series([1, 1, 1, 2, 2], name='values',
                                index=pd.Index([0, 1, 2, 2, 3],
                                               name='example_indices'))
    pd.testing.assert_series_equal(
        expected_series,
        lift_stats_generator._get_example_value_presence(
            t, types.FeaturePath(['x']), boundaries=None))

  def test_example_value_presence_none_value(self):
    t = pa.Table.from_arrays([
        pa.array([[1], None]),
    ], ['x'])
    expected_series = pd.Series([1], name='values',
                                index=pd.Index([0], name='example_indices'))
    pd.testing.assert_series_equal(
        expected_series,
        lift_stats_generator._get_example_value_presence(
            t, types.FeaturePath(['x']), boundaries=None))

  def test_example_value_presence_null_array(self):
    t = pa.Table.from_arrays([
        pa.array([None, None], type=pa.null()),
    ], ['x'])
    self.assertIsNone(
        lift_stats_generator._get_example_value_presence(
            t, types.FeaturePath(['x']), boundaries=None))

  def test_example_value_presence_struct_leaf(self):
    t = pa.Table.from_arrays([
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
    expected_series = pd.Series([1, 2, 3, 1, 4], name='values',
                                index=pd.Index([0, 0, 0, 1, 1],
                                               name='example_indices'))
    pd.testing.assert_series_equal(
        expected_series,
        lift_stats_generator._get_example_value_presence(
            t, types.FeaturePath(['x', 'y']), boundaries=None))


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
        pa.Table.from_arrays([
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

  def test_lift_int_y(self):
    examples = [
        pa.Table.from_arrays([
            pa.array([[11], [11], [22], [11]]),
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
                      x_int: 11
                      lift: 1.3333333
                      x_count: 3
                      x_and_y_count: 2
                    }
                    lift_values {
                      x_int: 22
                      lift: 0.0
                      x_count: 1
                      x_and_y_count: 0
                    }
                  }
                  lift_series {
                    y_int: 1
                    y_count: 2
                    lift_values {
                      x_int: 22
                      lift: 2.0
                      x_count: 1
                      x_and_y_count: 1
                    }
                    lift_values {
                      x_int: 11
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
        schema=schema, y_path=types.FeaturePath(['int_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_bool_y(self):
    examples = [
        pa.Table.from_arrays([
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
        schema=schema, y_path=types.FeaturePath(['bool_y']))
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_float_y(self):
    examples = [
        pa.Table.from_arrays([
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
        y_boundaries=[2, 4])
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_lift_no_categorical_features(self):
    examples = [
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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

  def test_lift_min_x_count(self):
    examples = [
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        pa.Table.from_arrays([
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
        ('slice1', pa.Table.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice2', pa.Table.from_arrays([
            pa.array([['a'], ['a'], ['a']]),
            pa.array([['cat'], ['dog'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice1', pa.Table.from_arrays([
            pa.array([['a'], ['a'], ['b'], ['a']]),
            pa.array([['cat'], ['dog'], ['cat'], ['dog']]),
        ], ['categorical_x', 'string_y'])),
        ('slice2', pa.Table.from_arrays([
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
