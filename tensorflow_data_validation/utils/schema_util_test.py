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
"""Tests for schema utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import schema_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

FLAGS = flags.FLAGS


SET_DOMAIN_VALID_TESTS = [
    {
        'testcase_name': 'int_domain',
        'input_schema_proto_text': '''feature { name: 'x' }''',
        'feature_name_or_path': 'x',
        'domain': schema_pb2.IntDomain(min=1, max=5),
        'output_schema_proto_text': '''
          feature { name: 'x' int_domain { min: 1 max: 5 } }'''
    },
    {
        'testcase_name': 'float_domain',
        'input_schema_proto_text': '''feature { name: 'x' }''',
        'feature_name_or_path': 'x',
        'domain': schema_pb2.FloatDomain(min=1.1, max=5.1),
        'output_schema_proto_text': '''
          feature { name: 'x' float_domain { min: 1.1 max: 5.1 } }'''
    },
    {
        'testcase_name': 'string_domain',
        'input_schema_proto_text': '''feature { name: 'x' }''',
        'feature_name_or_path': 'x',
        'domain': schema_pb2.StringDomain(value=['a', 'b']),
        'output_schema_proto_text': '''
          feature { name: 'x' string_domain { value: 'a' value: 'b' } }'''
    },
    {
        'testcase_name': 'bool_domain',
        'input_schema_proto_text': '''feature { name: 'x' }''',
        'feature_name_or_path': 'x',
        'domain': schema_pb2.BoolDomain(true_value='T', false_value='F'),
        'output_schema_proto_text': '''
          feature { name: 'x' bool_domain { true_value: 'T' false_value: 'F' } }
        '''
    },
    {
        'testcase_name': 'global_domain',
        'input_schema_proto_text': '''
          string_domain { name: 'global_domain' value: 'a' value: 'b' }
          feature { name: 'x' }''',
        'feature_name_or_path': 'x',
        'domain': 'global_domain',
        'output_schema_proto_text': '''
          string_domain { name: 'global_domain' value: 'a' value: 'b' }
          feature { name: 'x' domain: 'global_domain' }
        '''
    },
    {
        'testcase_name': 'set_domain_using_path',
        'input_schema_proto_text': '''
          feature {
            name: "feature1"
            type: STRUCT
            struct_domain {
              feature {
                name: "sub_feature1"
              }
            }
          }
          ''',
        'feature_name_or_path': types.FeaturePath(['feature1', 'sub_feature1']),
        'domain': schema_pb2.BoolDomain(true_value='T', false_value='F'),
        'output_schema_proto_text': '''
          feature {
            name: "feature1"
            type: STRUCT
            struct_domain {
              feature {
                name: "sub_feature1"
                bool_domain {
                  true_value: 'T'
                  false_value: 'F'
                }
              }
            }
          }
        '''
    }
]


class SchemaUtilTest(parameterized.TestCase):

  def test_get_feature(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
        }
        feature {
          name: "feature2"
        }
        """, schema_pb2.Schema())

    feature2 = schema_util.get_feature(schema, 'feature2')
    self.assertEqual(feature2.name, 'feature2')
    # Check to verify that we are operating on the same feature object.
    self.assertIs(feature2, schema_util.get_feature(schema, 'feature2'))

  def test_get_feature_using_path(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          type: STRUCT
          struct_domain {
            feature {
              name: "sub_feature1"
            }
          }
        }
        """, schema_pb2.Schema())
    sub_feature1 = schema_util.get_feature(
        schema, types.FeaturePath(['feature1', 'sub_feature1']))
    self.assertIs(sub_feature1, schema.feature[0].struct_domain.feature[0])

  def test_get_feature_not_present(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
        }
        """, schema_pb2.Schema())

    with self.assertRaisesRegex(ValueError, 'Feature.*not found in the schema'):
      _ = schema_util.get_feature(schema, 'feature2')

  def test_get_feature_using_path_not_present(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          type: STRUCT
          struct_domain {
            feature {
              name: "sub_feature1"
            }
          }
        }
        """, schema_pb2.Schema())
    with self.assertRaisesRegex(ValueError, 'Feature.*not found in the schema'):
      _ = schema_util.get_feature(
          schema, types.FeaturePath(['feature1', 'sub_feature2']))

  def test_get_feature_internal_step_not_struct(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
        }
        """, schema_pb2.Schema())
    with self.assertRaisesRegex(ValueError,
                                'does not refer to a valid STRUCT feature'):
      _ = schema_util.get_feature(
          schema, types.FeaturePath(['feature1', 'sub_feature2']))

  def test_get_feature_invalid_schema_input(self):
    with self.assertRaisesRegex(TypeError, 'should be a Schema proto'):
      _ = schema_util.get_feature({}, 'feature')

  def test_get_string_domain_schema_level_domain(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "domain1"
        }
        string_domain {
          name: "domain2"
        }
        feature {
          name: "feature1"
          domain: "domain2"
        }
        """, schema_pb2.Schema())

    domain2 = schema_util.get_domain(schema, 'feature1')
    self.assertIsInstance(domain2, schema_pb2.StringDomain)
    self.assertEqual(domain2.name, 'domain2')
    # Check to verify that we are operating on the same domain object.
    self.assertIs(domain2, schema_util.get_domain(schema, 'feature1'))

  def test_get_string_domain_feature_level_domain(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "domain2"
        }
        feature {
          name: "feature1"
          string_domain {
            name: "domain1"
          }
        }
        """, schema_pb2.Schema())

    domain1 = schema_util.get_domain(schema, 'feature1')
    self.assertIsInstance(domain1, schema_pb2.StringDomain)
    self.assertEqual(domain1.name, 'domain1')
    # Check to verify that we are operating on the same domain object.
    self.assertIs(domain1, schema_util.get_domain(schema, 'feature1'))

  def test_get_int_domain_feature_level_domain(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          int_domain {
            name: "domain1"
          }
        }
        """, schema_pb2.Schema())

    domain1 = schema_util.get_domain(schema, 'feature1')
    self.assertIsInstance(domain1, schema_pb2.IntDomain)
    self.assertEqual(domain1.name, 'domain1')
    # Check to verify that we are operating on the same domain object.
    self.assertIs(domain1, schema_util.get_domain(schema, 'feature1'))

  def test_get_float_domain_feature_level_domain(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          float_domain {
            name: "domain1"
          }
        }
        """, schema_pb2.Schema())

    domain1 = schema_util.get_domain(schema, 'feature1')
    self.assertIsInstance(domain1, schema_pb2.FloatDomain)
    self.assertEqual(domain1.name, 'domain1')
    # Check to verify that we are operating on the same domain object.
    self.assertIs(domain1, schema_util.get_domain(schema, 'feature1'))

  def test_get_bool_domain_feature_level_domain(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          bool_domain {
            name: "domain1"
          }
        }
        """, schema_pb2.Schema())

    domain1 = schema_util.get_domain(schema, 'feature1')
    self.assertIsInstance(domain1, schema_pb2.BoolDomain)
    self.assertEqual(domain1.name, 'domain1')
    # Check to verify that we are operating on the same domain object.
    self.assertIs(domain1, schema_util.get_domain(schema, 'feature1'))

  def test_get_domain_using_path(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          type: STRUCT
          struct_domain {
            feature {
              name: "sub_feature1"
              bool_domain {
                name: "domain1"
              }
            }
          }
        }
        """, schema_pb2.Schema())
    domain1 = schema_util.get_domain(
        schema, types.FeaturePath(['feature1', 'sub_feature1']))
    self.assertIs(
        domain1, schema.feature[0].struct_domain.feature[0].bool_domain)

  def test_get_domain_not_present(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "domain1"
        }
        feature {
          name: "feature1"
        }
        """, schema_pb2.Schema())

    with self.assertRaisesRegex(ValueError, 'has no domain associated'):
      _ = schema_util.get_domain(schema, 'feature1')

  def test_get_domain_invalid_schema_input(self):
    with self.assertRaisesRegex(TypeError, 'should be a Schema proto'):
      _ = schema_util.get_domain({}, 'feature')

  def test_write_load_schema_text(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
        }
        feature {
          name: "feature2"
        }
        """, schema_pb2.Schema())

    schema_path = os.path.join(FLAGS.test_tmpdir, 'schema.pbtxt')
    schema_util.write_schema_text(schema=schema, output_path=schema_path)
    loaded_schema = schema_util.load_schema_text(input_path=schema_path)
    self.assertEqual(schema, loaded_schema)

  def test_write_schema_text_invalid_schema_input(self):
    with self.assertRaisesRegex(TypeError, 'should be a Schema proto'):
      _ = schema_util.write_schema_text({}, 'schema.pbtxt')

  def test_get_bytes_features(self):
    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: BYTES
          image_domain { }
        }
        feature {
          name: "fb"
          type: BYTES
        }
        feature {
          name: "fc"
          type: INT
          int_domain { }
        }
        feature {
          name: "fd"
          type: FLOAT
        }
        feature {
          name: "fe"
          type: INT
          bool_domain {
            name: "fc_bool_domain"
          }
        }
        feature {
          name: "ff"
          type: STRUCT
          struct_domain {
            feature {
              name: "ff_fa"
              type: BYTES
              image_domain { }
            }
            feature {
              name: "ff_fb"
            }
          }
        }
        """, schema_pb2.Schema())
    self.assertEqual(
        schema_util.get_bytes_features(schema), [
            types.FeaturePath(['fa']),
            types.FeaturePath(['ff', 'ff_fa'])
        ])

  def test_get_bytes_features_categorical_value(self):
    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: BYTES
          string_domain {
            is_categorical: CATEGORICAL_UNSPECIFIED
          }
        }
        feature {
          name: "fb"
          type: BYTES
          string_domain {
            is_categorical: CATEGORICAL_YES
          }
        }
        feature {
          name: "fc"
          type: INT
          bool_domain {
            name: "fc_bool_domain"
          }
        }
        feature {
          name: "fd"
          type: BYTES
          string_domain {
            is_categorical: CATEGORICAL_NO
          }
        }
        feature {
          name: "fe"
          type: BYTES
        }
        feature {
          name: "ff"
          type: FLOAT
        }
        feature {
          name: "fg"
          type: BYTES
          string_domain {
          }
        }
        feature {
            name: "fh"
            type: BYTES
            domain: "fh"
        }
        feature {
            name: "fi"
            type: BYTES
            domain: "fi"
        }
        feature {
            name: "fj"
            type: BYTES
            domain: "fi"
        }
        string_domain{
            name: "fh"
            value: "a"
        }
        string_domain{
            name: "fi"
            value: "b"
            is_categorical: CATEGORICAL_YES
        }
        string_domain{
            name: "fj"
            value: "b"
            is_categorical: CATEGORICAL_YES
        }
        """, schema_pb2.Schema())
    expect_result = {
        types.FeaturePath(['fa']):
            schema_pb2.StringDomain.CATEGORICAL_UNSPECIFIED,
        types.FeaturePath(['fb']):
            schema_pb2.StringDomain.CATEGORICAL_YES,
        types.FeaturePath(['fd']):
            schema_pb2.StringDomain.CATEGORICAL_NO,
        types.FeaturePath(['fe']):
            schema_pb2.StringDomain.CATEGORICAL_UNSPECIFIED,
        types.FeaturePath(['fg']):
            schema_pb2.StringDomain.CATEGORICAL_UNSPECIFIED,
        types.FeaturePath(['fh']):
            schema_pb2.StringDomain.CATEGORICAL_UNSPECIFIED,
        types.FeaturePath(['fi']):
            schema_pb2.StringDomain.CATEGORICAL_YES,
        types.FeaturePath(['fj']):
            schema_pb2.StringDomain.CATEGORICAL_YES,
    }
    result = schema_util.get_bytes_features_categorical_value(schema)
    self.assertEqual(result, expect_result)

  def test_get_categorical_numeric_feature_types(self):
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
          type: INT
          bool_domain {
            name: "fc_bool_domain"
          }
        }
        feature {
          name: "fd"
          type: STRUCT
          struct_domain {
            feature {
              name: "fd_fa"
              type: INT
              int_domain {
                is_categorical: true
              }
            }
            feature {
              name: "fd_fb"
            }
          }
        }
        feature {
          name: "fe"
          type: FLOAT
        }
        feature {
          name: "fg"
          type: FLOAT
          float_domain {
             is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    self.assertEqual(
        schema_util.get_categorical_numeric_feature_types(schema), {
            types.FeaturePath(['fa']): schema_pb2.INT,
            types.FeaturePath(['fc']): schema_pb2.INT,
            types.FeaturePath(['fd', 'fd_fa']): schema_pb2.INT,
            types.FeaturePath(['fg']): schema_pb2.FLOAT,
        })

  def test_is_categorical_features(self):
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
        feature {
          name: "fa"
          type: INT
        }
        """, schema_pb2.Schema())
    expected = [True, True, False, False]
    self.assertEqual([
        schema_util.is_categorical_feature(feature)
        for feature in schema.feature
    ], expected)

  @parameterized.named_parameters(*SET_DOMAIN_VALID_TESTS)
  def test_set_domain(self, input_schema_proto_text, feature_name_or_path,
                      domain, output_schema_proto_text):
    actual_schema = schema_pb2.Schema()
    text_format.Merge(input_schema_proto_text, actual_schema)
    schema_util.set_domain(actual_schema, feature_name_or_path, domain)
    expected_schema = schema_pb2.Schema()
    text_format.Merge(output_schema_proto_text, expected_schema)
    self.assertEqual(actual_schema, expected_schema)

  def test_set_domain_invalid_schema(self):
    with self.assertRaisesRegex(TypeError, 'should be a Schema proto'):
      schema_util.set_domain({}, 'feature', schema_pb2.IntDomain())

  def test_set_domain_invalid_domain(self):
    with self.assertRaisesRegex(TypeError, 'domain is of type'):
      schema_util.set_domain(schema_pb2.Schema(), 'feature', {})

  def test_set_domain_invalid_global_domain(self):
    schema = schema_pb2.Schema()
    schema.feature.add(name='feature')
    schema.string_domain.add(name='domain1', value=['a', 'b'])
    with self.assertRaisesRegex(ValueError, 'Invalid global string domain'):
      schema_util.set_domain(schema, 'feature', 'domain2')

  def test_get_categorical_features(self):
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
        feature {
          name: "fd"
          type: INT
        }
        feature {
          name: "fd"
          type: STRUCT
          struct_domain {
            feature {
              name: "fd_fa"
              type: INT
              int_domain {
                is_categorical: true
              }
            }
            feature {
              name: "fd_fb"
            }
          }
        }
        feature {
          name: "fe"
          type: FLOAT
          float_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    expected = set([
        types.FeaturePath(['fa']),
        types.FeaturePath(['fb']),
        types.FeaturePath(['fd', 'fd_fa']),
        types.FeaturePath(['fe']),
    ])
    self.assertEqual(schema_util.get_categorical_features(schema), expected)

  def test_get_multivalent_features(self):
    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            shape {
              dim {
                size: 1
              }
            }
          }
          feature {
            name: "fb"
            type: BYTES
            value_count {
              min: 0
              max: 1
            }
          }
          feature {
            name: "fc"
            value_count {
              min: 1
              max: 18
            }
          }
          feature {
            name: "fd"
            value_count {
              min: 1
              max: 1
            }
          }
          feature {
            name: "fe"
            shape {
              dim {
                size: 2
              }
            }
          }
          feature {
            name: "ff"
            shape {
              dim {
                size: 1
              }
              dim {
                size: 1
              }
            }
          }
          feature {
            name: "fg"
            value_count {
              min: 2
            }
          }
          feature {
            name: "fh"
            value_count {
              min: 0
              max: 2
            }
          }
          feature {
            name: "fi"
            type: STRUCT
            struct_domain {
              feature {
                name: "fi_fa"
                value_count {
                  min: 0
                  max: 1
                }
              }
              feature {
                name: "fi_fb"
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          """, schema_pb2.Schema())
    expected = set([types.FeaturePath(['fc']),
                    types.FeaturePath(['fe']),
                    types.FeaturePath(['ff']),
                    types.FeaturePath(['fg']),
                    types.FeaturePath(['fh']),
                    types.FeaturePath(['fi', 'fi_fb'])])
    self.assertEqual(schema_util.get_multivalent_features(schema), expected)

  def test_look_up_feature(self):
    feature_1 = text_format.Parse("""name: "feature1" """, schema_pb2.Feature())
    feature_2 = text_format.Parse("""name: "feature2" """, schema_pb2.Feature())

    container = [feature_1, feature_2]
    self.assertEqual(
        schema_util.look_up_feature('feature1', container), feature_1)
    self.assertEqual(
        schema_util.look_up_feature('feature2', container), feature_2)
    self.assertIsNone(schema_util.look_up_feature('feature3', container), None)

  def test_generate_dummy_schema_with_paths(self):
    schema = text_format.Parse(
        """
    feature {
      name: "foo"
    }
    feature {
      name: "bar"
    }
    feature {
      name: "baz"
      struct_domain: {
        feature {
          name: "zip"
        }
        feature {
          name: "zop"
        }
      }
    }
    """, schema_pb2.Schema())
    self.assertEqual(
        schema_util.generate_dummy_schema_with_paths([
            types.FeaturePath(['foo']),
            types.FeaturePath(['bar']),
            types.FeaturePath(['baz', 'zip']),
            types.FeaturePath(['baz', 'zop'])
        ]), schema)


if __name__ == '__main__':
  absltest.main()
