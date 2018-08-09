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

from absl.testing import absltest
from tensorflow_data_validation.utils import schema_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class SchemaUtilTest(absltest.TestCase):

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

  def test_get_feature_not_present(self):
    schema = text_format.Parse(
        """
        feature {
          name: "feature1"
        }
        """, schema_pb2.Schema())

    with self.assertRaisesRegexp(ValueError,
                                 'Feature.*not found in the schema.*'):
      _ = schema_util.get_feature(schema, 'feature2')

  def test_get_feature_invalid_schema_input(self):
    with self.assertRaisesRegexp(TypeError, '.*should be a Schema proto.*'):
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

    with self.assertRaisesRegexp(ValueError,
                                 '.*has no domain associated.*'):
      _ = schema_util.get_domain(schema, 'feature1')

  def test_get_domain_invalid_schema_input(self):
    with self.assertRaisesRegexp(TypeError, '.*should be a Schema proto.*'):
      _ = schema_util.get_domain({}, 'feature')


if __name__ == '__main__':
  absltest.main()
