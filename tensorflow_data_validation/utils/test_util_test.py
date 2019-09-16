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
"""Tests for test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class TestAssertFeatureProtoEqual(absltest.TestCase):
  """Tests assert_feature_proto_equal."""

  class SampleTestUsingAssertFeatureProtoEqual(
      absltest.TestCase):
    """A mock test case.

    Calls assert_feature_proto_equal.
    """

    # This is a work around for unittest in Python 2. It requires the runTest
    # method to be implemented if the test is being called directly instead of
    # through unittest.main()/absltest.main().
    def runTest(self):
      pass

    def assert_on_equal_feature_protos(self):
      expected = text_format.Parse(
          """
              name: 'a'
              type: BYTES
              custom_stats {
                name: 'A'
                num: 2.5
              }
              custom_stats {
                name: 'B'
                num: 3.0
              }
             """, statistics_pb2.FeatureNameStatistics())
      actual = text_format.Parse(
          """
              name: 'a'
              type: BYTES
              custom_stats {
                name: 'B'
                num: 3.0
              }
              custom_stats {
                name: 'A'
                num: 2.5
              }
             """, statistics_pb2.FeatureNameStatistics())
      test_util.assert_feature_proto_equal(
          self, actual, expected)

    def assert_on_unequal_feature_protos(self):
      expected = text_format.Parse(
          """
              name: 'a'
              custom_stats {
                name: 'MI'
                num: 2.5
              }
             """, statistics_pb2.FeatureNameStatistics())
      actual = text_format.Parse(
          """
              name: 'a'
              custom_stats {
                name: 'MI'
                num: 2.0
              }
             """, statistics_pb2.FeatureNameStatistics())
      test_util.assert_feature_proto_equal(
          self, actual, expected)

  def setUp(self):
    super(TestAssertFeatureProtoEqual, self).setUp()
    self._test = self.SampleTestUsingAssertFeatureProtoEqual()

  def test_feature_protos_equal(self):
    self.assertIsNone(self._test.assert_on_equal_feature_protos())

  def test_feature_protos_unequal(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_unequal_feature_protos()


class TestAssertDatasetFeatureStatsProtoEqual(absltest.TestCase):
  """Tests assert_dataset_feature_stats_proto_equal."""

  class SampleTestUsingAssertDatasetFeatureStatsProtoEqual(absltest.TestCase):
    """A mock test case.

    Calls assert_dataset_feature_stats_proto_equal.
    """

    # This is a work around for unittest in Python 2. It requires the runTest
    # method to be implemented if the test is being called directly instead of
    # through unittest.main()/absltest.main().
    def runTest(self):
      pass

    def assert_on_two_protos_with_same_features_in_same_order(self):
      expected = text_format.Parse(
          """
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        path {
          step: 'fb'
        }
        type: STRING
        string_stats {
          unique: 5
        }
      }
      """, statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        path {
          step: 'fb'
        }
        type: STRING
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

    def assert_on_two_protos_with_same_features_in_different_order(self):
      expected = text_format.Parse(
          """
      features {
        path {
          step: 'fb'
        }
        type: STRING
        string_stats {
          unique: 5
        }
      }
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        path {
          step: 'fb'
        }
        type: STRING
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

    def assert_on_two_protos_with_different_features(self):
      expected = text_format.Parse(
          """
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        path {
          step: 'fb'
        }
        type: STRING
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

    def assert_on_two_protos_with_different_numbers_of_features(self):
      expected = text_format.Parse(
          """
        features {
          path {
            step: 'fa'
          }
          type: STRING
          string_stats {
            unique: 4
          }
        }
        features {
          path {
            step: 'fb'
          }
          type: STRING
          string_stats {
            unique: 5
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
        features {
          path {
            step: 'fa'
          }
          type: STRING
          string_stats {
            unique: 4
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
      test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

    def assert_on_two_protos_with_different_num_examples(self):
      expected = text_format.Parse(
          """
      num_examples: 1
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }
      """, statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      num_examples: 2
      features {
        path {
          step: 'fa'
        }
        type: STRING
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

  def setUp(self):
    super(TestAssertDatasetFeatureStatsProtoEqual, self).setUp()
    self._test = self.SampleTestUsingAssertDatasetFeatureStatsProtoEqual()

  def test_two_protos_with_same_features_in_same_order(self):
    self.assertIsNone(
        self._test.assert_on_two_protos_with_same_features_in_same_order())

  def test_two_protos_with_same_features_in_different_order(self):
    self.assertIsNone(
        self._test.assert_on_two_protos_with_same_features_in_different_order())

  def test_two_protos_with_different_features(self):
    with self.assertRaisesRegexp(AssertionError, 'Feature path .*'):
      self._test.assert_on_two_protos_with_different_features()

  def test_two_protos_with_different_numbers_of_features(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_with_different_numbers_of_features()

  def test_two_protos_with_different_num_examples(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_with_different_num_examples()


if __name__ == '__main__':
  absltest.main()
