"""Tests for test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class TestAssertFeatureProtoWithinErrorOnCustomStats(absltest.TestCase):
  """Tests assert_feature_proto_equal_with_error_on_custom_stats."""

  class SampleTestUsingAssertFeatureProtoWithinErrorOnCustomStats(
      absltest.TestCase):
    """A mock test case.

    Calls assert_feature_proto_equal_with_error_on_custom_stats.
    """

    # This is a work around for unittest in Python 2. It requires the runTest
    # method to be implemented if the test is being called directly instead of
    # through unittest.main()/absltest.main().
    def runTest(self):
      pass

    def assert_on_two_protos_within_valid_error(self):
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
                num: 2.45
              }
             """, statistics_pb2.FeatureNameStatistics())
      test_util.assert_feature_proto_equal_with_error_on_custom_stats(
          self, actual, expected)

    def assert_on_two_protos_not_within_valid_error(self):
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
      test_util.assert_feature_proto_equal_with_error_on_custom_stats(
          self, actual, expected)

    def assert_on_two_protos_within_valid_error_but_different_name(self):
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
              name: 'b'
              custom_stats {
                name: 'MI'
                num: 2.45
              }
                 """, statistics_pb2.FeatureNameStatistics())
      test_util.assert_feature_proto_equal_with_error_on_custom_stats(
          self, actual, expected)

  def setUp(self):
    super(TestAssertFeatureProtoWithinErrorOnCustomStats, self).setUp()
    self._test = self.SampleTestUsingAssertFeatureProtoWithinErrorOnCustomStats(
    )

  def test_proto_within_valid_error(self):
    self.assertIsNone(self._test.assert_on_two_protos_within_valid_error())

  def test_proto_not_within_valid_error(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_not_within_valid_error()

  def test_proto_within_valid_error_but_different_name(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_within_valid_error_but_different_name()


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
        name: 'fa'
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        name: 'fb'
        type: STRING
        string_stats {
          unique: 5
        }
      }
      """, statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        name: 'fa'
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        name: 'fb'
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
        name: 'fb'
        type: STRING
        string_stats {
          unique: 5
        }
      }
      features {
        name: 'fa'
        type: STRING
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        name: 'fa'
        type: STRING
        string_stats {
          unique: 4
        }
      }
      features {
        name: 'fb'
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
        name: 'fa'
        type: STRING
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
      features {
        name: 'fb'
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
          name: 'fa'
          type: STRING
          string_stats {
            unique: 4
          }
        }
        features {
          name: 'fb'
          type: STRING
          string_stats {
            unique: 5
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
      actual = text_format.Parse(
          """
        features {
          name: 'fa'
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
        name: 'fa'
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
        name: 'fa'
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
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_with_different_features()

  def test_two_protos_with_different_numbers_of_features(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_with_different_numbers_of_features()

  def test_two_protos_with_different_num_examples(self):
    with self.assertRaises(AssertionError):
      self._test.assert_on_two_protos_with_different_num_examples()


if __name__ == '__main__':
  absltest.main()
