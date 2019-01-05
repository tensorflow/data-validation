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
"""Utilities for writing statistics generator tests."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.types_compat import Callable, Dict, List

from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2


def make_example_dict_equal_fn(
    test,
    expected):
  """Makes a matcher function for comparing the example dict.

  Args:
    test: test case object.
    expected: the expected example dict.

  Returns:
    A matcher function for comparing the example dicts.
  """
  def _matcher(actual):
    """Matcher function for comparing the example dicts."""
    try:
      # Check number of examples.
      test.assertEqual(len(actual), len(expected))

      for i in range(len(actual)):
        for key in actual[i]:
          # Check each feature value.
          if isinstance(expected[i][key], np.ndarray):
            test.assertEqual(actual[i][key].dtype, expected[i][key].dtype)
            np.testing.assert_equal(actual[i][key], expected[i][key])
          else:
            test.assertEqual(actual[i][key], expected[i][key])

    except AssertionError as e:
      raise util.BeamAssertException('Failed assert: ' + str(e))

  return _matcher


def make_dataset_feature_stats_list_proto_equal_fn(
    test,
    expected_result
):
  """Makes a matcher function for comparing DatasetFeatureStatisticsList proto.

  Args:
    test: test case object
    expected_result: the expected DatasetFeatureStatisticsList proto.

  Returns:
    A matcher function for comparing DatasetFeatureStatisticsList proto.
  """

  def _matcher(actual):
    """Matcher function for comparing DatasetFeatureStatisticsList proto."""
    try:
      test.assertEqual(len(actual), 1)
      # Get the dataset stats from DatasetFeatureStatisticsList proto.
      actual_stats = actual[0].datasets[0]
      expected_stats = expected_result.datasets[0]

      assert_dataset_feature_stats_proto_equal(test, actual_stats,
                                               expected_stats)
    except AssertionError as e:
      raise util.BeamAssertException('Failed assert: ' + str(e))

  return _matcher


def assert_feature_proto_equal(
    test, actual,
    expected):
  """Ensures feature protos are equal.

  Args:
    test: The test case.
    actual: The actual feature proto.
    expected: The expected feature proto.
  """

  test.assertEqual(len(actual.custom_stats), len(expected.custom_stats))
  expected_custom_stats = {}
  for expected_custom_stat in expected.custom_stats:
    expected_custom_stats[expected_custom_stat.name] = expected_custom_stat

  for actual_custom_stat in actual.custom_stats:
    test.assertTrue(actual_custom_stat.name in expected_custom_stats)
    expected_custom_stat = expected_custom_stats[actual_custom_stat.name]
    compare.assertProtoEqual(
        test, actual_custom_stat, expected_custom_stat, normalize_numbers=True)
  del actual.custom_stats[:]
  del expected.custom_stats[:]

  # Compare the rest of the proto without numeric custom stats
  compare.assertProtoEqual(test, actual, expected, normalize_numbers=True)


def assert_dataset_feature_stats_proto_equal(
    test, actual,
    expected):
  """Compares DatasetFeatureStatistics protos.

  This function can be used to test whether two DatasetFeatureStatistics protos
  contain the same information, even if the order of the features differs.

  Args:
    test: The test case.
    actual: The actual DatasetFeatureStatistics proto.
    expected: The expected DatasetFeatureStatistics proto.
  """
  test.assertEqual(actual.num_examples, expected.num_examples)
  test.assertEqual(len(actual.features), len(expected.features))

  expected_features = {}
  for feature in expected.features:
    expected_features[feature.name] = feature

  for feature in actual.features:
    if feature.name not in expected_features:
      raise AssertionError
    assert_feature_proto_equal(test, feature, expected_features[feature.name])


class CombinerStatsGeneratorTest(absltest.TestCase):
  """Test class with extra combiner stats generator related functionality."""

  # Runs the provided combiner statistics generator and tests if the output
  # matches the expected result.
  def assertCombinerOutputEqual(
      self, batches,
      generator,
      expected_result):
    """Tests a combiner statistics generator.

    This runs the generator twice to cover different behavior. There must be at
    least two input batches in order to test the generator's merging behavior.

    Args:
      batches: A list of batches of test data.
      generator: The CombinerStatsGenerator to test.
      expected_result: Dict mapping feature name to FeatureNameStatistics proto
        that it is expected the generator will return for the feature.
    """
    # Run generator to check that merge_accumulators() works correctly.
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    self.assertEqual(len(result.features), len(expected_result))  # pylint: disable=g-generic-assert
    for actual_feature_stats in result.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected_result[actual_feature_stats.name],
          normalize_numbers=True)

    # Run generator to check that add_input() works correctly when adding
    # inputs to a non-empty accumulator.
    accumulator = generator.create_accumulator()

    for batch in batches:
      accumulator = generator.add_input(accumulator, batch)

    result = generator.extract_output(accumulator)
    self.assertEqual(len(result.features), len(expected_result))  # pylint: disable=g-generic-assert
    for actual_feature_stats in result.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected_result[actual_feature_stats.name],
          normalize_numbers=True)


class TransformStatsGeneratorTest(absltest.TestCase):
  """Test class with extra transform stats generator related functionality."""

  # Runs the provided transform statistics generator and tests if the output
  # matches the expected result.
  def assertTransformOutputEqual(
      self, examples,
      generator,
      expected_results):
    """Tests a transform statistics generator."""

    def _make_result_matcher(
        test,
        expected_results):
      """Makes matcher for a list of DatasetFeatureStatistics protos."""

      def _equal(actual_results):
        """Matcher for comparing a list of DatasetFeatureStatistics protos."""
        test.assertLen(expected_results, len(actual_results))
        # Sort both list of protos based on their string presentation to make
        # sure the sort is stable.
        sorted_expected_results = sorted(expected_results, key=str)
        sorted_actual_results = sorted(actual_results, key=str)
        for index, actual in enumerate(sorted_actual_results):
          compare.assertProtoEqual(
              test,
              actual,
              sorted_expected_results[index],
              normalize_numbers=True)

      return _equal

    with beam.Pipeline() as p:
      result = p | beam.Create(examples) | generator.ptransform
      util.assert_that(result, _make_result_matcher(self, expected_results))
