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
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.types_compat import Callable, Dict, List

from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2


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

      test.assertEqual(actual_stats.num_examples, expected_stats.num_examples)
      test.assertEqual(len(actual_stats.features), len(expected_stats.features))

      expected_features = {}
      for feature in expected_stats.features:
        expected_features[feature.name] = feature

      for feature in actual_stats.features:
        compare.assertProtoEqual(
            test,
            feature,
            expected_features[feature.name],
            normalize_numbers=True)
    except AssertionError, e:
      raise util.BeamAssertException('Failed assert: ' + str(e))

  return _matcher


class CombinerStatsGeneratorTest(absltest.TestCase):
  """Test class with extra combiner stats generator related functionality."""

  # Runs the provided combiner statistics generator and tests if the output
  # matches the expected result.
  def assertCombinerOutputEqual(
      self, batches,
      generator, expected_result):
    """Tests a combiner statistics generator."""
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    self.assertEqual(len(result.features), len(expected_result))
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
      self, batches,
      generator,
      expected_results):
    """Tests a transform statistics generator."""

    def _make_result_matcher(
        test,
        expected_results):
      """Makes matcher for a list of DatasetFeatureStatistics protos."""

      def _equal(actual_results):
        """Matcher for comparing a list of DatasetFeatureStatistics protos."""
        test.assertEquals(len(expected_results), len(actual_results))
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
      result = p | beam.Create(batches) | generator.ptransform
      util.assert_that(result, _make_result_matcher(self, expected_results))
