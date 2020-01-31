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
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from typing import Callable, Dict, List, Optional, Tuple, Union

from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2


def make_example_dict_equal_fn(
    test: absltest.TestCase,
    expected: List[types.Example]) -> Callable[[List[types.Example]], None]:
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
    test: absltest.TestCase,
    expected_result: statistics_pb2.DatasetFeatureStatisticsList
) -> Callable[[List[statistics_pb2.DatasetFeatureStatisticsList]], None]:
  """Makes a matcher function for comparing DatasetFeatureStatisticsList proto.

  Args:
    test: test case object
    expected_result: the expected DatasetFeatureStatisticsList proto.

  Returns:
    A matcher function for comparing DatasetFeatureStatisticsList proto.
  """

  def _matcher(actual: List[statistics_pb2.DatasetFeatureStatisticsList]):
    """Matcher function for comparing DatasetFeatureStatisticsList proto."""
    try:
      test.assertEqual(len(actual), 1)
      test.assertEqual(len(actual[0].datasets), len(expected_result.datasets))

      sorted_actual_datasets = sorted(actual[0].datasets, key=lambda d: d.name)
      sorted_expected_datasets = sorted(expected_result.datasets,
                                        key=lambda d: d.name)

      for i in range(len(sorted_actual_datasets)):
        assert_dataset_feature_stats_proto_equal(test,
                                                 sorted_actual_datasets[i],
                                                 sorted_expected_datasets[i])
    except AssertionError as e:
      raise util.BeamAssertException('Failed assert: ' + str(e))

  return _matcher


def assert_feature_proto_equal(
    test: absltest.TestCase, actual: statistics_pb2.FeatureNameStatistics,
    expected: statistics_pb2.FeatureNameStatistics) -> None:
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
    test: absltest.TestCase, actual: statistics_pb2.DatasetFeatureStatistics,
    expected: statistics_pb2.DatasetFeatureStatistics) -> None:
  """Compares DatasetFeatureStatistics protos.

  This function can be used to test whether two DatasetFeatureStatistics protos
  contain the same information, even if the order of the features differs.

  Args:
    test: The test case.
    actual: The actual DatasetFeatureStatistics proto.
    expected: The expected DatasetFeatureStatistics proto.
  """
  test.assertEqual(actual.name, expected.name)
  test.assertEqual(actual.num_examples, expected.num_examples)
  test.assertEqual(len(actual.features), len(expected.features))

  expected_features = {}
  for feature in expected.features:
    expected_features[types.FeaturePath.from_proto(feature.path)] = feature

  for feature in actual.features:
    feature_path = types.FeaturePath.from_proto(feature.path)
    if feature_path not in expected_features:
      raise AssertionError(
          'Feature path %s found in actual but not found in expected.' %
          feature_path)
    assert_feature_proto_equal(test, feature, expected_features[feature_path])


class CombinerStatsGeneratorTest(absltest.TestCase):
  """Test class with extra combiner stats generator related functionality."""

  # Runs the provided combiner statistics generator and tests if the output
  # matches the expected result.
  def assertCombinerOutputEqual(
      self, batches: List[types.ExampleBatch],
      generator: stats_generator.CombinerStatsGenerator,
      expected_feature_stats: Dict[types.FeaturePath,
                                   statistics_pb2.FeatureNameStatistics],
      expected_cross_feature_stats: Optional[Dict[
          types.FeatureCross, statistics_pb2.CrossFeatureStatistics]] = None
      ) -> None:
    """Tests a combiner statistics generator.

    This runs the generator twice to cover different behavior. There must be at
    least two input batches in order to test the generator's merging behavior.

    Args:
      batches: A list of batches of test data.
      generator: The CombinerStatsGenerator to test.
      expected_feature_stats: Dict mapping feature name to FeatureNameStatistics
        proto that it is expected the generator will return for the feature.
      expected_cross_feature_stats: Dict mapping feature cross to
        CrossFeatureStatistics proto that it is expected the generator will
        return for the feature cross.
    """
    if expected_cross_feature_stats is None:
      expected_cross_feature_stats = {}
    # Run generator to check that merge_accumulators() works correctly.
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    self.assertEqual(  # pylint: disable=g-generic-assert
        len(result.features), len(expected_feature_stats),
        '{}, {}'.format(result, expected_feature_stats))
    for actual_feature_stats in result.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected_feature_stats[types.FeaturePath.from_proto(
              actual_feature_stats.path)],
          normalize_numbers=True)

    self.assertEqual(  # pylint: disable=g-generic-assert
        len(result.cross_features), len(expected_cross_feature_stats),
        '{}, {}'.format(result, expected_cross_feature_stats))
    for actual_cross_feature_stats in result.cross_features:
      cross = (actual_cross_feature_stats.path_x.step[0],
               actual_cross_feature_stats.path_y.step[0])
      compare.assertProtoEqual(
          self,
          actual_cross_feature_stats,
          expected_cross_feature_stats[cross],
          normalize_numbers=True)

    # Run generator to check that add_input() works correctly when adding
    # inputs to a non-empty accumulator.
    accumulator = generator.create_accumulator()

    for batch in batches:
      accumulator = generator.add_input(accumulator, batch)

    result = generator.extract_output(accumulator)
    self.assertEqual(len(result.features), len(expected_feature_stats))  # pylint: disable=g-generic-assert
    for actual_feature_stats in result.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected_feature_stats[types.FeaturePath.from_proto(
              actual_feature_stats.path)],
          normalize_numbers=True)

    self.assertEqual(len(result.cross_features),
                     len(expected_cross_feature_stats))  # pylint: disable=g-generic-assert
    for actual_cross_feature_stats in result.cross_features:
      cross = (actual_cross_feature_stats.path_x.step[0],
               actual_cross_feature_stats.path_y.step[0])
      compare.assertProtoEqual(
          self,
          actual_cross_feature_stats,
          expected_cross_feature_stats[cross],
          normalize_numbers=True)


class _DatasetFeatureStatisticsComparatorWrapper(object):
  """Wraps a DatasetFeatureStatistics and provides a custom comparator.

  This is to facilitate assertCountEqual().
  """

  # Disable the built-in __hash__ (in python2). This forces __eq__ to be
  # used in assertCountEqual().
  __hash__ = None

  def __init__(self, wrapped: statistics_pb2.DatasetFeatureStatistics):
    self._wrapped = wrapped
    self._normalized = statistics_pb2.DatasetFeatureStatistics()
    self._normalized.MergeFrom(wrapped)
    compare.NormalizeNumberFields(self._normalized)

  def __eq__(self, other: '_DatasetFeatureStatisticsComparatorWrapper'):
    return compare.ProtoEq(self._normalized, other._normalized)  # pylint: disable=protected-access

  def __repr__(self):
    return self._normalized.__repr__()


class TransformStatsGeneratorTest(absltest.TestCase):
  """Test class with extra transform stats generator related functionality."""

  def setUp(self):
    super(TransformStatsGeneratorTest, self).setUp()
    self.maxDiff = None  # pylint: disable=invalid-name

  # Runs the provided slicing aware transform statistics generator and tests
  # if the output matches the expected result.
  def assertSlicingAwareTransformOutputEqual(
      self, examples: List[Union[types.SlicedExample, types.Example]],
      generator: stats_generator.TransformStatsGenerator,
      expected_results: List[Union[
          statistics_pb2.DatasetFeatureStatistics,
          Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]],
      add_default_slice_key_to_input: bool = False,
      add_default_slice_key_to_output: bool = False,
  ) -> None:
    """Tests a slicing aware transform statistics generator.

    Args:
      examples: Input sliced examples.
      generator: A TransformStatsGenerator.
      expected_results: Expected statistics proto results.
      add_default_slice_key_to_input: If True, adds the default slice key to
        the input examples.
      add_default_slice_key_to_output: If True, adds the default slice key to
        the result protos.
    """

    def _make_result_matcher(
        test: absltest.TestCase,
        expected_results: List[
            Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]):
      """Makes matcher for a list of DatasetFeatureStatistics protos."""

      def _equal(actual_results: List[
          Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]):
        """Matcher for comparing a list of DatasetFeatureStatistics protos."""
        if len(actual_results) == 1 and len(expected_results) == 1:
          # If appropriate use proto matcher for better errors
          test.assertEqual(expected_results[0][0], actual_results[0][0])
          compare.assertProtoEqual(test, expected_results[0][1],
                                   actual_results[0][1], normalize_numbers=True)
        else:
          test.assertCountEqual(
              [(k, _DatasetFeatureStatisticsComparatorWrapper(v))
               for k, v in expected_results],
              [(k, _DatasetFeatureStatisticsComparatorWrapper(v))
               for k, v in actual_results])

      return _equal

    if add_default_slice_key_to_input:
      examples = [(None, e) for e in examples]
    if add_default_slice_key_to_output:
      expected_results = [(None, p)
                          for p in expected_results]

    with beam.Pipeline() as p:
      result = p | beam.Create(examples) | generator.ptransform
      util.assert_that(result, _make_result_matcher(self, expected_results))


class CombinerFeatureStatsGeneratorTest(absltest.TestCase):
  """Test class for combiner feature stats generator related functionality."""

  # Runs the provided combiner feature statistics generator and tests if the
  # output matches the expected result.
  def assertCombinerOutputEqual(
      self, input_batches: List[types.ValueBatch],
      generator: stats_generator.CombinerFeatureStatsGenerator,
      expected_result: statistics_pb2.FeatureNameStatistics) -> None:
    """Tests a feature combiner statistics generator.

    This runs the generator twice to cover different behavior. There must be at
    least two input batches in order to test the generator's merging behavior.

    Args:
      input_batches: A list of batches of test data.
      generator: The CombinerFeatureStatsGenerator to test.
      expected_result: The FeatureNameStatistics proto that it is expected the
        generator will return.
    """
    # Run generator to check that merge_accumulators() works correctly.
    accumulators = [
        generator.add_input(generator.create_accumulator(),
                            types.FeaturePath(['']), input_batch)
        for input_batch in input_batches
    ]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    compare.assertProtoEqual(
        self, result, expected_result, normalize_numbers=True)

    # Run generator to check that add_input() works correctly when adding
    # inputs to a non-empty accumulator.
    accumulator = generator.create_accumulator()

    for input_batch in input_batches:
      accumulator = generator.add_input(
          accumulator, types.FeaturePath(['']), input_batch)

    result = generator.extract_output(accumulator)
    compare.assertProtoEqual(
        self, result, expected_result, normalize_numbers=True)


def make_arrow_tables_equal_fn(test: absltest.TestCase,
                               expected_tables: List[pa.Table]):
  """Makes a matcher function for comparing arrow tables."""
  def _matcher(actual_tables):
    """Arrow tables matcher fn."""
    test.assertLen(actual_tables, len(expected_tables))
    for i in range(len(expected_tables)):
      test.assertEqual(actual_tables[i].num_columns,
                       expected_tables[i].num_columns)
      for column_name, expected_column in zip(
          expected_tables[i].schema.names, expected_tables[i].columns):
        actual_column = actual_tables[i].column(column_name)
        test.assertEqual(len(actual_column.data.chunks),
                         len(expected_column.data.chunks))
        for j in range(len(expected_column.data.chunks)):
          actual_chunk = actual_column.data.chunk(j)
          expected_chunk = expected_column.data.chunk(j)
          test.assertTrue(
              actual_chunk.equals(expected_chunk),
              '{} vs {}'.format(actual_chunk, expected_chunk))

  return _matcher
