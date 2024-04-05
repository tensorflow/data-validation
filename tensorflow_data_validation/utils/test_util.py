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
"""Utilities for writing statistics generator and validation tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator

from tensorflow.python.util.protobuf import compare  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import statistics_pb2


# pytype: disable=attribute-error
def _clear(msg, field_name) -> bool:
  """Clear a field if set and return True if it was."""
  try:
    if msg.HasField(field_name):
      msg.ClearField(field_name)
      return True
  except ValueError:
    if msg.__getattribute__(field_name):
      msg.ClearField(field_name)
      return True
  return False
# pytype: enable=attribute-error


def _clear_histograms(
    dataset: statistics_pb2.DatasetFeatureStatistics
) -> Tuple[statistics_pb2.DatasetFeatureStatistics, bool]:
  """Returns input with cleared histograms returns true if any were set."""
  has_hist = False
  result = statistics_pb2.DatasetFeatureStatistics()
  result.MergeFrom(dataset)
  for feature in result.features:
    if feature.HasField('num_stats'):
      has_hist = _clear(feature.num_stats, 'histograms') or has_hist
      has_hist = _clear(feature.num_stats.weighted_numeric_stats,
                        'histograms') or has_hist
      common_stats = feature.num_stats.common_stats
    elif feature.HasField('string_stats'):
      common_stats = feature.string_stats.common_stats
    elif feature.HasField('struct_stats'):
      common_stats = feature.struct_stats.common_stats
    elif feature.HasField('bytes_stats'):
      common_stats = feature.bytes_stats.common_stats
    else:
      common_stats = None
    if common_stats is not None:
      has_hist = _clear(common_stats,
                        'feature_list_length_histogram') or has_hist
      has_hist = _clear(common_stats, 'num_values_histogram') or has_hist
    for custom in feature.custom_stats:
      has_hist = _clear(custom, 'histogram') or has_hist
  return result, has_hist


def make_dataset_feature_stats_list_proto_equal_fn(
    test: absltest.TestCase,
    expected_result: statistics_pb2.DatasetFeatureStatisticsList,
    expected_result_len: int = 1,
    expected_result_merge_fn: Optional[
        Callable[[Iterable[statistics_pb2.DatasetFeatureStatisticsList]],
                 statistics_pb2.DatasetFeatureStatisticsList]] = None,
    check_histograms: bool = True
) -> Callable[[Iterable[statistics_pb2.DatasetFeatureStatisticsList]], None]:
  """Makes a matcher function for comparing DatasetFeatureStatisticsList proto.

  Args:
    test: test case object
    expected_result: the expected DatasetFeatureStatisticsList proto.
    expected_result_len: The expected number of elements. If this is a number
      greater than 1, expected_result_merge_fn should be provided to merge the
      inputs into the form expected by expected_result.
    expected_result_merge_fn: Called on elements to merge multiple inputs into
      the form expected by expected_result.
    check_histograms: If True, asserts equality of histograms.
      Otherwise histograms are not checked, and are assumed to not be specified
      in expected output.

  Returns:
    A matcher function for comparing DatasetFeatureStatisticsList proto.
  """

  def _matcher(actual: Iterable[statistics_pb2.DatasetFeatureStatisticsList]):
    """Matcher function for comparing DatasetFeatureStatisticsList proto."""
    actual = list(actual)
    try:
      test.assertLen(
          actual, expected_result_len,
          'Expected exactly %d DatasetFeatureStatisticsList' %
          expected_result_len)
      if len(actual) == 1:
        actual = actual[0]
      else:
        actual = expected_result_merge_fn(actual)
      test.assertLen(actual.datasets, len(expected_result.datasets))

      sorted_actual_datasets = sorted(actual.datasets, key=lambda d: d.name)
      sorted_expected_datasets = sorted(expected_result.datasets,
                                        key=lambda d: d.name)

      for i in range(len(sorted_actual_datasets)):
        assert_dataset_feature_stats_proto_equal(test,
                                                 sorted_actual_datasets[i],
                                                 sorted_expected_datasets[i],
                                                 check_histograms)
    except AssertionError as e:
      raise util.BeamAssertException from e

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

  test.assertLen(actual.custom_stats, len(expected.custom_stats))
  expected_custom_stats = {}
  for expected_custom_stat in expected.custom_stats:
    expected_custom_stats[expected_custom_stat.name] = expected_custom_stat

  for actual_custom_stat in actual.custom_stats:
    test.assertIn(actual_custom_stat.name, expected_custom_stats)
    expected_custom_stat = expected_custom_stats[actual_custom_stat.name]
    compare.assertProtoEqual(
        test, expected_custom_stat, actual_custom_stat, normalize_numbers=True)
  del actual.custom_stats[:]
  del expected.custom_stats[:]

  # Compare the rest of the proto without numeric custom stats
  compare.assertProtoEqual(test, expected, actual, normalize_numbers=True)


def assert_dataset_feature_stats_proto_equal(
    test: absltest.TestCase,
    actual: statistics_pb2.DatasetFeatureStatistics,
    expected: statistics_pb2.DatasetFeatureStatistics,
    check_histograms: bool = True) -> None:
  """Compares DatasetFeatureStatistics protos.

  This function can be used to test whether two DatasetFeatureStatistics protos
  contain the same information, even if the order of the features differs.

  Args:
    test: The test case.
    actual: The actual DatasetFeatureStatistics proto.
    expected: The expected DatasetFeatureStatistics proto.
    check_histograms: If True, asserts equality of histograms.
      Otherwise histograms are not checked, and are assumed to not be specified
      in expected output.
  """
  if not check_histograms:
    expected, any_hist = _clear_histograms(expected)
    if any_hist:
      raise ValueError(
          'Histograms set in expected result with check_histogram=False.')
    actual, _ = _clear_histograms(actual)
  test.assertEqual(
      expected.name, actual.name, 'Expected name to be {}, found {} in '
      'DatasetFeatureStatistics {}'.format(expected.name, actual.name, actual))
  test.assertEqual(
      expected.num_examples, actual.num_examples,
      'Expected num_examples to be {}, found {} in DatasetFeatureStatistics {}'
      .format(expected.num_examples, actual.num_examples, actual))
  test.assertLen(actual.features, len(expected.features))

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


def make_skew_result_equal_fn(test, expected):
  """Makes a matcher function for comparing FeatureSkew result protos."""

  def _matcher(actual):
    try:
      test.assertLen(actual, len(expected))
      sorted_actual = sorted(actual, key=lambda a: a.feature_name)
      sorted_expected = sorted(expected, key=lambda e: e.feature_name)
      for i in range(len(sorted_actual)):
        test.assertEqual(sorted_actual[i], sorted_expected[i])
    except AssertionError as e:
      raise util.BeamAssertException(traceback.format_exc()) from e

  return _matcher


def make_confusion_count_result_equal_fn(test, expected):
  """Makes a matcher function for comparing ConfusionCount result protos."""

  def _matcher(actual):
    try:
      test.assertLen(actual, len(expected))
      # pylint: disable=g-long-lambda
      sort_key = lambda a: (a.feature_name, a.base.bytes_value, a.test.
                            bytes_value)
      # pylint: enable=g-long-lambda
      sorted_actual = sorted(actual, key=sort_key)
      sorted_expected = sorted(expected, key=sort_key)
      for i in range(len(sorted_actual)):
        test.assertEqual(sorted_actual[i], sorted_expected[i])
    except AssertionError as e:
      raise util.BeamAssertException(traceback.format_exc()) from e

  return _matcher


class CombinerStatsGeneratorTest(absltest.TestCase):
  """Test class with extra combiner stats generator related functionality."""

  # Runs the provided combiner statistics generator and tests if the output
  # matches the expected result.
  def assertCombinerOutputEqual(
      self, batches: List[pa.RecordBatch],
      generator: stats_generator.CombinerStatsGenerator,
      expected_feature_stats: Dict[types.FeaturePath,
                                   statistics_pb2.FeatureNameStatistics],
      expected_cross_feature_stats: Optional[Dict[
          types.FeatureCross, statistics_pb2.CrossFeatureStatistics]] = None,
      only_match_expected_feature_stats: bool = False,
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
      only_match_expected_feature_stats: if True, will only compare features
        that appear in `expected_feature_stats`.
    """
    generator.setup()

    if expected_cross_feature_stats is None:
      expected_cross_feature_stats = {}

    def _verify(output):
      """Verifies that the output meeds the expectations."""
      if only_match_expected_feature_stats:
        features_in_stats = set(
            [types.FeaturePath.from_proto(f.path) for f in output.features])
        self.assertTrue(set(expected_feature_stats.keys())
                        .issubset(features_in_stats))
      else:
        self.assertEqual(  # pylint: disable=g-generic-assert
            len(output.features), len(expected_feature_stats),
            '{}, {}'.format(output, expected_feature_stats))
      for actual_feature_stats in output.features:
        actual_path = types.FeaturePath.from_proto(actual_feature_stats.path)
        expected_stats = expected_feature_stats.get(actual_path)
        if (only_match_expected_feature_stats and expected_stats is None):
          continue
        compare.assertProtoEqual(
            self,
            expected_stats,
            actual_feature_stats,
            normalize_numbers=True)

      self.assertEqual(  # pylint: disable=g-generic-assert
          len(result.cross_features), len(expected_cross_feature_stats),
          '{}, {}'.format(result, expected_cross_feature_stats))
      for actual_cross_feature_stats in result.cross_features:
        cross = (actual_cross_feature_stats.path_x.step[0],
                 actual_cross_feature_stats.path_y.step[0])
        compare.assertProtoEqual(
            self,
            expected_cross_feature_stats[cross],
            actual_cross_feature_stats,
            normalize_numbers=True)
    # Run generator to check that merge_accumulators() works correctly.
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    _verify(result)

    # Run generator to check that compact() works correctly after
    # merging accumulators.
    accumulators = [
        generator.add_input(generator.create_accumulator(), batch)
        for batch in batches
    ]
    result = generator.extract_output(
        generator.compact(generator.merge_accumulators(accumulators)))
    _verify(result)

    # Run generator to check that add_input() works correctly when adding
    # inputs to a non-empty accumulator.
    accumulator = generator.create_accumulator()

    for batch in batches:
      accumulator = generator.add_input(accumulator, batch)

    result = generator.extract_output(accumulator)
    _verify(result)


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
      self,
      examples: List[Union[types.SlicedRecordBatch, pa.RecordBatch]],
      generator: stats_generator.TransformStatsGenerator,
      expected_results: List[Union[
          statistics_pb2.DatasetFeatureStatistics,
          Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]],
      metrics_verify_fn: Optional[Callable[[beam.metrics.metric.MetricResults],
                                           None]] = None,
      add_default_slice_key_to_input: bool = False,
      add_default_slice_key_to_output: bool = False,
  ) -> None:
    """Tests a slicing aware transform statistics generator.

    Args:
      examples: Input sliced examples.
      generator: A TransformStatsGenerator.
      expected_results: Expected statistics proto results.
      metrics_verify_fn: A callable which will be invoked on the resulting
        beam.metrics.metric.MetricResults object.
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

      def _equal(actual_results: Iterable[
          Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]):
        """Matcher for comparing a list of DatasetFeatureStatistics protos."""
        actual_results = list(actual_results)
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
      expected_results = [(None, p) for p in expected_results]

    options = beam.options.pipeline_options.PipelineOptions(
        runtime_type_check=True)
    with beam.Pipeline(options=options) as p:
      result = p | beam.Create(examples) | generator.ptransform
      util.assert_that(result, _make_result_matcher(self, expected_results))
      pipeline_result = p.run()
      if metrics_verify_fn:
        metrics_verify_fn(pipeline_result.metrics())


class CombinerFeatureStatsGeneratorTest(absltest.TestCase):
  """Test class for combiner feature stats generator related functionality."""

  # Runs the provided combiner feature statistics generator and tests if the
  # output matches the expected result.
  def assertCombinerOutputEqual(
      self,
      input_arrays: List[pa.Array],
      generator: stats_generator.CombinerFeatureStatsGenerator,
      expected_result: statistics_pb2.FeatureNameStatistics,
      feature_path: types.FeaturePath = types.FeaturePath(['']),
  ) -> None:
    """Tests a feature combiner statistics generator.

    This runs the generator twice to cover different behavior. There must be at
    least two input batches in order to test the generator's merging behavior.

    Args:
      input_arrays: A list of batches of test data. Each input represents a a
        single column or feature's values across a batch.
      generator: The CombinerFeatureStatsGenerator to test.
      expected_result: The FeatureNameStatistics proto that it is expected the
        generator will return.
      feature_path: The FeaturePath to use, if not specified, will set a default
        value.
    """
    self.assertIsInstance(input_arrays, list)
    generator.setup()
    # Run generator to check that merge_accumulators() works correctly.
    accumulators = [
        generator.add_input(generator.create_accumulator(), feature_path, arr)
        for arr in input_arrays
    ]
    # Assume that generators will never be called with empty inputs.
    accumulators = accumulators or [generator.create_accumulator()]
    result = generator.extract_output(
        generator.merge_accumulators(accumulators))
    compare.assertProtoEqual(
        self, expected_result, result, normalize_numbers=True)

    # Run generator to check that compact() works correctly after
    # merging accumulators.
    accumulators = [
        generator.add_input(generator.create_accumulator(), feature_path, arr)
        for arr in input_arrays
    ]
    # Assume that generators will never be called with empty inputs.
    accumulators = accumulators or [generator.create_accumulator()]
    result = generator.extract_output(
        generator.compact(generator.merge_accumulators(accumulators)))
    compare.assertProtoEqual(
        self, expected_result, result, normalize_numbers=True)

    # Run generator to check that add_input() works correctly when adding
    # inputs to a non-empty accumulator.
    accumulator = generator.create_accumulator()

    for arr in input_arrays:
      accumulator = generator.add_input(accumulator, feature_path, arr)

    result = generator.extract_output(accumulator)
    compare.assertProtoEqual(
        self, expected_result, result, normalize_numbers=True)


def make_arrow_record_batches_equal_fn(
    test: absltest.TestCase, expected_record_batches: List[pa.RecordBatch]):
  """Makes a matcher function for comparing arrow record batches."""

  def _matcher(actual_record_batches: Iterable[pa.RecordBatch]):
    """Arrow record batches matcher fn."""
    actual_record_batches = list(actual_record_batches)
    test.assertLen(actual_record_batches, len(expected_record_batches))
    for i in range(len(expected_record_batches)):
      actual_record_batch = actual_record_batches[i]
      expected_record_batch = expected_record_batches[i]
      test.assertEqual(
          expected_record_batch.num_columns,
          actual_record_batch.num_columns,
          'Expected {} columns, found {} in record_batch {}'.format(
              expected_record_batch.num_columns,
              actual_record_batch.num_columns, actual_record_batch))
      for column_name, expected_column in zip(
          expected_record_batch.schema.names, expected_record_batch.columns):
        field_index = actual_record_batch.schema.get_field_index(column_name)
        test.assertGreaterEqual(
            field_index, 0, 'Unable to find column {}'.format(column_name))
        actual_column = actual_record_batch.column(field_index)
        test.assertTrue(
            actual_column.equals(expected_column),
            '{}: {} vs {}'.format(column_name, actual_column, expected_column))

  return _matcher
