# Copyright 2022 Google LLC
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
"""Tests for feature_skew_detector."""

import traceback

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_data_validation.utils import beam_runner_util
from tensorflow_data_validation.skew import feature_skew_detector
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format

# Ranges of values for identifier features.
_IDENTIFIER_RANGE = 2
# Names of identifier features.
_IDENTIFIER1 = 'id1'
_IDENTIFIER2 = 'id2'
# Name of feature that is skewed in the test data.
_SKEW_FEATURE = 'skewed'
# Name of feature that appears only in the base data and not test.
_BASE_ONLY_FEATURE = 'base_only'
# Name of feature that appears only in the test data and not base.
_TEST_ONLY_FEATURE = 'test_only'
# Name of feature that has the same value in both base and test data.
_NO_SKEW_FEATURE = 'no_skew'
# Name of feature that has skew but should be ignored.
_IGNORE_FEATURE = 'ignore'
# Name of float feature that has values that are close in base and test
# data.
_CLOSE_FLOAT_FEATURE = 'close_float'


def _unpack_results(results_dict):
  """Unpacks results in the order skew_results, skew_pairs."""
  return (results_dict[feature_skew_detector.SKEW_RESULTS_KEY],
          results_dict[feature_skew_detector.SKEW_PAIRS_KEY])


def _remove_fields_from_skew_pair(skew_pair):
  new_skew_pair = feature_skew_results_pb2.SkewPair()
  new_skew_pair.CopyFrom(skew_pair)
  new_skew_pair.ClearField('base')
  new_skew_pair.ClearField('test')
  return new_skew_pair


def make_sample_equal_fn(test, expected_size, potential_samples):
  """Makes a matcher function for checking SkewPair results."""
  def _matcher(actual):
    try:
      test.assertLen(actual, expected_size)
      for each in actual:
        test.assertTrue(each in potential_samples)
    except AssertionError:
      raise util.BeamAssertException(traceback.format_exc())

  return _matcher


def get_test_input(include_skewed_features, include_close_floats):
  baseline_examples = list()
  test_examples = list()
  skew_pairs = list()
  for i in range(_IDENTIFIER_RANGE):
    for j in range(_IDENTIFIER_RANGE):
      shared_example = tf.train.Example()
      shared_example.features.feature[_IDENTIFIER1].int64_list.value.append(i)
      shared_example.features.feature[_IDENTIFIER2].int64_list.value.append(j)
      shared_example.features.feature[_NO_SKEW_FEATURE].int64_list.value.append(
          1)

      base_example = tf.train.Example()
      base_example.CopyFrom(shared_example)
      test_example = tf.train.Example()
      test_example.CopyFrom(shared_example)

      base_example.features.feature[_IGNORE_FEATURE].int64_list.value.append(0)
      test_example.features.feature[_IGNORE_FEATURE].int64_list.value.append(1)

    if include_close_floats:
      base_example.features.feature[
          _CLOSE_FLOAT_FEATURE].float_list.value.append(1.12345)
      test_example.features.feature[
          _CLOSE_FLOAT_FEATURE].float_list.value.append(1.12456)

    if include_skewed_features:
      # Add three different kinds of skew: value mismatch, appears only in
      # base, and appears only in test.
      base_example.features.feature[_SKEW_FEATURE].int64_list.value.append(0)
      test_example.features.feature[_SKEW_FEATURE].int64_list.value.append(1)
      base_example.features.feature[_BASE_ONLY_FEATURE].int64_list.value.append(
          0)
      test_example.features.feature[_TEST_ONLY_FEATURE].int64_list.value.append(
          1)

      skew_pair = feature_skew_results_pb2.SkewPair()
      # Because serialization of tf.Examples is not deterministic, we do not add
      # or compare the base/test fields of the skew pair in this test.
      skew_pair.matched_features.append(_NO_SKEW_FEATURE)
      skew_pair.mismatched_features.append(_SKEW_FEATURE)
      skew_pair.base_only_features.append(_BASE_ONLY_FEATURE)
      skew_pair.test_only_features.append(_TEST_ONLY_FEATURE)
      skew_pairs.append(skew_pair)

    baseline_examples.append(base_example)
    test_examples.append(test_example)
  return (baseline_examples, test_examples, skew_pairs)


def _make_ex(identifier: str,
             val_skew: str = '',
             val_noskew: str = '') -> tf.train.Example:
  """Makes an example with a skewed and unskewed feature."""
  ex = tf.train.Example()
  if identifier:
    ex.features.feature['id'].bytes_list.value.append(identifier.encode())
  if val_skew:
    ex.features.feature['value_skew'].bytes_list.value.append(val_skew.encode())
  if val_noskew:
    ex.features.feature['value_noskew'].bytes_list.value.append(
        val_noskew.encode())
  return ex


class FeatureSkewDetectorTest(parameterized.TestCase):

  def test_detect_feature_skew(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=True, include_close_floats=True)

    expected_result = [
        text_format.Parse(
            """
        feature_name: 'close_float'
        base_count: 2
        test_count: 2
        mismatch_count: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'skewed'
        base_count: 2
        test_count: 2
        mismatch_count: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'base_only'
        base_count: 2
        base_only: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'test_only'
        test_count: 2
        test_only: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'no_skew'
        base_count: 2
        test_count: 2
        match_count: 2
        diff_count: 0""", feature_skew_results_pb2.FeatureSkew()),
    ]

    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE]))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_result))

  def test_detect_no_skew(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=False, include_close_floats=False)

    expected_result = [
        text_format.Parse(
            """
        feature_name: 'no_skew'
        base_count: 2
        test_count: 2
        match_count: 2
        diff_count: 0""", feature_skew_results_pb2.FeatureSkew()),
    ]

    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Baseline' >> beam.Create(
          baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, skew_sample = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size=2))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_result),
          'CheckSkewResult')
      util.assert_that(skew_sample, make_sample_equal_fn(self, 0, []),
                       'CheckSkewSample')

  def test_obtain_skew_sample(self):
    baseline_examples, test_examples, skew_pairs = get_test_input(
        include_skewed_features=True, include_close_floats=False)

    sample_size = 1
    potential_samples = skew_pairs
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      _, skew_sample = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size))
      # Because serialization of tf.Examples is not deterministic, we remove the
      # base/test fields of the skew pair before comparing them to the expected
      # samples.
      skew_sample |= 'RemoveSelectedFields' >> beam.Map(
          _remove_fields_from_skew_pair
      )
      util.assert_that(
          skew_sample, make_sample_equal_fn(self, sample_size,
                                            potential_samples))

  def test_empty_inputs(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=True, include_close_floats=True)

    # Expect no skew results or sample in each case.
    expected_result = list()

    # Empty base collection.
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples_1 = p | 'Create Base' >> beam.Create([])
      test_examples_1 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result_1, skew_sample_1 = _unpack_results(
          (baseline_examples_1, test_examples_1)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size=1))
      util.assert_that(
          skew_result_1,
          test_util.make_skew_result_equal_fn(self, expected_result),
          'CheckSkewResult')
      util.assert_that(skew_sample_1,
                       make_sample_equal_fn(self, 0, expected_result),
                       'CheckSkewSample')

    # Empty test collection.
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples_2 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_2 = p | 'Create Test' >> beam.Create([])
      skew_result_2, skew_sample_2 = _unpack_results(
          (baseline_examples_2, test_examples_2)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size=1))
      util.assert_that(
          skew_result_2,
          test_util.make_skew_result_equal_fn(self, expected_result),
          'CheckSkewResult')
      util.assert_that(skew_sample_2,
                       make_sample_equal_fn(self, 0, expected_result),
                       'CheckSkewSample')

    # Empty base and test collections.
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples_3 = p | 'Create Base' >> beam.Create([])
      test_examples_3 = p | 'Create Test' >> beam.Create([])
      skew_result_3, skew_sample_3 = _unpack_results(
          (baseline_examples_3, test_examples_3)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size=1))
      util.assert_that(
          skew_result_3,
          test_util.make_skew_result_equal_fn(self, expected_result),
          'CheckSkewResult')
      util.assert_that(skew_sample_3,
                       make_sample_equal_fn(self, 0, expected_result),
                       'CheckSkewSample')

  def test_float_precision_configuration(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=True, include_close_floats=True)

    expected_result = [
        text_format.Parse(
            """
        feature_name: 'skewed'
        base_count: 2
        test_count: 2
        mismatch_count: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'base_only'
        base_count: 2
        base_only: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'test_only'
        test_count: 2
        test_only: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'no_skew'
        base_count: 2
        test_count: 2
        match_count: 2""", feature_skew_results_pb2.FeatureSkew()),
    ]

    expected_with_float = expected_result + [
        text_format.Parse(
            """
        feature_name: 'close_float'
        base_count: 2
        test_count: 2
        mismatch_count: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew())
    ]

    # Do not set a float_round_ndigits.
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples_1 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_1 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = _unpack_results(
          (baseline_examples_1, test_examples_1)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size=1))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_with_float))

    expected_with_float_and_option = expected_result + [
        text_format.Parse(
            """
              feature_name: 'close_float'
              base_count: 2
              test_count: 2
              match_count: 2
              """, feature_skew_results_pb2.FeatureSkew())
    ]

    # Set float_round_ndigits
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples_2 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_2 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = _unpack_results(
          (baseline_examples_2, test_examples_2)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE],
              sample_size=1,
              float_round_ndigits=2))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self,
                                              expected_with_float_and_option))

  def test_no_identifier_features(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=False, include_close_floats=False)
    with self.assertRaisesRegex(ValueError,
                                'At least one feature name must be specified'):
      with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
        baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
        test_examples = p | 'Create Test' >> beam.Create(test_examples)
        _ = ((baseline_examples, test_examples)
             | feature_skew_detector.DetectFeatureSkewImpl([]))

  def test_duplicate_identifiers_allowed_with_duplicates(self):
    base_example_1 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    base_example_2 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 50 } }
          }
        }
        """, tf.train.Example())
    test_example = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
          feature {
            key: "val2"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    expected_result = [
        text_format.Parse(
            """
        feature_name: 'val'
        base_count: 2
        test_count: 2
        match_count: 1
        mismatch_count: 1
        diff_count: 1""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'val2'
        base_count: 0
        test_count: 2
        test_only: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
    ]
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(
          [base_example_1, base_example_2])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'], [], allow_duplicate_identifiers=True))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_result))

  def test_duplicate_identifiers_not_allowed_with_duplicates(self):
    base_example_1 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    base_example_2 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 50 } }
          }
        }
        """, tf.train.Example())
    test_example = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
          feature {
            key: "val2"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(
          [base_example_1, base_example_2])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'], [], allow_duplicate_identifiers=False))
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, []))

    runner = p.run()
    runner.wait_until_finish()
    result_metrics = runner.metrics()
    actual_counter = result_metrics.query(
        beam.metrics.metric.MetricsFilter().with_name(
            'examplediff_skip_dupe_id'))['counters']
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, 1)

  def test_skips_missing_identifier_example(self):
    base_example_1 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    test_example = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example())
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'], [], allow_duplicate_identifiers=True))
      util.assert_that(skew_result,
                       test_util.make_skew_result_equal_fn(self, []))

    runner = p.run()
    runner.wait_until_finish()

  def test_empty_features_equivalent(self):
    base_example_1 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value {}
          }
        }
        """, tf.train.Example())
    test_example = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value {}
          }
        }
        """, tf.train.Example())
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, skew_pairs = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'], [], allow_duplicate_identifiers=True, sample_size=10))
      expected_result = [
          text_format.Parse(
              """
        feature_name: 'val'
        match_count: 1
        """, feature_skew_results_pb2.FeatureSkew()),
      ]
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_result))
      util.assert_that(skew_pairs, self.assertEmpty, label='assert_pairs_empty')

    runner = p.run()
    runner.wait_until_finish()

  def test_empty_features_not_equivalent_to_missing(self):
    base_example_1 = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value {}
          }
        }
        """, tf.train.Example())
    test_example = text_format.Parse(
        """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
        }
        """, tf.train.Example())
    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = _unpack_results(
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'], [], allow_duplicate_identifiers=True, sample_size=10))
      expected_result = [
          text_format.Parse(
              """
        feature_name: 'val'
        """, feature_skew_results_pb2.FeatureSkew()),
      ]
      util.assert_that(
          skew_result,
          test_util.make_skew_result_equal_fn(self, expected_result))

    runner = p.run()
    runner.wait_until_finish()

  def test_telemetry(self):
    shared_example = tf.train.Example()
    shared_example.features.feature[_IDENTIFIER1].int64_list.value.append(1)

    base_example = tf.train.Example()
    base_example.CopyFrom(shared_example)
    test_example = tf.train.Example()
    test_example.CopyFrom(base_example)

    # Add Identifier 2 to base example only.
    base_example.features.feature[_IDENTIFIER2].int64_list.value.append(2)

    p = beam.Pipeline(runner=beam_runner_util.get_test_runner())
    baseline_data = p | 'Create Base' >> beam.Create([base_example])
    test_data = p | 'Create Test' >> beam.Create([test_example])
    _ = ((baseline_data, test_data)
         | feature_skew_detector.DetectFeatureSkewImpl(
             [_IDENTIFIER1, _IDENTIFIER2]))
    runner = p.run()
    runner.wait_until_finish()
    result_metrics = runner.metrics()

    # Test example does not have Identifier 2.
    actual_counter = result_metrics.query(
        beam.metrics.metric.MetricsFilter().with_name(
            'examples_with_missing_identifier_features'))['counters']
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, 1)

  def test_confusion_analysis(self):

    baseline_examples = [
        _make_ex('id0', 'foo', 'foo'),
        _make_ex('id1', 'foo', 'foo'),
        _make_ex('id2', 'foo', 'foo'),
        _make_ex('id3', 'foo', 'foo'),
        _make_ex('id4', 'bar', 'bar'),
        _make_ex('id5', 'bar', 'bar'),
        _make_ex('id6', 'baz', 'baz'),
        _make_ex('id7', 'zip', 'zap'),
    ]
    test_examples = [
        _make_ex('id0', 'foo', 'foo'),
        _make_ex('id1', 'zim', 'foo'),
        _make_ex('id2', 'foo', 'foo'),
        _make_ex('id3', 'bar', 'foo'),
        _make_ex('id4', 'bar', 'bar'),
        _make_ex('id5', 'foo', 'bar'),
        _make_ex('id6', 'baz', 'baz'),
        _make_ex('id7', '', 'zap'),
    ]

    def _confusion_result(
        base: str, test: str, feature_name: str,
        count: int) -> feature_skew_results_pb2.ConfusionCount:
      result = feature_skew_results_pb2.ConfusionCount(
          feature_name=feature_name, count=count)
      result.base.bytes_value = base.encode('utf8')
      result.test.bytes_value = test.encode('utf8')
      return result

    expected_result = [
        _confusion_result('foo', 'foo', 'value_noskew', 4),
        _confusion_result('bar', 'bar', 'value_noskew', 2),
        _confusion_result('baz', 'baz', 'value_noskew', 1),
        _confusion_result('foo', 'foo', 'value_skew', 2),
        _confusion_result('foo', 'zim', 'value_skew', 1),
        _confusion_result('foo', 'bar', 'value_skew', 1),
        _confusion_result('bar', 'bar', 'value_skew', 1),
        _confusion_result('bar', 'foo', 'value_skew', 1),
        _confusion_result('baz', 'baz', 'value_skew', 1),
        _confusion_result('zip', '__MISSING_VALUE__', 'value_skew', 1),
        _confusion_result('zap', 'zap', 'value_noskew', 1),
    ]

    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      confusion_counts = (
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              ['id'],
              confusion_configs=[
                  feature_skew_detector.ConfusionConfig(name='value_skew'),
                  feature_skew_detector.ConfusionConfig(name='value_noskew')
              ]))[feature_skew_detector.CONFUSION_KEY]
      util.assert_that(
          confusion_counts,
          test_util.make_confusion_count_result_equal_fn(self, expected_result))

  @parameterized.named_parameters(
      {
          'testcase_name':
              'int64_feature',
          'input_example':
              text_format.Parse(
                  """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { int64_list { value: 100 } }
          }
        }
        """, tf.train.Example()),
          'expected_error_regex':
              'int64 features unsupported for confusion analysis'
      }, {
          'testcase_name':
              'float_feature',
          'input_example':
              text_format.Parse(
                  """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { float_list { value: 0.5 } }
          }
        }
        """, tf.train.Example()),
          'expected_error_regex':
              'float features unsupported for confusion analysis'
      }, {
          'testcase_name':
              'multivalent_feature',
          'input_example':
              text_format.Parse(
                  """
        features {
          feature {
            key: "id"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "val"
            value { bytes_list { value: "foo" value: "bar" } }
          }
        }
        """, tf.train.Example()),
          'expected_error_regex':
              'multivalent features unsupported for confusion analysis'
      })
  def test_confusion_analysis_errors(self, input_example, expected_error_regex):
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      # Use the direct runner here to get exception propagation.
      with beam.Pipeline() as p:
        baseline_examples = p | 'Create Base' >> beam.Create([input_example])
        test_examples = p | 'Create Test' >> beam.Create([input_example])
        _ = (
            (baseline_examples, test_examples)
            | feature_skew_detector.DetectFeatureSkewImpl(
                ['id'],
                confusion_configs=[
                    feature_skew_detector.ConfusionConfig(name='val'),
                ]))[feature_skew_detector.CONFUSION_KEY]

  def test_match_stats(self):
    baseline_examples = [
        _make_ex('id0'),
        _make_ex('id0'),
        _make_ex('id1'),
        _make_ex('id4'),
        _make_ex(''),
    ]
    test_examples = [
        _make_ex('id0'),
        _make_ex('id0'),
        _make_ex('id2'),
        _make_ex('id3'),
        _make_ex('id4'),
        _make_ex(''),
        _make_ex(''),
    ]

    with beam.Pipeline(runner=beam_runner_util.get_test_runner()) as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      match_stats = ((baseline_examples, test_examples)
                     | feature_skew_detector.DetectFeatureSkewImpl(
                         ['id'], []))[feature_skew_detector.MATCH_STATS_KEY]

      def _assert_fn(got_match_stats):
        expected_match_stats = text_format.Parse(
            """
        base_with_id_count: 4
        test_with_id_count: 5
        identifiers_count: 5
        matching_pairs_count: 5
        ids_missing_in_base_count: 2
        ids_missing_in_test_count: 1
        duplicate_id_count: 1
        base_missing_id_count: 1
        test_missing_id_count: 2
        """, feature_skew_results_pb2.MatchStats())
        self.assertEqual([expected_match_stats], got_match_stats)

      util.assert_that(match_stats, _assert_fn)

if __name__ == '__main__':
  absltest.main()
