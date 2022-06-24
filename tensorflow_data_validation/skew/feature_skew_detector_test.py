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
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
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
      skew_pair.base.CopyFrom(base_example)
      skew_pair.test.CopyFrom(test_example)
      skew_pair.matched_features.append(_NO_SKEW_FEATURE)
      skew_pair.mismatched_features.append(_SKEW_FEATURE)
      skew_pair.base_only_features.append(_BASE_ONLY_FEATURE)
      skew_pair.test_only_features.append(_TEST_ONLY_FEATURE)
      skew_pairs.append(skew_pair)

    baseline_examples.append(base_example)
    test_examples.append(test_example)
  return (baseline_examples, test_examples, skew_pairs)


class FeatureSkewDetectorTest(absltest.TestCase):

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

    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = ((baseline_examples, test_examples)
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

    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Baseline' >> beam.Create(
          baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, skew_sample = (
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples = p | 'Create Test' >> beam.Create(test_examples)
      _, skew_sample = (
          (baseline_examples, test_examples)
          | feature_skew_detector.DetectFeatureSkewImpl(
              [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE], sample_size))
      util.assert_that(
          skew_sample, make_sample_equal_fn(self, sample_size,
                                            potential_samples))

  def test_empty_inputs(self):
    baseline_examples, test_examples, _ = get_test_input(
        include_skewed_features=True, include_close_floats=True)

    # Expect no skew results or sample in each case.
    expected_result = list()

    # Empty base collection.
    with beam.Pipeline() as p:
      baseline_examples_1 = p | 'Create Base' >> beam.Create([])
      test_examples_1 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result_1, skew_sample_1 = (
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
    with beam.Pipeline() as p:
      baseline_examples_2 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_2 = p | 'Create Test' >> beam.Create([])
      skew_result_2, skew_sample_2 = (
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
    with beam.Pipeline() as p:
      baseline_examples_3 = p | 'Create Base' >> beam.Create([])
      test_examples_3 = p | 'Create Test' >> beam.Create([])
      skew_result_3, skew_sample_3 = (
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
    with beam.Pipeline() as p:
      baseline_examples_1 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_1 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = ((baseline_examples_1, test_examples_1)
                        | feature_skew_detector.DetectFeatureSkewImpl(
                            [_IDENTIFIER1, _IDENTIFIER2], [_IGNORE_FEATURE],
                            sample_size=1))
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
    with beam.Pipeline() as p:
      baseline_examples_2 = p | 'Create Base' >> beam.Create(baseline_examples)
      test_examples_2 = p | 'Create Test' >> beam.Create(test_examples)
      skew_result, _ = ((baseline_examples_2, test_examples_2)
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
      with beam.Pipeline() as p:
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create(
          [base_example_1, base_example_2])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = ((baseline_examples, test_examples)
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create(
          [base_example_1, base_example_2])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = ((baseline_examples, test_examples)
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
            'skipped_duplicate_identifier'))['counters']
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = ((baseline_examples, test_examples)
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, skew_pairs = (
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
    with beam.Pipeline() as p:
      baseline_examples = p | 'Create Base' >> beam.Create([base_example_1])
      test_examples = p | 'Create Test' >> beam.Create([test_example])
      skew_result, _ = (
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

    p = beam.Pipeline()
    baseline_data = p | 'Create Base' >> beam.Create([base_example])
    test_data = p | 'Create Test' >> beam.Create([test_example])
    _, _ = ((baseline_data, test_data)
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


if __name__ == '__main__':
  absltest.main()
