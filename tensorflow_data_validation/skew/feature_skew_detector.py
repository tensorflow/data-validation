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
"""Finds feature skew between baseline and test examples.

Feature skew is detected by joining baseline and test examples on a
fingerprint computed based on the provided identifier features. For each pair,
the feature skew detector compares the fingerprint of each baseline feature
value to the fingerprint of the corresponding test feature value.

If there is a mismatch in feature values, if the feature is only in the baseline
example, or if the feature is only in the test example, feature skew is
reported in the skew results and (optionally) a skew sample is output with
baseline-test example pairs that exhibit the feature skew.

For example, given the following examples with an identifier feature of 'id':
Baseline
  features {
    feature {
      key: "id"
      value { bytes_list {
        value: "id_1"
      }
    }
    feature {
      key: "float_values"
      value { float_list {
        value: 1.0
        value: 2.0
      }}
    }
  }

Test
  features {
    feature {
      key: "id"
      value { bytes_list {
        value: "id_1"
      }
    }
    feature {
      key: "float_values"
      value { float_list {
        value: 1.0
        value: 3.0
      }}
    }
  }

The following feature skew will be detected:
  feature_name: "float_values"
  baseline_count: 1
  test_count: 1
  mismatch_count: 1
  diff_count: 1
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import apache_beam as beam
import farmhash
import tensorflow as tf
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2


_BASELINE_KEY = "base"
_TEST_KEY = "test"

_EXAMPLES_WITH_MISSING_IDENTIFIER_COUNTER = beam.metrics.Metrics.counter(
    constants.METRICS_NAMESPACE, "examples_with_missing_identifier_features")

_PerFeatureSkew = List[Tuple[str, feature_skew_results_pb2.FeatureSkew]]
_PairOrFeatureSkew = Union[feature_skew_results_pb2.SkewPair,
                           Tuple[str, feature_skew_results_pb2.FeatureSkew]]


def _get_serialized_feature(feature: tf.train.Feature,
                            float_round_ndigits: Optional[int]) -> str:
  """Gets serialized feature, rounding floats as specified.

  Args:
    feature: The feature to serialize.
    float_round_ndigits: Number of digits of precision after the decimal point
      to which to round float values before serializing the feature.

  Returns:
    The serialized feature.
  """
  kind = feature.WhichOneof("kind")
  if (kind == "bytes_list" or kind == "int64_list"):
    return str(feature.SerializePartialToString(deterministic=True))
  elif kind == "float_list":
    if float_round_ndigits is None:
      return str(feature.SerializePartialToString(deterministic=True))
    else:
      rounded_feature = tf.train.Feature()
      for value in feature.float_list.value:
        rounded_feature.float_list.value.append(
            round(value, float_round_ndigits))
      return str(rounded_feature.SerializePartialToString(deterministic=True))
  else:
    raise ValueError("Unknown feature type detected: %s" % kind)


def _compute_skew_for_features(
    base_feature: tf.train.Feature, test_feature: tf.train.Feature,
    float_round_ndigits: Optional[int],
    feature_name: str) -> feature_skew_results_pb2.FeatureSkew:
  """Computes feature skew for a pair of baseline and test features.

  Args:
    base_feature: The feature to compare from the baseline example.
    test_feature: The feature to compare from the test example.
    float_round_ndigits: Number of digits precision after the decimal point to
      which to round float values before comparison.
    feature_name: The name of the feature for which to compute skew between the
      examples.

  Returns:
    A FeatureSkew proto containing information about skew for the specified
      feature.
  """
  skew_results = feature_skew_results_pb2.FeatureSkew()
  skew_results.feature_name = feature_name
  if not _empty_or_null(base_feature) and not _empty_or_null(test_feature):
    skew_results.base_count = 1
    skew_results.test_count = 1
    if (farmhash.fingerprint64(
        _get_serialized_feature(base_feature,
                                float_round_ndigits)) == farmhash.fingerprint64(
                                    _get_serialized_feature(
                                        test_feature, float_round_ndigits))):
      skew_results.match_count = 1
    else:
      skew_results.mismatch_count = 1
  elif not _empty_or_null(base_feature):
    skew_results.base_count = 1
    skew_results.base_only = 1
  elif not _empty_or_null(test_feature):
    skew_results.test_count = 1
    skew_results.test_only = 1
  elif (test_feature is None) == (base_feature is None):
    # Both features are None, or present with zero values.
    skew_results.match_count = 1
  return skew_results


def _compute_skew_for_examples(
    base_example: tf.train.Example, test_example: tf.train.Example,
    features_to_ignore: List[tf.train.Feature],
    float_round_ndigits: Optional[int]) -> Tuple[_PerFeatureSkew, bool]:
  """Computes feature skew for a pair of baseline and test examples.

  Args:
    base_example: The baseline example to compare.
    test_example: The test example to compare.
    features_to_ignore: The features not to compare.
    float_round_ndigits: Number of digits precision after the decimal point to
      which to round float values before comparison.

  Returns:
    A tuple containing a list of the skew information for each feature
    and a boolean indicating whether skew was found in any feature, in which
    case the examples are considered skewed.
  """
  all_feature_names = set()
  all_feature_names.update(base_example.features.feature.keys())
  all_feature_names.update(test_example.features.feature.keys())
  feature_names = all_feature_names.difference(set(features_to_ignore))

  result = list()
  is_skewed = False
  for name in feature_names:
    base_feature = base_example.features.feature.get(name)
    test_feature = test_example.features.feature.get(name)
    skew = _compute_skew_for_features(base_feature, test_feature,
                                      float_round_ndigits, name)
    if skew.match_count == 0:
      # If any features have a mismatch or are found only in the baseline or
      # test example, the examples are considered skewed.
      is_skewed = True
    result.append((name, skew))
  return result, is_skewed


def _merge_feature_skew_results(
    skew_results: Iterable[feature_skew_results_pb2.FeatureSkew]
) -> feature_skew_results_pb2.FeatureSkew:
  """Merges multiple FeatureSkew protos into a single FeatureSkew proto.

  Args:
    skew_results: An iterable of FeatureSkew protos.

  Returns:
    A FeatureSkew proto containing the result of merging the inputs.
  """
  result = feature_skew_results_pb2.FeatureSkew()
  for skew_result in skew_results:
    if not result.feature_name:
      result.feature_name = skew_result.feature_name
    elif result.feature_name != skew_result.feature_name:
      raise ValueError("Attempting to merge skew results with different names.")
    result.base_count += skew_result.base_count
    result.test_count += skew_result.test_count
    result.match_count += skew_result.match_count
    result.base_only += skew_result.base_only
    result.test_only += skew_result.test_only
    result.mismatch_count += skew_result.mismatch_count
  result.diff_count = (
      result.base_only + result.test_only + result.mismatch_count)
  return result


def _construct_skew_pair(
    per_feature_skew: List[Tuple[str, feature_skew_results_pb2.FeatureSkew]],
    base_example: tf.train.Example,
    test_example: tf.train.Example) -> feature_skew_results_pb2.SkewPair:
  """Constructs a SkewPair from baseline and test examples.

  Args:
    per_feature_skew: Skew results for each feature in the input examples.
    base_example: The baseline example to include.
    test_example: The test example to include.

  Returns:
    A SkewPair containing examples that exhibit some skew.
  """
  skew_pair = feature_skew_results_pb2.SkewPair()
  skew_pair.base.CopyFrom(base_example)
  skew_pair.test.CopyFrom(test_example)

  for feature_name, skew_result in per_feature_skew:
    if skew_result.match_count == 1:
      skew_pair.matched_features.append(feature_name)
    elif skew_result.base_only == 1:
      skew_pair.base_only_features.append(feature_name)
    elif skew_result.test_only == 1:
      skew_pair.test_only_features.append(feature_name)
    elif skew_result.mismatch_count == 1:
      skew_pair.mismatched_features.append(feature_name)

  return skew_pair


def _empty_or_null(feature: Optional[tf.train.Feature]) -> bool:
  """True if feature is None or holds no values."""
  if feature is None:
    return True
  if len(feature.bytes_list.value) + len(feature.int64_list.value) + len(
      feature.float_list.value) == 0:
    return True
  return False


class _ExtractIdentifiers(beam.DoFn):
  """DoFn that extracts a unique fingerprint for each example.

  This class computes fingerprints by combining the identifier features.
  """

  def __init__(self, identifier_features: List[types.FeatureName],
               float_round_ndigits: Optional[int]) -> None:
    """Initializes _ExtractIdentifiers.

    Args:
      identifier_features: The names of the features to use to compute a
        fingerprint for the example.
      float_round_ndigits: Number of digits precision after the decimal point to
        which to round float values before generating the fingerprint.
    """
    self._identifier_features = sorted(identifier_features)
    self._float_round_ndigits = float_round_ndigits

  def process(
      self,
      example: tf.train.Example) -> Iterable[Tuple[str, tf.train.Example]]:
    serialized_feature_values = []
    for identifier_feature in self._identifier_features:
      feature = example.features.feature.get(identifier_feature)
      if _empty_or_null(feature):
        _EXAMPLES_WITH_MISSING_IDENTIFIER_COUNTER.inc()
        return
      else:
        serialized_feature_values.append(
            _get_serialized_feature(feature, self._float_round_ndigits))
    yield (str(farmhash.fingerprint64("".join(serialized_feature_values))),
           example)


class _ComputeSkew(beam.DoFn):
  """DoFn that computes skew for each pair of examples."""

  def __init__(self, features_to_ignore: List[tf.train.Feature],
               float_round_ndigits: Optional[int],
               allow_duplicate_identifiers: bool) -> None:
    """Initializes _ComputeSkew.

    Args:
      features_to_ignore: Names of features that are ignored in skew detection.
      float_round_ndigits: Number of digits precision after the decimal point to
        which to round float values before detecting skew.
      allow_duplicate_identifiers: If set, skew detection will be done on
        examples for which there are duplicate identifier feature values. In
        this case, the counts in the FeatureSkew result are based on each
        baseline-test example pair analyzed. Examples with given identifier
        feature values must all fit in memory.
    """
    self._features_to_ignore = features_to_ignore
    self._float_round_ndigits = float_round_ndigits
    self._allow_duplicate_identifiers = allow_duplicate_identifiers
    self._skipped_duplicate_identifiers = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, "skipped_duplicate_identifier")

  def process(
      self, element: Tuple[str,
                           Dict[str,
                                List[Any]]]) -> Iterable[_PairOrFeatureSkew]:
    (_, examples) = element
    base_examples = examples.get(_BASELINE_KEY)
    test_examples = examples.get(_TEST_KEY)
    if not self._allow_duplicate_identifiers:
      if len(base_examples) > 1 or len(test_examples) > 1:
        self._skipped_duplicate_identifiers.inc(1)
        return
    if base_examples and test_examples:
      for base_example in base_examples:
        for test_example in test_examples:
          result, is_skewed = _compute_skew_for_examples(
              base_example, test_example, self._features_to_ignore,
              self._float_round_ndigits)
          if is_skewed:
            skew_pair = _construct_skew_pair(result, base_example,
                                             test_example)
            yield beam.pvalue.TaggedOutput("skew_pairs", skew_pair)
          for each in result:
            yield beam.pvalue.TaggedOutput("skew_results", each)


class DetectFeatureSkewImpl(beam.PTransform):
  """Identifies feature skew in baseline and test examples.

  This PTransform returns a tuple of PCollections containing:
    1. Aggregated skew statistics (containing, e.g., mismatch count, baseline
       only, test only) for each feature; and
    2. A sample of skewed example pairs (if sample_size is > 0).
  """

  def __init__(self,
               identifier_features: List[types.FeatureName],
               features_to_ignore: Optional[List[types.FeatureName]] = None,
               sample_size: int = 0,
               float_round_ndigits: Optional[int] = None,
               allow_duplicate_identifiers: bool = False) -> None:
    """Initializes DetectFeatureSkewImpl.

    Args:
      identifier_features: The names of the features to use to identify an
        example.
      features_to_ignore: The names of the features for which skew detection is
        not done.
      sample_size: Size of the sample of baseline-test example pairs that
        exhibit skew to include in the skew results.
      float_round_ndigits: Number of digits of precision after the decimal point
        to which to round float values before detecting skew.
      allow_duplicate_identifiers: If set, skew detection will be done on
        examples for which there are duplicate identifier feature values. In
        this case, the counts in the FeatureSkew result are based on each
        baseline-test example pair analyzed. Examples with given identifier
        feature values must all fit in memory.
    """
    if not identifier_features:
      raise ValueError("At least one feature name must be specified in "
                       "identifier_features.")
    self._identifier_features = identifier_features
    self._sample_size = sample_size
    self._float_round_ndigits = float_round_ndigits
    if features_to_ignore is not None:
      self._features_to_ignore = features_to_ignore + identifier_features
    else:
      self._features_to_ignore = identifier_features
    self._allow_duplicate_identifiers = allow_duplicate_identifiers

  def expand(
      self, pcollections: Tuple[beam.pvalue.PCollection,
                                beam.pvalue.PCollection]
  ) -> Tuple[beam.pvalue.PCollection, beam.pvalue.PCollection]:
    base_examples, test_examples = pcollections
    keyed_base_examples = (
        base_examples | "ExtractBaseIdentifiers" >> beam.ParDo(
            _ExtractIdentifiers(self._identifier_features,
                                self._float_round_ndigits)))
    keyed_test_examples = (
        test_examples | "ExtractTestIdentifiers" >> beam.ParDo(
            _ExtractIdentifiers(self._identifier_features,
                                self._float_round_ndigits)))
    results = (
        {
            "base": keyed_base_examples,
            "test": keyed_test_examples
        } | "JoinExamples" >> beam.CoGroupByKey()
        | "ComputeSkew" >> beam.ParDo(
            _ComputeSkew(self._features_to_ignore, self._float_round_ndigits,
                         self._allow_duplicate_identifiers)).with_outputs(
                             "skew_results", "skew_pairs"))
    skew_results = (
        results.skew_results | "MergeSkewResultsPerFeature" >>  # pytype: disable=attribute-error
        beam.CombinePerKey(_merge_feature_skew_results)
        | "DropKeys" >> beam.Values())
    skew_pairs = (
        results.skew_pairs | "SampleSkewPairs" >>  # pytype: disable=attribute-error
        beam.combiners.Sample.FixedSizeGlobally(self._sample_size)
        # Sampling results in a pcollection with a single element consisting of
        # a list of the samples. Convert this to a pcollection of samples.
        | "Flatten" >> beam.FlatMap(lambda x: x))

    return skew_results, skew_pairs
