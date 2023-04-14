# Copyright 2019 Google LLC
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
"""Module that computes statistics for features of natural language type.

The module uses a pluggable nl-classifier. If the match rate is high enough
and enough values have been considered then the feature is marked as natural
language by generating the appropriate domain_info as a custom_statistic and
the observed match ratio.

A simple heuristic based on average word length is used as default classifier.
The heuristic is too lenient, but efficient. A model based classifier could
be used for more accurate results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Optional
import numpy as np
import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util
from typing import Iterable, Text
from tensorflow_metadata.proto.v0 import statistics_pb2

# AverageWordHeuristicNLClassifier default initialization values
_AVG_WORD_LENGTH_MIN = 2.5
_AVG_WORD_LENGTH_MAX = 8
_MIN_WORDS_PER_VALUE = 3
_CROP_AT_LENGTH = 100
# We crop the feature values as checking all the values is expensive for
# features with high valency.
_CROP_AT_VALUES = 100

# NLDomainInferringStatsGenerator default initialization values.
_MATCH_RATIO = 0.8
_VALUES_THRESHOLD = 100

# Custom statistics exported by this generator.
_DOMAIN_INFO = 'domain_info'
_NL_MATCH_RATIO = 'natural_language_match_rate'


class _PartialNLStats(object):
  """Partial feature stats for natural language."""

  def __init__(self, matched: int = 0, considered: int = 0,
               invalidate=False) -> None:
    # The total number of values matching natural language heuristic.
    self.matched = matched
    # The total number of values considered for classification.
    self.considered = considered
    # True only if this feature should never be considered, e.g: some
    # value_lists have inconsistent types.
    self.invalidate = invalidate

  def __iadd__(self, other: '_PartialNLStats') -> '_PartialNLStats':
    """Merge two partial natual language stats."""
    self.matched += other.matched
    self.considered += other.considered
    self.invalidate |= other.invalidate
    return self


class NLClassifierInterface(six.with_metaclass(abc.ABCMeta)):
  """Interface for an NL classifier."""

  @abc.abstractmethod
  def classify(self, value: Text) -> bool:
    """Should return True iff value is classified as NL."""
    raise NotImplementedError()


class AverageWordHeuristicNLClassifier(NLClassifierInterface):
  """A simple heuristic based on average word length.

  A value is classified as NL iff all the conditions are met:
  1. It contains at least min_words_per_value.
  2. The average length is in [avg_word_length_min, avg_word_length_max].
  For efficiency, the value is cropped to at most crop_at_length chars.

  This heuristic is lenient and targets efficiency. For more accurate results
  consider replacing with a model-based classifier.
  """

  def __init__(self,
               avg_word_length_min: float = _AVG_WORD_LENGTH_MIN,
               avg_word_length_max: float = _AVG_WORD_LENGTH_MAX,
               min_words_per_value: int = _MIN_WORDS_PER_VALUE,
               crop_at_length: int = _CROP_AT_LENGTH) -> None:
    self._avg_word_length_min = avg_word_length_min
    self._avg_word_length_max = avg_word_length_max
    self._min_words_per_value = min_words_per_value
    self._crop_at_length = crop_at_length

  def classify(self, value: Text) -> bool:
    words = value[0:self._crop_at_length].split()
    if not words:
      return False
    # Expanded for loop efficiency.
    sum_word_length = 0
    for w in words:
      sum_word_length += len(w)
    avg_word_length = float(sum_word_length) / len(words)
    if (self._avg_word_length_min <= avg_word_length <=
        self._avg_word_length_max and len(words) >= self._min_words_per_value):
      return True
    return False


class NLDomainInferringStatsGenerator(
    stats_generator.CombinerFeatureStatsGenerator):
  """Generates feature level statistics for natural language stats.

  A combiner that uses a pluggable NL classifier to generate natural language
  stats for input examples. After the statistics are combined it classifies
  as NL iff both the stats represent enough values (self._values_threshold)
  and the match ratio is high enough (self._match_ratio).
  """

  def __init__(self,
               classifier: Optional[NLClassifierInterface] = None,
               match_ratio: float = _MATCH_RATIO,
               values_threshold: int = _VALUES_THRESHOLD) -> None:
    """Initializes a NLDomainInferringStatsGenerator.

    Args:
      classifier: A NLClassifier that classifies values as NL.
      match_ratio: In order for a feature to be marked as NL the classifier
        match ratio should meet or exceed this ratio. The ratio should be in
        [0, 1].
      values_threshold: In order for a feature to be marked as NL at least
        this many values should be considered.

    Raises:
      ValueError: If values_threshold <= 0 or match_ratio not in [0, 1].
    """
    super(NLDomainInferringStatsGenerator, self).__init__(type(self).__name__)
    if classifier is None:
      classifier = AverageWordHeuristicNLClassifier()
    if values_threshold <= 0:
      raise ValueError(
          'NLDomainInferringStatsGenerator expects values_threshold > 0.')
    if not 0.0 <= match_ratio <= 1.0:
      raise ValueError(
          'NLDomainInferringStatsGenerator expects a match_ratio in [0, 1].')
    self._classifier = classifier
    self._values_threshold = values_threshold
    self._match_ratio = match_ratio

  def create_accumulator(self) -> _PartialNLStats:
    """Return a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    return _PartialNLStats()

  def add_input(self, accumulator: _PartialNLStats,
                feature_path: types.FeaturePath,
                feature_array: pa.Array) -> _PartialNLStats:
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      feature_path: The path of the feature.
      feature_array: An arrow Array representing a batch of feature values
        which should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    if accumulator.invalidate:
      return accumulator
    feature_type = stats_util.get_feature_type_from_arrow_type(
        feature_path, feature_array.type)
    # Ignore null array.
    if feature_type is None:
      return accumulator
    # If we see a different type, invalidate.
    if feature_type != statistics_pb2.FeatureNameStatistics.STRING:
      accumulator.invalidate = True
      return accumulator

    def _is_non_utf8(value):
      return (isinstance(value, bytes) and
              stats_util.maybe_get_utf8(value) is None)

    is_non_utf_vec = np.vectorize(_is_non_utf8, otypes=[bool])
    classify_vec = np.vectorize(self._classifier.classify, otypes=[bool])
    values = np.asarray(array_util.flatten_nested(feature_array)[0]
                        .slice(0, _CROP_AT_VALUES))
    if np.any(is_non_utf_vec(values)):
      accumulator.invalidate = True
      return accumulator
    accumulator.considered += values.size
    accumulator.matched += np.sum(classify_vec(values))
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_PartialNLStats]) -> _PartialNLStats:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      result += accumulator
    return result

  def extract_output(self, accumulator: _PartialNLStats
                    ) -> statistics_pb2.FeatureNameStatistics:
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.FeatureNameStatistics()
    if (not accumulator.invalidate and
        accumulator.considered >= self._values_threshold):
      match_ratio = float(accumulator.matched) / accumulator.considered
      if match_ratio >= self._match_ratio:
        result.custom_stats.add(
            name=stats_util.DOMAIN_INFO, str='natural_language_domain {}')
        result.custom_stats.add(name=_NL_MATCH_RATIO, num=match_ratio)
    return result
