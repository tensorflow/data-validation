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
"""Tests for natural_language_stats_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation.statistics.generators import natural_language_domain_inferring_stats_generator as nlsg
from tensorflow_data_validation.utils import test_util
from typing import Text

from tensorflow_metadata.proto.v0 import statistics_pb2


class _FakeHeuristic(nlsg.NLClassifierInterface):

  def classify(self, single_value: Text) -> bool:
    return single_value == 'MATCH'


class NaturalLanguageStatsGeneratorTest(
    test_util.CombinerFeatureStatsGeneratorTest):

  def test_partial_stats_iadd(self):
    stats = nlsg._PartialNLStats(matched=4, considered=10, invalidate=False)
    stats2 = nlsg._PartialNLStats(matched=2, considered=12, invalidate=False)

    stats += stats2
    self.assertEqual(6, stats.matched)
    self.assertEqual(22, stats.considered)
    self.assertEqual(False, stats.invalidate)

  def test_average_word_heuristic_empty_input(self):
    self.assertFalse(nlsg.AverageWordHeuristicNLClassifier().classify(''))

  def test_average_word_heuristic_input_with_only_spaces(self):
    self.assertFalse(nlsg.AverageWordHeuristicNLClassifier().classify('  '))

  def test_average_word_heuristic_avg_word_length_check(self):
    text_avg_word_length_3_8 = 'Hello this is some text'
    self.assertFalse(
        nlsg.AverageWordHeuristicNLClassifier(
            avg_word_length_min=3.5,
            avg_word_length_max=3.7).classify(text_avg_word_length_3_8))
    self.assertTrue(
        nlsg.AverageWordHeuristicNLClassifier(
            avg_word_length_min=3.7,
            avg_word_length_max=3.9).classify(text_avg_word_length_3_8))
    self.assertFalse(
        nlsg.AverageWordHeuristicNLClassifier(
            avg_word_length_min=3.9,
            avg_word_length_max=4.1).classify(text_avg_word_length_3_8))

  def test_average_word_heuristic_min_words(self):
    text_5_words = 'Hello this is some text'
    self.assertTrue(
        nlsg.AverageWordHeuristicNLClassifier(
            min_words_per_value=3).classify(text_5_words))
    self.assertFalse(
        nlsg.AverageWordHeuristicNLClassifier(
            min_words_per_value=6).classify(text_5_words))

  def test_nl_generator_bad_initialization(self):
    """Tests bad initialization values."""
    with self.assertRaisesRegexp(
        ValueError,
        'NLDomainInferringStatsGenerator expects values_threshold > 0.'):
      nlsg.NLDomainInferringStatsGenerator(values_threshold=0)
    with self.assertRaisesRegexp(
        ValueError,
        r'NLDomainInferringStatsGenerator expects a match_ratio in \[0, 1\].'):
      nlsg.NLDomainInferringStatsGenerator(match_ratio=1.1)

  def test_nl_generator_empty_input(self):
    """Tests generator on empty input with fake heuristic."""
    generator = nlsg.NLDomainInferringStatsGenerator(_FakeHeuristic())
    self.assertCombinerOutputEqual([], generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_nl_generator_values_threshold_check(self):
    """Tests generator values threshold with fake heuristic."""
    # Expected to give 6 matches.
    input_batches = [
        pa.array([['MATCH', 'MATCH', 'MATCH'], ['MATCH']]),
        pa.array([['MATCH', 'MATCH']]),
        # Nones should be ignored.
        pa.array([None, None]),
    ]
    # Try generators with values_threshold=7 (should not create stats) and
    # 6 (should create stats)
    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), values_threshold=7)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), values_threshold=6)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info', str='natural_language_domain {}'),
            statistics_pb2.CustomStatistic(
                name='natural_language_match_rate', num=1.0)
        ]))

  def test_nl_generator_utf8_check(self):
    """Tests generator utf8 check with fake heuristic."""
    # Expected to give 6 matches.
    input_batches = [
        pa.array([['MATCH', 'MATCH', 'MATCH'], ['MATCH']]),
        pa.array([['MATCH', 'MATCH']]),
        # Non utf-8 string invalidates accumulator.
        pa.array([[b'\xF0']]),
    ]
    # Try generators with values_threshold=1 which should have generated
    # stats without the non utf-8 value.
    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_nl_generator_invalidation_check(self):
    """Tests generator invalidation with fake heuristic."""
    # Expected to give 6 matches.
    input_batches = [
        pa.array([['MATCH', 'MATCH', 'MATCH'], ['MATCH']]),
        pa.array([['MATCH', 'MATCH']]),
        # Incorrect type invalidates accumulator.
        pa.array([[42]]),
    ]
    # No domain_info is generated as the incorrect type of 42 value invalidated
    # the stats.
    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_nl_generator_match_ratio_check(self):
    """Tests generator match ratio with fake heuristic."""
    input_batches = [
        pa.array([['MATCH', 'MATCH', 'MATCH'], ['MATCH', 'Nope']]),
        pa.array([['MATCH', 'MATCH', 'MATCH']]),
        pa.array([['12345', 'No']]),
    ]
    # Set values_threshold=5 so it always passes.
    # Try generators with match_ratio 0.71 (should not create stats) and
    # 0.69 (should create stats)
    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), match_ratio=0.71, values_threshold=5)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

    generator = nlsg.NLDomainInferringStatsGenerator(
        _FakeHeuristic(), match_ratio=0.69, values_threshold=5)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info', str='natural_language_domain {}'),
            statistics_pb2.CustomStatistic(
                name='natural_language_match_rate', num=0.7)
        ]))

  def test_nl_generator_avg_word_heuristic_match(self):
    """Tests generator with avg word length heuristic."""
    generator = nlsg.NLDomainInferringStatsGenerator(values_threshold=2)
    input_batches = [
        pa.array([['This looks correct.', 'This one too, it should be text.'],
                  ['xosuhddsofuhg123fdgosh']]),
        pa.array([['This should be text as well', 'Here is another text']]),
        pa.array([['This should also be considered good.']]),
    ]

    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info', str='natural_language_domain {}'),
            statistics_pb2.CustomStatistic(
                name='natural_language_match_rate', num=0.8333333)
        ]))

  def test_nl_generator_avg_word_heuristic_non_match(self):
    """Tests generator with avg word length heuristic."""
    generator = nlsg.NLDomainInferringStatsGenerator(values_threshold=2)
    input_batches = [
        pa.array([['abc' * 10, 'xxxxxxxxx'], ['xosuhddsofuhg123fdgosh']]),
        pa.array([['Only one valid text?']]),
    ]

    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())


if __name__ == '__main__':
  absltest.main()
