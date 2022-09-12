# Copyright 2020 Google LLC
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

import tempfile

from absl.testing import absltest
import pyarrow as pa

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import natural_language_stats_generator as nlsg
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class NaturalLanguageStatsGeneratorTest(
    test_util.CombinerFeatureStatsGeneratorTest):

  def setUp(self):
    super(NaturalLanguageStatsGeneratorTest, self).setUp()
    self._schema = text_format.Parse(
        """
        feature {
          name: "string_nlp_feature_with_vocab"
          type: BYTES
          natural_language_domain {
            vocabulary: "my_vocab"
            coverage {
              excluded_int_tokens: [1]
            }
          }
        }
        feature {
          name: "string_nlp_feature_no_vocab"
          type: BYTES
          natural_language_domain {
            coverage {
              oov_string_tokens: ['Bar', 'Baz']
            }
          }
        }
        feature {
          name: "int_nlp_feature_with_vocab"
          type: INT
          natural_language_domain {
            vocabulary: "my_vocab"
            coverage {
              excluded_string_tokens: ['Foo', 'Caz']
              excluded_int_tokens: [1]
              oov_string_tokens: ['Bar', 'Baz']
            }
            sequence_length_constraints {
              excluded_string_value: ['Razzz']
            }
          }
        }
        feature {
          name: "int_nlp_feature_with_vocab_and_token_constraints"
          type: INT
          natural_language_domain {
            vocabulary: "my_vocab"
            coverage {
              excluded_int_tokens: [1]
            }
            token_constraints {
              string_value: 'Foo'
            }
            token_constraints {
              int_value: 1
            }
          }
        }
        feature {
          name: "int_nlp_feature_no_vocab"
          type: INT
          natural_language_domain {
            coverage {
              excluded_int_tokens: [1]
            }
          }
        }
        feature {
          name: "int_nlp_feature_empty_domain"
          type: INT
          natural_language_domain {}
        }
        feature {
          name: "non_nlp_feature"
          type: BYTES
        }
        """, schema_pb2.Schema())
    self._string_nlp_feature_with_vocab_path = types.FeaturePath(
        ['string_nlp_feature_with_vocab'])
    self._string_nlp_feature_no_vocab_path = types.FeaturePath(
        ['string_nlp_feature_no_vocab'])
    self._int_nlp_feature_with_vocab_path = types.FeaturePath(
        ['int_nlp_feature_with_vocab'])
    self._int_nlp_feature_with_vocab_and_token_constraints_path = (
        types.FeaturePath(['int_nlp_feature_with_vocab_and_token_constraints']))
    self._int_nlp_feature_no_vocab_path = types.FeaturePath(
        ['int_nlp_feature_no_vocab'])
    self._int_nlp_feature_empty_domain = types.FeaturePath(
        ['int_nlp_feature_empty_domain'])
    self._non_nlp_feature_path = types.FeaturePath(['non_nlp_feature'])

  def test_partial_stats_iadd(self):
    stats = nlsg._PartialNLStats(
        invalidate=False, num_in_vocab_tokens=2, total_num_tokens=3)
    stats.vocab_token_length_quantiles.AddValues(pa.array([1, 2, 2]))
    stats.token_occurrence_counts.AddValues(pa.array([b'foo', b'bar', b'bar']))
    stats.min_sequence_length = 3
    stats.max_sequence_length = 7
    stats.sequence_length_quantiles.AddValues(pa.array([1, 2, 2]))
    ts = nlsg._TokenStats()
    ts.frequency = 10
    ts.num_sequences = 2
    ts.per_sequence_min_frequency = 3
    ts.per_sequence_max_frequency = 7
    ts.positions[1] = 3
    ts.positions[2] = 7
    stats.token_statistics['foo'] = ts

    stats_2 = nlsg._PartialNLStats(
        invalidate=False, num_in_vocab_tokens=7, total_num_tokens=10)
    stats_2.vocab_token_length_quantiles.AddValues(pa.array([2, 3]))
    stats_2.token_occurrence_counts.AddValues(pa.array([b'bar', b'baz']))
    stats_2.min_sequence_length = None
    stats_2.max_sequence_length = 9
    stats_2.sequence_length_quantiles.AddValues(pa.array([2, 3]))
    ts1 = nlsg._TokenStats()
    ts1.frequency = 12
    ts1.num_sequences = 1
    ts1.per_sequence_min_frequency = 4
    ts1.per_sequence_max_frequency = 8
    ts1.positions[1] = 12
    stats_2.token_statistics['foo'] = ts1
    stats_2.token_statistics['bar'] = ts1

    stats += stats_2
    self.assertEqual(9, stats.num_in_vocab_tokens)
    self.assertEqual(13, stats.total_num_tokens)
    self.assertEqual(3, stats.min_sequence_length)
    self.assertEqual(9, stats.max_sequence_length)
    self.assertEqual(False, stats.invalidate)
    token_occurrence_counts = stats.token_occurrence_counts.Estimate(
    ).to_pylist()
    self.assertListEqual(token_occurrence_counts, [{
        'values': b'bar',
        'counts': 3.0
    }, {
        'values': b'baz',
        'counts': 1.0
    }, {
        'values': b'foo',
        'counts': 1.0
    }])
    quantiles = stats.vocab_token_length_quantiles.GetQuantiles(2)
    quantiles = quantiles.flatten().to_pylist()
    self.assertListEqual(quantiles, [1, 2, 3])

    quantiles = stats.sequence_length_quantiles.GetQuantiles(2)
    quantiles = quantiles.flatten().to_pylist()
    self.assertListEqual(quantiles, [1, 2, 3])

    foo_ts_result = stats.token_statistics['foo']
    bar_ts_result = stats.token_statistics['bar']

    self.assertEqual(foo_ts_result.frequency, 22)
    self.assertEqual(foo_ts_result.num_sequences, 3)
    self.assertEqual(foo_ts_result.per_sequence_min_frequency, 3)
    self.assertEqual(foo_ts_result.per_sequence_max_frequency, 8)
    self.assertEqual(foo_ts_result.positions[1], 15)
    self.assertEqual(foo_ts_result.positions[2], 7)

    self.assertEqual(bar_ts_result.frequency, 12)
    self.assertEqual(bar_ts_result.num_sequences, 1)
    self.assertEqual(bar_ts_result.per_sequence_min_frequency, 4)
    self.assertEqual(bar_ts_result.per_sequence_max_frequency, 8)
    self.assertEqual(bar_ts_result.positions[1], 12)

  def _create_expected_feature_name_statistics(
      self,
      feature_coverage=None,
      avg_token_length=None,
      min_sequence_length=None,
      max_sequence_length=None,
      token_len_quantiles=None,
      sequence_len_quantiles=None,
      sorted_token_names_and_counts=None,
      reported_sequences=None,
      token_statistics=None):
    nls = statistics_pb2.NaturalLanguageStatistics()
    if feature_coverage is not None:
      nls.feature_coverage = feature_coverage
    if avg_token_length:
      nls.avg_token_length = avg_token_length
    if min_sequence_length:
      nls.min_sequence_length = min_sequence_length
    if max_sequence_length:
      nls.max_sequence_length = max_sequence_length
    if token_len_quantiles:
      for low_value, high_value, sample_count in token_len_quantiles:
        nls.token_length_histogram.type = statistics_pb2.Histogram.QUANTILES
        nls.token_length_histogram.buckets.add(
            low_value=low_value,
            high_value=high_value,
            sample_count=sample_count)
    if sequence_len_quantiles:
      for low_value, high_value, sample_count in sequence_len_quantiles:
        nls.sequence_length_histogram.type = statistics_pb2.Histogram.QUANTILES
        nls.sequence_length_histogram.buckets.add(
            low_value=low_value,
            high_value=high_value,
            sample_count=sample_count)
    if sorted_token_names_and_counts:
      for index, (token_name,
                  count) in enumerate(sorted_token_names_and_counts):
        nls.rank_histogram.buckets.add(
            low_rank=index,
            high_rank=index,
            label=token_name,
            sample_count=count)
    if token_statistics:
      for k, v in token_statistics.items():
        ts = nls.token_statistics.add(
            frequency=v[0],
            fraction_of_sequences=v[1],
            per_sequence_min_frequency=v[2],
            per_sequence_max_frequency=v[3],
            per_sequence_avg_frequency=v[4])
        if isinstance(k, str):
          ts.string_token = k
        else:
          ts.int_token = k
        ts.positions.CopyFrom(v[5])
    if reported_sequences:
      for r in reported_sequences:
        nls.reported_sequences.append(str(r))

    custom_nl_stats = statistics_pb2.CustomStatistic(name='nl_statistics')
    custom_nl_stats.any.Pack(nls)
    return statistics_pb2.FeatureNameStatistics(custom_stats=[custom_nl_stats])

  def test_nl_generator_empty_input(self):
    generator = nlsg.NLStatsGenerator(None, None, 0, 0, 0)
    self.assertCombinerOutputEqual(
        [], generator, self._create_expected_feature_name_statistics())

  def test_nl_generator_invalidation_check_no_nld(self):
    """Tests generator invalidation with no natural language domain."""
    generator = nlsg.NLStatsGenerator(self._schema, None, 0, 0, 0)
    generator.setup()
    accumulator = generator.create_accumulator()
    self.assertFalse(accumulator.invalidate)
    valid_input = pa.array([['Foo'], ['Bar']])
    accumulator = generator.add_input(accumulator, self._non_nlp_feature_path,
                                      valid_input)
    self.assertTrue(accumulator.invalidate)

  def test_nl_generator_invalidation_check_empty_nld(self):
    """Tests generator invalidation whith empty natural language domain."""
    generator = nlsg.NLStatsGenerator(self._schema, None, 0, 0, 0)
    generator.setup()
    accumulator = generator.create_accumulator()
    self.assertFalse(accumulator.invalidate)
    valid_input = pa.array([[0], [1]])
    accumulator = generator.add_input(accumulator,
                                      self._int_nlp_feature_empty_domain,
                                      valid_input)
    self.assertTrue(accumulator.invalidate)

  def test_nl_generator_invalidation_check_float_input(self):
    """Tests generator invalidation with float inputs."""
    generator = nlsg.NLStatsGenerator(self._schema, None, 0, 0, 0)
    generator.setup()
    accumulator = generator.create_accumulator()
    self.assertFalse(accumulator.invalidate)
    valid_input = pa.array([['Foo'], ['Bar']])
    accumulator = generator.add_input(accumulator,
                                      self._string_nlp_feature_no_vocab_path,
                                      valid_input)
    self.assertFalse(accumulator.invalidate)
    invalid_input = pa.array([[1.0], [2.0], [3.0]])
    accumulator = generator.add_input(accumulator,
                                      self._string_nlp_feature_no_vocab_path,
                                      invalid_input)
    self.assertTrue(accumulator.invalidate)

  def test_nl_generator_string_feature_no_vocab(self):
    """Tests generator calculation with a string domain having no vocab."""
    input_batches = [pa.array([[b'Foo'], None, [b'Baz']])]
    generator = nlsg.NLStatsGenerator(self._schema, None, 0, 0, 0)
    expected_reported_sequences = [['Baz'], ['Foo']] * 2
    self.assertCombinerOutputEqual(
        input_batches, generator,
        self._create_expected_feature_name_statistics(
            feature_coverage=0.5,
            avg_token_length=3,
            min_sequence_length=1,
            max_sequence_length=1,
            reported_sequences=expected_reported_sequences),
        self._string_nlp_feature_no_vocab_path)

  def test_nl_generator_string_feature_vocab(self):
    """Tests generator calculation with a string domain having a vocab."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\nBazz\n')
      vocab_file.flush()

      input_batches = [pa.array([[b'Bar', b'Bazz'], None])]
      generator = nlsg.NLStatsGenerator(self._schema,
                                        {'my_vocab': vocab_file.name}, 0, 0, 0)
      expected_reported_sequences = [['Bar', 'Bazz']] * 2
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=1.0,
              avg_token_length=4,
              min_sequence_length=2,
              max_sequence_length=2,
              reported_sequences=expected_reported_sequences),
          self._string_nlp_feature_with_vocab_path)

  def test_nl_generator_int_feature_no_vocab(self):
    """Tests generator calculation with a int domain having no vocab."""
    input_batches = [pa.array([[1, 2, 3]])]
    generator = nlsg.NLStatsGenerator(self._schema, None, 0, 0, 0)
    expected_reported_sequences = [[1, 2, 3]] * 2
    self.assertCombinerOutputEqual(
        input_batches, generator,
        self._create_expected_feature_name_statistics(
            feature_coverage=0.0,
            min_sequence_length=3,
            max_sequence_length=3,
            reported_sequences=expected_reported_sequences),
        self._int_nlp_feature_no_vocab_path)

  def test_nl_generator_int_feature_vocab(self):
    """Tests generator calculation with an int domain and a vocab."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\nBaz\nBazz\n')
      vocab_file.flush()
      input_batches = [pa.array([[0, 1, 2, 3, 4]]), pa.array([[0, 1, 2, 3, 4]])]
      generator = nlsg.NLStatsGenerator(self._schema,
                                        {'my_vocab': vocab_file.name}, 0, 0, 0)
      expected_reported_sequences = [['Foo', 'Bar', 'Baz', 'Bazz', 4]] * 2
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=float(1) / 3,
              avg_token_length=4,
              min_sequence_length=5,
              max_sequence_length=5,
              reported_sequences=expected_reported_sequences),
          self._int_nlp_feature_with_vocab_path)

  def test_nl_generator_token_and_sequence_histograms(self):
    """Tests generator calculation of token and sequence histograms."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\nBaz\nBazz\nCar\nRazzz\n')
      vocab_file.flush()
      input_batches = [pa.array([[0, 1, 2, 4, 4], [3, 3, 3, 5]])]
      generator = nlsg.NLStatsGenerator(
          schema=self._schema,
          vocab_paths={'my_vocab': vocab_file.name},
          num_quantiles_histogram_buckets=2,
          num_rank_histogram_buckets=2,
          num_histogram_buckets=2)
      expected_reported_sequences = [['Foo', 'Bar', 'Baz', 'Car', 'Car'],
                                     ['Bazz', 'Bazz', 'Bazz', 'Razzz']] * 2
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=0.8571428571428571,
              avg_token_length=(3 + 3 + 4 + 4 + 4 + 5) / 6,
              min_sequence_length=3,
              max_sequence_length=5,
              token_len_quantiles=[(3, 4, 5), (4, 5, 1)],
              sequence_len_quantiles=[(3, 5, 1.5), (5, 5, 0.5)],
              sorted_token_names_and_counts=[('Bazz', 3), ('Car', 2)],
              reported_sequences=expected_reported_sequences),
          self._int_nlp_feature_with_vocab_path)

  def test_nl_generator_token_stats(self):
    """Tests generator calculation of token statistics."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\n')
      vocab_file.flush()
      input_batches = [pa.array([[0, 1, 0], [1, 0, 0]])]
      generator = nlsg.NLStatsGenerator(
          schema=self._schema,
          vocab_paths={'my_vocab': vocab_file.name},
          num_quantiles_histogram_buckets=0,
          num_rank_histogram_buckets=0,
          num_histogram_buckets=3)
      expected_reported_sequences = [['Foo', 'Bar', 'Foo'],
                                     ['Bar', 'Foo', 'Foo']] * 2
      position_histogram_1 = statistics_pb2.Histogram()
      position_histogram_1.buckets.add(
          low_value=0, high_value=float(1) / 3, sample_count=1)
      position_histogram_1.buckets.add(
          low_value=float(1) / 3, high_value=float(2) / 3, sample_count=1)
      position_histogram_foo = statistics_pb2.Histogram()
      position_histogram_foo.buckets.add(
          low_value=0, high_value=float(1) / 3, sample_count=1)
      position_histogram_foo.buckets.add(
          low_value=float(1) / 3, high_value=float(2) / 3, sample_count=1)
      position_histogram_foo.buckets.add(
          low_value=float(2) / 3, high_value=1, sample_count=2)
      expected_token_stats = {
          1: (2, 1.0, 1, 1, 1, position_histogram_1),
          'Foo': (4, 1.0, 2, 2, 2, position_histogram_foo)
      }
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=1.0,
              avg_token_length=3,
              min_sequence_length=3,
              max_sequence_length=3,
              reported_sequences=expected_reported_sequences,
              token_statistics=expected_token_stats),
          self._int_nlp_feature_with_vocab_and_token_constraints_path)


if __name__ == '__main__':
  absltest.main()
