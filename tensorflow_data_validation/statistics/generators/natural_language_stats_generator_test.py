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

from google.protobuf import any_pb2
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
    self._int_nlp_feature_no_vocab_path = types.FeaturePath(
        ['int_nlp_feature_no_vocab'])
    self._int_nlp_feature_empty_domain = types.FeaturePath(
        ['int_nlp_feature_empty_domain'])
    self._non_nlp_feature_path = types.FeaturePath(['non_nlp_feature'])

  def test_partial_stats_iadd(self):
    stats = nlsg._PartialNLStats(
        invalidate=False,
        num_in_vocab_tokens=2,
        total_num_tokens=3,
        sum_in_vocab_token_lengths=4)
    stats_2 = nlsg._PartialNLStats(
        invalidate=False,
        num_in_vocab_tokens=7,
        total_num_tokens=10,
        sum_in_vocab_token_lengths=11)

    stats += stats_2
    self.assertEqual(9, stats.num_in_vocab_tokens)
    self.assertEqual(13, stats.total_num_tokens)
    self.assertEqual(15, stats.sum_in_vocab_token_lengths)
    self.assertEqual(False, stats.invalidate)

  def _create_expected_feature_name_statistics(self,
                                               feature_coverage=None,
                                               avg_token_length=None):
    custom_stats = []
    nls = statistics_pb2.NaturalLanguageStatistics()
    if feature_coverage is not None:
      nls.feature_coverage = feature_coverage
      custom_stats.append(
          statistics_pb2.CustomStatistic(
              name='nl_feature_coverage', num=feature_coverage))
    if avg_token_length is not None:
      nls.avg_token_length = avg_token_length
      custom_stats.append(
          statistics_pb2.CustomStatistic(
              name='nl_avg_token_length', num=avg_token_length))
    my_proto = any_pb2.Any()
    custom_stats.append(
        statistics_pb2.CustomStatistic(
            name='nl_statistics', any=my_proto.Pack(nls)))
    return statistics_pb2.FeatureNameStatistics(custom_stats=custom_stats)

  def test_nl_generator_empty_input(self):
    generator = nlsg.NLStatsGenerator()
    self.assertCombinerOutputEqual(
        [], generator, self._create_expected_feature_name_statistics())

  def test_nl_generator_invalidation_check_no_nld(self):
    """Tests generator invalidation with no natural language domain."""
    generator = nlsg.NLStatsGenerator(schema=self._schema)
    generator.setup()
    accumulator = generator.create_accumulator()
    self.assertFalse(accumulator.invalidate)
    valid_input = pa.array([['Foo'], ['Bar']])
    accumulator = generator.add_input(accumulator, self._non_nlp_feature_path,
                                      valid_input)
    self.assertTrue(accumulator.invalidate)

  def test_nl_generator_invalidation_check_empty_nld(self):
    """Tests generator invalidation whith empty natural language domain."""
    generator = nlsg.NLStatsGenerator(schema=self._schema)
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
    generator = nlsg.NLStatsGenerator(schema=self._schema)
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
    input_batches = [pa.array([['Foo'], None, ['Baz']])]
    generator = nlsg.NLStatsGenerator(schema=self._schema)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        self._create_expected_feature_name_statistics(
            feature_coverage=0.5, avg_token_length=3.0),
        self._string_nlp_feature_no_vocab_path)

  def test_nl_generator_string_feature_vocab(self):
    """Tests generator calculation with a string domain having a vocab."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\nBazz\n')
      vocab_file.flush()

      input_batches = [pa.array([['Bar'], None, ['Bazz']])]
      generator = nlsg.NLStatsGenerator(
          vocab_paths={'my_vocab': vocab_file.name}, schema=self._schema)
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=1.0, avg_token_length=4.0),
          self._string_nlp_feature_with_vocab_path)

  def test_nl_generator_int_feature_no_vocab(self):
    """Tests generator calculation with a int domain having no vocab."""
    input_batches = [pa.array([[1], [2], [3]])]
    generator = nlsg.NLStatsGenerator(schema=self._schema)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        self._create_expected_feature_name_statistics(feature_coverage=0.0),
        self._int_nlp_feature_no_vocab_path)

  def test_nl_generator_int_feature_vocab(self):
    """Tests generator calcualtion with an int domain and a vocab."""
    with tempfile.NamedTemporaryFile() as vocab_file:
      vocab_file.write(b'Foo\nBar\nBaz\nBazz\n')
      vocab_file.flush()
      input_batches = [pa.array([[0], [1], [2], [3], [4]])]
      generator = nlsg.NLStatsGenerator(
          vocab_paths={'my_vocab': vocab_file.name}, schema=self._schema)
      self.assertCombinerOutputEqual(
          input_batches, generator,
          self._create_expected_feature_name_statistics(
              feature_coverage=float(1) / 3, avg_token_length=4.0),
          self._int_nlp_feature_with_vocab_path)


if __name__ == '__main__':
  absltest.main()
