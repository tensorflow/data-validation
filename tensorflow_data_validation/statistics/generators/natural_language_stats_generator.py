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
"""Module that computes statistics for features of natural language type.

This module computes natural language statistics for features where the
natural_language_domain is specified. These statistics are stored as
custom_stats entries of the FeatureNameStatistics message corresponding to the
specified feature. We store a custom_stats called nl_statistics that contains
a populated tensorflow.metadata.v0.NaturalLanguageStatistics proto.

In addition, we store the following custom_stats so that they will be
automatically populated by Facets. (%T% in the names below denotes the token
name).

custom_stats name                    type             NaturalLanguageStatistics
-----------------                    -------------    -------------------------
nl_feature_coverage                  double           feature_coverage
nl_avg_token_length                  double           avg_token_length
nl_token_length_histogram            Histogram        token_length_histogram
nl_location_misses                   double           location_misses
nl_reported_sequences                string           Reported_sequences
                                                      (line-separated)
nl_rank_tokens                       RankHistogram    rank_histogram
nl_%T%_token_frequency               double           TokenStatistics
                                                      .frequency
nl_%T%_fraction_of_examples          double           TokenStatistics
                                                      .fraction_of_examples
nl_%T%_per_example_min_frequency     double           TokenStatistics
                                                      .per_example_min_frequency
nl_%T%_per_example_avg_frequency     double           TokenStatistics
                                                      .per_example_avg_frequency
nl_%T%_per_example_max_frequency     double           TokenStatistics
                                                      .per_example_max_frequency
nl_%T%_token_positions               Histogram        TokenStatistics.positions
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Dict, Iterable, List, Optional, Set, Text

import pyarrow as pa

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import vocab_util

from tfx_bsl import sketches

from google.protobuf import any_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

_NL_DOMAIN = 'natural_language_domain'
_NUM_MISRAGRIES_SKETCH_BUCKETS = 16384
_QUANTILES_SKETCH_ERROR = 0.01
_QUANTILES_SKETCH_NUM_ELEMENTS = 2 ^ 32
_QUANTILES_SKETCH_NUM_STREAMS = 1


# TODO(b/175875824): Determine if we should remove NL features from the default
# Top-K computation which is largely redundant.
class _PartialNLStats(object):
  """Partial feature stats for natural language."""

  def __init__(
      self,
      invalidate=False,
      num_in_vocab_tokens: int = 0,
      total_num_tokens: int = 0,
      sum_in_vocab_token_lengths: int = 0,
  ) -> None:
    # True only if this feature should never be considered, e.g: some
    # value_lists have inconsistent types or feature doesn't have an
    # NL domain.
    self.invalidate = invalidate
    self.num_in_vocab_tokens = num_in_vocab_tokens
    self.total_num_tokens = total_num_tokens
    self.sum_in_vocab_token_lengths = sum_in_vocab_token_lengths
    self.vocab_token_length_quantiles = sketches.QuantilesSketch(
        _QUANTILES_SKETCH_ERROR, _QUANTILES_SKETCH_NUM_ELEMENTS,
        _QUANTILES_SKETCH_NUM_STREAMS)
    self.token_occurrence_counts = sketches.MisraGriesSketch(
        _NUM_MISRAGRIES_SKETCH_BUCKETS)

  def __iadd__(self, other: '_PartialNLStats') -> '_PartialNLStats':
    """Merge two partial natual language stats."""
    self.invalidate |= other.invalidate

    self.num_in_vocab_tokens += other.num_in_vocab_tokens
    self.total_num_tokens += other.total_num_tokens
    self.sum_in_vocab_token_lengths += other.sum_in_vocab_token_lengths
    self.vocab_token_length_quantiles.Merge(other.vocab_token_length_quantiles)
    self.token_occurrence_counts.Merge(other.token_occurrence_counts)
    return self


def _update_accumulator_with_in_vocab_string_tokens(
    accumulator: _PartialNLStats, token_list: List[Text]):
  accumulator.num_in_vocab_tokens += len(token_list)
  accumulator.token_occurrence_counts.AddValues(pa.array(token_list))

  token_len_list = [len(t) for t in token_list]
  accumulator.sum_in_vocab_token_lengths += sum(token_len_list)
  accumulator.vocab_token_length_quantiles.AddValues(pa.array(token_len_list))


def _compute_int_listscalar_statistics(
    row: List[int], accumulator: _PartialNLStats,
    excluded_string_tokens: Set[Text], excluded_int_tokens: Set[int],
    oov_string_tokens: Set[Text], unused_vocab: Optional[Dict[Text, int]],
    rvocab: Optional[Dict[int, Text]]):
  """Compute statistics for an integer listscalar."""
  filtered_entry_str_list = []
  for entry in row:
    if entry in excluded_int_tokens:
      continue
    # Vocabulary exists.
    if rvocab is not None:
      if entry in rvocab:
        entry_str = rvocab[entry]
        if entry_str in excluded_string_tokens:
          continue
        if entry_str not in oov_string_tokens:
          filtered_entry_str_list.append(entry_str)
    accumulator.total_num_tokens += 1
  if filtered_entry_str_list:
    _update_accumulator_with_in_vocab_string_tokens(accumulator,
                                                    filtered_entry_str_list)


def _compute_str_listscalar_statistics(
    row: List[Text], accumulator: _PartialNLStats,
    excluded_string_tokens: Set[Text], excluded_int_tokens: Set[int],
    oov_string_tokens: Set[Text], vocab: Optional[Dict[Text, int]],
    unused_rvocab: Optional[Dict[int, Text]]):
  """Compute statistics for a string listscalar."""
  filtered_entry_list = []
  for entry in row:
    if entry in excluded_string_tokens:
      continue
    if (vocab is not None and entry in vocab and
        vocab[entry] in excluded_int_tokens):
      continue
    if entry not in oov_string_tokens:
      filtered_entry_list.append(entry)
    accumulator.total_num_tokens += 1
  if filtered_entry_list:
    _update_accumulator_with_in_vocab_string_tokens(accumulator,
                                                    filtered_entry_list)


def _populate_token_length_histogram(
    nls: statistics_pb2.NaturalLanguageStatistics, accumulator: _PartialNLStats,
    num_quantiles_histogram_buckets: int):
  """Populate the token length histogram."""
  quantiles = accumulator.vocab_token_length_quantiles.GetQuantiles(
      num_quantiles_histogram_buckets)
  quantiles = quantiles.flatten().to_pylist()

  if quantiles:
    quantiles_histogram = quantiles_util.generate_quantiles_histogram(
        quantiles, accumulator.num_in_vocab_tokens,
        num_quantiles_histogram_buckets)
    nls.token_length_histogram.CopyFrom(quantiles_histogram)


def _populate_token_rank_histogram(
    nls: statistics_pb2.NaturalLanguageStatistics, accumulator: _PartialNLStats,
    num_rank_histogram_buckets: int):
  """Populate the token rank histogram."""
  entries = accumulator.token_occurrence_counts.Estimate().to_pylist()
  for i, e in enumerate(entries[:num_rank_histogram_buckets]):
    nls.rank_histogram.buckets.add(
        low_rank=i, high_rank=i, label=e['values'], sample_count=e['counts'])


class NLStatsGenerator(stats_generator.CombinerFeatureStatsGenerator):
  """Generates feature level statistics for natural language stats.

  A combiner that computes statistics based on the specified
  natural_language_domain.
  """

  def __init__(self, schema: Optional[schema_pb2.Schema],
               vocab_paths: Optional[Dict[Text, Text]],
               num_quantiles_histogram_buckets: int,
               num_rank_histogram_buckets: int) -> None:
    """Initializes a NLStatsGenerator.

    Args:
      schema: An optional schema for the dataset.
      vocab_paths: A dictonary mapping vocab names to vocab paths.
      num_quantiles_histogram_buckets: Number of quantiles to use for
        histograms.
      num_rank_histogram_buckets: Number of buckets to allow for rank
        histograms.
    """
    self._schema = schema
    self._vocab_paths = vocab_paths
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    assert num_rank_histogram_buckets <= _NUM_MISRAGRIES_SKETCH_BUCKETS, (
        'num_rank_histogram_buckets cannot be greater than %d' %
        _NUM_MISRAGRIES_SKETCH_BUCKETS)
    self._num_rank_histogram_buckets = num_rank_histogram_buckets
    self._nld_vocabularies = {}
    self._nld_excluded_string_tokens = {}
    self._nld_excluded_int_tokens = {}
    self._nld_oov_string_tokens = {}
    self._vocabs = {}
    self._rvocabs = {}
    self._feature_type_fns = {
        statistics_pb2.FeatureNameStatistics.INT:
            _compute_int_listscalar_statistics,
        statistics_pb2.FeatureNameStatistics.STRING:
            _compute_str_listscalar_statistics
    }
    self._valid_feature_paths = set()

  def setup(self) -> None:
    """Prepares an instance for combining."""
    if self._schema is not None:
      for k, v in schema_util.get_all_leaf_features(self._schema):
        if v.WhichOneof('domain_info') == _NL_DOMAIN:
          nld = v.natural_language_domain
          self._nld_vocabularies[k] = nld.vocabulary
          coverage_constraints = nld.coverage
          self._nld_excluded_string_tokens[k] = set(
              coverage_constraints.excluded_string_tokens)
          self._nld_excluded_int_tokens[k] = set(
              coverage_constraints.excluded_int_tokens)
          self._nld_oov_string_tokens[k] = set(
              coverage_constraints.oov_string_tokens)
          if (self._nld_vocabularies[k] or
              self._nld_excluded_string_tokens[k] or
              self._nld_excluded_int_tokens[k] or
              self._nld_oov_string_tokens[k]):
            self._valid_feature_paths.add(k)

    if self._vocab_paths is not None:
      for k, v in self._vocab_paths.items():
        self._vocabs[k], self._rvocabs[k] = vocab_util.load_vocab(v)

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
      feature_array: An arrow Array representing a batch of feature values which
        should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    if feature_path not in self._valid_feature_paths:
      accumulator.invalidate = True
      return accumulator

    feature_type = stats_util.get_feature_type_from_arrow_type(
        feature_path, feature_array.type)
    # Ignore null array.
    if feature_type is None:
      return accumulator

    if feature_type not in self._feature_type_fns:
      accumulator.invalidate = True
      return accumulator

    feature_type_fn = self._feature_type_fns[feature_type]

    vocab = None
    rvocab = None
    if self._nld_vocabularies[feature_path]:
      vocab_name = self._nld_vocabularies[feature_path]
      vocab = self._vocabs[vocab_name]
      rvocab = self._rvocabs[vocab_name]

    excluded_string_tokens = self._nld_excluded_string_tokens[feature_path]
    excluded_int_tokens = self._nld_excluded_int_tokens[feature_path]
    oov_string_tokens = self._nld_oov_string_tokens[feature_path]

    # TODO(b/175875824): Benchmark and optimize performance.
    for row in feature_array.to_pylist():
      if row is not None:
        feature_type_fn(row, accumulator, excluded_string_tokens,
                        excluded_int_tokens, oov_string_tokens, vocab, rvocab)
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_PartialNLStats]) -> _PartialNLStats:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    result = _PartialNLStats()
    for accumulator in accumulators:
      result += accumulator
    return result

  def compact(self, accumulator: _PartialNLStats) -> _PartialNLStats:
    accumulator.vocab_token_length_quantiles.Compact()
    return accumulator

  def extract_output(
      self,
      accumulator: _PartialNLStats) -> statistics_pb2.FeatureNameStatistics:
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.FeatureNameStatistics()
    if accumulator.invalidate:
      return result

    nls = statistics_pb2.NaturalLanguageStatistics()
    if accumulator.total_num_tokens:
      nls.feature_coverage = (
          float(accumulator.num_in_vocab_tokens) / accumulator.total_num_tokens)
      result.custom_stats.add(
          name='nl_feature_coverage', num=nls.feature_coverage)
    if accumulator.num_in_vocab_tokens:
      nls.avg_token_length = (
          float(accumulator.sum_in_vocab_token_lengths) /
          accumulator.num_in_vocab_tokens)
      result.custom_stats.add(
          name='nl_avg_token_length', num=nls.avg_token_length)
    if self._num_quantiles_histogram_buckets:
      _populate_token_length_histogram(nls, accumulator,
                                       self._num_quantiles_histogram_buckets)
      if nls.token_length_histogram.buckets:
        result.custom_stats.add(
            name='nl_token_length_histogram',
            histogram=nls.token_length_histogram)
    if self._num_rank_histogram_buckets:
      _populate_token_rank_histogram(nls, accumulator,
                                     self._num_rank_histogram_buckets)
      if nls.rank_histogram.buckets:
        result.custom_stats.add(
            name='nl_rank_tokens', rank_histogram=nls.rank_histogram)
    my_proto = any_pb2.Any()
    result.custom_stats.add(name='nl_statistics', any=my_proto.Pack(nls))
    return result
