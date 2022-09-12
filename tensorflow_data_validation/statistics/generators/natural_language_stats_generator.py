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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Dict, Iterable, List, Optional, Set, Text, Union

import pyarrow as pa
import six

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import quantiles_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.utils import vocab_util

from tfx_bsl import sketches

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order

_NL_DOMAIN = 'natural_language_domain'
_INT_VALUE = 'int_value'

_NUM_MISRAGRIES_SKETCH_BUCKETS = 16384
_QUANTILES_SKETCH_ERROR = 0.01
_QUANTILES_SKETCH_NUM_ELEMENTS = 2 ^ 32
_QUANTILES_SKETCH_NUM_STREAMS = 1
_NUM_REPORTED_SEQUENCES_PER_TYPE = 5


_ReportedSequence = tfx_namedtuple.namedtuple(
    '_ReportedSequence', ['sequence', 'hash_value', 'metric'])


def _sort_and_truncate_reported_sequence(sequence: List[_ReportedSequence]):
  sequence.sort(key=lambda x: x.metric)
  deduped_values = []
  hash_values = set()
  for s in sequence:
    if s.hash_value in hash_values:
      continue
    hash_values.add(s.hash_value)
    deduped_values.append(s)
  return deduped_values[:_NUM_REPORTED_SEQUENCES_PER_TYPE]


class _TokenStats(object):
  """Tracks statistics for individual tokens."""

  def __init__(self):
    self.frequency = 0
    self.num_sequences = 0
    self.per_sequence_min_frequency = None
    self.per_sequence_max_frequency = None
    self.positions = collections.Counter()

  def __iadd__(self, other: '_TokenStats') -> '_TokenStats':
    """Merge two _TokenStats."""
    self.frequency += other.frequency
    self.num_sequences += other.num_sequences
    for attr, fn in [('per_sequence_min_frequency', min),
                     ('per_sequence_max_frequency', max)]:
      self_freq = getattr(self, attr)
      other_freq = getattr(other, attr)
      if (self_freq is not None and other_freq is not None):
        setattr(self, attr, fn(self_freq, other_freq))
      elif self_freq is None:
        setattr(self, attr, other_freq)
    self.positions += other.positions
    return self


# TODO(b/175875824): Determine if we should remove NL features from the default
# Top-K computation which is largely redundant.
class _PartialNLStats(object):
  """Partial feature stats for natural language."""

  def __init__(self,
               invalidate=False,
               num_in_vocab_tokens: int = 0,
               total_num_tokens: int = 0,
               sum_in_vocab_token_lengths: int = 0,
               num_examples: int = 0) -> None:
    # True only if this feature should never be considered, e.g: some
    # value_lists have inconsistent types or feature doesn't have an
    # NL domain.
    self.invalidate = invalidate
    self.num_in_vocab_tokens = num_in_vocab_tokens
    self.total_num_tokens = total_num_tokens
    self.sum_in_vocab_token_lengths = sum_in_vocab_token_lengths
    self.num_examples = num_examples
    self.vocab_token_length_quantiles = sketches.QuantilesSketch(
        _QUANTILES_SKETCH_ERROR, _QUANTILES_SKETCH_NUM_ELEMENTS,
        _QUANTILES_SKETCH_NUM_STREAMS)
    self.min_sequence_length = None
    self.max_sequence_length = None
    self.sequence_length_quantiles = sketches.QuantilesSketch(
        _QUANTILES_SKETCH_ERROR, _QUANTILES_SKETCH_NUM_ELEMENTS,
        _QUANTILES_SKETCH_NUM_STREAMS)
    self.token_occurrence_counts = sketches.MisraGriesSketch(
        _NUM_MISRAGRIES_SKETCH_BUCKETS)
    self.token_statistics = collections.defaultdict(_TokenStats)
    self.reported_sequences_coverage = []
    self.reported_sequences_avg_token_length = []

  def __iadd__(self, other: '_PartialNLStats') -> '_PartialNLStats':
    """Merge two partial natual language stats."""
    self.invalidate |= other.invalidate

    self.num_in_vocab_tokens += other.num_in_vocab_tokens
    self.total_num_tokens += other.total_num_tokens
    self.sum_in_vocab_token_lengths += other.sum_in_vocab_token_lengths
    self.num_examples += other.num_examples
    self.vocab_token_length_quantiles.Merge(other.vocab_token_length_quantiles)
    if self.min_sequence_length is None:
      self.min_sequence_length = other.min_sequence_length
    elif other.min_sequence_length is not None:
      self.min_sequence_length = min(self.min_sequence_length,
                                     other.min_sequence_length)
    if self.max_sequence_length is None:
      self.max_sequence_length = other.max_sequence_length
    elif other.max_sequence_length is not None:
      self.max_sequence_length = max(self.max_sequence_length,
                                     other.max_sequence_length)

    self.sequence_length_quantiles.Merge(other.sequence_length_quantiles)
    self.token_occurrence_counts.Merge(other.token_occurrence_counts)

    for t in other.token_statistics:
      if t not in self.token_statistics:
        self.token_statistics[t] = other.token_statistics[t]
      else:
        self.token_statistics[t] += other.token_statistics[t]

    for list_name in [
        'reported_sequences_coverage', 'reported_sequences_avg_token_length'
    ]:
      cur_list = getattr(self, list_name)
      cur_list += getattr(other, list_name)
      cur_list = _sort_and_truncate_reported_sequence(cur_list)
      setattr(self, list_name, cur_list)
    return self


def _update_accumulator_with_in_vocab_string_tokens(
    accumulator: _PartialNLStats, token_list: List[Text]):
  accumulator.num_in_vocab_tokens += len(token_list)
  accumulator.token_occurrence_counts.AddValues(pa.array(token_list))

  token_len_list = [len(t) for t in token_list]
  accumulator.sum_in_vocab_token_lengths += sum(token_len_list)
  accumulator.vocab_token_length_quantiles.AddValues(pa.array(token_len_list))


def _update_accumulator_with_token_statistics(accumulator: _PartialNLStats,
                                              row: List[Union[int, Text]],
                                              tokens: Union[Set[int],
                                                            Set[Text]],
                                              num_histogram_buckets):
  """Compute token statistics for a specific row."""
  for t in tokens:
    norm_indices = [float(i) / len(row) for i, v in enumerate(row) if v == t]
    num_occur = len(norm_indices)
    accumulator.token_statistics[t].frequency += num_occur
    accumulator.token_statistics[t].num_sequences += (1 if num_occur else 0)
    for attr, fn in [('per_sequence_min_frequency', min),
                     ('per_sequence_max_frequency', max)]:
      accum_freq = getattr(accumulator.token_statistics[t], attr)
      if accum_freq is not None:
        setattr(accumulator.token_statistics[t], attr,
                fn(accum_freq, num_occur))
      else:
        setattr(accumulator.token_statistics[t], attr, num_occur)
    for i in norm_indices:
      accumulator.token_statistics[t].positions[int(i *
                                                    num_histogram_buckets)] += 1


def _update_accumulator_reported_sequences(accumulator: _PartialNLStats,
                                           resolved_entry: List[Union[Text,
                                                                      int]],
                                           oov_string_tokens: Set[Text]):
  """Update reported sequences in accumulator."""
  token_lens = [
      len(i) for i in resolved_entry
      if (isinstance(i, str) and i not in oov_string_tokens)
  ]

  coverage = (float(len(token_lens)) / len(resolved_entry))
  if token_lens:
    avg_token_len = float(sum(token_lens)) / len(token_lens)
  else:
    avg_token_len = 0

  for attr, metric in [('reported_sequences_coverage', coverage),
                       ('reported_sequences_avg_token_length', avg_token_len)]:
    cur_list = getattr(accumulator, attr)
    cur_list.append(
        _ReportedSequence(
            sequence=resolved_entry,
            hash_value=hash(str(resolved_entry)),
            metric=metric))
    cur_list = _sort_and_truncate_reported_sequence(cur_list)
    setattr(accumulator, attr, cur_list)


def _update_accumulator_with_sequence_lengths(
    accumulator: _PartialNLStats, sequence_length_excluded_int_tokens: Set[int],
    sequence_length_excluded_string_tokens: Set[Text], max_sequence_length: int,
    int_row: Optional[List[Union[int, Text]]],
    string_row: Optional[List[Union[Text, int]]]):
  """Update sequence length quantiles in accumulator.

  We expect that int_row and string row preserve the position of the the token
  within the seqence and hence allow the lists to contain both ints and strings.

  Args:
    accumulator: The accumulator to update.
    sequence_length_excluded_int_tokens: The int tokens to not consider when
      calculating the length.
    sequence_length_excluded_string_tokens: The string tokens to not consider
      when calculating the length.
    max_sequence_length: The max sequence length to use if no excluded tokens
      are present.
    int_row: The row of integer tokens. Note: the row can include strings if
      there is an incomplete mapping from strings to ints (this preserves the
      position).
    string_row: The row of string tokens. Note: the row can include ints if if
      there is an incomplete mapping from ints to strings (this preserves the
      position).
  """
  sequence_length = max_sequence_length
  if int_row is not None:
    matches = [e for e in int_row if e in sequence_length_excluded_int_tokens]
    sequence_length -= len(matches)
  if string_row is not None:
    matches = [
        e for e in string_row if e in sequence_length_excluded_string_tokens
    ]
    sequence_length -= len(matches)
  accumulator.sequence_length_quantiles.AddValues(pa.array([sequence_length]))
  accumulator.min_sequence_length = (
      sequence_length if not accumulator.min_sequence_length else min(
          accumulator.min_sequence_length, sequence_length))
  accumulator.max_sequence_length = (
      sequence_length if not accumulator.max_sequence_length else max(
          accumulator.max_sequence_length, sequence_length))


def _compute_int_statistics(
    row: List[int], accumulator: _PartialNLStats,
    excluded_string_tokens: Set[Text], excluded_int_tokens: Set[int],
    oov_string_tokens: Set[Text], unused_vocab: Optional[Dict[Text, int]],
    rvocab: Optional[Dict[int, Text]], int_tokens: Set[int],
    string_tokens: Set[Text], sequence_length_excluded_int_tokens: Set[int],
    sequence_length_excluded_string_tokens: Set[Text],
    num_histogram_buckets: int):
  """Compute statistics for an integer entry."""
  accumulator.num_examples += 1
  if row:
    _update_accumulator_with_token_statistics(accumulator, row, int_tokens,
                                              num_histogram_buckets)
    string_row = None
    if rvocab:
      string_row = [rvocab.get(r, r) for r in row]
      _update_accumulator_with_token_statistics(accumulator, string_row,
                                                string_tokens,
                                                num_histogram_buckets)

    _update_accumulator_reported_sequences(accumulator,
                                           string_row if string_row else row,
                                           oov_string_tokens)
    _update_accumulator_with_sequence_lengths(
        accumulator, sequence_length_excluded_int_tokens,
        sequence_length_excluded_string_tokens, len(row), row, string_row)

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


def _compute_str_statistics(
    row: List[Text], accumulator: _PartialNLStats,
    excluded_string_tokens: Set[Text], excluded_int_tokens: Set[int],
    oov_string_tokens: Set[Text], vocab: Optional[Dict[Text, int]],
    unused_rvocab: Optional[Dict[int, Text]], int_tokens: Set[int],
    string_tokens: Set[Text], sequence_length_excluded_int_tokens: Set[int],
    sequence_length_excluded_string_tokens: Set[Text], num_histogram_buckets):
  """Compute statistics for string features."""
  accumulator.num_examples += 1
  row = [six.ensure_text(e) for e in row]
  if row:
    _update_accumulator_with_token_statistics(accumulator, row, string_tokens,
                                              num_histogram_buckets)
    _update_accumulator_reported_sequences(accumulator, row, oov_string_tokens)
    int_row = None
    if vocab:
      int_row = [vocab.get(r, r) for r in row]
      _update_accumulator_with_token_statistics(accumulator, int_row,
                                                int_tokens,
                                                num_histogram_buckets)
    _update_accumulator_with_sequence_lengths(
        accumulator, sequence_length_excluded_int_tokens,
        sequence_length_excluded_string_tokens, len(row), int_row, row)

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
  quantiles, weights = (
      accumulator.vocab_token_length_quantiles.GetQuantilesAndCumulativeWeights(
          num_quantiles_histogram_buckets))
  quantiles = quantiles.flatten().to_numpy(zero_copy_only=False)
  weights = weights.flatten().to_numpy(zero_copy_only=False)
  if quantiles.size:
    quantiles_histogram = quantiles_util.generate_quantiles_histogram(
        quantiles, weights)
    nls.token_length_histogram.CopyFrom(quantiles_histogram)


def _populate_sequence_length_histogram(
    nls: statistics_pb2.NaturalLanguageStatistics, accumulator: _PartialNLStats,
    num_quantiles_histogram_buckets: int):
  """Populate sequence length histogram."""

  quantiles, weights = (
      accumulator.sequence_length_quantiles.GetQuantilesAndCumulativeWeights(
          num_quantiles_histogram_buckets))
  quantiles = quantiles.flatten().to_numpy(zero_copy_only=False)
  weights = weights.flatten().to_numpy(zero_copy_only=False)

  if quantiles.size:
    quantiles_histogram = quantiles_util.generate_quantiles_histogram(
        quantiles, weights)
    nls.sequence_length_histogram.CopyFrom(quantiles_histogram)


def _populate_token_rank_histogram(
    nls: statistics_pb2.NaturalLanguageStatistics, accumulator: _PartialNLStats,
    num_rank_histogram_buckets: int):
  """Populate the token rank histogram."""
  entries = accumulator.token_occurrence_counts.Estimate().to_pylist()
  for i, e in enumerate(entries[:num_rank_histogram_buckets]):
    nls.rank_histogram.buckets.add(
        low_rank=i, high_rank=i, label=e['values'], sample_count=e['counts'])


def _populate_token_position_histogram(
    token_proto: statistics_pb2.NaturalLanguageStatistics.TokenStatistics,
    stats: _TokenStats, num_histogram_buckets: int):
  """Populate the token position histogram."""
  positions = list(stats.positions.items())
  positions.sort(key=lambda x: x[0])
  for k, v in positions:
    low_value = float(k) / num_histogram_buckets
    high_value = float(k + 1) / num_histogram_buckets
    token_proto.positions.buckets.add(
        low_value=low_value, high_value=high_value, sample_count=v)


def _populate_token_statistics(
    name: Text,
    num_histogram_buckets: int,
    num_examples: int,
    token_proto: statistics_pb2.NaturalLanguageStatistics.TokenStatistics,
    stats: _TokenStats):
  """Populates the token statistics for a specified token."""
  if isinstance(name, int):
    token_proto.int_token = name
  else:
    token_proto.string_token = name
  if stats.num_sequences:
    token_proto.frequency = stats.frequency
    token_proto.fraction_of_sequences = (
        float(stats.num_sequences) / num_examples)
    token_proto.per_sequence_min_frequency = stats.per_sequence_min_frequency
    token_proto.per_sequence_max_frequency = stats.per_sequence_max_frequency
    token_proto.per_sequence_avg_frequency = (
        float(stats.frequency) / stats.num_sequences)
    _populate_token_position_histogram(token_proto, stats,
                                       num_histogram_buckets)


class NLStatsGenerator(stats_generator.CombinerFeatureStatsGenerator):
  """Generates feature level statistics for natural language stats.

  A combiner that computes statistics based on the specified
  natural_language_domain.
  """

  def __init__(self, schema: Optional[schema_pb2.Schema],
               vocab_paths: Optional[Dict[Text, Text]],
               num_histogram_buckets: int, num_quantiles_histogram_buckets: int,
               num_rank_histogram_buckets: int) -> None:
    """Initializes a NLStatsGenerator.

    Args:
      schema: An optional schema for the dataset.
      vocab_paths: A dictonary mapping vocab names to vocab paths.
      num_histogram_buckets: Number of buckets to use for histograms.
      num_quantiles_histogram_buckets: Number of quantiles to use for
        histograms.
      num_rank_histogram_buckets: Number of buckets to allow for rank
        histograms.
    """
    self._schema = schema
    self._vocab_paths = vocab_paths
    self._num_histogram_buckets = num_histogram_buckets
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    assert num_rank_histogram_buckets <= _NUM_MISRAGRIES_SKETCH_BUCKETS, (
        'num_rank_histogram_buckets cannot be greater than %d' %
        _NUM_MISRAGRIES_SKETCH_BUCKETS)
    self._num_rank_histogram_buckets = num_rank_histogram_buckets
    self._nld_vocabularies = {}
    self._nld_excluded_string_tokens = {}
    self._nld_excluded_int_tokens = {}
    self._nld_oov_string_tokens = {}
    self._nld_specified_int_tokens = collections.defaultdict(set)
    self._nld_specified_str_tokens = collections.defaultdict(set)
    self._nld_sequence_length_excluded_int_tokens = {}
    self._nld_sequence_length_excluded_string_tokens = {}
    self._vocabs = {}
    self._rvocabs = {}
    self._feature_type_fns = {
        statistics_pb2.FeatureNameStatistics.INT: _compute_int_statistics,
        statistics_pb2.FeatureNameStatistics.STRING: _compute_str_statistics
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
          sequence_length_constraints = nld.sequence_length_constraints
          self._nld_sequence_length_excluded_int_tokens[k] = set(
              sequence_length_constraints.excluded_int_value)
          self._nld_sequence_length_excluded_string_tokens[k] = set(
              sequence_length_constraints.excluded_string_value)
          if (self._nld_vocabularies[k] or
              self._nld_excluded_string_tokens[k] or
              self._nld_excluded_int_tokens[k] or
              self._nld_oov_string_tokens[k]):
            self._valid_feature_paths.add(k)
          for t in nld.token_constraints:
            if t.WhichOneof('value') == _INT_VALUE:
              self._nld_specified_int_tokens[k].add(t.int_value)
            else:
              self._nld_specified_str_tokens[k].add(t.string_value)

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
    int_tokens = self._nld_specified_int_tokens[feature_path]
    string_tokens = self._nld_specified_str_tokens[feature_path]
    sequence_length_excluded_int_tokens = (
        self._nld_sequence_length_excluded_int_tokens[feature_path])
    sequence_length_excluded_string_tokens = (
        self._nld_sequence_length_excluded_string_tokens[feature_path])

    # TODO(b/175875824): Benchmark and optimize performance.
    for row in feature_array.to_pylist():
      if row is not None:
        feature_type_fn(row, accumulator, excluded_string_tokens,
                        excluded_int_tokens, oov_string_tokens, vocab, rvocab,
                        int_tokens, string_tokens,
                        sequence_length_excluded_int_tokens,
                        sequence_length_excluded_string_tokens,
                        self._num_histogram_buckets)
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

  def compact(self, accumulator: _PartialNLStats) -> _PartialNLStats:
    accumulator.vocab_token_length_quantiles.Compact()
    accumulator.sequence_length_quantiles.Compact()
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
    if accumulator.num_in_vocab_tokens:
      nls.avg_token_length = (
          float(accumulator.sum_in_vocab_token_lengths) /
          accumulator.num_in_vocab_tokens)
    if accumulator.min_sequence_length:
      nls.min_sequence_length = accumulator.min_sequence_length
    if accumulator.max_sequence_length:
      nls.max_sequence_length = accumulator.max_sequence_length
    if self._num_quantiles_histogram_buckets:
      _populate_token_length_histogram(nls, accumulator,
                                       self._num_quantiles_histogram_buckets)
      _populate_sequence_length_histogram(nls, accumulator,
                                          self._num_quantiles_histogram_buckets)
    if self._num_rank_histogram_buckets:
      _populate_token_rank_histogram(nls, accumulator,
                                     self._num_rank_histogram_buckets)
    if accumulator.token_statistics:
      for name, stats in accumulator.token_statistics.items():
        _populate_token_statistics(name, self._num_histogram_buckets,
                                   accumulator.num_examples,
                                   nls.token_statistics.add(), stats)

    for r in (accumulator.reported_sequences_coverage +
              accumulator.reported_sequences_avg_token_length):
      str_seq = str(r[0])
      nls.reported_sequences.append(str_seq)
    custom_nl_stats = result.custom_stats.add(name='nl_statistics')
    custom_nl_stats.any.Pack(nls)
    return result
