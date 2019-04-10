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

"""Implementation of statistics generators."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import itertools

import apache_beam as beam
import six
from six.moves import zip
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.statistics.generators import image_stats_generator
from tensorflow_data_validation.statistics.generators import natural_language_stats_generator
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators import time_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_combiner_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_stats_generator

from tensorflow_data_validation.utils import batch_util
from tensorflow_data_validation.utils import slicing_util
from tensorflow_data_validation.types_compat import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


@beam.typehints.with_input_types(types.BeamExample)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators over input examples."""

  def __init__(
      self,
      options = stats_options.StatsOptions()
      ):
    self._options = options

  def expand(self, dataset):
    # If a set of whitelist features are provided, keep only those features.
    if self._options.feature_whitelist:
      dataset |= ('RemoveNonWhitelistedFeatures' >> beam.Map(
          _filter_features, feature_whitelist=self._options.feature_whitelist))

    if self._options.slice_functions:
      # Add default slicing function.
      slice_functions = [slicing_util.default_slicer]
      slice_functions.extend(self._options.slice_functions)
      dataset = (
          dataset
          | 'GenerateSliceKeys' >> beam.FlatMap(
              slicing_util.generate_slices, slice_functions=slice_functions))
    else:
      # TODO(pachristopher): Remove this special case if this doesn't give any
      # performance improvement.
      dataset = (dataset
                 | 'KeyWithVoid' >> beam.Map(lambda v: (None, v)))

    return dataset | GenerateSlicedStatisticsImpl(self._options)


# This transform will be used by the example validation API to compute
# statistics over anomalous examples. Specifically, it is used to compute
# statistics over examples found for each anomaly (i.e., the anomaly type
# will be the slice key).
@beam.typehints.with_input_types(types.SlicedExample)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateSlicedStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators to sliced input examples."""

  def __init__(
      self,
      options = stats_options.StatsOptions(),
      is_slicing_enabled = False,
      ):
    """Initializes GenerateSlicedStatisticsImpl.

    Args:
      options: `tfdv.StatsOptions` for generating data statistics.
      is_slicing_enabled: Whether to include slice keys in the resulting proto,
        even if slice functions are not provided in `options`. If slice
        functions are provided in `options`, slice keys are included regardless
        of this value.
    """
    self._options = options
    self._is_slicing_enabled = (
        is_slicing_enabled or self._options.slice_functions is not None)

  def expand(self, dataset):
    # Handles generators by their type:
    #   - CombinerStatsGenerators will be wrapped in a single CombinePerKey by
    #     _CombinerStatsGeneratorsCombineFn.
    #   - TransformStatsGenerator will be invoked separately with `dataset`.
    combiner_stats_generators = []
    result_protos = []
    for generator in get_generators(self._options):
      if isinstance(generator, stats_generator.CombinerStatsGenerator):
        combiner_stats_generators.append(generator)
      elif isinstance(generator, stats_generator.TransformStatsGenerator):
        result_protos.append(
            dataset
            | generator.name >> generator.ptransform)
      else:
        raise TypeError('Statistics generator must extend one of '
                        'CombinerStatsGenerator or TransformStatsGenerator, '
                        'found object of type %s' %
                        generator.__class__.__name__)
    if combiner_stats_generators:
      # TODO(b/115685296): Obviate the need for the fanout=8 workaround.
      result_protos.append(dataset
                           | 'RunCombinerStatsGenerators'
                           >> beam.CombinePerKey(
                               _CombinerStatsGeneratorsCombineFn(
                                   combiner_stats_generators,
                                   self._options.desired_batch_size
                                   )).with_hot_key_fanout(fanout=8))

    # result_protos is a list of PCollections of (slice key,
    # DatasetFeatureStatistics proto) pairs. We now flatten the list into a
    # single PCollection, combine the DatasetFeatureStatistics protos by key,
    # and then merge the DatasetFeatureStatistics protos in the PCollection into
    # a single DatasetFeatureStatisticsList proto.
    return (result_protos
            | 'FlattenFeatureStatistics' >> beam.Flatten()
            | 'MergeDatasetFeatureStatisticsProtos' >>
            beam.CombinePerKey(_merge_dataset_feature_stats_protos)
            | 'AddSliceKeyToStatsProto' >> beam.Map(
                _add_slice_key,
                self._is_slicing_enabled)
            | 'ToList' >> beam.combiners.ToList()
            | 'MakeDatasetFeatureStatisticsListProto' >>
            beam.Map(_make_dataset_feature_statistics_list_proto))


def get_generators(options,
                   in_memory = False
                  ):
  """Initializes the list of stats generators, including custom generators.

  Args:
    options: A StatsOptions object.
    in_memory: Whether the generators will be used to generate statistics in
      memory (True) or using Beam (False).

  Returns:
    A list of stats generator objects.
  """
  generators = _get_default_generators(options, in_memory)
  if options.generators is not None:
    # Add custom stats generators.
    generators.extend(options.generators)
  if options.enable_semantic_domain_stats:
    generators += [
        image_stats_generator.ImageStatsGenerator(),
        natural_language_stats_generator.NLStatsGenerator(),
        time_stats_generator.TimeStatsGenerator(),
    ]
  # Replace all CombinerFeatureStatsGenerator with a single
  # CombinerFeatureStatsWrapperGenerator.
  feature_generators = [
      x for x in generators
      if isinstance(x, stats_generator.CombinerFeatureStatsGenerator)
  ]
  if feature_generators:
    generators = [
        x for x in generators
        if not isinstance(x, stats_generator.CombinerFeatureStatsGenerator)
    ] + [
        CombinerFeatureStatsWrapperGenerator(
            feature_generators, weight_feature=options.weight_feature)
    ]
  if in_memory:
    for generator in generators:
      if not isinstance(generator, stats_generator.CombinerStatsGenerator):
        raise TypeError('Statistics generator used in '
                        'generate_statistics_in_memory must '
                        'extend CombinerStatsGenerator, found object of type '
                        '%s.' % generator.__class__.__name__)
  return generators


def _get_default_generators(
    options, in_memory = False
):
  """Initializes default list of stats generators.

  Args:
    options: A StatsOptions object.
    in_memory: Whether the generators will be used to generate statistics in
      memory (True) or using Beam (False).

  Returns:
    A list of stats generator objects.
  """
  stats_generators = [
      basic_stats_generator.BasicStatsGenerator(
          schema=options.schema,
          weight_feature=options.weight_feature,
          num_values_histogram_buckets=options.num_values_histogram_buckets,
          num_histogram_buckets=options.num_histogram_buckets,
          num_quantiles_histogram_buckets=\
            options.num_quantiles_histogram_buckets,
          epsilon=options.epsilon)
  ]
  if in_memory:
    stats_generators.append(
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            schema=options.schema,
            weight_feature=options.weight_feature,
            num_top_values=options.num_top_values,
            frequency_threshold=options.frequency_threshold,
            weighted_frequency_threshold=options.weighted_frequency_threshold,
            num_rank_histogram_buckets=options.num_rank_histogram_buckets))
  else:
    stats_generators.extend([
        top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
            schema=options.schema,
            weight_feature=options.weight_feature,
            num_top_values=options.num_top_values,
            frequency_threshold=options.frequency_threshold,
            weighted_frequency_threshold=options.weighted_frequency_threshold,
            num_rank_histogram_buckets=options.num_rank_histogram_buckets),
    ])
  return stats_generators


def _filter_features(
    example,
    feature_whitelist):
  """Removes features that are not whitelisted.

  Args:
    example: Input example.
    feature_whitelist: A list of feature names to whitelist.

  Returns:
    An example containing only the whitelisted features of the input example.
  """
  return {
      feature_name: example[feature_name]
      for feature_name in feature_whitelist
      if feature_name in example
  }


def _add_slice_key(
    stats_proto_per_slice,
    is_slicing_enabled
):
  """Add slice key to stats proto."""
  result = statistics_pb2.DatasetFeatureStatistics()
  result.CopyFrom(stats_proto_per_slice[1])
  if is_slicing_enabled:
    result.name = stats_proto_per_slice[0]
  return result


def _merge_dataset_feature_stats_protos(
    stats_protos
):
  """Merges together a list of DatasetFeatureStatistics protos.

  Args:
    stats_protos: A list of DatasetFeatureStatistics protos to merge.

  Returns:
    The merged DatasetFeatureStatistics proto.
  """
  stats_per_feature = {}
  # Iterate over each DatasetFeatureStatistics proto and merge the
  # FeatureNameStatistics protos per feature.
  for stats_proto in stats_protos:
    for feature_stats_proto in stats_proto.features:
      if feature_stats_proto.name not in stats_per_feature:
        stats_per_feature[feature_stats_proto.name] = feature_stats_proto
      else:
        stats_per_feature[feature_stats_proto.name].MergeFrom(
            feature_stats_proto)

  # Create a new DatasetFeatureStatistics proto.
  result = statistics_pb2.DatasetFeatureStatistics()
  num_examples = None
  for feature_stats_proto in stats_per_feature.values():
    # Add the merged FeatureNameStatistics proto for the feature
    # into the DatasetFeatureStatistics proto.
    new_feature_stats_proto = result.features.add()
    new_feature_stats_proto.CopyFrom(feature_stats_proto)

    # Get the number of examples from one of the features that
    # has common stats.
    if num_examples is None:
      stats_type = feature_stats_proto.WhichOneof('stats')
      stats_proto = None
      if stats_type == 'num_stats':
        stats_proto = feature_stats_proto.num_stats
      else:
        stats_proto = feature_stats_proto.string_stats

      if stats_proto.HasField('common_stats'):
        num_examples = (stats_proto.common_stats.num_non_missing +
                        stats_proto.common_stats.num_missing)

  # Set the num_examples field.
  if num_examples is not None:
    result.num_examples = num_examples
  return result


def _make_dataset_feature_statistics_list_proto(
    stats_protos
):
  """Constructs a DatasetFeatureStatisticsList proto.

  Args:
    stats_protos: List of DatasetFeatureStatistics protos.

  Returns:
    The DatasetFeatureStatisticsList proto containing the input stats protos.
  """
  # Create a new DatasetFeatureStatisticsList proto.
  result = statistics_pb2.DatasetFeatureStatisticsList()

  for stats_proto in stats_protos:
    # Add the input DatasetFeatureStatistics proto.
    new_stats_proto = result.datasets.add()
    new_stats_proto.CopyFrom(stats_proto)
  return result


class _CombinerStatsGeneratorsCombineFnAcc(object):
  """accumulator for _CombinerStatsGeneratorsCombineFn."""

  __slots__ = ['partial_accumulators', 'input_examples']

  def __init__(self, partial_accumulators):
    # Partial accumulator states of the underlying CombinerStatsGenerators.
    self.partial_accumulators = partial_accumulators
    # Input examples to be processed.
    self.input_examples = []


@beam.typehints.with_input_types(types.Example)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatistics)
class _CombinerStatsGeneratorsCombineFn(beam.CombineFn):
  """A beam.CombineFn wrapping a list of CombinerStatsGenerators with batching.

  This wrapper does two things:
    1. Wraps a list of combiner stats generators. Its accumulator is a list
       of accumulators for each wrapped stats generators.
    2. Batches input examples before passing it to the underlying
       stats generators.

  We do this by accumulating examples in the combiner state until we
  accumulate a large enough batch, at which point we send them through the
  add_input step of each of the underlying combiner stats generators. When
  merging, we merge the accumulators of the stats generators and accumulate
  examples accordingly. We finally process any remaining examples
  before producing the final output value.

  This wrapper is needed to support slicing as we need the ability to
  perform slice-aware batching. But currently there is no way to do key-aware
  batching in Beam. Hence, this wrapper does batching and combining together.

  See also:
  BEAM-3737: Key-aware batching function
  (https://issues.apache.org/jira/browse/BEAM-3737).
  """

  # This needs to be large enough to allow for efficient TF invocations during
  # batch flushing, but shouldn't be too large as it also acts as cap on the
  # maximum memory usage of the computation.
  # TODO(b/73789023): Ideally we should automatically infer the batch size.
  _DEFAULT_DESIRED_ADD_INPUT_BATCH_SIZE = 1000

  # This needs to be large enough to allow for efficient merging of
  # accumulators in the individual stats generators, but shouldn't be too large
  # as it also acts as cap on the maximum memory usage of the computation.
  # TODO(b/73789023): Ideally we should automatically infer the batch size.
  _DEFAULT_DESIRED_MERGE_ACCUMULATOR_BATCH_SIZE = 100

  def __init__(
      self,
      generators,
      desired_batch_size = None):
    self._generators = generators

    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    # TODO(b/73789023): Figure out how to make this batch size dynamic.
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_ADD_INPUT_BATCH_SIZE

    # Metrics
    self._combine_add_input_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_add_input_batch_size')
    self._combine_merge_accumulator_batch_size = (
        beam.metrics.Metrics.distribution(
            constants.METRICS_NAMESPACE,
            'combine_merge_accumulator_batch_size'))
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')

  def _for_each_generator(self,
                          func,
                          *args):
    """Apply `func` for each wrapped generators.

    Args:
      func: a function that takes N + 1 arguments where N is the size of `args`.
        the first argument is the stats generator.
      *args: Iterables parallel to wrapped stats generators (i.e. the i-th item
        corresponds to the self._generators[i]).
    Returns:
      A list whose i-th element is the result of
      func(self._generators[i], args[0][i], args[1][i], ...).
    """
    return [func(gen, *args_for_func) for gen, args_for_func in zip(
        self._generators, zip(*args))]

  def create_accumulator(self
                        ):  # pytype: disable=invalid-annotation
    return _CombinerStatsGeneratorsCombineFnAcc(
        [g.create_accumulator() for g in self._generators])

  def _maybe_do_batch(
      self,
      accumulator,
      force = False):
    """Maybe updates accumulator in place.

    Checks if accumulator has enough examples for a batch, and if so, does the
    stats computation for the batch and updates accumulator in place.

    Args:
      accumulator: Accumulator. Will be updated in place.
      force: Force computation of stats even if accumulator has less examples
        than the batch size.
    """
    batch_size = len(accumulator.input_examples)
    if (force and batch_size > 0) or batch_size >= self._desired_batch_size:
      self._combine_add_input_batch_size.update(batch_size)
      merged_batch = batch_util.merge_single_batch(accumulator.input_examples)

      accumulator.partial_accumulators = self._for_each_generator(
          lambda gen, acc: gen.add_input(acc, merged_batch),
          accumulator.partial_accumulators)
      del accumulator.input_examples[:]

  def add_input(
      self, accumulator,
      input_example
      ):
    accumulator.input_examples.append(input_example)
    self._maybe_do_batch(accumulator)
    return accumulator

  def merge_accumulators(
      self,
      accumulators
      ):
    result = self.create_accumulator()
    # Make sure accumulators is an iterator (so it remembers its position).
    accumulators = iter(accumulators)
    while True:
      # Repeatedly take the next N from `accumulators` (an iterator).
      # If there are less than N remaining, all is taken.
      batched_accumulators = list(itertools.islice(
          accumulators, self._DEFAULT_DESIRED_MERGE_ACCUMULATOR_BATCH_SIZE))
      if not batched_accumulators:
        break

      # Batch together remaining examples in each accumulator, and
      # feed to each generator. Note that there might still be remaining
      # examples after this, but a compact() might follow and flush the
      # remaining examples, and extract_output() in the end will flush anyways.
      batched_partial_accumulators = []
      for acc in batched_accumulators:
        result.input_examples.extend(acc.input_examples)
        self._maybe_do_batch(result)
        batched_partial_accumulators.append(acc.partial_accumulators)

      batched_accumulators_by_generator = list(
          zip(*batched_partial_accumulators))

      result.partial_accumulators = self._for_each_generator(
          lambda gen, b, m: gen.merge_accumulators(itertools.chain((b,), m)),
          result.partial_accumulators,
          batched_accumulators_by_generator)

    return result

  # TODO(pachristopher): Consider adding CombinerStatsGenerator.compact method.
  def compact(
      self,
      accumulator
      ):
    self._maybe_do_batch(accumulator, force=True)
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self,
      accumulator
  ):  # pytype: disable=invalid-annotation
    # Make sure we have processed all the examples.
    self._maybe_do_batch(accumulator, force=True)
    return _merge_dataset_feature_stats_protos(
        self._for_each_generator(lambda gen, acc: gen.extract_output(acc),
                                 accumulator.partial_accumulators))


def generate_partial_statistics_in_memory(
    examples,
    options,
    stats_generators
):
  """Generates statistics for an in-memory list of examples.

  Args:
    examples: A list of input examples.
    options: Options for generating data statistics.
    stats_generators: A list of statistics generators.

  Returns:
    A list of accumulators containing partial statistics.
  """
  batch = batch_util.merge_single_batch(examples)
  # If whitelist features are provided, keep only those features.
  if options.feature_whitelist:
    batch = {
        feature_name: batch[feature_name]
        for feature_name in options.feature_whitelist
    }
  return [
      generator.add_input(generator.create_accumulator(), batch)
      for generator in stats_generators  # pytype: disable=attribute-error
  ]


def generate_statistics_in_memory(
    examples,
    options = stats_options.StatsOptions()
):
  """Generates statistics for an in-memory list of examples.

  Args:
    examples: A list of input examples.
    options: Options for generating data statistics.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  stats_generators = get_generators(options, in_memory=True)
  partial_stats = generate_partial_statistics_in_memory(
      examples, options, stats_generators)
  return extract_statistics_output(partial_stats, stats_generators)


def extract_statistics_output(
    partial_stats,
    stats_generators
):
  """Extracts final stats output from the accumulators holding partial stats."""
  outputs = [
      gen.extract_output(stats)
      for (gen, stats) in zip(stats_generators, partial_stats)  # pytype: disable=attribute-error
  ]
  return _make_dataset_feature_statistics_list_proto(
      [_merge_dataset_feature_stats_protos(outputs)])


# Type for the wrapper_accumulator of a CombinerFeatureStatsWrapperGenerator.
# See documentation below for more details.
WrapperAccumulator = Dict[Text, List[Any]]


class CombinerFeatureStatsWrapperGenerator(
    stats_generator.CombinerStatsGenerator):
  """A combiner that wraps multiple CombinerFeatureStatsGenerators.

  This combiner wraps multiple CombinerFeatureStatsGenerators by generating
  and updating wrapper_accumulators where:
  wrapper_accumulator[feature_name][feature_generator_index] contains the
  generator specific accumulator for the pair (feature_name,
  feature_generator_index).
  """

  def __init__(self,
               feature_stats_generators,
               name = 'CombinerFeatureStatsWrapperGenerator',
               schema = None,
               weight_feature = None):
    """Initializes a CombinerFeatureStatsWrapperGenerator.

    Args:
      feature_stats_generators: A list of CombinerFeatureStatsGenerator.
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      weight_feature: An optional feature name whose numeric value represents
        the weight of an example. Currently the weight feature is ignored by
        feature level stats generators.
    """
    super(CombinerFeatureStatsWrapperGenerator, self).__init__(name, schema)
    self._feature_stats_generators = feature_stats_generators
    self._weight_feature = weight_feature

  def _perhaps_initialize_for_feature_name(
      self, wrapper_accumulator,
      feature_name):
    """Initializes the feature_name key if it does not exist."""
    # Note: This manual initialization could have been avoided if
    # wrapper_accumulator was a defaultdict, but this breaks pickling.
    if feature_name not in wrapper_accumulator:
      wrapper_accumulator[feature_name] = [
          generator.create_accumulator()
          for generator in self._feature_stats_generators
      ]

  def create_accumulator(self):
    """Returns a fresh, empty wrapper_accumulator.

    Returns:
      An empty wrapper_accumulator.
    """
    return {}

  def add_input(self, wrapper_accumulator,
                input_batch):
    """Returns result of folding a batch of inputs into wrapper_accumulator.

    Args:
      wrapper_accumulator: The current wrapper accumulator.
      input_batch: A Python dict whose keys are strings denoting feature names
        and values are lists representing a batch of examples, which should be
        added to the accumulator.

    Returns:
      The wrapper_accumulator after updating the statistics for the batch of
      inputs.
    """
    for feature_name, values in six.iteritems(input_batch):
      if feature_name == self._weight_feature:
        continue
      self._perhaps_initialize_for_feature_name(wrapper_accumulator,
                                                feature_name)
      for index, generator in enumerate(self._feature_stats_generators):
        wrapper_accumulator[feature_name][index] = generator.add_input(
            generator.create_accumulator(), values)
    return wrapper_accumulator

  def merge_accumulators(
      self,
      wrapper_accumulators):
    """Merges several wrapper_accumulators to a single one.

    Args:
      wrapper_accumulators: The wrapper accumulators to merge.

    Returns:
      The merged accumulator.
    """
    result = self.create_accumulator()
    for wrapper_accumulator in wrapper_accumulators:
      for feature_name, accumulator_for_feature in six.iteritems(
          wrapper_accumulator):
        self._perhaps_initialize_for_feature_name(result, feature_name)
        for index, generator in enumerate(self._feature_stats_generators):
          result[feature_name][index] = generator.merge_accumulators(
              [result[feature_name][index], accumulator_for_feature[index]])
    return result

  def extract_output(self, wrapper_accumulator
                    ):
    """Returns result of converting wrapper_accumulator into the output value.

    Args:
      wrapper_accumulator: The final wrapper_accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, accumulator_for_feature in six.iteritems(
        wrapper_accumulator):
      feature_stats = result.features.add()
      feature_stats.name = feature_name
      for index, generator in enumerate(self._feature_stats_generators):
        feature_stats.MergeFrom(
            generator.extract_output(accumulator_for_feature[index]))
    return result
