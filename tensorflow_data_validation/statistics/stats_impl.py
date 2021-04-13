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

import itertools
import math
import random
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.statistics.generators import image_stats_generator
from tensorflow_data_validation.statistics.generators import lift_stats_generator
from tensorflow_data_validation.statistics.generators import natural_language_domain_inferring_stats_generator
from tensorflow_data_validation.statistics.generators import natural_language_stats_generator
from tensorflow_data_validation.statistics.generators import sparse_feature_stats_generator
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators import time_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_combiner_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_sketch_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_stats_generator
from tensorflow_data_validation.statistics.generators import weighted_feature_stats_generator
from tensorflow_data_validation.utils import slicing_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import table_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


_DEFAULT_MG_SKETCH_SIZE = 1024
_DEFAULT_KMV_SKETCH_SIZE = 16384


@beam.typehints.with_input_types(pa.RecordBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators over input examples."""

  def __init__(
      self,
      options: stats_options.StatsOptions = stats_options.StatsOptions()
      ) -> None:
    self._options = options

  def expand(self, dataset: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    # If a set of allowed features are provided, keep only those features.
    if self._options.feature_allowlist:
      dataset |= ('FilterFeaturesByAllowList' >> beam.Map(
          _filter_features, feature_allowlist=self._options.feature_allowlist))

    if self._options.experimental_slice_functions:
      # Add default slicing function.
      slice_functions = [slicing_util.default_slicer]
      slice_functions.extend(self._options.experimental_slice_functions)
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
@beam.typehints.with_input_types(types.BeamSlicedRecordBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateSlicedStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators to sliced input examples."""

  def __init__(
      self,
      options: stats_options.StatsOptions = stats_options.StatsOptions(),
      is_slicing_enabled: bool = False,
      ) -> None:
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
        is_slicing_enabled or bool(self._options.experimental_slice_functions))

  def expand(self, dataset: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
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
      # TODO(b/162543416): Obviate the need for explicit fanout.
      fanout = max(
          32, 5 * int(math.ceil(math.sqrt(len(combiner_stats_generators)))))
      result_protos.append(dataset
                           | 'RunCombinerStatsGenerators'
                           >> beam.CombinePerKey(
                               _CombinerStatsGeneratorsCombineFn(
                                   combiner_stats_generators,
                                   self._options.desired_batch_size
                                   )).with_hot_key_fanout(fanout))

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


def get_generators(options: stats_options.StatsOptions,
                   in_memory: bool = False
                  ) -> List[stats_generator.StatsGenerator]:
  """Initializes the list of stats generators, including custom generators.

  Args:
    options: A StatsOptions object.
    in_memory: Whether the generators will be used to generate statistics in
      memory (True) or using Beam (False).

  Returns:
    A list of stats generator objects.
  """
  generators = [NumExamplesStatsGenerator(options.weight_feature)]
  if options.add_default_generators:
    generators.extend(_get_default_generators(options, in_memory))
  if options.generators:
    # Add custom stats generators.
    generators.extend(options.generators)
  if options.enable_semantic_domain_stats:
    semantic_domain_feature_stats_generators = [
        image_stats_generator.ImageStatsGenerator(),
        natural_language_domain_inferring_stats_generator
        .NLDomainInferringStatsGenerator(),
        time_stats_generator.TimeStatsGenerator(),
    ]
    # Wrap semantic domain feature stats generators as a separate combiner
    # stats generator, so that we can apply sampling only for those and other
    # feature stats generators are not affected by it.
    generators.append(
        CombinerFeatureStatsWrapperGenerator(
            semantic_domain_feature_stats_generators,
            sample_rate=options.semantic_domain_stats_sample_rate))
  if options.schema is not None:
    if _schema_has_sparse_features(options.schema):
      generators.append(
          sparse_feature_stats_generator.SparseFeatureStatsGenerator(
              options.schema))
    if _schema_has_natural_language_domains(options.schema):
      generators.append(
          natural_language_stats_generator.NLStatsGenerator(
              options.schema, options.vocab_paths,
              options.num_histogram_buckets,
              options.num_quantiles_histogram_buckets,
              options.num_rank_histogram_buckets))
    if options.schema.weighted_feature:
      generators.append(
          weighted_feature_stats_generator.WeightedFeatureStatsGenerator(
              options.schema))
    if options.label_feature and not in_memory:
      # The LiftStatsGenerator is not a CombinerStatsGenerator and therefore
      # cannot currenty be used for in_memory executions.
      generators.append(
          lift_stats_generator.LiftStatsGenerator(
              y_path=types.FeaturePath([options.label_feature]),
              schema=options.schema,
              example_weight_map=options.example_weight_map,
              output_custom_stats=True))

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
        CombinerFeatureStatsWrapperGenerator(feature_generators)
    ]
  if in_memory:
    for generator in generators:
      if not isinstance(generator, stats_generator.CombinerStatsGenerator):
        raise TypeError('Statistics generator used in '
                        'generate_statistics_in_memory must '
                        'extend CombinerStatsGenerator, found object of '
                        'type %s.' % generator.__class__.__name__)
  return generators


def _get_default_generators(
    options: stats_options.StatsOptions, in_memory: bool = False
) -> List[stats_generator.StatsGenerator]:
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
          example_weight_map=options.example_weight_map,
          num_values_histogram_buckets=options.num_values_histogram_buckets,
          num_histogram_buckets=options.num_histogram_buckets,
          num_quantiles_histogram_buckets=options
          .num_quantiles_histogram_buckets,
          epsilon=options.epsilon),
  ]
  if options.experimental_use_sketch_based_topk_uniques:
    stats_generators.append(
        top_k_uniques_sketch_stats_generator.TopKUniquesSketchStatsGenerator(
            schema=options.schema,
            example_weight_map=options.example_weight_map,
            num_top_values=options.num_top_values,
            num_rank_histogram_buckets=options.num_rank_histogram_buckets,
            frequency_threshold=options.frequency_threshold,
            weighted_frequency_threshold=options.weighted_frequency_threshold,
            num_misragries_buckets=_DEFAULT_MG_SKETCH_SIZE,
            num_kmv_buckets=_DEFAULT_KMV_SKETCH_SIZE))
  elif in_memory:
    stats_generators.append(
        top_k_uniques_combiner_stats_generator
        .TopKUniquesCombinerStatsGenerator(
            schema=options.schema,
            example_weight_map=options.example_weight_map,
            num_top_values=options.num_top_values,
            frequency_threshold=options.frequency_threshold,
            weighted_frequency_threshold=options.weighted_frequency_threshold,
            num_rank_histogram_buckets=options.num_rank_histogram_buckets))
  else:
    stats_generators.append(
        top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
            schema=options.schema,
            example_weight_map=options.example_weight_map,
            num_top_values=options.num_top_values,
            frequency_threshold=options.frequency_threshold,
            weighted_frequency_threshold=options.weighted_frequency_threshold,
            num_rank_histogram_buckets=options.num_rank_histogram_buckets),
    )
  return stats_generators


def _schema_has_sparse_features(schema: schema_pb2.Schema) -> bool:
  """Returns whether there are any sparse features in the specified schema."""

  def _has_sparse_features(
      feature_container: Iterable[schema_pb2.Feature]
  ) -> bool:
    """Helper function used to determine whether there are sparse features."""
    for f in feature_container:
      if isinstance(f, schema_pb2.SparseFeature):
        return True
      if f.type == schema_pb2.STRUCT:
        if f.struct_domain.sparse_feature:
          return True
        return _has_sparse_features(f.struct_domain.feature)
    return False

  if schema.sparse_feature:
    return True
  return _has_sparse_features(schema.feature)


def _schema_has_natural_language_domains(schema: schema_pb2.Schema) -> bool:
  """Returns whether there are features in the schema with a nl domain."""
  for f in schema.feature:
    if f.WhichOneof('domain_info') == 'natural_language_domain':
      return True
  return False


def _filter_features(
    record_batch: pa.RecordBatch,
    feature_allowlist: List[types.FeatureName]) -> pa.RecordBatch:
  """Removes features that are not on the allowlist.

  Args:
    record_batch: Input Arrow RecordBatch.
    feature_allowlist: A set of feature names to keep.

  Returns:
    An Arrow RecordBatch containing only features on the allowlist.
  """
  schema = record_batch.schema
  column_names = set(schema.names)
  columns_to_select = []
  column_names_to_select = []
  for feature_name in feature_allowlist:
    if feature_name in column_names:
      columns_to_select.append(
          record_batch.column(schema.get_field_index(feature_name)))
      column_names_to_select.append(feature_name)
  return pa.RecordBatch.from_arrays(columns_to_select, column_names_to_select)


def _add_slice_key(
    stats_proto_per_slice: Tuple[types.SliceKey,
                                 statistics_pb2.DatasetFeatureStatistics],
    is_slicing_enabled: bool
) -> statistics_pb2.DatasetFeatureStatistics:
  """Add slice key to stats proto."""
  result = statistics_pb2.DatasetFeatureStatistics()
  result.CopyFrom(stats_proto_per_slice[1])
  if is_slicing_enabled:
    result.name = stats_proto_per_slice[0]
  return result


def _merge_dataset_feature_stats_protos(
    stats_protos: Iterable[statistics_pb2.DatasetFeatureStatistics]
) -> statistics_pb2.DatasetFeatureStatistics:
  """Merges together a list of DatasetFeatureStatistics protos.

  Args:
    stats_protos: A list of DatasetFeatureStatistics protos to merge.

  Returns:
    The merged DatasetFeatureStatistics proto.
  """
  stats_per_feature = {}
  # Create a new DatasetFeatureStatistics proto.
  result = statistics_pb2.DatasetFeatureStatistics()
  # Iterate over each DatasetFeatureStatistics proto and merge the
  # FeatureNameStatistics protos per feature and add the cross feature stats.
  for stats_proto in stats_protos:
    if stats_proto.cross_features:
      result.cross_features.extend(stats_proto.cross_features)
    for feature_stats_proto in stats_proto.features:
      feature_path = types.FeaturePath.from_proto(feature_stats_proto.path)
      if feature_path not in stats_per_feature:
        # Make a copy for the "cache" since we are modifying it in 'else' below.
        new_feature_stats_proto = statistics_pb2.FeatureNameStatistics()
        new_feature_stats_proto.CopyFrom(feature_stats_proto)
        stats_per_feature[feature_path] = new_feature_stats_proto
      else:
        stats_for_feature = stats_per_feature[feature_path]
        # MergeFrom would concatenate repeated fields which is not what we want
        # for path.step.
        del stats_for_feature.path.step[:]
        stats_for_feature.MergeFrom(feature_stats_proto)

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


def _update_example_and_missing_count(
    stats: statistics_pb2.DatasetFeatureStatistics) -> None:
  """Updates example count of the dataset and missing count for all features."""
  if not stats.features:
    return
  dummy_feature = stats_util.get_feature_stats(stats, _DUMMY_FEATURE_PATH)
  num_examples = stats_util.get_custom_stats(dummy_feature, _NUM_EXAMPLES_KEY)
  weighted_num_examples = stats_util.get_custom_stats(
      dummy_feature, _WEIGHTED_NUM_EXAMPLES_KEY)
  stats.features.remove(dummy_feature)
  for feature_stats in stats.features:
    # For features nested under a STRUCT feature, their num_missing is computed
    # in the basic stats generator (because their num_missing is relative to
    # their parent's value count).
    if len(feature_stats.path.step) > 1:
      continue
    common_stats = None
    which_oneof_stats = feature_stats.WhichOneof('stats')
    if which_oneof_stats is None:
      # There are not common_stats for this feature (which can be the case when
      # generating only custom_stats for a sparse or weighted feature). In that
      # case, simply continue without modifying the common stats.
      continue
    common_stats = getattr(feature_stats, which_oneof_stats).common_stats
    assert num_examples >= common_stats.num_non_missing, (
        'Total number of examples: {} is less than number of non missing '
        'examples: {} for feature {}.'.format(
            num_examples, common_stats.num_non_missing,
            '.'.join(feature_stats.path.step)))
    num_missing = int(num_examples - common_stats.num_non_missing)
    common_stats.num_missing = num_missing
    if common_stats.presence_and_valency_stats:
      common_stats.presence_and_valency_stats[0].num_missing = num_missing
    if weighted_num_examples != 0:
      weighted_num_missing = (
          weighted_num_examples -
          common_stats.weighted_common_stats.num_non_missing)
      common_stats.weighted_common_stats.num_missing = weighted_num_missing
      if common_stats.weighted_presence_and_valency_stats:
        common_stats.weighted_presence_and_valency_stats[0].num_missing = (
            weighted_num_missing)

  stats.num_examples = int(num_examples)
  stats.weighted_num_examples = weighted_num_examples


def _make_dataset_feature_statistics_list_proto(
    stats_protos: List[statistics_pb2.DatasetFeatureStatistics]
) -> statistics_pb2.DatasetFeatureStatisticsList:
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
    # We now update the example count for the dataset and the missing count
    # for all the features, using the number of examples computed separately
    # using NumExamplesStatsGenerator. Note that we compute the number of
    # examples separately to avoid ignoring example counts for features which
    # may be completely missing in a shard. We set the missing count of a
    # feature to be num_examples - non_missing_count.
    _update_example_and_missing_count(new_stats_proto)
  if not stats_protos:
    # Handle the case in which there are no examples. In that case, we want to
    # output a DatasetFeatureStatisticsList proto with a dataset containing
    # num_examples == 0 instead of an empty DatasetFeatureStatisticsList proto.
    result.datasets.add(num_examples=0)
  return result


_DUMMY_FEATURE_PATH = types.FeaturePath(['__TFDV_INTERNAL_FEATURE__'])
_NUM_EXAMPLES_KEY = '__NUM_EXAMPLES__'
_WEIGHTED_NUM_EXAMPLES_KEY = '__WEIGHTED_NUM_EXAMPLES__'


class NumExamplesStatsGenerator(stats_generator.CombinerStatsGenerator):
  """Computes total number of examples."""

  def __init__(self,
               weight_feature: Optional[types.FeatureName] = None) -> None:
    self._weight_feature = weight_feature

  def create_accumulator(self) -> List[float]:
    return [0, 0]  # [num_examples, weighted_num_examples]

  def add_input(self, accumulator: List[float],
                examples: pa.RecordBatch) -> List[float]:
    accumulator[0] += examples.num_rows
    if self._weight_feature:
      weights_column = examples.column(
          examples.schema.get_field_index(self._weight_feature))
      accumulator[1] += np.sum(np.asarray(weights_column.flatten()))
    return accumulator

  def merge_accumulators(self, accumulators: Iterable[List[float]]
                        ) -> List[float]:
    result = self.create_accumulator()
    for acc in accumulators:
      result[0] += acc[0]
      result[1] += acc[1]
    return result

  def extract_output(self, accumulator: List[float]
                    ) -> statistics_pb2.DatasetFeatureStatistics:
    result = statistics_pb2.DatasetFeatureStatistics()
    dummy_feature = result.features.add()
    dummy_feature.path.CopyFrom(_DUMMY_FEATURE_PATH.to_proto())
    dummy_feature.custom_stats.add(name=_NUM_EXAMPLES_KEY, num=accumulator[0])
    dummy_feature.custom_stats.add(name=_WEIGHTED_NUM_EXAMPLES_KEY,
                                   num=accumulator[1])
    return result


class _CombinerStatsGeneratorsCombineFnAcc(object):
  """accumulator for _CombinerStatsGeneratorsCombineFn."""

  __slots__ = [
      'partial_accumulators', 'input_record_batches', 'curr_batch_size',
      'curr_byte_size'
  ]

  def __init__(self, partial_accumulators: List[Any]):
    # Partial accumulator states of the underlying CombinerStatsGenerators.
    self.partial_accumulators = partial_accumulators
    # Input record batches to be processed.
    self.input_record_batches = []
    # Current batch size.
    self.curr_batch_size = 0
    # Current total byte size of all the pa.RecordBatches accumulated.
    self.curr_byte_size = 0


@beam.typehints.with_input_types(pa.RecordBatch)
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

  # This needs to be large enough to allow for efficient merging of
  # accumulators in the individual stats generators, but shouldn't be too large
  # as it also acts as cap on the maximum memory usage of the computation.
  # TODO(b/73789023): Ideally we should automatically infer the batch size.
  _DESIRED_MERGE_ACCUMULATOR_BATCH_SIZE = 100

  # The combiner accumulates record batches from the upstream and merges them
  # when certain conditions are met. A merged record batch would allow better
  # vectorized processing, but we have to pay for copying and the RAM to
  # contain the merged record batch. If the total byte size of accumulated
  # record batches exceeds this threshold a merge will be forced to avoid
  # consuming too much memory.
  _MERGE_RECORD_BATCH_BYTE_SIZE_THRESHOLD = 20 << 20  # 20MiB

  def __init__(
      self,
      generators: List[stats_generator.CombinerStatsGenerator],
      desired_batch_size: Optional[int] = None) -> None:
    self._generators = generators

    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    # TODO(b/73789023): Figure out how to make this batch size dynamic.
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = constants.DEFAULT_DESIRED_INPUT_BATCH_SIZE

    # Metrics
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._combine_byte_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_byte_size')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')
    self._num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_instances')

    # TODO(b/175966315): Remove this once all supported versions have setup
    # defined.
    if not getattr(
        super(_CombinerStatsGeneratorsCombineFn, self), 'setup', None):
      self.setup()

  def _for_each_generator(self,
                          func: Callable[..., Any],
                          *args: Iterable[Any]) -> List[Any]:
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

  def _should_do_batch(self, accumulator: _CombinerStatsGeneratorsCombineFnAcc,
                       force: bool) -> bool:
    curr_batch_size = accumulator.curr_batch_size
    if force and curr_batch_size > 0:
      return True

    if curr_batch_size >= self._desired_batch_size:
      return True

    if (accumulator.curr_byte_size >=
        self._MERGE_RECORD_BATCH_BYTE_SIZE_THRESHOLD):
      return True

    return False

  def _maybe_do_batch(
      self,
      accumulator: _CombinerStatsGeneratorsCombineFnAcc,
      force: bool = False) -> None:
    """Maybe updates accumulator in place.

    Checks if accumulator has enough examples for a batch, and if so, does the
    stats computation for the batch and updates accumulator in place.

    Args:
      accumulator: Accumulator. Will be updated in place.
      force: Force computation of stats even if accumulator has less examples
        than the batch size.
    """
    if self._should_do_batch(accumulator, force):
      self._combine_batch_size.update(accumulator.curr_batch_size)
      self._combine_byte_size.update(accumulator.curr_byte_size)
      if len(accumulator.input_record_batches) == 1:
        record_batch = accumulator.input_record_batches[0]
      else:
        record_batch = table_util.MergeRecordBatches(
            accumulator.input_record_batches)
      accumulator.partial_accumulators = self._for_each_generator(
          lambda gen, gen_acc: gen.add_input(gen_acc, record_batch),
          accumulator.partial_accumulators)
      del accumulator.input_record_batches[:]
      accumulator.curr_batch_size = 0
      accumulator.curr_byte_size = 0

  def setup(self):
    """Prepares each generator for combining."""
    for gen in self._generators:
      gen.setup()

  def create_accumulator(self) -> _CombinerStatsGeneratorsCombineFnAcc:
    return _CombinerStatsGeneratorsCombineFnAcc(
        [g.create_accumulator() for g in self._generators])

  def add_input(
      self, accumulator: _CombinerStatsGeneratorsCombineFnAcc,
      input_record_batch: pa.RecordBatch
  ) -> _CombinerStatsGeneratorsCombineFnAcc:
    accumulator.input_record_batches.append(input_record_batch)
    num_rows = input_record_batch.num_rows
    accumulator.curr_batch_size += num_rows
    accumulator.curr_byte_size += table_util.TotalByteSize(input_record_batch)
    self._maybe_do_batch(accumulator)
    self._num_instances.inc(num_rows)
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[_CombinerStatsGeneratorsCombineFnAcc]
      ) -> _CombinerStatsGeneratorsCombineFnAcc:
    result = self.create_accumulator()
    # Make sure accumulators is an iterator (so it remembers its position).
    accumulators = iter(accumulators)
    while True:
      # Repeatedly take the next N from `accumulators` (an iterator).
      # If there are less than N remaining, all is taken.
      batched_accumulators = list(itertools.islice(
          accumulators, self._DESIRED_MERGE_ACCUMULATOR_BATCH_SIZE))
      if not batched_accumulators:
        break

      # Batch together remaining examples in each accumulator, and
      # feed to each generator. Note that there might still be remaining
      # examples after this, but a compact() might follow and flush the
      # remaining examples, and extract_output() in the end will flush anyways.
      batched_partial_accumulators = []
      for acc in batched_accumulators:
        result.input_record_batches.extend(acc.input_record_batches)
        result.curr_batch_size += acc.curr_batch_size
        result.curr_byte_size += acc.curr_byte_size
        self._maybe_do_batch(result)
        batched_partial_accumulators.append(acc.partial_accumulators)

      batched_accumulators_by_generator = list(
          zip(*batched_partial_accumulators))

      result.partial_accumulators = self._for_each_generator(
          lambda gen, b, m: gen.merge_accumulators(itertools.chain((b,), m)),
          result.partial_accumulators,
          batched_accumulators_by_generator)

    return result

  def compact(
      self,
      accumulator: _CombinerStatsGeneratorsCombineFnAcc
      ) -> _CombinerStatsGeneratorsCombineFnAcc:
    self._maybe_do_batch(accumulator, force=True)
    accumulator.partial_accumulators = self._for_each_generator(
        lambda gen, acc: gen.compact(acc), accumulator.partial_accumulators)
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self,
      accumulator: _CombinerStatsGeneratorsCombineFnAcc
  ) -> statistics_pb2.DatasetFeatureStatistics:  # pytype: disable=invalid-annotation
    # Make sure we have processed all the examples.
    self._maybe_do_batch(accumulator, force=True)
    return _merge_dataset_feature_stats_protos(
        self._for_each_generator(lambda gen, acc: gen.extract_output(acc),
                                 accumulator.partial_accumulators))


def generate_partial_statistics_in_memory(
    record_batch: pa.RecordBatch, options: stats_options.StatsOptions,
    stats_generators: List[stats_generator.CombinerStatsGenerator]
) -> List[Any]:
  """Generates statistics for an in-memory list of examples.

  Args:
    record_batch: Arrow RecordBatch.
    options: Options for generating data statistics.
    stats_generators: A list of combiner statistics generators.

  Returns:
    A list of accumulators containing partial statistics.
  """
  result = []
  if options.feature_allowlist:
    schema = record_batch.schema
    columns = [
        record_batch.column(schema.get_field_index(f))
        for f in options.feature_allowlist
    ]
    record_batch = pa.RecordBatch.from_arrays(columns,
                                              list(options.feature_allowlist))
  for generator in stats_generators:
    result.append(
        generator.add_input(generator.create_accumulator(), record_batch))
  return result


def generate_statistics_in_memory(
    record_batch: pa.RecordBatch,
    options: stats_options.StatsOptions = stats_options.StatsOptions()
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Generates statistics for an in-memory list of examples.

  Args:
    record_batch: Arrow RecordBatch.
    options: Options for generating data statistics.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  stats_generators = cast(List[stats_generator.CombinerStatsGenerator],
                          get_generators(options, in_memory=True))
  partial_stats = generate_partial_statistics_in_memory(record_batch, options,
                                                        stats_generators)
  return extract_statistics_output(partial_stats, stats_generators)


def extract_statistics_output(
    partial_stats: List[Any],
    stats_generators: List[stats_generator.CombinerStatsGenerator]
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Extracts final stats output from the accumulators holding partial stats."""

  # We call compact before extract_output to guarentee that `compact()` is
  # called at least once, for testing coverage.
  outputs = [
      gen.extract_output(gen.compact(stats))
      for (gen, stats) in zip(stats_generators, partial_stats)  # pytype: disable=attribute-error
  ]
  return _make_dataset_feature_statistics_list_proto(
      [_merge_dataset_feature_stats_protos(outputs)])


# Type for the wrapper_accumulator of a CombinerFeatureStatsWrapperGenerator.
# See documentation below for more details.
WrapperAccumulator = Dict[types.FeaturePath, List[Any]]


class CombinerFeatureStatsWrapperGenerator(
    stats_generator.CombinerStatsGenerator):
  """A combiner that wraps multiple CombinerFeatureStatsGenerators.

  This combiner wraps multiple CombinerFeatureStatsGenerators by generating
  and updating wrapper_accumulators where:
  wrapper_accumulator[feature_path][feature_generator_index] contains the
  generator specific accumulator for the pair (feature_path,
  feature_generator_index).
  """

  def __init__(self,
               feature_stats_generators: List[
                   stats_generator.CombinerFeatureStatsGenerator],
               name: Text = 'CombinerFeatureStatsWrapperGenerator',
               schema: Optional[schema_pb2.Schema] = None,
               sample_rate: Optional[float] = None) -> None:
    """Initializes a CombinerFeatureStatsWrapperGenerator.

    Args:
      feature_stats_generators: A list of CombinerFeatureStatsGenerator.
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
      sample_rate: An optional sampling rate. If specified, statistics is
        computed over the sample.
    """
    super(CombinerFeatureStatsWrapperGenerator, self).__init__(name, schema)
    self._feature_stats_generators = feature_stats_generators
    self._sample_rate = sample_rate

  def _get_wrapped_accumulators(self, wrapper_accumulator: WrapperAccumulator,
                                feature_path: types.FeaturePath) -> List[Any]:
    """Initializes the feature_path key if it does not exist."""
    result = wrapper_accumulator.get(feature_path, None)
    if result is not None:
      return result
    # Note: This manual initialization could have been avoided if
    # wrapper_accumulator was a defaultdict, but this breaks pickling.
    result = [
        generator.create_accumulator()
        for generator in self._feature_stats_generators
    ]
    wrapper_accumulator[feature_path] = result
    return result

  def setup(self):
    """Prepares every CombinerFeatureStatsGenerator instance for combining."""
    for gen in self._feature_stats_generators:
      gen.setup()

  def create_accumulator(self) -> WrapperAccumulator:
    """Returns a fresh, empty wrapper_accumulator.

    Returns:
      An empty wrapper_accumulator.
    """
    return {}

  def add_input(self, wrapper_accumulator: WrapperAccumulator,
                input_record_batch: pa.RecordBatch) -> WrapperAccumulator:
    """Returns result of folding a batch of inputs into wrapper_accumulator.

    Args:
      wrapper_accumulator: The current wrapper accumulator.
      input_record_batch: An arrow RecordBatch representing a batch of examples,
      which should be added to the accumulator.

    Returns:
      The wrapper_accumulator after updating the statistics for the batch of
      inputs.
    """
    if self._sample_rate is not None and random.random() <= self._sample_rate:
      return wrapper_accumulator

    for feature_path, feature_array, _ in arrow_util.enumerate_arrays(
        input_record_batch,
        example_weight_map=None,
        enumerate_leaves_only=True):
      wrapped_accumulators = self._get_wrapped_accumulators(
          wrapper_accumulator, feature_path)
      for index, generator in enumerate(self._feature_stats_generators):
        wrapped_accumulators[index] = generator.add_input(
            wrapped_accumulators[index], feature_path, feature_array)

    return wrapper_accumulator

  def merge_accumulators(
      self,
      wrapper_accumulators: Iterable[WrapperAccumulator]) -> WrapperAccumulator:
    """Merges several wrapper_accumulators to a single one.

    Args:
      wrapper_accumulators: The wrapper accumulators to merge.

    Returns:
      The merged accumulator.
    """
    result = self.create_accumulator()
    for wrapper_accumulator in wrapper_accumulators:
      for feature_path, accumulator_for_feature in wrapper_accumulator.items():
        wrapped_accumulators = self._get_wrapped_accumulators(
            result, feature_path)
        for index, generator in enumerate(self._feature_stats_generators):
          wrapped_accumulators[index] = generator.merge_accumulators(
              [wrapped_accumulators[index], accumulator_for_feature[index]])
    return result

  def compact(self,
              wrapper_accumulator: WrapperAccumulator) -> WrapperAccumulator:
    """Returns a compacted wrapper_accumulator.

    This overrides the base class's implementation. This is optionally called
    before an accumulator is sent across the wire.

    Args:
      wrapper_accumulator: The wrapper accumulator to compact.
    """
    for accumulator_for_feature in wrapper_accumulator.values():
      for index, generator in enumerate(self._feature_stats_generators):
        accumulator_for_feature[index] = generator.compact(
            accumulator_for_feature[index])

    return wrapper_accumulator

  def extract_output(self, wrapper_accumulator: WrapperAccumulator
                    ) -> statistics_pb2.DatasetFeatureStatistics:
    """Returns result of converting wrapper_accumulator into the output value.

    Args:
      wrapper_accumulator: The final wrapper_accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_path, accumulator_for_feature in wrapper_accumulator.items():
      feature_stats = result.features.add()
      feature_stats.path.CopyFrom(feature_path.to_proto())
      for index, generator in enumerate(self._feature_stats_generators):
        feature_stats.MergeFrom(
            generator.extract_output(accumulator_for_feature[index]))
    return result
