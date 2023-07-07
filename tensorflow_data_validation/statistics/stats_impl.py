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

import math
import random
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Text, Tuple, Union

import apache_beam as beam
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils import preprocessing_util
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.statistics.generators import basic_stats_generator
from tensorflow_data_validation.statistics.generators import image_stats_generator
from tensorflow_data_validation.statistics.generators import lift_stats_generator
from tensorflow_data_validation.statistics.generators import natural_language_domain_inferring_stats_generator
from tensorflow_data_validation.statistics.generators import natural_language_stats_generator
from tensorflow_data_validation.statistics.generators import sparse_feature_stats_generator
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators import time_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_sketch_stats_generator
from tensorflow_data_validation.statistics.generators import top_k_uniques_stats_generator
from tensorflow_data_validation.statistics.generators import weighted_feature_stats_generator
from tensorflow_data_validation.utils import feature_partition_util
from tensorflow_data_validation.utils import metrics_util
from tensorflow_data_validation.utils import slicing_util

from tfx_bsl import beam as tfx_bsl_beam
from tfx_bsl.arrow import table_util
from tfx_bsl.statistics import merge_util
from tfx_bsl.telemetry import collection

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

tfx_bsl_beam.fix_code_type_pickling()

_DEFAULT_MG_SKETCH_SIZE = 1024
_DEFAULT_KMV_SKETCH_SIZE = 16384


class GenerateStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators over input examples."""

  def __init__(
      self,
      options: stats_options.StatsOptions = stats_options.StatsOptions()
      ) -> None:
    self._options = options

  def expand(
      self, dataset: beam.PCollection[pa.RecordBatch]
  ) -> beam.PCollection[statistics_pb2.DatasetFeatureStatisticsList]:
    # Generate derived features, if applicable.
    if self._options.schema is not None:
      dataset, derivers_configured = preprocessing_util.add_derived_features(
          dataset, self._options.schema)
      if derivers_configured:
        metadata_generator = preprocessing_util.get_metadata_generator()
        assert metadata_generator is not None
        self._options.generators = self._options.generators or []
        self._options.generators.append(metadata_generator)

    # If a set of allowed features are provided, keep only those features.
    if self._options.feature_allowlist:
      dataset |= 'FilterFeaturesByAllowList' >> beam.Map(
          _filter_features, feature_allowlist=self._options.feature_allowlist)

    _ = dataset | 'TrackTotalBytes' >> collection.TrackRecordBatchBytes(
        constants.METRICS_NAMESPACE, 'record_batch_input_bytes')

    if self._options.slicing_config:
      slice_fns, slice_sqls = (
          slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
              self._options.slicing_config))
    else:
      slice_fns, slice_sqls = (self._options.experimental_slice_functions,
                               self._options.experimental_slice_sqls)

    if slice_fns:
      # Add default slicing function.
      slice_functions = [slicing_util.default_slicer]
      slice_functions.extend(slice_fns)
      dataset = (
          dataset
          | 'GenerateSliceKeys' >> beam.FlatMap(
              slicing_util.generate_slices, slice_functions=slice_functions))
    elif slice_sqls:
      dataset = (
          dataset
          | 'GenerateSlicesSql' >> beam.ParDo(
              slicing_util.GenerateSlicesSqlDoFn(slice_sqls=slice_sqls)))
    else:
      dataset = (dataset
                 | 'KeyWithVoid' >> beam.Map(lambda v: (None, v)))
    _ = dataset | 'TrackDistinctSliceKeys' >> _TrackDistinctSliceKeys()  # pylint: disable=no-value-for-parameter
    return dataset | GenerateSlicedStatisticsImpl(self._options)


def _increment_counter(counter_name: Text, element: int):  # pylint: disable=invalid-name
  counter = beam.metrics.Metrics.counter(
      constants.METRICS_NAMESPACE, counter_name)
  counter.inc(element)
  return element


@beam.ptransform_fn
def _TrackDistinctSliceKeys(  # pylint: disable=invalid-name
    slice_keys_and_values: beam.PCollection[types.SlicedRecordBatch]
) -> beam.pvalue.PCollection[int]:
  """Gathers slice key telemetry post slicing."""

  return (slice_keys_and_values
          | 'ExtractSliceKeys' >> beam.Keys()
          | 'RemoveDuplicates' >> beam.Distinct()
          | 'Size' >> beam.combiners.Count.Globally()
          | 'IncrementCounter' >> beam.Map(
              lambda x: _increment_counter('num_distinct_slice_keys', x)))


class _YieldPlaceholderFn(beam.DoFn):
  """Yields a single empty proto if input (count) is zero."""

  def process(self, count: int):
    if count == 0:
      yield ('', statistics_pb2.DatasetFeatureStatistics())


@beam.ptransform_fn
def _AddPlaceholderStatistics(  # pylint: disable=invalid-name
    statistics: beam.PCollection[Tuple[
        types.SliceKey, statistics_pb2.DatasetFeatureStatistics]]):
  """Adds a placeholder empty dataset for empty input, otherwise noop."""
  count = statistics | beam.combiners.Count.Globally()
  maybe_placeholder = count | beam.ParDo(_YieldPlaceholderFn())
  return (statistics, maybe_placeholder) | beam.Flatten()


def _split_generator_types(
    generators: List[stats_generator.StatsGenerator], num_partitions: int
) -> Tuple[List[stats_generator.TransformStatsGenerator],
           List[stats_generator.CombinerStatsGenerator],
           List[stats_generator.CombinerStatsGenerator]]:
  """Split generators by type.

  Args:
    generators: A list of generators.
    num_partitions: The number of feature partitions to split by.

  Returns:
    A three tuple consisting of 1) TransformStatsGenerators 2)
    CombinerStatsGenerators that should not be feature-partitioned 3)
    CombinerStatsGenerators that should be feature-partitioned

  Raises:
    TypeError: If provided generators are not instances of
      TransformStatsGenerators or CombinerStatsGenerator.
  """
  transform_generators = []
  unpartitioned_combiners = []
  partitioned_combiners = []
  for generator in generators:
    if isinstance(generator, stats_generator.TransformStatsGenerator):
      transform_generators.append(generator)
    elif isinstance(generator, stats_generator.CombinerStatsGenerator):
      if num_partitions > 1:
        try:
          _ = generator._copy_for_partition_index(0, num_partitions)  # pylint: disable=protected-access
          partitioned_combiners.append(generator)
        except NotImplementedError:
          unpartitioned_combiners.append(generator)
      else:
        unpartitioned_combiners.append(generator)
    else:
      raise TypeError('Statistics generator must extend one of '
                      'CombinerStatsGenerator or TransformStatsGenerator, '
                      'found object of type %s' % generator.__class__.__name__)
  return transform_generators, unpartitioned_combiners, partitioned_combiners


# This transform will be used by the example validation API to compute
# statistics over anomalous examples. Specifically, it is used to compute
# statistics over examples found for each anomaly (i.e., the anomaly type
# will be the slice key).
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
        even if slice functions or slicing SQL queries are not provided in
        `options`. If slice functions or slicing SQL queries are provided in
        `options`, slice keys are included regardless of this value.
    """
    self._options = options
    self._is_slicing_enabled = (
        is_slicing_enabled or
        bool(self._options.experimental_slice_functions) or
        bool(self._options.experimental_slice_sqls) or
        bool(self._options.slicing_config))

  def _to_partitioned_combiner_stats_generator_combine_fn(
      self, generators: List[stats_generator.CombinerStatsGenerator]
  ) -> List['_CombinerStatsGeneratorsCombineFn']:
    """Produce one CombineFn per partition wrapping partitioned generators."""
    if not generators:
      return []
    result = []
    for idx in range(self._options.experimental_num_feature_partitions):
      index_generators = [
          g._copy_for_partition_index(  # pylint: disable=protected-access
              idx, self._options.experimental_num_feature_partitions)
          for g in generators
      ]
      result.append(
          _CombinerStatsGeneratorsCombineFn(index_generators,
                                            self._options.desired_batch_size))
    return result

  def expand(self, dataset: beam.PCollection[types.SlicedRecordBatch]):
    # Collect telemetry on what generators are in use.
    generators = get_generators(self._options)
    generator_name_counts = {'generator_%s' % g.name: 1 for g in generators}
    _ = (
        dataset | metrics_util.IncrementJobCounters(generator_name_counts))

    # Handles generators by their type:
    #   - CombinerStatsGenerators will be wrapped in a single CombinePerKey by
    #     _CombinerStatsGeneratorsCombineFn.
    #   - TransformStatsGenerator will be invoked separately with `dataset`.
    result_protos = []
    (transform_generators, unpartitioned_combiners,
     partitioned_combiners) = _split_generator_types(
         generators, self._options.experimental_num_feature_partitions)
    # Set up combineFns.
    combine_fns = []
    if unpartitioned_combiners:
      combine_fns.append(
          _CombinerStatsGeneratorsCombineFn(unpartitioned_combiners,
                                            self._options.desired_batch_size))
    if partitioned_combiners:
      combine_fns.extend(
          self._to_partitioned_combiner_stats_generator_combine_fn(
              partitioned_combiners))

    # Apply transform generators.
    for generator in transform_generators:
      result_protos.append(dataset | generator.name >> generator.ptransform)

    # Apply combiner stats generators.
    # TODO(b/162543416): Obviate the need for explicit fanout.
    fanout = max(128,
                 20 * int(math.ceil(math.sqrt(len(combine_fns)))))
    for i, combine_fn in enumerate(combine_fns):
      result_protos.append(
          dataset
          | 'RunCombinerStatsGenerators[%d]' %
          i >> beam.CombinePerKey(combine_fn).with_hot_key_fanout(fanout))
    result_protos = result_protos | 'FlattenFeatureStatistics' >> beam.Flatten()
    result_protos = (
        result_protos
        | 'AddPlaceholderStatistics' >> _AddPlaceholderStatistics())  # pylint: disable=no-value-for-parameter
    # Combine result_protos into a configured number of partitions.
    return (result_protos
            | 'AddSliceKeyToStatsProto' >> beam.Map(_add_slice_key,
                                                    self._is_slicing_enabled)
            | 'MakeDatasetFeatureStatisticsListProto' >>
            beam.Map(_make_singleton_dataset_feature_statistics_list_proto)
            | 'SplitIntoFeaturePartitions' >> beam.ParDo(
                feature_partition_util.KeyAndSplitByFeatureFn(
                    self._options.experimental_result_partitions))
            | 'MergeStatsProtos' >> beam.CombinePerKey(
                merge_util.merge_dataset_feature_statistics_list)
            | 'Values' >> beam.Values())


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
  generators = []
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
  if options.use_sketch_based_topk_uniques or in_memory:
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
    feature_allowlist: Union[List[types.FeatureName], List[types.FeaturePath]]
) -> pa.RecordBatch:
  """Removes features that are not on the allowlist.

  Args:
    record_batch: Input Arrow RecordBatch.
    feature_allowlist: A set of feature names to keep.

  Returns:
    An Arrow RecordBatch containing only features on the allowlist.
  """
  columns_to_select = []
  column_names_to_select = []
  for feature_name in feature_allowlist:
    if isinstance(feature_name, types.FeaturePath):
      # TODO(b/255895499): Support paths.
      raise NotImplementedError
    col = arrow_util.get_column(record_batch, feature_name, missing_ok=True)
    if col is None:
      continue
    columns_to_select.append(col)
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


def _make_singleton_dataset_feature_statistics_list_proto(
    statistics: statistics_pb2.DatasetFeatureStatistics
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Wrap statistics in a DatasetFeatureStatisticsList proto."""
  result = statistics_pb2.DatasetFeatureStatisticsList()
  new_stats_proto = result.datasets.add()
  new_stats_proto.CopyFrom(statistics)
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

  # The combiner accumulates record batches from the upstream and merges them
  # when certain conditions are met. A merged record batch would allow better
  # vectorized processing, but we have to pay for copying and the RAM to
  # contain the merged record batch. If the total byte size of accumulated
  # record batches exceeds this threshold a merge will be forced to avoid
  # consuming too much memory.
  #
  # TODO(b/162543416): Perhaps this should be increased (eg to 32 or 64 MiB)?
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
    self._num_do_batch_force = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_do_batch_force')
    self._num_do_batch_count = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_do_batch_count')
    self._num_do_batch_bytes = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_do_batch_bytes')

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
      self._num_do_batch_force.inc(1)
      return True

    if curr_batch_size >= self._desired_batch_size:
      self._num_do_batch_count.inc(1)
      return True

    if (accumulator.curr_byte_size >=
        self._MERGE_RECORD_BATCH_BYTE_SIZE_THRESHOLD):
      self._num_do_batch_bytes.inc(1)
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
    accumulator.curr_byte_size += input_record_batch.nbytes
    self._maybe_do_batch(accumulator)
    self._num_instances.inc(num_rows)
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[_CombinerStatsGeneratorsCombineFnAcc]
      ) -> _CombinerStatsGeneratorsCombineFnAcc:
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      result.input_record_batches.extend(accumulator.input_record_batches)
      result.curr_batch_size += accumulator.curr_batch_size
      result.curr_byte_size += accumulator.curr_byte_size
      self._maybe_do_batch(result)
      result.partial_accumulators = self._for_each_generator(
          lambda gen, x, y: gen.merge_accumulators([x, y]),
          result.partial_accumulators,
          accumulator.partial_accumulators)

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
      self, accumulator: _CombinerStatsGeneratorsCombineFnAcc
  ) -> statistics_pb2.DatasetFeatureStatistics:
    # Make sure we have processed all the examples.
    self._maybe_do_batch(accumulator, force=True)
    generator_outputs = self._for_each_generator(
        lambda gen, acc: gen.extract_output(acc),
        accumulator.partial_accumulators)
    # TODO(b/202910677): We should consider returning a list directly and not
    # merging at all.
    merged = merge_util.merge_dataset_feature_statistics(generator_outputs)
    if len(merged.datasets) != 1:
      raise ValueError(
          'Expected a single slice key in _CombinerStatsGeneratorsCombineFn, '
          'got %d' % len(merged.datasets))
    return merged.datasets[0]


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
    columns, features = [], []
    for feature_name in options.feature_allowlist:
      c = arrow_util.get_column(record_batch, feature_name, missing_ok=True)
      if c is not None:
        columns.append(c)
        features.append(feature_name)
    record_batch = pa.RecordBatch.from_arrays(columns, features)
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
  return merge_util.merge_dataset_feature_statistics(outputs)


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
    if self._sample_rate is not None and random.random() > self._sample_rate:
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
