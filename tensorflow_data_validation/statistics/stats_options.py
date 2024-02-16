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

"""Statistics generation options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import types as python_types
from typing import Dict, List, Optional, Text, Union

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import example_weight_map
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import slicing_util
from tfx_bsl.arrow import sql_util
from tfx_bsl.coders import example_coder
from tfx_bsl.public.proto import slicing_spec_pb2

from google.protobuf import json_format
from tensorflow_metadata.proto.v0 import schema_pb2


_SCHEMA_JSON_KEY = 'schema_json'
_SLICING_CONFIG_JSON_KEY = 'slicing_config_json'
_PER_FEATURE_WEIGHT_OVERRIDE_JSON_KEY = 'per_feature_weight_override_json'
_TYPE_NAME_KEY = 'TYPE_NAME'


# TODO(b/181559345): Currently we use a single epsilon (error tolerance)
# parameter for all histograms. Set this parameter specific to each
# histogram based on the number of buckets.


# TODO(b/118833241): Set MI default configs when MI is a default generator.
class StatsOptions(object):
  """Options for generating statistics."""

  def __init__(
      self,
      generators: Optional[List[stats_generator.StatsGenerator]] = None,
      schema: Optional[schema_pb2.Schema] = None,
      label_feature: Optional[types.FeatureName] = None,
      weight_feature: Optional[types.FeatureName] = None,
      slice_functions: Optional[List[types.SliceFunction]] = None,
      sample_rate: Optional[float] = None,
      num_top_values: int = 20,
      frequency_threshold: int = 1,
      weighted_frequency_threshold: float = 1.0,
      num_rank_histogram_buckets: int = 1000,
      num_values_histogram_buckets: int = 10,
      num_histogram_buckets: int = 10,
      num_quantiles_histogram_buckets: int = 10,
      epsilon: float = 0.01,
      infer_type_from_schema: bool = False,
      desired_batch_size: Optional[int] = None,
      enable_semantic_domain_stats: bool = False,
      semantic_domain_stats_sample_rate: Optional[float] = None,
      per_feature_weight_override: Optional[
          Dict[types.FeaturePath, types.FeatureName]
      ] = None,
      vocab_paths: Optional[Dict[types.VocabName, types.VocabPath]] = None,
      add_default_generators: bool = True,
      # TODO(b/255895499): Support "from schema" for feature_allowlist.
      feature_allowlist: Optional[
          Union[List[types.FeatureName], List[types.FeaturePath]]
      ] = None,
      experimental_use_sketch_based_topk_uniques: Optional[bool] = None,
      use_sketch_based_topk_uniques: Optional[bool] = None,
      experimental_slice_functions: Optional[List[types.SliceFunction]] = None,
      experimental_slice_sqls: Optional[List[Text]] = None,
      experimental_result_partitions: int = 1,
      experimental_num_feature_partitions: int = 1,
      slicing_config: Optional[slicing_spec_pb2.SlicingConfig] = None,
      experimental_filter_read_paths: bool = False,
      per_feature_stats_config: Optional[types.PerFeatureStatsConfig] = None,
  ):
    """Initializes statistics options.

    Args:
      generators: An optional list of statistics generators. A statistics
        generator must extend either CombinerStatsGenerator or
        TransformStatsGenerator.
      schema: An optional tensorflow_metadata Schema proto. Currently we use the
        schema to infer categorical and bytes features.
      label_feature: An optional feature name which represents the label.
      weight_feature: An optional feature name whose numeric value represents
        the weight of an example.
      slice_functions: DEPRECATED. Use `experimental_slice_functions`.
      sample_rate: An optional sampling rate. If specified, statistics is
        computed over the sample.
      num_top_values: An optional number of most frequent feature values to keep
        for string features.
      frequency_threshold: An optional minimum number of examples the most
        frequent values must be present in.
      weighted_frequency_threshold: An optional minimum weighted number of
        examples the most frequent weighted values must be present in. This
        option is only relevant when a weight_feature is specified.
      num_rank_histogram_buckets: An optional number of buckets in the rank
        histogram for string features.
      num_values_histogram_buckets: An optional number of buckets in a quantiles
        histogram for the number of values per Feature, which is stored in
        CommonStatistics.num_values_histogram.
      num_histogram_buckets: An optional number of buckets in a standard
        NumericStatistics.histogram with equal-width buckets.
      num_quantiles_histogram_buckets: An optional number of buckets in a
        quantiles NumericStatistics.histogram.
      epsilon: An optional error tolerance for the computation of quantiles,
        typically a small fraction close to zero (e.g. 0.01). Higher values of
        epsilon increase the quantile approximation, and hence result in more
        unequal buckets, but could improve performance, and resource
        consumption.
      infer_type_from_schema: A boolean to indicate whether the feature types
        should be inferred from the schema. If set to True, an input schema must
        be provided. This flag is used only when invoking TFDV through
        `tfdv.generate_statistics_from_csv`.
      desired_batch_size: An optional maximum number of examples to include in
        each batch that is passed to the statistics generators. When invoking
        TFDV using its end-to-end APIs (e.g.
        `generate_statistics_from_tfrecord`), this option also controls the
        decoder batch size -- if provided, the decoded RecordBatches that are to
        be fed to TFDV will have the fixed batch size. When invoking TFDV using
        `tfdv.GenerateStatistics`, this option only controls the maximum size of
        RecordBatches constructed within StatsGenerators (a generator may
        combine RecordBatches).
      enable_semantic_domain_stats: If True statistics for semantic domains are
        generated (e.g: image, text domains).
      semantic_domain_stats_sample_rate: An optional sampling rate for semantic
        domain statistics. If specified, semantic domain statistics is computed
        over a sample.
      per_feature_weight_override: If specified, the "example weight" paired
        with a feature will be first looked up in this map and if not found,
        fall back to `weight_feature`.
      vocab_paths: An optional dictionary mapping vocab names to paths. Used in
        the schema when specifying a NaturalLanguageDomain. The paths can either
        be to GZIP-compressed TF record files that have a tfrecord.gz suffix or
        to text files.
      add_default_generators: Whether to invoke the default set of stats
        generators in the run. Generators invoked consists of 1) the default
        generators (controlled by this option); 2) user-provided generators (
        controlled by the `generators` option); 3) semantic generators
        (controlled by `enable_semantic_domain_stats`) and 4) schema-based
        generators that are enabled based on information provided in the schema.
      feature_allowlist: An optional list of names of the features to calculate
        statistics for, or a list of paths.
      experimental_use_sketch_based_topk_uniques: Deprecated, prefer
        use_sketch_based_topk_uniques.
      use_sketch_based_topk_uniques: if True, use the sketch based top-k and
        uniques stats generator.
      experimental_slice_functions: An optional list of functions that generate
        slice keys for each example. Each slice function should take
        pyarrow.RecordBatch as input and return an Iterable[Tuple[Text,
        pyarrow.RecordBatch]]. Each tuple contains the slice key and the
        corresponding sliced RecordBatch. Only one of
        experimental_slice_functions or experimental_slice_sqls must be
        specified.
      experimental_slice_sqls: List of slicing SQL queries. The query must have
        the following pattern: "SELECT STRUCT({feature_name} [AS {slice_key}])
        [FROM example.feature_name [, example.feature_name, ... ] [WHERE ... ]]"
        The “example.feature_name” inside the FROM statement is used to flatten
        the repeated fields. For non-repeated fields, you can directly write the
        query as follows: “SELECT STRUCT(non_repeated_feature_a,
        non_repeated_feature_b)” In the query, the “example” is a key word that
        binds to each input "row". The semantics of this variable will depend on
        the decoding of the input data to the Arrow representation (e.g., for
        tf.Example, each key is decoded to a separate column). Thus, structured
        data can be readily accessed by iterating/unnesting the fields of the
        "example" variable. Example 1: Slice on each value of a feature "SELECT
        STRUCT(gender) FROM example.gender" Example 2: Slice on each value of
        one feature and a specified value of another. "SELECT STRUCT(gender,
        country) FROM example.gender, example.country WHERE country = 'USA'"
        Only one of experimental_slice_functions or experimental_slice_sqls must
        be specified.
      experimental_result_partitions: The number of feature partitions to
        combine output DatasetFeatureStatisticsLists into. If set to 1 (default)
        output is globally combined. If set to value greater than one, up to
        that many shards are returned, each containing a subset of features.
      experimental_num_feature_partitions: If > 1, partitions computations by
        supported generators to act on this many bundles of features. For best
        results this should be set to at least several times less than the
        number of features in a dataset, and never more than the available beam
        parallelism.
      slicing_config: an optional SlicingConfig. SlicingConfig includes
        slicing_specs specified with feature keys, feature values or slicing SQL
        queries.
      experimental_filter_read_paths: If provided, tries to push down either
        paths passed via feature_allowlist or via the schema (in that priority)
        to the underlying read operation. Support depends on the file reader.
      per_feature_stats_config: Supports granular control of what statistics are
        enabled per feature. Experimental.
    """
    self.generators = generators
    self.feature_allowlist = feature_allowlist
    self.schema = schema
    self.label_feature = label_feature
    self.weight_feature = weight_feature
    if slice_functions is not None and experimental_slice_functions is not None:
      raise ValueError(
          'Specify only one of slice_functions or experimental_slice_functions')
    self.experimental_slice_functions = None
    if slice_functions is not None:
      self.experimental_slice_functions = slice_functions
    elif experimental_slice_functions is not None:
      self.experimental_slice_functions = experimental_slice_functions
    self.sample_rate = sample_rate
    self.num_top_values = num_top_values
    self.frequency_threshold = frequency_threshold
    self.weighted_frequency_threshold = weighted_frequency_threshold
    self.num_rank_histogram_buckets = num_rank_histogram_buckets
    self.num_values_histogram_buckets = num_values_histogram_buckets
    self.num_histogram_buckets = num_histogram_buckets
    self.num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    self.epsilon = epsilon
    self.infer_type_from_schema = infer_type_from_schema
    self.desired_batch_size = desired_batch_size
    self.enable_semantic_domain_stats = enable_semantic_domain_stats
    self.semantic_domain_stats_sample_rate = semantic_domain_stats_sample_rate
    self._per_feature_weight_override = per_feature_weight_override
    self.vocab_paths = vocab_paths
    self.add_default_generators = add_default_generators
    if (use_sketch_based_topk_uniques is not None and
        experimental_use_sketch_based_topk_uniques is not None):
      raise ValueError(
          'Must set at most one of use_sketch_based_topk_uniques and'
          ' experimental_use_sketch_based_topk_uniques')
    # TODO(b/239609486): Change the None default to True.
    if (
        experimental_use_sketch_based_topk_uniques
        or use_sketch_based_topk_uniques
    ):
      self.use_sketch_based_topk_uniques = True
    else:
      self.use_sketch_based_topk_uniques = False
    self.experimental_slice_sqls = experimental_slice_sqls
    self.experimental_num_feature_partitions = (
        experimental_num_feature_partitions
    )
    self.experimental_result_partitions = experimental_result_partitions
    self.slicing_config = slicing_config
    self.experimental_filter_read_paths = experimental_filter_read_paths
    self.per_feature_stats_config = per_feature_stats_config

  def __repr__(self):
    return '<{}>'.format(', '.join(
        '{}={!r}'.format(k, v) for k, v in self.__dict__.items()))

  def to_json(self) -> Text:
    """Convert from an object to JSON representation of the __dict__ attribute.

    Custom generators and slice_functions cannot being converted. As a result,
    a ValueError will be raised when these options are specified and TFDV is
    running in a setting where the stats options have been json-serialized,
    first. This will happen in the case where TFDV is run as a TFX component.
    The schema proto and slicing_config will be json_encoded.

    Returns:
      A JSON representation of a filtered version of __dict__.
    """
    options_dict = copy.copy(self.__dict__)
    options_dict[_TYPE_NAME_KEY] = 'StatsOptions'
    if options_dict['_slice_functions'] is not None:
      raise ValueError(
          'StatsOptions cannot be converted with experimental_slice_functions.'
      )
    if options_dict['_generators'] is not None:
      raise ValueError(
          'StatsOptions cannot be converted with generators.'
      )
    if self.schema is not None:
      del options_dict['_schema']
      options_dict[_SCHEMA_JSON_KEY] = json_format.MessageToJson(self.schema)
    if self.slicing_config is not None:
      del options_dict['_slicing_config']
      options_dict[_SLICING_CONFIG_JSON_KEY] = json_format.MessageToJson(
          self.slicing_config)
    if self._per_feature_weight_override is not None:
      del options_dict['_per_feature_weight_override']
      options_dict[_PER_FEATURE_WEIGHT_OVERRIDE_JSON_KEY] = {
          k.to_json(): v for k, v in self._per_feature_weight_override.items()
      }
    if self._per_feature_stats_config is not None:
      raise ValueError(
          'StatsOptions cannot be converted with per_feature_stats_config.'
      )
    return json.dumps(options_dict)

  @classmethod
  def from_json(cls, options_json: Text) -> 'StatsOptions':
    """Construct an instance of stats options from a JSON representation.

    Args:
      options_json: A JSON representation of the __dict__ attribute of a
        StatsOptions instance.

    Returns:
      A StatsOptions instance constructed by setting the __dict__ attribute to
      the deserialized value of options_json.
    """
    options_dict = json.loads(options_json)
    type_name = options_dict.pop(_TYPE_NAME_KEY, None)
    if type_name is not None and type_name != 'StatsOptions':
      raise ValueError('JSON does not encode a StatsOptions')
    if _SCHEMA_JSON_KEY in options_dict:
      options_dict['_schema'] = json_format.Parse(
          options_dict[_SCHEMA_JSON_KEY], schema_pb2.Schema())
      del options_dict[_SCHEMA_JSON_KEY]
    if _SLICING_CONFIG_JSON_KEY in options_dict:
      options_dict['_slicing_config'] = json_format.Parse(
          options_dict[_SLICING_CONFIG_JSON_KEY],
          slicing_spec_pb2.SlicingConfig())
      del options_dict[_SLICING_CONFIG_JSON_KEY]
    per_feature_weight_override_json = options_dict.get(
        _PER_FEATURE_WEIGHT_OVERRIDE_JSON_KEY)
    if per_feature_weight_override_json is not None:
      options_dict['_per_feature_weight_override'] = {
          types.FeaturePath.from_json(k): v
          for k, v in per_feature_weight_override_json.items()
      }
      del options_dict[_PER_FEATURE_WEIGHT_OVERRIDE_JSON_KEY]
    options = cls()
    options.__dict__ = options_dict
    return options

  @property
  def generators(self) -> Optional[List[stats_generator.StatsGenerator]]:
    return self._generators

  @generators.setter
  def generators(
      self, generators: Optional[List[stats_generator.StatsGenerator]]) -> None:
    if generators is not None:
      if not isinstance(generators, list):
        raise TypeError('generators is of type %s, should be a list.' %
                        type(generators).__name__)
      for generator in generators:
        if not isinstance(generator, (
            stats_generator.CombinerStatsGenerator,
            stats_generator.TransformStatsGenerator,
            stats_generator.CombinerFeatureStatsGenerator,
        )):
          raise TypeError(
              'Statistics generator must extend one of '
              'CombinerStatsGenerator, TransformStatsGenerator, or '
              'CombinerFeatureStatsGenerator found object of type %s.' %
              generator.__class__.__name__)
    self._generators = generators

  @property
  def feature_allowlist(
      self
  ) -> Optional[Union[List[types.FeatureName], List[types.FeaturePath]]]:
    return self._feature_allowlist

  @feature_allowlist.setter
  def feature_allowlist(
      self, feature_allowlist: Optional[Union[List[types.FeatureName],
                                              List[types.FeaturePath]]]
  ) -> None:
    if feature_allowlist is not None and not isinstance(feature_allowlist,
                                                        list):
      raise TypeError('feature_allowlist is of type %s, should be a list.' %
                      type(feature_allowlist).__name__)
    self._feature_allowlist = feature_allowlist

  @property
  def schema(self) -> Optional[schema_pb2.Schema]:
    return self._schema

  @schema.setter
  def schema(self, schema: Optional[schema_pb2.Schema]) -> None:
    if schema is not None and not isinstance(schema, schema_pb2.Schema):
      raise TypeError('schema is of type %s, should be a Schema proto.' %
                      type(schema).__name__)
    self._schema = schema

  @property
  def vocab_paths(self) -> Optional[Dict[types.VocabName, types.VocabPath]]:
    return self._vocab_paths

  @vocab_paths.setter
  def vocab_paths(
      self, vocab_paths: Optional[Dict[types.VocabName,
                                       types.VocabPath]]) -> None:
    if vocab_paths is not None and not isinstance(vocab_paths, dict):
      raise TypeError('vocab_paths is of type %s, should be a dict.' %
                      type(vocab_paths).__name__)
    self._vocab_paths = vocab_paths

  @property
  def experimental_slice_functions(self) -> Optional[List[types.SliceFunction]]:
    return self._slice_functions

  @experimental_slice_functions.setter
  def experimental_slice_functions(
      self, slice_functions: Optional[List[types.SliceFunction]]) -> None:
    if hasattr(self, 'experimental_slice_sqls'):
      _validate_slicing_options(slice_functions, self.experimental_slice_sqls)
    if slice_functions is not None:
      if not isinstance(slice_functions, list):
        raise TypeError(
            'experimental_slice_functions is of type %s, should be a list.' %
            type(slice_functions).__name__)
      for slice_function in slice_functions:
        if not isinstance(slice_function, python_types.FunctionType):
          raise TypeError(
              'experimental_slice_functions must contain functions only.')
    self._slice_functions = slice_functions

  @property
  def experimental_slice_sqls(self) -> Optional[List[Text]]:
    return self._slice_sqls

  @experimental_slice_sqls.setter
  def experimental_slice_sqls(self, slice_sqls: Optional[List[Text]]) -> None:
    if hasattr(self, 'experimental_slice_functions'):
      _validate_slicing_options(self.experimental_slice_functions, slice_sqls)
    if slice_sqls and self.schema:
      for slice_sql in slice_sqls:
        _validate_sql(slice_sql, self.schema)
    self._slice_sqls = slice_sqls

  @property
  def slicing_config(self) -> Optional[slicing_spec_pb2.SlicingConfig]:
    return self._slicing_config

  @slicing_config.setter
  def slicing_config(
      self, slicing_config: Optional[slicing_spec_pb2.SlicingConfig]) -> None:
    _validate_slicing_config(slicing_config)

    if slicing_config and self.experimental_slice_functions:
      raise ValueError(
          'Specify only one of slicing_config or experimental_slice_functions.')

    if slicing_config and self.experimental_slice_sqls:
      raise ValueError(
          'Specify only one of slicing_config or experimental_slice_sqls.')

    self._slicing_config = slicing_config

  @property
  def sample_rate(self) -> Optional[float]:
    return self._sample_rate

  @sample_rate.setter
  def sample_rate(self, sample_rate: Optional[float]):
    if sample_rate is not None:
      if not 0 < sample_rate <= 1:
        raise ValueError('Invalid sample_rate %f' % sample_rate)
    self._sample_rate = sample_rate

  @property
  def num_values_histogram_buckets(self) -> int:
    return self._num_values_histogram_buckets

  @num_values_histogram_buckets.setter
  def num_values_histogram_buckets(self,
                                   num_values_histogram_buckets: int) -> None:
    # TODO(b/120164508): Disallow num_values_histogram_buckets = 1 because it
    # causes the underlying quantile op to fail. If the quantile op is modified
    # to support num_quantiles = 1, then allow num_values_histogram_buckets = 1.
    if num_values_histogram_buckets <= 1:
      raise ValueError('Invalid num_values_histogram_buckets %d' %
                       num_values_histogram_buckets)
    self._num_values_histogram_buckets = num_values_histogram_buckets

  @property
  def num_histogram_buckets(self) -> int:
    return self._num_histogram_buckets

  @num_histogram_buckets.setter
  def num_histogram_buckets(self, num_histogram_buckets: int) -> None:
    if num_histogram_buckets < 1:
      raise ValueError(
          'Invalid num_histogram_buckets %d' % num_histogram_buckets)
    self._num_histogram_buckets = num_histogram_buckets

  @property
  def num_quantiles_histogram_buckets(self) -> int:
    return self._num_quantiles_histogram_buckets

  @num_quantiles_histogram_buckets.setter
  def num_quantiles_histogram_buckets(
      self, num_quantiles_histogram_buckets: int) -> None:
    if num_quantiles_histogram_buckets < 1:
      raise ValueError('Invalid num_quantiles_histogram_buckets %d' %
                       num_quantiles_histogram_buckets)
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets

  @property
  def desired_batch_size(self) -> Optional[int]:
    return self._desired_batch_size

  @desired_batch_size.setter
  def desired_batch_size(self, desired_batch_size: Optional[int]) -> None:
    if desired_batch_size is not None and desired_batch_size < 1:
      raise ValueError('Invalid desired_batch_size %d' %
                       desired_batch_size)
    self._desired_batch_size = desired_batch_size

  @property
  def semantic_domain_stats_sample_rate(self) -> Optional[float]:
    return self._semantic_domain_stats_sample_rate

  @semantic_domain_stats_sample_rate.setter
  def semantic_domain_stats_sample_rate(
      self, semantic_domain_stats_sample_rate: Optional[float]):
    if semantic_domain_stats_sample_rate is not None:
      if not 0 < semantic_domain_stats_sample_rate <= 1:
        raise ValueError('Invalid semantic_domain_stats_sample_rate %f'
                         % semantic_domain_stats_sample_rate)
    self._semantic_domain_stats_sample_rate = semantic_domain_stats_sample_rate

  @property
  def example_weight_map(self):
    return example_weight_map.ExampleWeightMap(
        self.weight_feature, self._per_feature_weight_override)

  @property
  def add_default_generators(self) -> bool:
    return self._add_default_generators

  @add_default_generators.setter
  def add_default_generators(self, add_default_generators: bool) -> None:
    self._add_default_generators = add_default_generators

  @property
  def use_sketch_based_topk_uniques(self) -> bool:
    return self._use_sketch_based_topk_uniques

  @use_sketch_based_topk_uniques.setter
  def use_sketch_based_topk_uniques(
      self, use_sketch_based_topk_uniques: bool) -> None:
    # Check that if sketch based generators are turned off we don't have any
    # categorical float features in the schema.
    if (self.schema and not use_sketch_based_topk_uniques and
        schema_pb2.FLOAT in schema_util.get_categorical_numeric_feature_types(
            self.schema).values()):
      raise ValueError('Categorical float features set in schema require '
                       'use_sketch_based_topk_uniques')
    self._use_sketch_based_topk_uniques = use_sketch_based_topk_uniques

  # TODO(b/239609486): Deprecate this alias.
  @property
  def experimental_use_sketch_based_topk_uniques(self) -> bool:
    return self.use_sketch_based_topk_uniques

  @experimental_use_sketch_based_topk_uniques.setter
  def experimental_use_sketch_based_topk_uniques(
      self, use_sketch_based_topk_uniques: bool
  ) -> None:
    self.use_sketch_based_topk_uniques = use_sketch_based_topk_uniques

  @property
  def experimental_result_partitions(self) -> int:
    return self._experimental_result_partitions

  @experimental_result_partitions.setter
  def experimental_result_partitions(self, num_partitions: int) -> None:
    if num_partitions > 0:
      self._experimental_result_partitions = num_partitions
    else:
      raise ValueError(
          'Unsupported experimental_result_partitions <= 0: %d' %
          num_partitions)

  @property
  def experimental_num_feature_partitions(self) -> int:
    return self._experimental_num_feature_partitions

  @experimental_num_feature_partitions.setter
  def experimental_num_feature_partitions(self,
                                          feature_partitions: int) -> None:
    if feature_partitions <= 0:
      raise ValueError('experimental_num_feature_partitions must be > 0.')
    self._experimental_num_feature_partitions = feature_partitions

  @property
  def experimental_filter_read_paths(self) -> bool:
    return self._experimental_filter_read_paths

  @experimental_filter_read_paths.setter
  def experimental_filter_read_paths(self, filter_read: bool) -> None:
    self._experimental_filter_read_paths = filter_read

  @property
  def per_feature_stats_config(self) -> types.PerFeatureStatsConfig:
    return (
        self._per_feature_stats_config or types.PerFeatureStatsConfig.default()
    )

  @per_feature_stats_config.setter
  def per_feature_stats_config(
      self, features_config: types.PerFeatureStatsConfig
  ) -> None:
    self._per_feature_stats_config = features_config


def _validate_sql(sql_query: Text, schema: schema_pb2.Schema):
  arrow_schema = example_coder.ExamplesToRecordBatchDecoder(
      schema.SerializeToString()).ArrowSchema()
  formatted_query = slicing_util.format_slice_sql_query(sql_query)
  try:
    sql_util.RecordBatchSQLSliceQuery(formatted_query, arrow_schema)
  except Exception as e:  # pylint: disable=broad-except
    # The schema passed to TFDV initially may be incomplete, so we can't crash
    # on what may be an error caused by missing features.
    logging.error('One of the slice SQL query %s raised an exception: %s.',
                  sql_query, repr(e))


def _validate_slicing_options(
    slice_fns: Optional[List[types.SliceFunction]] = None,
    slice_sqls: Optional[List[Text]] = None):
  if slice_fns and slice_sqls:
    raise ValueError('Only one of experimental_slice_functions or '
                     'experimental_slice_sqls must be specified.')


def _validate_slicing_config(
    slicing_config: Optional[slicing_spec_pb2.SlicingConfig]):
  """Validates slicing config.

  Args:
    slicing_config: an optional list of slicing specifications. Slicing
    specifications can be provided by feature keys, feature values or slicing
    SQL queries.
  Returns:
    None if slicing_config is None.
  Raises:
    ValueError: If both slicing functions and slicing sql queries are specified
    in the slicing config.
  """
  if slicing_config is None:
    return

  has_slice_fns, has_slice_sqls = False, False

  for slicing_spec in slicing_config.slicing_specs:
    if (not has_slice_fns) and (slicing_spec.feature_keys or
                                slicing_spec.feature_values):
      has_slice_fns = True
    if (not has_slice_sqls) and slicing_spec.slice_keys_sql:
      has_slice_sqls = True

    if has_slice_fns and has_slice_sqls:
      raise ValueError(
          'Only one of slicing features or slicing sql queries can be '
          'specified in the slicing config.')
