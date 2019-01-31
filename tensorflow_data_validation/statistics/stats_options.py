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

import logging
import types as python_types
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.types_compat import List, Optional

from tensorflow_metadata.proto.v0 import schema_pb2


# TODO(b/68277922): Currently we use a single epsilon (error tolerance)
# parameter for all histograms. Set this parameter specific to each
# histogram based on the number of buckets.


# TODO(b/118833241): Set MI default configs when MI is a default generator
class StatsOptions(object):
  """Options for generating statistics."""

  def __init__(
      self,
      generators = None,
      feature_whitelist = None,
      schema = None,
      weight_feature = None,
      slice_functions = None,
      sample_count = None,
      sample_rate = None,
      num_top_values = 20,
      num_rank_histogram_buckets = 1000,
      num_values_histogram_buckets = 10,
      num_histogram_buckets = 10,
      num_quantiles_histogram_buckets = 10,
      epsilon = 0.01,
      infer_type_from_schema = False
      ):
    """Initializes statistics options.

    Args:
      generators: An optional list of statistics generators. A statistics
        generator must extend either CombinerStatsGenerator or
        TransformStatsGenerator.
      feature_whitelist: An optional list of names of the features to calculate
        statistics for.
      schema: An optional tensorflow_metadata Schema proto. Currently we use the
        schema to infer categorical and bytes features.
      weight_feature: An optional feature name whose numeric value represents
          the weight of an example.
      slice_functions: An optional list of functions that generate slice keys
        for each example. Each slice function should take an example dict as
        input and return a list of zero or more slice keys.
      sample_count: An optional number of examples to include in the sample. If
        specified, statistics is computed over the sample. Only one of
        sample_count or sample_rate can be specified.
      sample_rate: An optional sampling rate. If specified, statistics is
        computed over the sample. Only one of sample_count or sample_rate can
        be specified.
      num_top_values: An optional number of most frequent feature values to keep
        for string features.
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
          should be inferred from the schema. If set to True, an input schema
          must be provided. This flag is used only when generating statistics
          on CSV data.
    """
    self.generators = generators
    self.feature_whitelist = feature_whitelist
    self.schema = schema
    self.weight_feature = weight_feature
    self.slice_functions = slice_functions
    self.sample_count = sample_count
    self.sample_rate = sample_rate
    self.num_top_values = num_top_values
    self.num_rank_histogram_buckets = num_rank_histogram_buckets
    self.num_values_histogram_buckets = num_values_histogram_buckets
    self.num_histogram_buckets = num_histogram_buckets
    self.num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    self.epsilon = epsilon
    self.infer_type_from_schema = infer_type_from_schema

  @property
  def generators(self):
    return self._generators

  @generators.setter
  def generators(
      self, generators):
    if generators is not None:
      if not isinstance(generators, list):
        raise TypeError('generators is of type %s, should be a list.' %
                        type(generators).__name__)
      for generator in generators:
        if not isinstance(generator, (stats_generator.CombinerStatsGenerator,
                                      stats_generator.TransformStatsGenerator)):
          raise TypeError(
              'Statistics generator must extend one of '
              'CombinerStatsGenerator or TransformStatsGenerator, '
              'found object of type %s.' % generator.__class__.__name__)
    self._generators = generators

  @property
  def feature_whitelist(self):
    return self._feature_whitelist

  @feature_whitelist.setter
  def feature_whitelist(
      self, feature_whitelist):
    if feature_whitelist is not None and not isinstance(feature_whitelist,
                                                        list):
      raise TypeError('feature_whitelist is of type %s, should be a list.' %
                      type(feature_whitelist).__name__)
    self._feature_whitelist = feature_whitelist

  @property
  def schema(self):
    return self._schema

  @schema.setter
  def schema(self, schema):
    if schema is not None and not isinstance(schema, schema_pb2.Schema):
      raise TypeError('schema is of type %s, should be a Schema proto.' %
                      type(schema).__name__)
    self._schema = schema

  @property
  def slice_functions(self):
    return self._slice_functions

  @slice_functions.setter
  def slice_functions(
      self, slice_functions):
    if slice_functions is not None:
      logging.warning('Slicing is currently not supported.')
      if not isinstance(slice_functions, list):
        raise TypeError('slice_functions is of type %s, should be a list.' %
                        type(slice_functions).__name__)
      for slice_function in slice_functions:
        if not isinstance(slice_function, python_types.FunctionType):
          raise TypeError('slice_functions must contain functions only.')

  @property
  def sample_count(self):
    return self._sample_count

  @sample_count.setter
  def sample_count(self, sample_count):
    if sample_count is not None:
      if hasattr(self, 'sample_rate') and self.sample_rate is not None:
        raise ValueError('Only one of sample_count or sample_rate can be '
                         'specified.')
      if sample_count < 1:
        raise ValueError('Invalid sample_count %d' % sample_count)
    self._sample_count = sample_count

  @property
  def sample_rate(self):
    return self._sample_rate

  @sample_rate.setter
  def sample_rate(self, sample_rate):
    if sample_rate is not None:
      if hasattr(self, 'sample_count') and self.sample_count is not None:
        raise ValueError('Only one of sample_count or sample_rate can be '
                         'specified.')
      if not 0 < sample_rate <= 1:
        raise ValueError('Invalid sample_rate %f' % sample_rate)
    self._sample_rate = sample_rate

  @property
  def num_values_histogram_buckets(self):
    return self._num_values_histogram_buckets

  @num_values_histogram_buckets.setter
  def num_values_histogram_buckets(self,
                                   num_values_histogram_buckets):
    # TODO(b/120164508): Disallow num_values_histogram_buckets = 1 because it
    # causes the underlying quantile op to fail. If the quantile op is modified
    # to support num_quantiles = 1, then allow num_values_histogram_buckets = 1.
    if num_values_histogram_buckets <= 1:
      raise ValueError('Invalid num_values_histogram_buckets %d' %
                       num_values_histogram_buckets)
    self._num_values_histogram_buckets = num_values_histogram_buckets

  @property
  def num_histogram_buckets(self):
    return self._num_histogram_buckets

  @num_histogram_buckets.setter
  def num_histogram_buckets(self, num_histogram_buckets):
    if num_histogram_buckets < 1:
      raise ValueError(
          'Invalid num_histogram_buckets %d' % num_histogram_buckets)
    self._num_histogram_buckets = num_histogram_buckets

  @property
  def num_quantiles_histogram_buckets(self):
    return self._num_quantiles_histogram_buckets

  @num_quantiles_histogram_buckets.setter
  def num_quantiles_histogram_buckets(
      self, num_quantiles_histogram_buckets):
    if num_quantiles_histogram_buckets < 1:
      raise ValueError('Invalid num_quantiles_histogram_buckets %d' %
                       num_quantiles_histogram_buckets)
    self._num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
