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

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.types_compat import List, Optional

from tensorflow_metadata.proto.v0 import schema_pb2




class StatsOptions(object):
  """Options for generating statistics."""

  def __init__(
      self,
      generators = None,
      feature_whitelist = None,
      schema = None,
      weight_feature = None,
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
    self.sample_count = sample_count
    self.sample_rate = sample_rate
    self.num_top_values = num_top_values
    self.num_rank_histogram_buckets = num_rank_histogram_buckets
    self.num_values_histogram_buckets = num_values_histogram_buckets
    self.num_histogram_buckets = num_histogram_buckets
    self.num_quantiles_histogram_buckets = num_quantiles_histogram_buckets
    self.epsilon = epsilon
    self.infer_type_from_schema = infer_type_from_schema
