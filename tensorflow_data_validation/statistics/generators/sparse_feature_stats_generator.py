# Copyright 2019 Google LLC
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
"""Module that computes statistics used to validate sparse features.

Currently, this module generates the following statistics for each
sparse feature:
- missing_value: Number of examples missing the value_feature.
- missing_index: A RankHistogram from index_name to the number of examples
                 missing the corresponding index_feature.
- min_length_diff: A RankHistogram from index_name to the minimum of
                   len(index_feature) - len(value_feature).
- max_length_diff: A RankHistogram from index_name to the maximum of
                   len(index_feature) - len(value_feature).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Iterable, List, Text, Tuple, Union
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.statistics.generators.constituents import count_missing_generator
from tensorflow_data_validation.statistics.generators.constituents import length_diff_generator

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order

# LINT.IfChange(custom_stat_names)
_MAX_LENGTH_DIFF_NAME = 'max_length_diff'
_MIN_LENGTH_DIFF_NAME = 'min_length_diff'
_MISSING_INDEX_NAME = 'missing_index'
_MISSING_VALUE_NAME = 'missing_value'
# LINT.ThenChange(../../anomalies/schema.cc:sparse_feature_custom_stat_names)

# Named tuple containing the FeaturePaths for the value and index features
# that comprise a given sparse feature.
_SparseFeatureComponents = tfx_namedtuple.namedtuple(
    '_SparseFeatureComponents', ['value_feature', 'index_features'])


def _get_all_sparse_features(
    schema: schema_pb2.Schema
) -> List[Tuple[types.FeaturePath, schema_pb2.SparseFeature]]:
  """Returns all sparse features in a schema."""

  def _recursion_helper(
      parent_path: types.FeaturePath, container: Union[schema_pb2.Schema,
                                                       schema_pb2.StructDomain]
  ) -> List[Tuple[types.FeaturePath, schema_pb2.SparseFeature]]:
    """Helper function that is used in finding sparse features in a tree."""
    result = []
    for sf in container.sparse_feature:
      # Sparse features do not have a struct_domain, so they cannot be parent
      # features. Thus, once this reaches a sparse feature, add it to the
      # result.
      result.append((parent_path.child(sf.name), sf))
    for f in container.feature:
      if f.type == schema_pb2.STRUCT:
        result.extend(
            _recursion_helper(parent_path.child(f.name), f.struct_domain))
    return result

  return _recursion_helper(types.FeaturePath([]), schema)


def _get_components(
    sparse_features: Iterable[Tuple[types.FeaturePath,
                                    schema_pb2.SparseFeature]]
) -> Dict[types.FeaturePath, _SparseFeatureComponents]:
  """Returns the index and value feature paths that comprise sparse features."""
  # A dict mapping sparse feature paths to their component index and value
  # feature paths.
  sparse_feature_components = dict()
  # The index and value features for a given sparse feature have the same parent
  # path as the sparse feature.
  for path, feature in sparse_features:
    parent_path = path.parent()
    value_feature = parent_path.child(feature.value_feature.name)
    index_features = set()
    for index_feature in feature.index_feature:
      index_features.add(parent_path.child(index_feature.name))
    sparse_feature_components[path] = _SparseFeatureComponents(
        value_feature, index_features)
  return sparse_feature_components


class SparseFeatureStatsGenerator(stats_generator.CompositeStatsGenerator):
  """Generates statistics for sparse features."""

  def __init__(self,
               schema: schema_pb2.Schema,
               name: Text = 'SparseFeatureStatsGenerator') -> None:
    """Initializes a sparse feature statistics generator.

    Args:
      schema: A required schema for the dataset.
      name: An optional unique name associated with the statistics generator.
    """
    self._sparse_feature_components = _get_components(
        _get_all_sparse_features(schema))

    # Create length diff generators for each index / value pair and count
    # missing generator for all paths.
    constituents = []
    for _, (value, indices) in self._sparse_feature_components.items():
      required_paths = [value] + list(indices)
      constituents.append(
          count_missing_generator.CountMissingGenerator(value, required_paths))
      for index in indices:
        constituents.append(
            length_diff_generator.LengthDiffGenerator(index, value,
                                                      required_paths))
        constituents.append(
            count_missing_generator.CountMissingGenerator(
                index, required_paths))

    super(SparseFeatureStatsGenerator, self).__init__(name, constituents,
                                                      schema)

  def extract_composite_output(self, accumulator):
    stats = statistics_pb2.DatasetFeatureStatistics()
    for feature_path, (value,
                       indices) in self._sparse_feature_components.items():
      required_paths = [value] + list(indices)
      feature_stats = stats.features.add(path=feature_path.to_proto())
      feature_stats.custom_stats.add(
          name=_MISSING_VALUE_NAME,
          num=accumulator[count_missing_generator.CountMissingGenerator.key(
              value, required_paths)])
      index_features_num_missing_histogram = statistics_pb2.RankHistogram()
      max_length_diff_histogram = statistics_pb2.RankHistogram()
      min_length_diff_histogram = statistics_pb2.RankHistogram()
      for index in sorted(indices):
        index_label = index.steps()[-1]
        missing_bucket = index_features_num_missing_histogram.buckets.add()
        missing_bucket.label = index_label
        missing_bucket.sample_count = accumulator[
            count_missing_generator.CountMissingGenerator.key(
                index, required_paths)]

        min_diff, max_diff = accumulator[
            length_diff_generator.LengthDiffGenerator.key(
                index, value, required_paths)]
        max_length_bucket = max_length_diff_histogram.buckets.add()
        max_length_bucket.label = index_label
        max_length_bucket.sample_count = max_diff

        min_length_bucket = min_length_diff_histogram.buckets.add()
        min_length_bucket.label = index_label
        min_length_bucket.sample_count = min_diff

      feature_stats.custom_stats.add(
          name=_MISSING_INDEX_NAME,
          rank_histogram=index_features_num_missing_histogram)
      feature_stats.custom_stats.add(
          name=_MAX_LENGTH_DIFF_NAME, rank_histogram=max_length_diff_histogram)
      feature_stats.custom_stats.add(
          name=_MIN_LENGTH_DIFF_NAME, rank_histogram=min_length_diff_histogram)
    return stats
