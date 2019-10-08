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

import collections
import numpy as np

from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tfx_bsl.arrow import array_util
from typing import Dict, Iterable, List, Optional, Text, Tuple, Set, Union

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# Named tuple containing the FeaturePaths for the value and index features
# that comprise a given sparse feature.
_SparseFeatureComponents = collections.namedtuple(
    '_SparseFeatureComponents', ['value_feature', 'index_features'])


def _look_up_sparse_feature(
    feature_name: types.FeatureName,
    container: Iterable[schema_pb2.SparseFeature]
) -> Optional[schema_pb2.SparseFeature]:
  for f in container:
    if f.name == feature_name:
      return f
  return None


def _get_sparse_feature(
    schema: schema_pb2.Schema, feature_path: types.FeaturePath
) -> schema_pb2.SparseFeature:
  """Returns a sparse feature from the schema."""
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  feature_container = None
  parent = feature_path.parent()
  if parent:
    # Sparse features do not have a struct_domain and so can be only leaves.
    # Thus, we can assume that all parent steps are features, not sparse
    # features.
    feature_container = schema.feature
    for step in parent.steps():
      f = schema_util.look_up_feature(step, feature_container)
      if f is None:
        raise ValueError('Feature %s not found in the schema.' % feature_path)
      if f.type != schema_pb2.STRUCT:
        raise ValueError(
            'Step %s in feature %s does not refer to a valid STRUCT feature' %
            (step, feature_path))
      feature_container = f.struct_domain.sparse_feature

  if feature_container is None:
    feature_container = schema.sparse_feature
  feature = _look_up_sparse_feature(feature_path.steps()[-1], feature_container)
  if feature is None:
    raise ValueError('Sparse Feature %s not found in the schema.' %
                     feature_path)
  return feature


def _get_all_index_and_value_feature_paths(
    sparse_features: Iterable[Tuple[types.FeaturePath,
                                    schema_pb2.SparseFeature]]
) -> Tuple[Set[types.FeaturePath], Set[types.FeaturePath], Dict[
    types.FeaturePath, _SparseFeatureComponents]]:
  """Returns the index and value feature paths that comprise sparse features."""
  # The set of feature paths for any feature that is an index feature for at
  # least one sparse feature.
  all_index_features = set()
  # The set of feature paths for any feature that is a value feature for at
  # least one sparse feature.
  all_value_features = set()
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
    all_value_features.add(value_feature)
    all_index_features = all_index_features.union(index_features)
  return (all_index_features, all_value_features, sparse_feature_components)


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


class _PartialSparseFeatureStats(object):
  """Holds partial statistics for a single sparse feature."""

  __slots__ = [
      'value_feature_num_missing', 'index_features_num_missing',
      'index_features_min_length_diff', 'index_features_max_length_diff'
  ]

  def __init__(
      self,
      value_feature_num_missing: int = 0,
      index_features_num_missing: Optional[Dict[types.FeaturePath, int]] = None,
      index_features_min_length_diff: Optional[Dict[types.FeaturePath,
                                                    int]] = None,
      index_features_max_length_diff: Optional[Dict[types.FeaturePath,
                                                    int]] = None
  ) -> None:
    # The number of examples missing for the value feature.
    self.value_feature_num_missing = value_feature_num_missing
    # The number of examples missing for each index feature.
    self.index_features_num_missing = index_features_num_missing
    if self.index_features_num_missing is None:
      self.index_features_num_missing = collections.Counter()
    # The minimum of len(index feature) - len(value feature) within any example
    # for each index feature.
    self.index_features_min_length_diff = index_features_min_length_diff
    if self.index_features_min_length_diff is None:
      self.index_features_min_length_diff = {}
    # The maximum of len(index feature) - len(value feature) within any example
    # for each index feature.
    self.index_features_max_length_diff = index_features_max_length_diff
    if self.index_features_max_length_diff is None:
      self.index_features_max_length_diff = {}

  def __add__(
      self,
      other: '_PartialSparseFeatureStats') -> '_PartialSparseFeatureStats':
    """Merge two partial sparse feature stats and return the merged result."""
    self.value_feature_num_missing += other.value_feature_num_missing
    self.index_features_num_missing.update(other.index_features_num_missing)
    combined_index_feature_keys = list(
        self.index_features_min_length_diff) + list(
            other.index_features_min_length_diff)
    merged_index_features_min_length_diff = {}
    merged_index_features_max_length_diff = {}
    for index_feature in combined_index_feature_keys:
      min_length_diffs = [
          min_length_diff.get(index_feature) for min_length_diff in [
              self.index_features_min_length_diff,
              other.index_features_min_length_diff
          ]
      ]
      # At least one of the _PartialSparseFeatureStats should have
      # index_features_min_length_diff populated for each index feature, so we
      # can call min() on the list of min_length_diffs and know that that list
      # will contain at least one value (and so won't fail). (This is also the
      # case for max length diff.)
      merged_index_features_min_length_diff[index_feature] = min(
          min_length_diff for min_length_diff in min_length_diffs
          if min_length_diff is not None)
      max_length_diffs = [
          max_length_diff.get(index_feature) for max_length_diff in [
              self.index_features_max_length_diff,
              other.index_features_max_length_diff
          ]
      ]
      merged_index_features_max_length_diff[index_feature] = max(
          max_length_diff for max_length_diff in max_length_diffs
          if max_length_diff is not None)
    self.index_features_min_length_diff = merged_index_features_min_length_diff
    self.index_features_max_length_diff = merged_index_features_max_length_diff
    return self


class SparseFeatureStatsGenerator(stats_generator.CombinerStatsGenerator):
  """Generates statistics for sparse features."""

  def __init__(self,
               schema: schema_pb2.Schema,
               name: Text = 'SparseFeatureStatsGenerator') -> None:
    """Initializes a sparse feature statistics generator.

    Args:
      schema: A required schema for the dataset.
      name: An optional unique name associated with the statistics generator.
    """
    super(SparseFeatureStatsGenerator, self).__init__(name, schema)
    self._name = name
    self._schema = schema
    (self._all_index_feature_paths, self._all_value_feature_paths,
     self._sparse_feature_component_paths) = (
         _get_all_index_and_value_feature_paths(
             _get_all_sparse_features(schema)))

  def create_accumulator(
      self) -> Dict[types.FeaturePath, _PartialSparseFeatureStats]:
    """Returns an accumulator mapping sparse features to their partial stats."""
    return {}

  def add_input(
      self, accumulator: Dict[types.FeaturePath,
                              _PartialSparseFeatureStats], input_table: pa.Table
  ) -> Dict[types.FeaturePath, _PartialSparseFeatureStats]:
    """Returns result of folding a batch of inputs into the current accumulator.

    Args:
      accumulator: The current accumulator.
      input_table: An Arrow Table whose columns are features and rows are
        examples.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    component_feature_value_list_lengths = dict()
    component_feature_num_missing = dict()
    batch_example_count = input_table.num_rows
    # Do a single pass through the input table to determine the value list
    # lengths and number missing for every feature that is an index or value
    # feature in any sparse feature in the schema.
    for feature_path, leaf_array, _ in arrow_util.enumerate_arrays(
        input_table, weight_column=None, enumerate_leaves_only=True):
      if (feature_path in self._all_index_feature_paths or
          feature_path in self._all_value_feature_paths):
        # If the column is a NullArray, skip it when populating the
        # component_feature_ dicts. Features that are missing from those dicts
        # are treated as entirely missing for the batch.
        if not pa.types.is_null(leaf_array.type):
          component_feature_value_list_lengths[
              feature_path] = arrow_util.primitive_array_to_numpy(
                  array_util.ListLengthsFromListArray(leaf_array))
          component_feature_num_missing[feature_path] = leaf_array.null_count

    # Now create a partial sparse feature stats object for each sparse feature
    # using the value list lengths and numbers missing information collected
    # above.
    for feature_path in self._sparse_feature_component_paths:
      value_feature_path = self._sparse_feature_component_paths[
          feature_path].value_feature
      index_feature_paths = self._sparse_feature_component_paths[
          feature_path].index_features
      missing_value_count = component_feature_num_missing.get(
          value_feature_path)
      # If this batch does not have the value feature at all,
      # missing_value_count is the number of examples in the batch.
      # Also populate the value list lengths for the value feature with all 0s
      # since a missing feature is considered to have a value list length of 0.
      if missing_value_count is None:
        missing_value_count = batch_example_count
        component_feature_value_list_lengths[value_feature_path] = np.full(
            batch_example_count, 0)
      missing_index_counts = collections.Counter()
      min_length_diff = dict()
      max_length_diff = dict()
      for index_feature_path in index_feature_paths:
        missing_index_count = component_feature_num_missing.get(
            index_feature_path)
        # If this batch does not have this index feature at all,
        # missing_index_count for that index feature is the number of
        # examples in the batch.
        # Also populate the value list lengths for the index feature with all 0s
        # since a missing feature is considered to have a value list length of
        # 0.
        if missing_index_count is None:
          missing_index_counts[index_feature_path] = batch_example_count
          component_feature_value_list_lengths[index_feature_path] = np.full(
              batch_example_count, 0)
        else:
          missing_index_counts[index_feature_path] = missing_index_count
        length_differences = np.subtract(
            component_feature_value_list_lengths[index_feature_path],
            component_feature_value_list_lengths[value_feature_path])
        min_length_diff[index_feature_path] = np.min(length_differences)
        max_length_diff[index_feature_path] = np.max(length_differences)

      stats_for_feature = _PartialSparseFeatureStats(missing_value_count,
                                                     missing_index_counts,
                                                     min_length_diff,
                                                     max_length_diff)
      existing_stats_for_feature = accumulator.get(feature_path)
      if existing_stats_for_feature is None:
        accumulator[feature_path] = stats_for_feature
      else:
        accumulator[feature_path] += stats_for_feature
    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[Dict[types.FeaturePath,
                                        _PartialSparseFeatureStats]]
  ) -> Dict[types.FeaturePath, _PartialSparseFeatureStats]:
    """Merges several accumulators into a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    result = {}
    for accumulator in accumulators:
      for feature_path, partial_stats in accumulator.items():
        existing = result.get(feature_path, _PartialSparseFeatureStats())
        result[feature_path] = existing + partial_stats
    return result

  def extract_output(
      self, accumulator: Dict[types.FeaturePath, _PartialSparseFeatureStats]
  ) -> statistics_pb2.DatasetFeatureStatistics:
    result = statistics_pb2.DatasetFeatureStatistics()
    for feature_path, partial_stats in accumulator.items():
      feature_result = statistics_pb2.FeatureNameStatistics()
      feature_result.path.CopyFrom(feature_path.to_proto())
      feature_result.custom_stats.add(
          name='missing_value', num=partial_stats.value_feature_num_missing)
      index_features_num_missing_histogram = statistics_pb2.RankHistogram()
      max_length_diff_histogram = statistics_pb2.RankHistogram()
      min_length_diff_histogram = statistics_pb2.RankHistogram()
      # Sort to get deterministic ordering of the buckets in the custom stat.
      for index_feature in sorted(partial_stats.index_features_num_missing):
        # The label is the last step in the feature path (and shares the parent
        # with the sparse feature).
        label = index_feature.steps()[-1]
        missing_bucket = index_features_num_missing_histogram.buckets.add()
        missing_bucket.label = label
        missing_bucket.sample_count = partial_stats.index_features_num_missing[
            index_feature]
        max_length_bucket = max_length_diff_histogram.buckets.add()
        max_length_bucket.label = label
        max_length_bucket.sample_count = (
            partial_stats.index_features_max_length_diff[index_feature])
        min_length_bucket = min_length_diff_histogram.buckets.add()
        min_length_bucket.label = label
        min_length_bucket.sample_count = (
            partial_stats.index_features_min_length_diff[index_feature])
      feature_result.custom_stats.add(
          name='missing_index',
          rank_histogram=index_features_num_missing_histogram)
      feature_result.custom_stats.add(
          name='max_length_diff', rank_histogram=max_length_diff_histogram)
      feature_result.custom_stats.add(
          name='min_length_diff', rank_histogram=min_length_diff_histogram)

      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_result)
    return result
