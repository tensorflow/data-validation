# Copyright 2021 Google LLC
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
"""Module that computes Mutual Information using knn estimation."""

import collections
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from absl import logging
import apache_beam as beam
import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.utils import feature_partition_util
from tensorflow_data_validation.utils import mutual_information_util
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

_ADJUSTED_MUTUAL_INFORMATION_KEY = u"adjusted_mutual_information"


# pylint: disable=g-bare-generic
def _get_flattened_feature_values_without_nulls(
    feature_array: pa.Array) -> List[Any]:
  """Flattens the feature array into a List and removes null values.

  Args:
    feature_array: Arrow Array.

  Returns:
    A list containing the flattened feature values with nulls removed.
  """
  non_missing_values = np.asarray(array_util.flatten_nested(feature_array)[0])
  return list(non_missing_values[~pd.isnull(non_missing_values)])


def _get_categorical_feature_encoding(
    category_frequencies: Dict[Any, int],
    max_encoding_length: int) -> Dict[Any, int]:
  """Gets the encoding for a categorical feature based on category frequency.

  Assigns a unique index for the max_encoding_length-1 most frequently occurring
  categories. This index corresponds to the index of the encoding this category
  maps to.

  Args:
    category_frequencies: A dict where the key is the category and the value is
      the number of times the category occurred.
    max_encoding_length: The maximum length of an encoded feature value.

  Returns:
    A dict where the key is the category and the value is an int which
      corresponds to the index of the encoding which the category maps to.
  """
  categorical_feature_encoding = {}
  for index, value in enumerate(
      sorted(category_frequencies, key=category_frequencies.get,
             reverse=True)[:max_encoding_length - 1]):
    categorical_feature_encoding[value] = index
  return categorical_feature_encoding


def _apply_categorical_encoding_to_feature_array(
    feature_array: pa.Array, categorical_encoding: Dict[Any, int],
    encoding_length: int) -> List[Any]:
  """Applies the provided encoding to the feature array.

  For each example, the frequency of each category is computed. Using the
  categorical_encoding dict, an encoding is created for the example by storing
  these counts in the appropriate index of the encoding.

  Args:
    feature_array: Arrow Array.
    categorical_encoding: A dict where the key is the category and the value is
      the index in the encoding to which the category corresponds to.
    encoding_length: The length of the list containing the encoded feature
      values.

  Returns:
    A list containing the encoded feature values for each example.
  """
  if pa.types.is_null(feature_array.type):
    return []
  result = [None for _ in range(len(feature_array))]
  flattened, non_missing_parent_indices = array_util.flatten_nested(
      feature_array, True)
  non_missing_values = flattened.to_pylist()
  non_missing_parent_indices = list(non_missing_parent_indices)
  for (value, index) in zip(non_missing_values, non_missing_parent_indices):
    if result[index] is None:
      result[index] = []
    result[index].append(value)
  for i in range(len(result)):
    if result[i] is None:
      result[i] = [None] * encoding_length
    else:
      category_frequencies = collections.Counter(result[i])
      encoded_values = [0] * encoding_length
      for category in category_frequencies:
        if category in categorical_encoding:
          encoded_values[categorical_encoding[category]] = (
              category_frequencies[category])
        elif not pd.isnull(category):
          encoded_values[-1] += category_frequencies[category]
      result[i] = encoded_values
  return result


def _encode_multivalent_categorical_feature(
    feature_array: pa.Array, max_encoding_length: int) -> List[int]:
  """Encodes multivalent categorical features into fixed length representation.

  Categorical multivalent features are encoded using a bag-of-words strategy.
  Encodings are obtained by counting the occurences of each unique value in the
  feature domain for each example. If the number of unique values in the
  feature's domain exceeds max_encoding_length, the top
  (max_encoding_length - 1) ocurring categories will be used to encode
  examples. The presence of other less frequently occurring values will
  contribute to the frequency count of the last category.

  Args:
    feature_array: Arrow Array.
    max_encoding_length: The maximum length of an encoded feature value.

  Returns:
    A list containing the encoded feature values for each example.
  """
  flattened_feature_values = _get_flattened_feature_values_without_nulls(
      feature_array)
  category_frequencies = dict(
      zip(*np.unique(flattened_feature_values, return_counts=True)))
  if not category_frequencies:
    encoding_length = max_encoding_length
  else:
    encoding_length = min(max_encoding_length, len(category_frequencies))
  categorical_encoding = _get_categorical_feature_encoding(
      category_frequencies, max_encoding_length)
  return _apply_categorical_encoding_to_feature_array(feature_array,
                                                      categorical_encoding,
                                                      encoding_length)


def _apply_numerical_encoding_to_feature_array(
    feature_array: pa.Array, histogram_bin_boundaries: np.ndarray,
    encoding_length: int) -> List[int]:
  """Determines encoding of numeric feature array from histogram bins.

  Using the provided histogram_bin_boundaries, a histogram is constructed for
  each example to obtain an encoding for a feature value.

  Args:
    feature_array: Arrow Array.
    histogram_bin_boundaries: A monotonically increasing np.ndarray representing
      the boundaries of each bin in the histogram.
    encoding_length: The length of the list containing the encoded feature
      values.

  Returns:
    A list conatining the encoded feature values for each example.
  """
  if pa.types.is_null(feature_array.type):
    return []
  result = [None for _ in range(len(feature_array))]  # type: List
  flattened, non_missing_parent_indices = array_util.flatten_nested(
      feature_array, True)
  assert non_missing_parent_indices is not None
  non_missing_values = np.asarray(flattened)
  non_missing_parent_indices = non_missing_parent_indices.astype(np.int32)
  values_indices = np.stack((non_missing_values, non_missing_parent_indices),
                            axis=-1)
  nan_mask = pd.isnull(non_missing_values)
  for (value, index) in values_indices[~nan_mask]:
    index = int(index)
    if result[index] is None:
      result[index] = []
    result[index].append(value)
  for (value, index) in values_indices[nan_mask]:
    index = int(index)
    if result[index] is None:
      result[index] = []
  for i in range(len(result)):
    if result[i] is None:
      result[i] = [None] * encoding_length
    else:
      result[i] = np.bincount(
          np.digitize(result[i], histogram_bin_boundaries) - 1,
          minlength=encoding_length).tolist()
  return result  # pytype: disable=bad-return-type


def _encode_multivalent_numeric_feature(
    feature_array: pa.Array, encoding_length: int) -> Optional[List[int]]:
  """Encodes numeric multivalent features into a fixed length representation.

  Numeric multivalent features are encoded using bucketization.
  max_encoding_length bins of equal sized intervals are constructed from the
  feature values. For each example, a histogram is constructed. These bin
  counts represent an encoding for the example.

  Args:
    feature_array: Arrow Array.
    encoding_length: The length of the list containing the encoded feature
      values.

  Returns:
    A list containing the encoded feature values for each example. Returns None
    if unable to encode the feature_array.
  """
  flattened_feature_values = _get_flattened_feature_values_without_nulls(
      feature_array)
  try:
    _, histogram_bin_boundaries = np.histogram(
        flattened_feature_values, bins=encoding_length - 1)
  except IndexError as e:
    # np.histogram cannot handle values > 2**53 if the min and max of the
    # examples are the same. https://github.com/numpy/numpy/issues/8627
    logging.exception("Unable to encode examples: %s with error: %s",
                      flattened_feature_values, e)
    return None
  return _apply_numerical_encoding_to_feature_array(feature_array,
                                                    histogram_bin_boundaries,
                                                    encoding_length)


def _encode_univalent_feature(feature_array: pa.Array) -> List[Any]:
  """Encodes univalent feature values into a fixed length representation.

  Univalent features are cast into a Python list. They are not affected by the
  encoding with the exception of null values which are replaced by None.

  Args:
    feature_array: Arrow Array.

  Returns:
    A list containing the feature values where null values are replaced by None.
  """
  result = [[None] for _ in range(len(feature_array))]
  flattened, non_missing_parent_indices = array_util.flatten_nested(
      feature_array, True)
  non_missing_values = np.asarray(flattened)
  nan_mask = pd.isnull(non_missing_values)
  non_nan_pairs = np.stack((non_missing_values, non_missing_parent_indices),
                           axis=-1)[~nan_mask]
  for (value, index) in non_nan_pairs:
    result[int(index)] = [value]
  return result


# TODO(b/120484896): Use embeddings in MI pre-processing of variable length
# multivalent features.
def _encode_examples(
    examples_record_batch: pa.RecordBatch,
    multivalent_features: Set[types.FeaturePath],
    categorical_features: Set[types.FeaturePath],
    features_to_ignore: Set[types.FeaturePath],
    max_encoding_length: int) -> Dict[types.FeaturePath, List[Any]]:
  """Encodes feature values into a fixed length representation.

  The MI implementation cannot handle variable length multivalent
  features, so features are encoded to a fixed length representation.

  Univalent features are not affected by the encoding with the exception of null
  values which are replaced by None.

  Categorical multivalent features are encoded using a bag-of-words strategy.
  Encodings are obtained by counting the occurences of each unique value in the
  feature domain for each example. If the number of unique values in the
  feature's domain exceeds max_encoding_length, the top
  (max_encoding_length - 1) occurring categories will be used to encode
  examples. The presence of other less frequently occurring values will
  contribute to the frequency count of the final category.

  Numeric multivalent features are encoded using bucketization.
  max_encoding_length bins of equal sized intervals are constructed from the
  feature values. For each example, a histogram is constructed. These bin
  counts represent an encoding for the example.

  Args:
    examples_record_batch: Arrow record_batch containing a batch of examples.
    multivalent_features: A set containing paths of all multivalent features.
    categorical_features: A set containing paths of all categorical features.
    features_to_ignore: A set containing paths of features to ignore.
    max_encoding_length: The maximum length of an encoded feature value. This
      should be set to limit the memory usage of MI computation.

  Returns:
    A Dict[FeatureName, List] where the key is the feature name and the
    value is a 2D List containing the encoded feature values of each example.
    If a feature is unable to be encoded, it will not appear in the resulting
    Dict.
  """
  result = {}
  for feature_name, feature_column in zip(examples_record_batch.schema.names,
                                          examples_record_batch.columns):
    # Note that multivalent_features and categorical_features might contain
    # complex paths (for features nested under STRUCT features), however
    # because STRUCT features can be neither multivalent nor categorical,
    # we are essentially filtering out any STRUCT features and their
    # descendents.
    feature_path = types.FeaturePath([feature_name])
    if features_to_ignore and feature_path in features_to_ignore:
      continue
    if feature_path in multivalent_features:
      if feature_path in categorical_features:
        result[feature_path] = _encode_multivalent_categorical_feature(
            feature_column, max_encoding_length)
      else:
        encoded_list = _encode_multivalent_numeric_feature(
            feature_column, max_encoding_length)
        if encoded_list is None:
          logging.error("Feature: %s was not encoded", feature_name)
        else:
          result[feature_path] = encoded_list
    else:
      result[feature_path] = _encode_univalent_feature(feature_column)
  return result


class _PartitionFn(beam.DoFn):
  """Custom partitioner DoFn for MutualInformation."""

  def __init__(self, row_partitions: int, column_partitions: int,
               label_column: str, seed: int):
    self._row_partitions = row_partitions
    self._column_partitions = column_partitions
    self._label_column = frozenset([label_column])
    self._rng = np.random.default_rng(seed=seed)

  def setup(self):
    if self._column_partitions > 1:
      self._partitioner = feature_partition_util.ColumnHasher(
          self._column_partitions)
    else:
      self._partitioner = None

  def process(
      self, element: types.SlicedRecordBatch
  ) -> Iterable[Tuple[Tuple[types.SliceKey, int], pa.RecordBatch]]:
    """Performs row-wise random key assignment and column-wise slicing.

    Each input RecordBatch is mapped to up to self._column_partitions output
    RecordBatch, each of which contains a subset of columns. Only the label
    column is duplicated across RecordBatches, so this is nearly a partitioning
    of columns. If self._column_partitions == 1, the output RecordBatch is
    unmodified.

    The total partition key space is _row_partitions * _column_partitions.

    Args:
      element: An input sliced record batch.

    Yields:
      A sequence of partitioned RecordBatches.

    """

    row_partition = self._rng.integers(0, self._row_partitions, dtype=int)
    if self._partitioner is None:
      slice_key, record_batch = element
      yield (slice_key, row_partition), record_batch
    else:
      for ((slice_key, column_partition),
           record_batch) in feature_partition_util.generate_feature_partitions(
               element, self._partitioner, self._label_column):
        partition = row_partition * self._column_partitions + column_partition
        yield (slice_key, partition), record_batch


# pylint: disable=invalid-name
@beam.typehints.with_input_types(types.SlicedRecordBatch)
@beam.typehints.with_output_types(Tuple[Tuple[types.SliceKey, int],
                                        pa.RecordBatch])
@beam.ptransform_fn
def _PartitionTransform(pcol, row_partitions: int, column_partitions: int,
                        label_feature: types.FeaturePath, seed: int):
  """Ptransform wrapping _default_assign_to_partition."""
  # We need to find the column name associated with the label path.
  steps = label_feature.steps()
  if not steps:
    raise ValueError("Empty label feature")
  label = steps[0]
  return pcol | "PartitionRowsCols" >> beam.ParDo(
      _PartitionFn(row_partitions, column_partitions, label, seed))
# pylint: enable=invalid-name


class MutualInformation(partitioned_stats_generator.PartitionedStatsFn):
  """Computes Mutual Information(MI) between each feature and the label.

  This statistic is the estimated Adjusted Mutual Information(AMI) between all
  features and the label. AMI prevents overestimation of MI for high entropy
  features. It is defined as MI(feature, labels) - MI(feature, shuffle(labels)).

  To use this statistic, use the `NonStreamingCustomStatsGenerator`. This
  generator can then be specified in the `stats_options` when calling
  `GenerateStatistics`.

  Example usage:
  ```
  generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
    MutualInformation(...))
  ```
  """

  def __init__(self,
               label_feature: types.FeaturePath,
               schema: Optional[schema_pb2.Schema] = None,
               max_encoding_length: int = 512,
               seed: int = 12345,
               multivalent_features: Optional[Set[types.FeaturePath]] = None,
               categorical_features: Optional[Set[types.FeaturePath]] = None,
               features_to_ignore: Optional[Set[types.FeaturePath]] = None,
               normalize_by_max: bool = False,
               allow_invalid_partitions: bool = False,
               custom_stats_key: str = _ADJUSTED_MUTUAL_INFORMATION_KEY,
               column_partitions: int = 1):
    """Initializes MutualInformation.

    Args:
      label_feature: The key used to identify labels in the ExampleBatch.
      schema: An optional schema describing the the dataset. Either a schema or
        a list of categorical and multivalent features must be provided.
      max_encoding_length: An int value to specify the maximum length of
        encoding to represent a feature value.
      seed: An int value to seed the RNG used in MI computation.
      multivalent_features: An optional set of features that are multivalent.
      categorical_features: An optional set of the features that are
        categorical.
      features_to_ignore: An optional set of features that should be ignored by
        the mutual information calculation.
      normalize_by_max: If True, AMI values are normalized to a range 0 to 1 by
        dividing by the maximum possible information AMI(Y, Y).
      allow_invalid_partitions: If True, generator tolerates input partitions
        that are invalid (e.g. size of partion is < the k for the KNN), where
        invalid partitions return no stats. The min_partitions_stat_presence arg
        to PartitionedStatisticsAnalyzer controls how many partitions may be
        invalid while still reporting the metric.
      custom_stats_key: A string that determines the key used in the custom
        statistic. This defaults to `_ADJUSTED_MUTUAL_INFORMATION_KEY`.
      column_partitions: If > 1, self.partitioner returns a PTransform that
        partitions input RecordBatches by column (feature), in addition to the
        normal row partitioning (by batch). The total number of effective
        partitions is column_partitions * row_partitions, where row_partitions
        is passed to self.partitioner.

    Raises:
      ValueError: If label_feature does not exist in the schema.
    """
    self._label_feature = label_feature
    self._schema = schema
    self._normalize_by_max = normalize_by_max
    if multivalent_features is not None:
      self._multivalent_features = multivalent_features
    elif self._schema is not None:
      self._multivalent_features = schema_util.get_multivalent_features(
          self._schema)
    else:
      raise ValueError(
          "Either multivalent feature set or schema must be provided")
    if categorical_features is not None:
      self._categorical_features = categorical_features
    elif self._schema is not None:
      self._categorical_features = schema_util.get_categorical_features(
          self._schema)
    else:
      raise ValueError(
          "Either categorical feature set or schema must be provided")
    if schema:
      assert schema_util.get_feature(self._schema, self._label_feature)
    self._label_feature_is_categorical = (
        self._label_feature in self._categorical_features)
    self._max_encoding_length = max_encoding_length
    self._seed = seed
    self._features_to_ignore = features_to_ignore
    self._allow_invalid_partitions = allow_invalid_partitions
    self._custom_stats_key = custom_stats_key
    self._column_partitions = column_partitions

  def _is_unique_array(self, array: np.ndarray):
    values = np.asarray(array.flatten(), dtype=bytes)
    return len(np.unique(values)) == len(values)

  def _label_feature_is_unique(self, record_batch: pa.RecordBatch):
    for feature_name, feature_array in zip(record_batch.schema.names,
                                           record_batch.columns):
      feature_path = types.FeaturePath([feature_name])
      if (feature_path == self._label_feature and
          self._label_feature in self._categorical_features and
          self._label_feature not in self._multivalent_features):
        if self._is_unique_array(feature_array):
          return True
    return False

  def compute(
      self, examples_record_batch: pa.RecordBatch
  ) -> statistics_pb2.DatasetFeatureStatistics:
    """Computes MI and AMI between all valid features and labels.

    Args:
      examples_record_batch: Arrow record_batch containing a batch of examples.

    Returns:
      DatasetFeatureStatistics proto containing AMI and MI for each feature.

    Raises:
      ValueError: If label_feature does not exist in examples.
    """
    if self._label_feature_is_unique(examples_record_batch):
      result = {}
      for feature_name in examples_record_batch.schema.names:
        feature_path = types.FeaturePath([feature_name])
        if feature_path != self._label_feature:
          result[feature_path] = {self._custom_stats_key: 0.0}
      return stats_util.make_dataset_feature_stats_proto(result)

    encoded_examples = _encode_examples(examples_record_batch,
                                        self._multivalent_features,
                                        self._categorical_features,
                                        self._features_to_ignore,
                                        self._max_encoding_length)
    if self._normalize_by_max:
      labels = encoded_examples[self._label_feature]
    else:
      labels = encoded_examples.pop(self._label_feature)
    mi_result = self._calculate_mi(encoded_examples, labels, self._seed)
    if self._normalize_by_max:
      mi_result = self._normalize_mi_values(mi_result)
    return stats_util.make_dataset_feature_stats_proto(mi_result)

  def partitioner(self, num_partitions: int) -> beam.PTransform:
    # pylint: disable=no-value-for-parameter
    return _PartitionTransform(num_partitions, self._column_partitions,
                               self._label_feature, self._seed)
    # pylint: enable=no-value-for-parameter

  def _normalize_mi_values(self, raw_mi: Dict[types.FeaturePath, Dict[str,
                                                                      float]]):
    """Normalizes values to a 0 to 1 scale by dividing by AMI(label, label)."""
    max_ami = raw_mi.pop(self._label_feature)[self._custom_stats_key]
    normalized_mi = {}
    for feature_name, value in raw_mi.items():
      if max_ami > 0:
        normalized_value = value[self._custom_stats_key] / max_ami
      else:
        normalized_value = 0.0
      normalized_mi[feature_name] = {
          self._custom_stats_key: normalized_value
      }
    return normalized_mi

  def _calculate_mi(self,
                    examples_dict: Dict[types.FeaturePath, List[List[Any]]],
                    labels: List[List[Any]],
                    seed: int,
                    k: int = 3) -> Dict[types.FeaturePath, Dict[str, float]]:
    """Estimates the AMI and stores results in dict.

    Args:
      examples_dict: A dictionary containing features, and it's list of values.
      labels: A List where the ith index represents the encoded label for the
        ith example. Each encoded label is of type:
        List[Optional[Union[LabelType, int]]], depending on if it is univalent
        or multivalent.
      seed: An int value to seed the RNG used in MI computation.
      k: The number of nearest neighbors. Must be >= 3.

    Returns:
      Dict[FeatureName, Dict[str,float]] where the keys of the dicts are the
      feature name and values are a dict where the key is
      self._custom_stats_key and the values are the MI and AMI for
      that
      feature.
    """
    result = {}

    if not examples_dict:
      return result

    # Put each column into its own 1D array.
    label_list = list(np.array(labels).T)

    # Multivalent features are encoded into multivalent numeric features.
    label_categorical_mask = [
        (self._label_feature in self._categorical_features and
         self._label_feature not in self._multivalent_features)
        for _ in label_list
    ]

    num_rows = len(list(examples_dict.values())[0])
    if num_rows < k and self._allow_invalid_partitions:
      logging.warn(
          "Partition had %s examples for k = %s. Skipping AMI computation.",
          num_rows, k)
      return result
    for feature_column in examples_dict:
      feature_array = np.array(examples_dict[feature_column])
      # A feature that is always empty cannot be predictive.
      if feature_array.size == 0:
        result[feature_column] = {self._custom_stats_key: 0.0}
        continue
      # If a categorical feature is fully unique, it cannot be predictive.
      if (feature_column in self._categorical_features and
          self._is_unique_array(feature_array)):
        result[feature_column] = {self._custom_stats_key: 0.0}
        continue

      # If a feature is always null, it cannot be predictive.
      all_values_are_null = False if np.sum(~pd.isnull(feature_array)) else True
      if all_values_are_null:
        result[feature_column] = {self._custom_stats_key: 0.0}
        continue

      feature_list = list(feature_array.T)
      feature_categorical_mask = [
          (feature_column in self._categorical_features and
           feature_column not in self._multivalent_features)
          for _ in feature_list
      ]

      ami = mutual_information_util.adjusted_mutual_information(
          label_list,
          feature_list,
          label_categorical_mask,
          feature_categorical_mask,
          k=k,
          seed=seed)
      result[feature_column] = {self._custom_stats_key: ami}
    return result
