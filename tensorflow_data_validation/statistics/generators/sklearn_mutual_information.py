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
"""Module that computes Mutual Information using sk-learn implementation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util

from typing import Dict, List, Set, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

try:
  # pylint:disable=g-import-not-at-top
  from sklearn.feature_selection import mutual_info_classif
  from sklearn.feature_selection import mutual_info_regression
except ImportError as e:
  raise ImportError('To use this StatsGenerator, make sure scikit-learn is '
                    'installed, or install TFDV using "pip install '
                    'tensorflow-data-validation[mutual-information]": {}'
                    .format(e))

MUTUAL_INFORMATION_KEY = "sklearn_mutual_information"
ADJUSTED_MUTUAL_INFORMATION_KEY = "sklearn_adjusted_mutual_information"
CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE = "__missing_category__"


def _flatten_and_impute(examples: pa.RecordBatch,
                        categorical_features: Set[types.FeaturePath]
                       ) -> Dict[types.FeaturePath, np.ndarray]:
  """Flattens and imputes the values in the input Arrow RecordBatch.

  Replaces missing values with CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
  for categorical features and 10*max(feature_values) for numeric features.
  We impute missing values with an extreme value that is far from observed
  values so it does not incorrectly impact KNN results. 10*max(feature_values)
  is used instead of sys.max_float because max_float is large enough to cause
  unexpected float arithmetic errors.

  Args:
    examples: Arrow RecordBatch containing a batch of examples where all
      features are univalent.
    categorical_features: Set of categorical feature names.

  Returns:
    A Dict[FeaturePath, np.ndarray] where the key is the feature path and the
    value is a 1D numpy array corresponding to the feature values.
  """
  num_rows = examples.num_rows
  result = {}
  for column_name, feature_array in zip(examples.schema.names,
                                        examples.columns):
    feature_path = types.FeaturePath([column_name])
    imputation_fill_value = (
        CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
        if feature_path in categorical_features else sys.maxsize)
    if pa.types.is_null(feature_array.type):
      # If null array, impute all values.
      imputed_values_array = np.full(
          shape=num_rows,
          fill_value=imputation_fill_value)
      result[feature_path] = imputed_values_array
    else:
      # to_pandas returns a readonly array. Create a copy as we will be imputing
      # the NaN values.
      flattened_array, non_missing_parent_indices = arrow_util.flatten_nested(
          feature_array, return_parent_indices=True)
      assert non_missing_parent_indices is not None
      non_missing_values = np.copy(np.asarray(flattened_array))
      is_categorical_feature = feature_path in categorical_features
      result_dtype = non_missing_values.dtype
      if non_missing_parent_indices.size < num_rows and is_categorical_feature:
        result_dtype = np.object
      flattened_array = np.ndarray(shape=num_rows, dtype=result_dtype)
      num_values = np.asarray(
          array_util.ListLengthsFromListArray(feature_array))
      missing_parent_indices = np.where(num_values == 0)[0]
      if feature_path not in categorical_features:
        # Also impute any NaN values.
        nan_mask = np.isnan(non_missing_values)
        if not np.all(nan_mask):
          imputation_fill_value = non_missing_values[~nan_mask].max() * 10
        non_missing_values[nan_mask.nonzero()[0]] = imputation_fill_value
      flattened_array[non_missing_parent_indices] = non_missing_values
      if missing_parent_indices.any():
        flattened_array[missing_parent_indices] = imputation_fill_value
      result[feature_path] = flattened_array
  return result


class SkLearnMutualInformation(partitioned_stats_generator.PartitionedStatsFn):
  """Computes Mutual Information(MI) between each feature and the label.

  The non-streaming sk-learn implementation of MI is used to
  estimate Adjusted Mutual Information(AMI) and MI between all valid features
  and the label. AMI prevents overestimation of MI for high entropy features.
  It is defined as MI(feature, labels) - MI(feature, np.random.shuffle(labels)).

  SkLearnMutualInformation will "gracefully fail" on all features
  that are multivalent since they are not supported by sk-learn. The compute
  method will not report statistics for these invalid features.
  """

  def __init__(self, label_feature: types.FeaturePath,
               schema: schema_pb2.Schema, seed: int):
    """Initializes SkLearnMutualInformation.

    Args:
      label_feature: The key used to identify labels in the ExampleBatch.
      schema: The schema of the dataset.
      seed: An int value to seed the RNG used in MI computation.

    Raises:
      ValueError: If label_feature does not exist in the schema.
    """
    self._label_feature = label_feature
    self._schema = schema
    self._categorical_features = schema_util.get_categorical_features(schema)
    assert schema_util.get_feature(self._schema, self._label_feature)
    self._label_feature_is_categorical = (
        self._label_feature in self._categorical_features)
    self._seed = seed
    self._schema_features = set([
        feature_path for (feature_path,
                          _) in schema_util.get_all_leaf_features(schema)
    ])

    # Seed the RNG used for shuffling and for MI computations.
    np.random.seed(seed)

  def compute(self, examples: pa.RecordBatch
             ) -> statistics_pb2.DatasetFeatureStatistics:
    """Computes MI and AMI between all valid features and labels.

    Args:
      examples: Arrow RecordBatch containing a batch of examples.

    Returns:
      DatasetFeatureStatistics proto containing AMI and MI for each valid
        feature in the dataset. Some features may filtered out by
        _remove_unsupported_feature_columns if they are inavlid. In this case,
        AMI and MI will not be calculated for the invalid feature.

    Raises:
      ValueError: If label_feature contains unsupported data.
    """
    examples = self._remove_unsupported_feature_columns(examples, self._schema)

    flattened_examples = _flatten_and_impute(examples,
                                             self._categorical_features)
    if self._label_feature not in flattened_examples:
      raise ValueError("Label column contains unsupported data.")
    labels = flattened_examples.pop(self._label_feature)
    df = pd.DataFrame(flattened_examples)
    # Boolean list used to mark features as discrete for sk-learn MI computation
    discrete_feature_mask = self._convert_categorical_features_to_numeric(df)
    return stats_util.make_dataset_feature_stats_proto(
        self._calculate_mi(df, labels, discrete_feature_mask, seed=self._seed))

  def _calculate_mi(self, df: pd.DataFrame, labels: np.ndarray,
                    discrete_feature_mask: List[bool],
                    seed: int) -> Dict[types.FeaturePath, Dict[Text, float]]:
    """Calls the sk-learn implementation of MI and stores results in dict.

    Args:
      df: A pd.DataFrame containing feature values where each column corresponds
        to a feature and each row corresponds to an example.
      labels: A List where the ith index represents the label for the ith
        example.
      discrete_feature_mask: A boolean list where the ith element is true iff
        the ith feature column in the input df is a categorical feature.
      seed: An int value to seed the RNG used in MI computation.

    Returns:
      Dict[FeatureName, Dict[str,float]] where the keys of the dicts are the
      feature name and values are a dict where the keys are
      MUTUAL_INFORMATION_KEY and ADJUSTED_MUTUAL_INFORMATION_KEY and the values
      are the MI and AMI for that feature.
    """
    result = {}
    if self._label_feature_is_categorical:
      mi_per_feature = mutual_info_classif(
          df.values,
          labels,
          discrete_features=discrete_feature_mask,
          copy=True,
          random_state=seed)

      np.random.shuffle(labels)

      shuffled_mi_per_feature = mutual_info_classif(
          df.values,
          labels,
          discrete_features=discrete_feature_mask,
          copy=False,
          random_state=seed)
    else:
      # Skip if sample size is smaller than the default value of n_neighbors.
      if df.values.shape[0] < 4:
        return result
      mi_per_feature = mutual_info_regression(
          df.values,
          labels,
          discrete_features=discrete_feature_mask,
          copy=True,
          random_state=seed)

      np.random.shuffle(labels)

      shuffled_mi_per_feature = mutual_info_regression(
          df.values,
          labels,
          discrete_features=discrete_feature_mask,
          copy=False,
          random_state=seed)

    for i, (mi, shuffled_mi) in enumerate(
        zip(mi_per_feature, shuffled_mi_per_feature)):
      result[df.columns[i]] = {
          MUTUAL_INFORMATION_KEY: mi.clip(min=0),
          ADJUSTED_MUTUAL_INFORMATION_KEY: mi - shuffled_mi
      }
    return result

  def _convert_categorical_features_to_numeric(self,
                                               df: pd.DataFrame) -> List[bool]:
    """Encodes all categorical features in input dataframe to numeric values.

    Categorical features are inferred from the schema. They are transformed
    using the np.unique function which maps each value in the feature's domain
    to a numeric id. Encoded categorical features are marked by a boolean mask
    which is returned and used by scikit-learn to identify discrete features.

    Args:
      df: A pd.DataFrame containing feature values where each column corresponds
        to a feature and each row corresponds to an example.

    Returns:
      A boolean list where the ith element is true iff the ith feature column in
      the input df is a categorical feature.
    """
    is_categorical_feature = [False for _ in df]
    columns_to_drop = []
    indices_to_drop = []
    for i, column in enumerate(df):
      if column in self._categorical_features:
        # Encode categorical columns.
        str_array = [(
            x.decode("utf-8", "replace") if isinstance(x, bytes) else
            (x if x is not None else CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE))
                     for x in df[column].values]
        unique_elements, df[column] = np.unique(str_array, return_inverse=True)
        is_categorical_feature[i] = True
        # Drop the categroical features that all its values are unique if the
        # label is not categorical.
        # Otherwise such feature will cause error during MI calculation.
        if unique_elements.size == df[column].shape[
            0] and not self._label_feature_is_categorical:
          columns_to_drop.append(column)
          indices_to_drop.append(i)
    df.drop(columns_to_drop, axis=1, inplace=True)
    is_categorical_feature = np.delete(is_categorical_feature, indices_to_drop)
    return is_categorical_feature

  def _remove_unsupported_feature_columns(
      self, examples: pa.RecordBatch, schema: schema_pb2.Schema
      ) -> pa.RecordBatch:
    """Removes feature columns that contain unsupported values.

    All feature columns that are multivalent are dropped since they are
    not supported by sk-learn.

    All columns of STRUCT type are also dropped.

    Args:
      examples: Arrow RecordBatch containing a batch of examples.
      schema: The schema for the data.

    Returns:
      Arrow RecordBatch.
    """
    columns = set(examples.schema.names)

    multivalent_features = schema_util.get_multivalent_features(schema)
    unsupported_columns = set()
    for f in multivalent_features:
      # Drop the column if they were in the examples.
      if f.steps()[0] in columns:
        unsupported_columns.add(f.steps()[0])
    for column_name, column in zip(examples.schema.names,
                                   examples.columns):
      # only support 1-nested non-struct arrays.
      column_type = column.type
      if (arrow_util.get_nest_level(column_type) != 1 or
          stats_util.get_feature_type_from_arrow_type(
              types.FeaturePath([column_name]), column_type)
          == statistics_pb2.FeatureNameStatistics.STRUCT):
        unsupported_columns.add(column_name)
      # Drop columns that were not in the schema.
      if types.FeaturePath([column_name]) not in self._schema_features:
        unsupported_columns.add(column_name)

    supported_columns = []
    supported_column_names = []
    for column_name, column in zip(examples.schema.names,
                                   examples.columns):
      if column_name not in unsupported_columns:
        supported_columns.append(column)
        supported_column_names.append(column_name)

    return pa.RecordBatch.from_arrays(supported_columns, supported_column_names)
