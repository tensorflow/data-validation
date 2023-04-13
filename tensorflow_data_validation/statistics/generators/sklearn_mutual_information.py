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
from typing import Dict, List, Optional, Sequence, Set, Text, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util

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

_MUTUAL_INFORMATION_KEY = "sklearn_mutual_information"
_ADJUSTED_MUTUAL_INFORMATION_KEY = "sklearn_adjusted_mutual_information"
_NORMALIZED_ADJUSTED_MUTUAL_INFORMATION_KEY = "sklearn_normalized_adjusted_mutual_information"
_CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE = "__missing_category__"
_KNN_N_NEIGHBORS = 3


def _flatten_and_impute(examples: pa.RecordBatch,
                        categorical_features: Set[types.FeaturePath]
                       ) -> Dict[types.FeaturePath, np.ndarray]:
  """Flattens and imputes the values in the input Arrow RecordBatch.

  Replaces missing values with _CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
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
        _CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
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
      flattened_array, non_missing_parent_indices = array_util.flatten_nested(
          feature_array, return_parent_indices=True)
      assert non_missing_parent_indices is not None
      non_missing_values = np.copy(np.asarray(flattened_array))
      is_categorical_feature = feature_path in categorical_features
      result_dtype = non_missing_values.dtype
      if non_missing_parent_indices.size < num_rows and is_categorical_feature:
        result_dtype = object
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
      _MUTUAL_INFORMATION_KEY, _ADJUSTED_MUTUAL_INFORMATION_KEY,
      _NORMALIZED_ADJUSTED_MUTUAL_INFORMATION_KEY and the values are the MI,
      AMI, and normalized AMI for that feature.
    """
    result = {}

    # Calculate MI for each feature.
    mi_per_feature = _sklearn_calculate_mi_wrapper(
        df.values,
        labels,
        discrete_features=discrete_feature_mask,
        copy=True,
        seed=seed,
        is_label_categorical=self._label_feature_is_categorical)

    if mi_per_feature is None:
      # MI could not be calculated.
      return result

    # There are multiple ways to normalized AMI. We choose to calculate it as:
    # Normalized AMI(X, Y) = AMI(X, Y) / (Max{H(X), H(Y)} - shuffle_mi(X, Y))
    # Where H(X) is the entropy of X.
    #
    # We can derive entropy from MI(X, X) as follows:
    # MI(X, X) = H(X) - H(X|X) = H(X)

    # Calculate H(feature), for each feature.
    entropy_per_feature = []
    for col in df.columns:
      col_is_categorical = col in self._categorical_features
      entropy = _sklearn_calculate_mi_wrapper(
          np.array([[x] for x in df[col].values]),
          df[col].values,
          discrete_features=col_is_categorical,
          copy=True,
          seed=seed,
          is_label_categorical=col_is_categorical)
      # The entropy might not exist for a feature. This is because now we are
      # treating each feature as a label. The features could be a mix of
      # categorical and numerical features, thus MI is calculated on a case by
      # case basis, and may not exist in some cases.
      # Setting it to 0 will not affect the normalized AMI result, since we are
      # looking for max entropy.
      entropy_per_feature.append(entropy[0] if entropy else 0)

    # Calculate H(label)
    if self._label_feature_is_categorical:
      # Encode categorical labels as numerical.
      _, integerized_label = np.unique(labels, return_inverse=True)
      labels_as_feature = np.array([[x] for x in integerized_label])
    else:
      labels_as_feature = np.array([[x] for x in labels])
    label_entropy = _sklearn_calculate_mi_wrapper(
        labels_as_feature,
        labels,
        discrete_features=self._label_feature_is_categorical,
        copy=True,
        seed=seed,
        is_label_categorical=self._label_feature_is_categorical)
    # label_entropy is guaranteed to exist. If it does not exist, then
    # mi_per_feature would have been None (and we would have exited this).
    assert len(label_entropy) == 1
    label_entropy = label_entropy[0]

    # Shuffle the labels and calculate the MI. This allows us to adjust
    # the MI for any memorization in the model.
    np.random.shuffle(labels)
    shuffled_mi_per_feature = _sklearn_calculate_mi_wrapper(
        df.values,
        labels,
        discrete_features=discrete_feature_mask,
        copy=False,
        seed=seed,
        is_label_categorical=self._label_feature_is_categorical)

    for i, (mi, shuffle_mi, entropy) in enumerate(
        zip(mi_per_feature, shuffled_mi_per_feature, entropy_per_feature)):
      max_entropy = max(label_entropy, entropy)
      ami = mi - shuffle_mi

      # Bound normalized AMI to be in [0, 1].
      # shuffle_mi <= max_entropy always holds.
      if max_entropy == shuffle_mi:
        # In the case of equality, MI(X, Y) <= max_entropy == shuffle_mi.
        # So AMI = MI(X, Y) - shuffle_mi < 0. We cap it at 0.
        normalized_ami = 0
      else:
        normalized_ami = min(1, max(0, ami / (max_entropy - shuffle_mi)))

      result[df.columns[i]] = {
          _MUTUAL_INFORMATION_KEY: mi.clip(min=0),
          _ADJUSTED_MUTUAL_INFORMATION_KEY: ami,
          _NORMALIZED_ADJUSTED_MUTUAL_INFORMATION_KEY: normalized_ami
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
        def maybe_decode_or_impute(x):
          if isinstance(x, bytes):
            return x.decode("utf-8", "replace")
          elif x is not None:
            return x
          else:
            return _CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
        str_array = [maybe_decode_or_impute(x) for x in df[column].values]
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


def _sklearn_calculate_mi_wrapper(
    feature: np.ndarray, label: np.ndarray,
    discrete_features: Union[bool, Sequence[bool]], seed: int, copy: bool,
    is_label_categorical: bool) -> Optional[np.ndarray]:
  """Wraps sklearn calculate mi with some additional validation.

  Args:
    feature: The features.
    label: The labels.
    discrete_features: If bool, then determines whether to consider all
      features discrete or continuous. If array, then it should be either a
      boolean mask with shape (n_features,) or array with indices of discrete
      features.
    seed: Determines random number generation for adding small noise to
      continuous variables in order to remove repeated values. Pass an int for
      reproducible results across multiple function calls.
    copy: Whether to make a copy of the given data. If set to False, the
      initial data will be overwritten.
    is_label_categorical: True if the label is a categorical feature.

  Returns:
    A numpy array of mutual information of each feature. Will return None if MI
    cannot be calculated.
  """
  if is_label_categorical:
    calc_mi_fn = mutual_info_classif
  else:
    # Skip if sample size is smaller than number of required neighbors plus
    # itself.
    if len(feature) <= _KNN_N_NEIGHBORS:
      return None
    calc_mi_fn = mutual_info_regression

  return calc_mi_fn(
      feature,
      label,
      discrete_features=discrete_features,
      n_neighbors=_KNN_N_NEIGHBORS,
      copy=copy,
      random_state=seed)
