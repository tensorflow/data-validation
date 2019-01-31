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

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import stats_util

from tensorflow_data_validation.types_compat import Dict, List

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


MUTUAL_INFORMATION_KEY = "sklearn_mutual_information"
ADJUSTED_MUTUAL_INFORMATION_KEY = "sklearn_adjusted_mutual_information"
CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE = "__missing_category__"


# TODO(b/117650247): sk-learn MI can't handle NaN, None or multivalent features
def _remove_unsupported_feature_columns(examples,
                                        schema):
  """Removes feature columns that contain unsupported values.

  All feature columns that are multivalent are dropped since they are
  not supported by sk-learn.

  Args:
    examples: ExampleBatch containing the values of each example per feature.
    schema: The schema for the data.
  """
  unsupported_features = schema_util.get_multivalent_features(schema)
  for feature_name in unsupported_features:
    del examples[feature_name]


def _flatten_examples(
    examples):
  """Flattens the values in an ExampleBatch to a 1D python list.

  Args:
    examples: An ExampleBatch where all features are univalent.

  Returns:
    A Dict[FeatureName, List] where the key is the feature name and the value is
    a 1D python List corresponding to the feature value for each example.
  """

  flattened_examples = {}
  for feature_name, feature_values in examples.items():
    flattened_examples[feature_name] = [
        feature_value[0] if feature_value is not None and
        not any(pd.isnull(feature_value)) else None
        for feature_value in feature_values
    ]
  return flattened_examples


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

  def __init__(self, label_feature,
               schema, seed):
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
    self._label_feature_is_categorical = schema_util.is_categorical_feature(
        schema_util.get_feature(self._schema, self._label_feature))
    self._seed = seed

    # Seed the RNG used for shuffling and for MI computations.
    np.random.seed(seed)

  def compute(self, examples
             ):
    """Computes MI and AMI between all valid features and labels.

    Args:
      examples: ExampleBatch containing the feature values for each feature.

    Returns:
      DatasetFeatureStatistics proto containing AMI and MI for each valid
        feature in the dataset. Some features may filtered out by
        _remove_unsupported_feature_columns if they are inavlid. In this case,
        AMI and MI will not be calculated for the invalid feature.

    Raises:
      ValueError: If label_feature contains unsupported data.
    """
    if self._label_feature not in examples:
      raise ValueError("Label column does not exist.")

    _remove_unsupported_feature_columns(examples, self._schema)

    if self._label_feature not in examples:
      raise ValueError("Label column contains unsupported data.")

    flattened_examples = _flatten_examples(examples)
    # TODO(b/119414212): Use Ranklab struct feature to handle null values for MI
    imputed_examples = self._impute(flattened_examples)
    labels = imputed_examples.pop(self._label_feature)
    df = pd.DataFrame(imputed_examples)
    # Boolean list used to mark features as discrete for sk-learn MI computation
    discrete_feature_mask = self._convert_categorical_features_to_numeric(df)
    return stats_util.make_dataset_feature_stats_proto(
        self._calculate_mi(df, labels, discrete_feature_mask, seed=self._seed))

  def _impute(self, examples
             ):
    """Imputes missing feature values.

    Replaces missing values with CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
    for categorical features and 10*max(feature_values) for numeric features.
    We impute missing values with an extreme value that is far from observed
    values so it does not incorrectly impact KNN results. 10*max(feature_values)
    is used instead of sys.max_float because max_float is large enough to cause
    unexpected float arithmetic errors.

    Args:
      examples: A dict where the key is the feature name and the values are the
        feature values.

    Returns:
      A dict where the key is the feature name and the values are the
        feature values with missing values imputed.
    """

    for feature, feature_values in examples.items():
      if schema_util.is_categorical_feature(
          schema_util.get_feature(self._schema, feature)):
        imputation_fill_value = CATEGORICAL_FEATURE_IMPUTATION_FILL_VALUE
      else:
        imputation_fill_value = max(
            value for value in feature_values if value is not None) * 10
      examples[feature] = [
          value if value is not None else imputation_fill_value
          for value in feature_values
      ]
    return examples

  def _calculate_mi(self, df, labels,
                    discrete_feature_mask,
                    seed):
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
          MUTUAL_INFORMATION_KEY: mi,
          ADJUSTED_MUTUAL_INFORMATION_KEY: mi - shuffled_mi
      }
    return result

  def _convert_categorical_features_to_numeric(self,
                                               df):
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

    for i, column in enumerate(df):
      if schema_util.is_categorical_feature(
          schema_util.get_feature(self._schema, column)):
        # Encode categorical columns
        df[column] = np.unique(df[column].values, return_inverse=True)[1]
        is_categorical_feature[i] = True
    return is_categorical_feature
