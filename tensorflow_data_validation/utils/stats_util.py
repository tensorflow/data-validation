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
"""Utilities for stats generators."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import Dict, Optional
from tensorflow_metadata.proto.v0 import statistics_pb2


_NP_TYPE_TO_FEATURE_TYPE = {
    np.float: statistics_pb2.FeatureNameStatistics.FLOAT,
    np.float16: statistics_pb2.FeatureNameStatistics.FLOAT,
    np.float32: statistics_pb2.FeatureNameStatistics.FLOAT,
    np.float64: statistics_pb2.FeatureNameStatistics.FLOAT,
    np.int: statistics_pb2.FeatureNameStatistics.INT,
    np.int8: statistics_pb2.FeatureNameStatistics.INT,
    np.int16: statistics_pb2.FeatureNameStatistics.INT,
    np.int32: statistics_pb2.FeatureNameStatistics.INT,
    np.int64: statistics_pb2.FeatureNameStatistics.INT,
    np.object: statistics_pb2.FeatureNameStatistics.STRING,
    np.object_: statistics_pb2.FeatureNameStatistics.STRING,
    np.str: statistics_pb2.FeatureNameStatistics.STRING,
    np.str_: statistics_pb2.FeatureNameStatistics.STRING,
    np.string_: statistics_pb2.FeatureNameStatistics.STRING,
    np.unicode_: statistics_pb2.FeatureNameStatistics.STRING,
}


def get_feature_type(
    dtype):
  """Get feature type from numpy dtype.

  Args:
    dtype: Numpy dtype.

  Returns:
    A statistics_pb2.FeatureNameStatistics.Type value.
  """
  return _NP_TYPE_TO_FEATURE_TYPE.get(dtype.type)


def make_dataset_feature_stats_proto(
    stats_values
):
  """Builds DatasetFeatureStatistics proto with custom stats from input dict.

  Args:
    stats_values: A Dict[FeatureName, Dict[str,float]] where the keys are
    feature names, and values are Dicts with keys denoting name of the custom
    statistic and values denoting the value of the custom statistic
    for the feature.
      Ex. {
            'feature_1': {
                'Mutual Information': 0.5,
                'Correlation': 0.1 },
            'feature_2': {
                'Mutual Information': 0.8,
                'Correlation': 0.6 }
          }

  Returns:
    DatasetFeatureStatistics proto containing the custom statistics for each
    feature in the dataset.
  """
  result = statistics_pb2.DatasetFeatureStatistics()

  # Sort alphabetically by feature name to have deterministic ordering
  feature_names = sorted(stats_values.keys())

  for feature_name in feature_names:
    feature_stats_proto = _make_feature_stats_proto(stats_values[feature_name],
                                                    feature_name)
    new_feature_stats_proto = result.features.add()
    new_feature_stats_proto.CopyFrom(feature_stats_proto)

  return result


def _make_feature_stats_proto(
    stats_values,
    feature_name):
  """Creates the FeatureNameStatistics proto for one feature.

  Args:
    stats_values: A Dict[str,float] where the key of the dict is the name of the
      custom statistic and the value is the numeric value of the custom
      statistic of that feature. Ex. {
              'Mutual Information': 0.5,
              'Correlation': 0.1 }
    feature_name: The name of the feature.

  Returns:
    A FeatureNameStatistic proto containing the custom statistics for a
    feature.
  """

  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name

  # Sort alphabetically by statistic name to have deterministic ordering
  stat_names = sorted(stats_values.keys())
  for stat_name in stat_names:
    result.custom_stats.add(name=stat_name, num=stats_values[stat_name])
  return result


def get_weight_feature(input_batch,
                       weight_feature):
  """Gets the weight feature from the input batch.

  Args:
    input_batch: Input batch of examples.
    weight_feature: Name of the weight feature.

  Returns:
    A numpy array containing the weights of the examples in the input batch.

  Raises:
    ValueError: If the weight feature is not present in the input batch or is
        not a valid weight feature (must be of numeric type and have a
        single value).
  """
  try:
    weights = input_batch[weight_feature]
  except KeyError:
    raise ValueError('Weight feature "{}" not present in the input '
                     'batch.'.format(weight_feature))

  # Check if we have a valid weight feature.
  for w in weights:
    if w is None:
      raise ValueError('Weight feature "{}" missing in an '
                       'example.'.format(weight_feature))
    elif (get_feature_type(w.dtype) ==
          statistics_pb2.FeatureNameStatistics.STRING):
      raise ValueError('Weight feature "{}" must be of numeric type. '
                       'Found {}.'.format(weight_feature, w))
    elif w.size != 1:
      raise ValueError('Weight feature "{}" must have a single value. '
                       'Found {}.'.format(weight_feature, w))
  return weights
