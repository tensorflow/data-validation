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
"""Utilities for stats generators.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import Dict, Optional
from tensorflow_metadata.proto.v0 import statistics_pb2


def make_feature_type(
    dtype):
  """Get feature type from numpy dtype.

  Args:
    dtype: Numpy dtype.

  Returns:
    A statistics_pb2.FeatureNameStatistics.Type value.
  """
  if not isinstance(dtype, np.dtype):
    raise TypeError(
        'dtype is of type %s, should be a numpy.dtype' % type(dtype).__name__)

  if np.issubdtype(dtype, np.integer):
    return statistics_pb2.FeatureNameStatistics.INT
  elif np.issubdtype(dtype, np.floating):
    return statistics_pb2.FeatureNameStatistics.FLOAT
  # The numpy dtype for strings is variable length.
  # We need to compare the dtype.type to be sure it's a string type.
  elif (dtype == np.object or dtype.type == np.string_ or
        dtype.type == np.unicode_):
    return statistics_pb2.FeatureNameStatistics.STRING
  return None


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

  for feature_name, custom_stat_to_value in stats_values.items():
    feature_stats_proto = _make_feature_stats_proto(custom_stat_to_value,
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
