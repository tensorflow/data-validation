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
from tensorflow_data_validation.types_compat import Dict, List, Optional
from google.protobuf import text_format
# TODO(b/125849585): Update to import from TF directly.
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import statistics_pb2


_NP_DTYPE_KIND_TO_FEATURE_TYPE = {
    'f': statistics_pb2.FeatureNameStatistics.FLOAT,
    'i': statistics_pb2.FeatureNameStatistics.INT,
    'u': statistics_pb2.FeatureNameStatistics.INT,
    'S': statistics_pb2.FeatureNameStatistics.STRING,
    'O': statistics_pb2.FeatureNameStatistics.STRING,
    'U': statistics_pb2.FeatureNameStatistics.STRING,
}


# LINT.IfChange
# Semantic domain information can be passed to schema inference using a
# CustomStatistic with name=DOMAIN_INFO.
DOMAIN_INFO = 'domain_info'
# LINT.ThenChange(../anomalies/custom_domain_util.cc)


def is_valid_utf8(value):
  """Returns True iff the value is valid utf8."""
  try:
    value.decode('utf-8')
  except UnicodeError:
    return False
  return True


def get_feature_type(
    dtype):
  """Get feature type from numpy dtype.

  Args:
    dtype: Numpy dtype.

  Returns:
    A statistics_pb2.FeatureNameStatistics.Type value.
  """
  return _NP_DTYPE_KIND_TO_FEATURE_TYPE.get(dtype.kind)


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
    A list containing the weights of the examples in the input batch.

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
  return weights  # pytype: disable=bad-return-type


def write_stats_text(stats,
                     output_path):
  """Writes a DatasetFeatureStatisticsList proto to a file in text format.

  Args:
    stats: A DatasetFeatureStatisticsList proto.
    output_path: File path to write the DatasetFeatureStatisticsList proto.

  Raises:
    TypeError: If the input proto is not of the expected type.
  """
  if not isinstance(stats, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'stats is of type %s, should be a '
        'DatasetFeatureStatisticsList proto.' % type(stats).__name__)

  stats_proto_text = text_format.MessageToString(stats)
  file_io.write_string_to_file(output_path, stats_proto_text)


def load_stats_text(
    input_path):
  """Loads the specified DatasetFeatureStatisticsList proto stored in text format.

  Args:
    input_path: File path from which to load the DatasetFeatureStatisticsList
      proto.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
  stats_text = file_io.read_file_to_string(input_path)
  text_format.Parse(stats_text, stats_proto)
  return stats_proto
