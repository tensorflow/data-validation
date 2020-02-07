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
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils import io_util
from typing import Dict, Optional, Text, Union
from google.protobuf import text_format
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


def maybe_get_utf8(value: bytes) -> Optional[Text]:
  """Returns the value decoded as utf-8, or None if it cannot be decoded.

  Args:
    value: The bytes value to decode.
  Returns:
    The value decoded as utf-8, or None, if the value cannot be decoded.
  """
  try:
    decoded_value = value.decode('utf-8')
  except UnicodeError:
    return None
  return decoded_value


def get_feature_type(
    dtype: np.dtype) -> Optional[types.FeatureNameStatisticsType]:
  """Get feature type from numpy dtype.

  Args:
    dtype: Numpy dtype.

  Returns:
    A statistics_pb2.FeatureNameStatistics.Type value.
  """
  return _NP_DTYPE_KIND_TO_FEATURE_TYPE.get(dtype.kind)


def get_feature_type_from_arrow_type(
    feature_path: types.FeaturePath,
    arrow_type: pa.DataType) -> Optional[types.FeatureNameStatisticsType]:
  """Get feature type from Arrow type.

  Args:
    feature_path: path of the feature.
    arrow_type: Arrow DataType.

  Returns:
    A statistics_pb2.FeatureNameStatistics.Type value or None if arrow_type
    is null (which means it cannot be determined for now).

  Raises:
    TypeError: if the type is not supported.
  """
  if pa.types.is_null(arrow_type):
    return None
  if not arrow_util.is_list_like(arrow_type):
    raise TypeError('Expected feature column to be a '
                    '(Large)List<primitive|struct> or null, but feature {} '
                    'was {}.'.format(feature_path, arrow_type))

  value_type = arrow_type.value_type
  if pa.types.is_integer(value_type):
    return statistics_pb2.FeatureNameStatistics.INT
  elif pa.types.is_floating(value_type):
    return statistics_pb2.FeatureNameStatistics.FLOAT
  elif arrow_util.is_binary_like(value_type):
    return statistics_pb2.FeatureNameStatistics.STRING
  elif pa.types.is_struct(value_type):
    return statistics_pb2.FeatureNameStatistics.STRUCT

  raise TypeError('Feature {} has unsupported arrow type: {}'.format(
      feature_path, arrow_type))


def make_dataset_feature_stats_proto(
    stats_values: Dict[types.FeaturePath, Dict[Text, float]]
) -> statistics_pb2.DatasetFeatureStatistics:
  """Builds DatasetFeatureStatistics proto with custom stats from input dict.

  Args:
    stats_values: A Dict[FeaturePath, Dict[str,float]] where the keys are
    feature paths, and values are Dicts with keys denoting name of the custom
    statistic and values denoting the value of the custom statistic
    for the feature.
      Ex. {
            FeaturePath(('feature_1',)): {
                'Mutual Information': 0.5,
                'Correlation': 0.1 },
            FeaturePath(('feature_2',)): {
                'Mutual Information': 0.8,
                'Correlation': 0.6 }
          }

  Returns:
    DatasetFeatureStatistics proto containing the custom statistics for each
    feature in the dataset.
  """
  result = statistics_pb2.DatasetFeatureStatistics()

  # Sort alphabetically by feature name to have deterministic ordering
  feature_paths = sorted(stats_values.keys())

  for feature_path in feature_paths:
    feature_stats_proto = _make_feature_stats_proto(stats_values[feature_path],
                                                    feature_path)
    new_feature_stats_proto = result.features.add()
    new_feature_stats_proto.CopyFrom(feature_stats_proto)

  return result


def _make_feature_stats_proto(
    stats_values: Dict[Text, float],
    feature_path: types.FeaturePath) -> statistics_pb2.FeatureNameStatistics:
  """Creates the FeatureNameStatistics proto for one feature.

  Args:
    stats_values: A Dict[str,float] where the key of the dict is the name of the
      custom statistic and the value is the numeric value of the custom
      statistic of that feature. Ex. {
              'Mutual Information': 0.5,
              'Correlation': 0.1 }
    feature_path: The path of the feature.

  Returns:
    A FeatureNameStatistic proto containing the custom statistics for a
    feature.
  """

  result = statistics_pb2.FeatureNameStatistics()
  result.path.CopyFrom(feature_path.to_proto())

  # Sort alphabetically by statistic name to have deterministic ordering
  stat_names = sorted(stats_values.keys())
  for stat_name in stat_names:
    result.custom_stats.add(name=stat_name, num=stats_values[stat_name])
  return result


def write_stats_text(stats: statistics_pb2.DatasetFeatureStatisticsList,
                     output_path: bytes) -> None:
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
  io_util.write_string_to_file(output_path, stats_proto_text)


def load_stats_text(
    input_path: bytes) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Loads the specified DatasetFeatureStatisticsList proto stored in text format.

  Args:
    input_path: File path from which to load the DatasetFeatureStatisticsList
      proto.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
  stats_text = io_util.read_file_to_string(input_path)
  text_format.Parse(stats_text, stats_proto)
  return stats_proto


def get_feature_stats(stats: statistics_pb2.DatasetFeatureStatistics,
                      feature_path: types.FeaturePath
                     ) -> statistics_pb2.FeatureNameStatistics:
  """Get feature statistics from the dataset statistics.

  Args:
    stats: A DatasetFeatureStatistics protocol buffer.
    feature_path: The path of the feature whose statistics to obtain from the
      dataset statistics.

  Returns:
    A FeatureNameStatistics protocol buffer.

  Raises:
    TypeError: If the input statistics is not of the expected type.
    ValueError: If the input feature is not found in the dataset statistics.
  """
  if not isinstance(stats, statistics_pb2.DatasetFeatureStatistics):
    raise TypeError('statistics is of type %s, should be a '
                    'DatasetFeatureStatistics proto.' %
                    type(stats).__name__)

  for feature_stats in stats.features:
    if feature_path == types.FeaturePath.from_proto(feature_stats.path):
      return feature_stats

  raise ValueError('Feature %s not found in the dataset statistics.' %
                   feature_path)


def get_custom_stats(
    feature_stats: statistics_pb2.FeatureNameStatistics,
    custom_stats_name: Text
) -> Union[float, Text, statistics_pb2.Histogram, statistics_pb2.RankHistogram]:
  """Get custom statistics from the feature statistics.

  Args:
    feature_stats: A FeatureNameStatistics protocol buffer.
    custom_stats_name: The name of the custom statistics to obtain from the
      feature statistics proto.

  Returns:
    The custom statistic.

  Raises:
    TypeError: If the input feature statistics is not of the expected type.
    ValueError: If the custom statistic is not found in the feature statistics.
  """
  if not isinstance(feature_stats, statistics_pb2.FeatureNameStatistics):
    raise TypeError('feature_stats is of type %s, should be a '
                    'FeatureNameStatistics proto.' %
                    type(feature_stats).__name__)

  for custom_stats in feature_stats.custom_stats:
    if custom_stats.name == custom_stats_name:
      return getattr(custom_stats, custom_stats.WhichOneof('val'))

  raise ValueError('Custom statistics %s not found in the feature statistics.' %
                   custom_stats_name)
