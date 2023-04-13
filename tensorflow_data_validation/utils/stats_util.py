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

import logging
from typing import Dict, Iterable, Optional, Sequence, Text, Tuple, Union

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils import artifacts_io_impl
from tensorflow_data_validation.utils import io_util
from tfx_bsl import statistics
from tfx_bsl.arrow import array_util
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
  if not array_util.is_list_like(arrow_type):
    raise TypeError('Expected feature column to be a '
                    '(Large)List<primitive|struct> or null, but feature {} '
                    'was {}.'.format(feature_path, arrow_type))

  value_type = array_util.get_innermost_nested_type(arrow_type)
  if pa.types.is_integer(value_type):
    return statistics_pb2.FeatureNameStatistics.INT
  elif pa.types.is_floating(value_type):
    return statistics_pb2.FeatureNameStatistics.FLOAT
  elif arrow_util.is_binary_like(value_type):
    return statistics_pb2.FeatureNameStatistics.STRING
  elif pa.types.is_struct(value_type):
    return statistics_pb2.FeatureNameStatistics.STRUCT
  elif pa.types.is_null(value_type):
    return None

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
                     output_path: Text) -> None:
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
    input_path: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
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


def load_stats_binary(
    input_path: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Loads a serialized DatasetFeatureStatisticsList proto from a file.

  Args:
    input_path: File path from which to load the DatasetFeatureStatisticsList
      proto.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
  stats_proto.ParseFromString(io_util.read_file_to_string(
      input_path, binary_mode=True))
  return stats_proto


def load_stats_tfrecord(
    input_path: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Loads data statistics proto from TFRecord file.

  Args:
    input_path: Data statistics file path.

  Returns:
    A DatasetFeatureStatisticsList proto.
  """
  it = artifacts_io_impl.get_io_provider('tfrecords').record_iterator_impl(
      [input_path])
  result = next(it)
  try:
    next(it)
    raise ValueError('load_stats_tfrecord expects a single record.')
  except StopIteration:
    return result
  except Exception as e:
    raise e


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


def get_slice_stats(
    stats: statistics_pb2.DatasetFeatureStatisticsList,
    slice_key: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Get statistics associated with a specific slice.

  Args:
    stats: A DatasetFeatureStatisticsList protocol buffer.
    slice_key: Slice key of the slice.

  Returns:
    Statistics of the specific slice.

  Raises:
    ValueError: If the input statistics proto does not have the specified slice
      statistics.
  """
  for slice_stats in stats.datasets:
    if slice_stats.name == slice_key:
      result = statistics_pb2.DatasetFeatureStatisticsList()
      result.datasets.add().CopyFrom(slice_stats)
      return result
  raise ValueError('Invalid slice key.')


def load_statistics(
    input_path: Text) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Loads data statistics proto from file.

  Args:
    input_path: Data statistics file path. The file should be a one-record
      TFRecord file or a plain file containing the statistics proto in Proto
      Text Format.

  Returns:
    A DatasetFeatureStatisticsList proto.

  Raises:
    IOError: If the input path does not exist.
  """
  if not tf.io.gfile.exists(input_path):
    raise IOError('Invalid input path {}.'.format(input_path))
  try:
    return load_stats_tfrecord(input_path)
  except Exception:  # pylint: disable=broad-except
    logging.info('File %s did not look like a TFRecord. Try reading as a plain '
                 'file.', input_path)
    return load_stats_text(input_path)


def _normalize_feature_id(
    name_or_path_or_steps: Union[str, types.FeaturePath, Iterable[str]]
) -> types.FeaturePath:
  if isinstance(name_or_path_or_steps, str):
    return types.FeaturePath([name_or_path_or_steps])
  if isinstance(name_or_path_or_steps, types.FeaturePath):
    return name_or_path_or_steps
  return types.FeaturePath(name_or_path_or_steps)


class DatasetListView(object):
  """View of statistics for multiple datasets (slices)."""

  def __init__(self, stats_proto: statistics_pb2.DatasetFeatureStatisticsList):
    self._statistics = stats_proto
    self._slice_map = {}  # type: Dict[str, DatasetView]
    self._initialized = False

  def _init_index(self):
    """Initializes internal mappings."""
    # Lazily initialize in case we don't need an index.
    if self._initialized:
      return
    for dataset in self._statistics.datasets:
      if dataset.name in self._slice_map:
        raise ValueError('Duplicate slice name %s' % dataset.name)
      self._slice_map[dataset.name] = DatasetView(dataset)
    self._initialized = True

  def proto(self) -> statistics_pb2.DatasetFeatureStatisticsList:
    """Retrieve the underlying proto."""
    return self._statistics

  def get_slice(self, slice_key: str) -> Optional['DatasetView']:
    self._init_index()
    return self._slice_map.get(slice_key, None)

  def get_default_slice(self) -> Optional['DatasetView']:
    self._init_index()
    if len(self._slice_map) == 1:
      for _, v in self._slice_map.items():
        return v
    return self._slice_map.get(constants.DEFAULT_SLICE_KEY, None)

  def get_default_slice_or_die(self) -> 'DatasetView':
    # TODO(b/221453427): Update uses, or consider changing get_default_slice.
    default_slice = self.get_default_slice()
    if default_slice is None:
      raise ValueError('Missing default slice')
    return default_slice

  def list_slices(self) -> Iterable[str]:
    self._init_index()
    return self._slice_map.keys()


class DatasetView(object):
  """View of statistics for a dataset (slice)."""

  def __init__(self, stats_proto: statistics_pb2.DatasetFeatureStatistics):
    self._feature_map = {}  # type: Dict[types.FeaturePath, int]
    self._cross_feature_map = {
    }  # type: Dict[Tuple[types.FeaturePath, types.FeaturePath], int]
    self._statistics = stats_proto
    self._initialized = False

  def _init_index(self):
    """Initializes internal indices. Noop if already initialized."""
    if self._initialized:
      return
    field_identifier = None
    for j, feature in enumerate(self._statistics.features):
      if field_identifier is None:
        field_identifier = feature.WhichOneof('field_id')
      elif feature.WhichOneof('field_id') != field_identifier:
        raise ValueError(
            'Features must be specified with either path or name within a'
            ' Dataset.')

      if field_identifier == 'name':
        feature_id = types.FeaturePath([feature.name])
      else:
        feature_id = types.FeaturePath.from_proto(feature.path)

      if feature_id in self._feature_map:
        raise ValueError('Duplicate feature %s' % feature_id)
      self._feature_map[feature_id] = j
    for j, cross_feature in enumerate(self._statistics.cross_features):
      feature_id = (types.FeaturePath.from_proto(cross_feature.path_x),
                    types.FeaturePath.from_proto(cross_feature.path_y))
      if feature_id in self._cross_feature_map:
        raise ValueError('Duplicate feature %s' % feature_id)
      self._cross_feature_map[feature_id] = j
    self._initialized = True

  def proto(self) -> statistics_pb2.DatasetFeatureStatistics:
    """Retrieve the underlying proto."""
    return self._statistics

  def get_feature(
      self, feature_id: Union[str, types.FeaturePath, Iterable[str]]
  ) -> Optional['FeatureView']:
    """Retrieve a feature if it exists.

    Features specified within the underlying proto by name (instead of path) are
    normalized to a length 1 path, and can be referred to as such.

    Args:
      feature_id: A types.FeaturePath, Iterable[str] consisting of path steps,
        or a str, which is converted to a length one path.

    Returns:
      A FeatureView, or None if feature_id is not present.
    """
    feature_id = _normalize_feature_id(feature_id)
    self._init_index()
    index = self._feature_map.get(feature_id, None)
    if index is None:
      return None
    return FeatureView(self._statistics.features[index])

  def get_cross_feature(
      self, x_path: Union[str, types.FeaturePath,
                          Iterable[str]], y_path: Union[str, types.FeaturePath,
                                                        Iterable[str]]
  ) -> Optional['CrossFeatureView']:
    """Retrieve a cross-feature if it exists, or None."""

    x_path = _normalize_feature_id(x_path)
    y_path = _normalize_feature_id(y_path)
    self._init_index()
    feature_id = (x_path, y_path)
    index = self._cross_feature_map.get(feature_id, None)
    if index is None:
      return None
    return CrossFeatureView(self._statistics.cross_features[index])

  def list_features(self) -> Iterable[types.FeaturePath]:
    """Lists feature identifiers."""
    self._init_index()
    return self._feature_map.keys()

  def list_cross_features(
      self) -> Iterable[Tuple[types.FeaturePath, types.FeaturePath]]:
    """Lists cross-feature identifiers."""
    self._init_index()
    return self._cross_feature_map.keys()

  def get_derived_feature(
      self, deriver_name: str,
      source_paths: Sequence[types.FeaturePath]) -> Optional['FeatureView']:
    """Retrieve a derived feature based on a deriver name and its inputs.

    Args:
      deriver_name: The name of a deriver. Matches validation_derived_source
        deriver_name.
      source_paths: Source paths for derived features. Matches
        validation_derived_source.source_path.

    Returns:
      FeatureView of derived feature.

    Raises:
      ValueError if multiple derived features match.
    """
    # TODO(b/221453427): Consider indexing if performance becomes an issue.
    results = []
    for feature in self.proto().features:
      if feature.validation_derived_source is None:
        continue
      if feature.validation_derived_source.deriver_name != deriver_name:
        continue
      if (len(source_paths) != len(
          feature.validation_derived_source.source_path)):
        continue
      all_match = True
      for i in range(len(source_paths)):
        if (source_paths[i] != types.FeaturePath.from_proto(
            feature.validation_derived_source.source_path[i])):
          all_match = False
          break
      if all_match:
        results.append(FeatureView(feature))
      if len(results) > 1:
        raise ValueError('Ambiguous result, %d features matched' % len(results))
    if len(results) == 1:
      return results.pop()
    return None


class FeatureView(object):
  """View of a single feature.

  This class provides accessor methods, as well as access to the underlying
  proto. Where possible, accessors should be used in place of proto access (for
  example, x.numeric_statistics() instead of x.proto().num_stats) in order to
  support future extension of the proto.
  """

  def __init__(self, stats_proto: statistics_pb2.FeatureNameStatistics):
    self._statistics = stats_proto

  def proto(self) -> statistics_pb2.FeatureNameStatistics:
    """Retrieve the underlying proto."""
    return self._statistics

  def custom_statistic(self,
                       name: str) -> Optional[statistics_pb2.CustomStatistic]:
    """Retrieve a custom_statistic by name."""
    result = None
    for stat in self._statistics.custom_stats:
      if stat.name == name:
        if result is None:
          result = stat
        else:
          raise ValueError('Duplicate custom_stats for name %s' % name)
    return result

  # TODO(b/202910677): Add convenience methods for retrieving first-party custom
  # statistics (e.g., MI, NLP).

  def numeric_statistics(self) -> Optional[statistics_pb2.NumericStatistics]:
    """Retrieve numeric statistics if available."""
    if self._statistics.WhichOneof('stats') == 'num_stats':
      return self._statistics.num_stats
    return None

  def string_statistics(self) -> Optional[statistics_pb2.StringStatistics]:
    """Retrieve string statistics if available."""
    if self._statistics.WhichOneof('stats') == 'string_stats':
      return self._statistics.string_stats
    return None

  def bytes_statistics(self) -> Optional[statistics_pb2.BytesStatistics]:
    """Retrieve byte statistics if available."""
    if self._statistics.WhichOneof('stats') == 'bytes_stats':
      return self._statistics.bytes_stats
    return None

  def struct_statistics(self) -> Optional[statistics_pb2.StructStatistics]:
    """Retrieve struct statistics if available."""
    if self._statistics.WhichOneof('stats') == 'struct_stats':
      return self._statistics.struct_stats
    return None

  def common_statistics(self) -> Optional[statistics_pb2.CommonStatistics]:
    """Retrieve common statistics if available."""
    which = self._statistics.WhichOneof('stats')
    if which == 'num_stats':
      return self._statistics.num_stats.common_stats
    if which == 'string_stats':
      return self._statistics.string_stats.common_stats
    if which == 'bytes_stats':
      return self._statistics.bytes_stats.common_stats
    if which == 'struct_stats':
      return self._statistics.struct_stats.common_stats
    return None


class CrossFeatureView(object):
  """View of a single cross feature."""

  def __init__(self, stats_proto: statistics_pb2.CrossFeatureStatistics):
    self._statistics = stats_proto

  def proto(self) -> statistics_pb2.CrossFeatureStatistics:
    """Retrieve the underlying proto."""
    return self._statistics


def load_sharded_statistics(
    input_path_prefix: Optional[str] = None,
    input_paths: Optional[Iterable[str]] = None,
    io_provider: Optional[artifacts_io_impl.StatisticsIOProvider] = None
) -> DatasetListView:
  """Read a sharded DatasetFeatureStatisticsList from disk as a DatasetListView.

  Args:
    input_path_prefix: If passed, loads files starting with this prefix and
      ending with a pattern corresponding to the output of the provided
        io_provider.
    input_paths: A list of file paths of files containing sharded
      DatasetFeatureStatisticsList protos.
    io_provider: Optional StatisticsIOProvider. If unset, a default will be
      constructed.

  Returns:
    A DatasetListView containing the merged proto.
  """
  if input_path_prefix is None == input_paths is None:
    raise ValueError('Must provide one of input_paths_prefix, input_paths.')
  if io_provider is None:
    io_provider = artifacts_io_impl.get_io_provider()
  if input_path_prefix is not None:
    input_paths = io_provider.glob(input_path_prefix)
  if not input_paths:
    raise ValueError('No input paths found paths=%s, pattern=%s' %
                     (input_paths, input_path_prefix))
  acc = statistics.DatasetListAccumulator()
  stats_iter = io_provider.record_iterator_impl(input_paths)
  for stats_list in stats_iter:
    for dataset in stats_list.datasets:
      acc.MergeDatasetFeatureStatistics(dataset.SerializeToString())
  stats = statistics_pb2.DatasetFeatureStatisticsList()
  stats.ParseFromString(acc.Get())
  return DatasetListView(stats)
