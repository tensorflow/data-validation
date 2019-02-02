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

"""Module for computing number of unique values per string feature."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils.stats_util import get_feature_type
from tensorflow_data_validation.types_compat import Iterator, Optional, Set, Text, Tuple, Union
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def _make_feature_stats_proto(
    feature_name, count,
    is_categorical):
  """Makes a FeatureNameStatistics proto containing the uniques stats."""
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name
  # If we have a categorical feature, we preserve the type to be the original
  # INT type.
  result.type = (statistics_pb2.FeatureNameStatistics.INT if is_categorical
                 else statistics_pb2.FeatureNameStatistics.STRING)
  result.string_stats.unique = count
  return result


def _make_dataset_feature_stats_proto_with_single_feature(
    feature_name_to_value_count,
    categorical_features
):
  """Generates a DatasetFeatureStatistics proto containing a single feature."""
  (slice_key, feature_name), count = feature_name_to_value_count
  result = statistics_pb2.DatasetFeatureStatistics()
  result.features.add().CopyFrom(
      _make_feature_stats_proto(
          feature_name, count, feature_name in categorical_features))
  return slice_key, result


# Input type check is commented out, as beam python will fail the type check
# when input is an empty dict.
# @beam.typehints.with_input_types(types.SlicedExample)
@beam.typehints.with_output_types(
    Tuple[types.SliceKey, statistics_pb2.DatasetFeatureStatistics])
class _UniquesStatsGeneratorImpl(beam.PTransform):
  """A PTransform that computes the number of unique values
  for string features.
  """

  def __init__(self, schema):
    """Initializes unique stats generator ptransform.

    Args:
      schema: An schema for the dataset. None if no schema is available.
    """
    self._categorical_features = set(
        schema_util.get_categorical_numeric_features(schema) if schema else [])

  def _convert_to_feature_name_value_tuples(
      self, sliced_example
  ):
    """Converts input to tuples containing slice key, feature name and value."""
    slice_key, example = sliced_example
    for feature_name, values in example.items():
      is_categorical = feature_name in self._categorical_features
      # Check if we have a numpy array with at least one value.
      if not isinstance(values, np.ndarray) or values.size == 0:
        continue
      # If the feature is neither categorical nor of string type, then
      # skip the feature.
      if not (is_categorical or get_feature_type(
          values.dtype) == statistics_pb2.FeatureNameStatistics.STRING):
        continue

      if is_categorical:
        values = values.astype(str)
      for value in values:
        yield slice_key, feature_name, value

  def expand(self, pcoll):
    """Computes number of unique values for string features."""
    # Count the number of appearance of each feature_value. Output is a
    # pcollection of DatasetFeatureStatistics protos
    return (
        pcoll
        | 'Uniques_ConvertToFeatureNameValueTuples' >>
        beam.FlatMap(self._convert_to_feature_name_value_tuples)
        | 'Uniques_RemoveDuplicateFeatureNameValueTuples' >>
        beam.RemoveDuplicates()
        # Drop the values to only have the slice_key and feature_name with each
        # repeated the number of unique values times.
        | 'Uniques_DropValues' >> beam.Map(lambda entry: (entry[0], entry[1]))
        | 'Uniques_CountPerFeatureName' >> beam.combiners.Count().PerElement()
        | 'Uniques_ConvertToSingleFeatureStats' >> beam.Map(
            _make_dataset_feature_stats_proto_with_single_feature,
            categorical_features=self._categorical_features))


class UniquesStatsGenerator(stats_generator.TransformStatsGenerator):
  """A transform statistics generator that computes the number of unique values
  for string features."""

  def __init__(self,
               name = 'UniquesStatsGenerator',
               schema = None):
    """Initializes unique stats generator.

    Args:
      name: An optional unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
    """
    super(UniquesStatsGenerator, self).__init__(
        name, schema=schema, ptransform=_UniquesStatsGeneratorImpl(schema))
