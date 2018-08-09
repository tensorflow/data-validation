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
from tensorflow_data_validation.types_compat import List, Optional
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def get_categorical_numeric_features(
    schema):
  """Get the list of numeric features that should be treated as categorical.

  Args:
    schema: The schema for the data.

  Returns:
    A list of int features that should be considered categorical.
  """
  categorical_features = []
  for feature in schema.feature:
    if (feature.type == schema_pb2.INT and feature.HasField('int_domain') and
        feature.int_domain.is_categorical):
      categorical_features.append(feature.name)
  return categorical_features


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
