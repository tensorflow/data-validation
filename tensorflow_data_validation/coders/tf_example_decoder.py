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

"""Decode TF Examples into in-memory representation for tf data validation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_data_validation import types


def _convert_to_numpy_array(feature):
  """Converts a single TF feature to its numpy array representation."""
  kind = feature.WhichOneof('kind')
  if kind == 'int64_list':
    return np.asarray(feature.int64_list.value, dtype=np.integer)
  elif kind == 'float_list':
    return np.asarray(feature.float_list.value, dtype=np.floating)
  elif kind == 'bytes_list':
    return np.asarray(feature.bytes_list.value, dtype=np.object)
  else:
    # Return an empty array for feature with no value list. In numpy, an empty
    # array has a dtype of float, thus we explicitly set it to np.object here.
    return np.array([], dtype=np.object)


class TFExampleDecoder(object):
  """A decoder for decoding TF examples into tf data validation datasets.
  """

  def decode(self, serialized_example_proto):
    """Decodes serialized tf.Example to tf data validation input dict."""
    example = tf.train.Example()
    example.ParseFromString(serialized_example_proto)
    feature_map = example.features.feature
    return {
        feature_name: _convert_to_numpy_array(feature_map[feature_name])
        for feature_name in feature_map
    }
