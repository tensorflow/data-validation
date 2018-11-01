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

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import Optional


def _convert_to_example_dict_value(feature
                                  ):
  """Converts a single TF feature to its example Dict value."""
  kind = feature.WhichOneof('kind')
  if kind == 'int64_list':
    return np.asarray(feature.int64_list.value, dtype=np.integer)
  elif kind == 'float_list':
    return np.asarray(feature.float_list.value, dtype=np.floating)
  elif kind == 'bytes_list':
    return np.asarray(feature.bytes_list.value, dtype=np.object)
  elif kind is None:
    # If we have a feature with no value list, we consider it to be a missing
    # value.
    return None
  else:
    raise ValueError('Unsupported value type found in feature: {}'.format(kind))


class TFExampleDecoder(object):
  """A decoder for decoding TF examples into tf data validation datasets.
  """

  def decode(self, serialized_example_proto):
    """Decodes serialized tf.Example to tf data validation input dict."""
    example = tf.train.Example()
    example.ParseFromString(serialized_example_proto)
    feature_map = example.features.feature
    return {
        feature_name: _convert_to_example_dict_value(feature_map[feature_name])
        for feature_name in feature_map
    }


@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(types.Example)
class DecodeTFExample(beam.PTransform):
  """Decodes TF examples into an in-memory dict representation. """

  def __init__(self):
    """Initializes DecodeTFExample ptransform."""
    self._decoder = TFExampleDecoder()

  def expand(self, examples):
    """Decodes serialized TF examples into an in-memory dict representation.

    Args:
      examples: A PCollection of strings representing serialized TF examples.

    Returns:
      A PCollection of dicts representing the TF examples.
    """
    return examples | 'ParseTFExamples' >> beam.Map(self._decoder.decode)
