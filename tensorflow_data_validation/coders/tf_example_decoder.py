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


def _make_example_dict_value(feature):
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


# TODO(b/118385481): Maybe reuse tft.ExampleProtoCoder when data schema is
# provided. The difference between this decoder and tensorflow transform's
# ExampleProtoCoder class is that this decoder does not accept data schema as
# input, thus we do not know the list of features and their types in advance.
class TFExampleDecoder(object):
  """A decoder for decoding TF examples into tf data validation datasets.
  """

  def __init__(self):
    # Using pre-allocated tf.train.Example object for performance reasons.
    # Note that we don't violate the Beam programming model as we don't
    # return this mutable object.
    self._decode_example_cache = tf.train.Example()

  def __reduce__(self):
    return TFExampleDecoder, ()

  def decode(self, serialized_example_proto):
    """Decodes serialized tf.Example to tf data validation input dict."""
    self._decode_example_cache.ParseFromString(serialized_example_proto)
    feature_map = self._decode_example_cache.features.feature
    return {
        feature_name: _make_example_dict_value(feature_map[feature_name])
        for feature_name in feature_map
    }


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(types.BeamExample)
def DecodeTFExample(examples
                   ):  # pylint: disable=invalid-name
  """Decodes serialized TF examples into an in-memory dict representation.

  Args:
    examples: A PCollection of strings representing serialized TF examples.

  Returns:
    A PCollection of dicts representing the TF examples.
  """
  decoder = TFExampleDecoder()
  return examples | 'ParseTFExamples' >> beam.Map(decoder.decode)
