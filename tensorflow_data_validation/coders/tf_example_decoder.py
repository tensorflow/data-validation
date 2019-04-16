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
from tensorflow_data_validation import types
from tensorflow_data_validation.pywrap import pywrap_tensorflow_data_validation


DecodeExample = pywrap_tensorflow_data_validation.TFDV_DecodeExample  # pylint: disable=invalid-name


# TODO(pachristopher): This fast coder can also benefit TFT. Consider moving
# this coder to tf.Beam once it is available.
class TFExampleDecoder(object):
  """A decoder for decoding TF examples into tf data validation datasets.
  """

  def decode(self, serialized_example_proto):
    """Decodes serialized tf.Example to tf data validation input dict."""
    return DecodeExample(serialized_example_proto)


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
