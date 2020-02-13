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
"""Tests for TFExampleDecoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.coders import tf_example_decoder_test_data
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format


class TFExampleDecoderTest(parameterized.TestCase):
  """Tests for TFExampleDecoder."""

  @parameterized.named_parameters(
      *tf_example_decoder_test_data.BEAM_TF_EXAMPLE_DECODER_TESTS)
  def test_decode_example_with_beam_pipeline(self, example_proto_text,
                                             decoded_table):
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)
    with beam.Pipeline() as p:
      result = (p
                | beam.Create([example.SerializeToString()])
                | tf_example_decoder.DecodeTFExample())
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, [decoded_table]))

if __name__ == '__main__':
  absltest.main()
