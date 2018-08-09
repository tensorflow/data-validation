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
import numpy as np
import tensorflow as tf
from tensorflow_data_validation.coders import tf_example_decoder

from google.protobuf import text_format


class TFExampleDecoderTest(absltest.TestCase):
  """Tests for TFExampleDecoder."""

  def _check_decoding_results(self, actual, expected):
    # Check that the numpy array dtypes match.
    self.assertEqual(len(actual), len(expected))
    for key in actual:
      self.assertEqual(actual[key].dtype, expected[key].dtype)
    np.testing.assert_equal(actual, expected)

  def test_decode_example_empty_input(self):
    example = tf.train.Example()
    decoder = tf_example_decoder.TFExampleDecoder()
    self._check_decoding_results(
        decoder.decode(example.SerializeToString()), {})

  def test_decode_example(self):
    example_proto_text = """
    features {
      feature { key: "int_feature_1"
                value { int64_list { value: [ 0 ] } } }
      feature { key: "int_feature_2"
                value { int64_list { value: [ 1, 2, 3 ] } } }
      feature { key: "float_feature_1"
                value { float_list { value: [ 4.0 ] } } }
      feature { key: "float_feature_2"
                value { float_list { value: [ 5.0, 6.0 ] } } }
      feature { key: "str_feature_1"
                value { bytes_list { value: [ 'female' ] } } }
      feature { key: "str_feature_2"
                value { bytes_list { value: [ 'string', 'list' ] } } }
    }
    """
    expected_decoded = {
        'int_feature_1': np.array([0], dtype=np.integer),
        'int_feature_2': np.array([1, 2, 3], dtype=np.integer),
        'float_feature_1': np.array([4.0], dtype=np.floating),
        'float_feature_2': np.array([5.0, 6.0], dtype=np.floating),
        'str_feature_1': np.array(['female'], dtype=np.object),
        'str_feature_2': np.array(['string', 'list'], dtype=np.object),
    }
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)

    decoder = tf_example_decoder.TFExampleDecoder()
    self._check_decoding_results(
        decoder.decode(example.SerializeToString()), expected_decoded)

  def test_decode_example_empty_feature(self):
    example_proto_text = """
    features {
      feature { key: "int_feature" value { int64_list { value: [ 0 ] } } }
      feature { key: "int_feature_empty" value { } }
      feature { key: "float_feature" value { float_list { value: [ 4.0 ] } } }
      feature { key: "float_feature_empty" value { } }
      feature { key: "str_feature" value { bytes_list { value: [ 'male' ] } } }
      feature { key: "str_feature_empty" value { } }
    }
    """
    expected_decoded = {
        'int_feature': np.array([0], dtype=np.integer),
        'int_feature_empty': np.array([], dtype=np.object),
        'float_feature': np.array([4.0], dtype=np.floating),
        'float_feature_empty': np.array([], dtype=np.object),
        'str_feature': np.array(['male'], dtype=np.object),
        'str_feature_empty': np.array([], dtype=np.object),
    }
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)

    decoder = tf_example_decoder.TFExampleDecoder()
    self._check_decoding_results(
        decoder.decode(example.SerializeToString()), expected_decoded)


if __name__ == '__main__':
  absltest.main()
