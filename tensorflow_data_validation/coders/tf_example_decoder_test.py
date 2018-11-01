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
import numpy as np
import tensorflow as tf
from tensorflow_data_validation.coders import tf_example_decoder
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format


TF_EXAMPLE_DECODER_TESTS = [
    {
        'testcase_name': 'empty_input',
        'example_proto_text': '''features {}''',
        'decoded_example': {}
    },
    {
        'testcase_name': 'int_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { int64_list { value: [ 1, 2, 3 ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([1, 2, 3], dtype=np.integer)}
    },
    {
        'testcase_name': 'float_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { float_list { value: [ 4.0, 5.0 ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([4.0, 5.0], dtype=np.floating)}
    },
    {
        'testcase_name': 'str_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { bytes_list { value: [ 'string', 'list' ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([b'string', b'list'],
                                          dtype=np.object)}
    },
    {
        'testcase_name': 'int_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { int64_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=np.integer)}
    },
    {
        'testcase_name': 'float_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { float_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=np.floating)}
    },
    {
        'testcase_name': 'str_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { bytes_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=np.object)}
    },
    {
        'testcase_name': 'feature_missing',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { }
            }
          }
        ''',
        'decoded_example': {'x': None}
    },
]


class TFExampleDecoderTest(parameterized.TestCase):
  """Tests for TFExampleDecoder."""

  def _check_decoding_results(self, actual, expected):
    # Check that the numpy array dtypes match.
    self.assertEqual(len(actual), len(expected))
    for key in actual:
      if expected[key] is None:
        self.assertEqual(actual[key], None)
      else:
        self.assertEqual(actual[key].dtype, expected[key].dtype)
        np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(*TF_EXAMPLE_DECODER_TESTS)
  def test_decode_example(self, example_proto_text, decoded_example):
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)
    decoder = tf_example_decoder.TFExampleDecoder()
    self._check_decoding_results(
        decoder.decode(example.SerializeToString()), decoded_example)

  def test_decode_example_with_beam_pipeline(self):
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
        'str_feature_1': np.array([b'female'], dtype=np.object),
        'str_feature_2': np.array([b'string', b'list'], dtype=np.object),
    }
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)
    with beam.Pipeline() as p:
      result = (p
                | beam.Create([example.SerializeToString()])
                | tf_example_decoder.DecodeTFExample())
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, [expected_decoded]))


if __name__ == '__main__':
  absltest.main()
