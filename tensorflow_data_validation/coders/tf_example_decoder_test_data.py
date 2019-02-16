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
"""Test data for TFExampleDecoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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
        'decoded_example': {'x': np.array([4.0, 5.0], dtype=np.float32)}
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
        'decoded_example': {'x': np.array([], dtype=np.float32)}
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

BEAM_TF_EXAMPLE_DECODER_TESTS = [
    {
        'testcase_name': 'beam_test',
        'example_proto_text': '''
          features {
            feature {
              key: "int_feature_1"
              value { int64_list { value: [ 0 ] } }
            }
            feature {
              key: "int_feature_2"
              value { int64_list { value: [ 1, 2, 3 ] } }
            }
            feature {
              key: "float_feature_1"
              value { float_list { value: [ 4.0 ] } }
            }
            feature {
              key: "float_feature_2"
              value { float_list { value: [ 5.0, 6.0 ] } }
            }
            feature {
              key: "str_feature_1"
              value { bytes_list { value: [ 'female' ] } }
            }
            feature {
              key: "str_feature_2"
              value { bytes_list { value: [ 'string', 'list' ] } }
            }
          }
        ''',
        'decoded_example': {
            'int_feature_1': np.array([0], dtype=np.integer),
            'int_feature_2': np.array([1, 2, 3], dtype=np.integer),
            'float_feature_1': np.array([4.0], dtype=np.float32),
            'float_feature_2': np.array([5.0, 6.0], dtype=np.float32),
            'str_feature_1': np.array([b'female'], dtype=np.object),
            'str_feature_2': np.array([b'string', b'list'], dtype=np.object),
        }
    },
]
