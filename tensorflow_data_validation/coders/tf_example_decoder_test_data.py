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

import pyarrow as pa


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
        'decoded_table': pa.Table.from_arrays([
            pa.array([[0]], pa.list_(pa.int64())),
            pa.array([[1, 2, 3]], pa.list_(pa.int64())),
            pa.array([[4.0]], pa.list_(pa.float32())),
            pa.array([[5.0, 6.0]], pa.list_(pa.float32())),
            pa.array([[b'female']], pa.list_(pa.binary())),
            pa.array([[b'string', b'list']], pa.list_(pa.binary()))
        ], ['int_feature_1', 'int_feature_2', 'float_feature_1',
            'float_feature_2', 'str_feature_1', 'str_feature_2'])
    },
]
