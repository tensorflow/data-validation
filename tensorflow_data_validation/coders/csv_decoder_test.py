# coding=utf-8
#
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

"""Tests for CSV decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation.coders import csv_decoder
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

_TEST_CASES = [
    dict(
        testcase_name='simple',
        input_lines=['1,2.0,hello', '5,12.34,world'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], [5]], pa.large_list(pa.int64())),
                pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
                pa.array([[b'hello'], [b'world']],
                         pa.large_list(pa.large_binary())),
            ], ['int_feature', 'float_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='with_schema',
        input_lines=['1,1,2.0,hello', '5,5,12.34,world'],
        column_names=[
            'int_feature_parsed_as_float', 'int_feature', 'float_feature',
            'str_feature'
        ],
        schema=text_format.Parse(
            """
        feature { name: "int_feature_parsed_as_float" type: FLOAT }
        feature { name: "int_feature" type: INT }
        feature { name: "float_feature" type: FLOAT }
        feature { name: "str_feature" type: BYTES }
        """, schema_pb2.Schema()),
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], [5]], pa.large_list(pa.float32())),
                pa.array([[1], [5]], pa.large_list(pa.int64())),
                pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
                pa.array([[b'hello'], [b'world']],
                         pa.large_list(pa.large_binary())),
            ], [
                'int_feature_parsed_as_float', 'int_feature', 'float_feature',
                'str_feature'
            ])
        ]),
    dict(
        testcase_name='missing_values',
        input_lines=['1,,hello', ',12.34,'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], None], pa.large_list(pa.int64())),
                pa.array([None, [12.34]], pa.large_list(pa.float32())),
                pa.array([[b'hello'], None], pa.large_list(pa.large_binary())),
            ], ['int_feature', 'float_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='int_and_float_in_same_column',
        input_lines=['2,1.5', '1.5,2'],
        column_names=['float_feature1', 'float_feature2'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[2.0], [1.5]], pa.large_list(pa.float32())),
                pa.array([[1.5], [2.0]], pa.large_list(pa.float32())),
            ], ['float_feature1', 'float_feature2'])
        ]),
    dict(
        testcase_name='int_and_string_in_same_column',
        input_lines=['2,abc', 'abc,2'],
        column_names=['str_feature1', 'str_feature2'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'2'], [b'abc']], pa.large_list(pa.large_binary())),
                pa.array([[b'abc'], [b'2']], pa.large_list(pa.large_binary())),
            ], ['str_feature1', 'str_feature2'])
        ]),
    dict(
        testcase_name='float_and_string_in_same_column',
        input_lines=['2.3,abc', 'abc,2.3'],
        column_names=['str_feature1', 'str_feature2'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'2.3'], [b'abc']], pa.large_list(
                    pa.large_binary())),
                pa.array([[b'abc'], [b'2.3']], pa.large_list(
                    pa.large_binary())),
            ], ['str_feature1', 'str_feature2'])
        ]),
    dict(
        testcase_name='unicode',
        input_lines=[u'1,שקרכלשהו,22.34,text field'],
        column_names=[
            'int_feature', 'unicode_feature', 'float_feature', 'str_feature'
        ],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1]], pa.large_list(pa.int64())),
                pa.array([[22.34]], pa.large_list(pa.float32())),
                pa.array([[u'שקרכלשהו'.encode('utf-8')]],
                         pa.large_list(pa.large_binary())),
                pa.array([[b'text field']], pa.large_list(pa.large_binary())),
            ], [
                'int_feature', 'float_feature', 'unicode_feature', 'str_feature'
            ])
        ]),
    dict(
        testcase_name='csv_record_with_quotes',
        input_lines=['1,"ab,cd,ef"', '5,"wx,xy,yz"'],
        column_names=['int_feature', 'str_feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], [5]], pa.large_list(pa.int64())),
                pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']],
                         pa.large_list(pa.large_binary())),
            ], ['int_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='space_delimiter',
        input_lines=['1 "ab,cd,ef"', '5 "wx,xy,yz"'],
        column_names=['int_feature', 'str_feature'],
        delimiter=' ',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], [5]], pa.large_list(pa.int64())),
                pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']],
                         pa.large_list(pa.large_binary())),
            ], ['int_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='tab_delimiter',
        input_lines=['1\t"this is a \ttext"', '5\t'],
        column_names=['int_feature', 'str_feature'],
        delimiter='\t',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1], [5]], pa.large_list(pa.int64())),
                pa.array([[b'this is a \ttext'], None],
                         pa.large_list(pa.large_binary())),
            ], ['int_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='negative_values',
        input_lines=['-34', '45'],
        column_names=['feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[-34], [45]], pa.large_list(pa.int64())),
            ], ['feature'])
        ]),
    dict(
        testcase_name='int64_max',
        input_lines=['34', str(sys.maxsize)],
        column_names=['feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[34], [sys.maxsize]], pa.large_list(pa.int64())),
            ], ['feature'])
        ]),
    dict(
        testcase_name='large_int_categorical_pos',
        input_lines=['34', str(sys.maxsize + 1)],
        column_names=['feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'34'], [str(sys.maxsize + 1).encode('utf-8')]],
                         pa.large_list(pa.large_binary())),
            ], ['feature'])
        ]),
    dict(
        testcase_name='large_int_categorical_neg',
        input_lines=['34', str(-(sys.maxsize + 2))],
        column_names=['feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'34'], [str(-(sys.maxsize + 2)).encode('utf-8')]],
                         pa.large_list(pa.large_binary())),
            ], ['feature'])
        ]),
    dict(
        testcase_name='large_int_categorical_pos_and_neg',
        input_lines=[str(sys.maxsize + 1),
                     str(-(sys.maxsize + 2))],
        column_names=['feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[str(sys.maxsize + 1).encode('utf-8')],
                          [str(-(sys.maxsize + 2)).encode('utf-8')]],
                         pa.large_list(pa.large_binary())),
            ], ['feature'])
        ]),
    dict(
        testcase_name='empty_row',
        input_lines=[',,', '1,2.0,hello'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([None, [1]], pa.large_list(pa.int64())),
                pa.array([None, [2.0]], pa.large_list(pa.float32())),
                pa.array([None, [b'hello']], pa.large_list(pa.large_binary())),
            ], ['int_feature', 'float_feature', 'str_feature'])
        ]),
    dict(
        testcase_name='skip_blank_line',
        input_lines=['', '1,2'],
        column_names=['int_feature1', 'int_feature2'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1]], pa.large_list(pa.int64())),
                pa.array([[2]], pa.large_list(pa.int64())),
            ], ['int_feature1', 'int_feature2'])
        ]),
    dict(
        testcase_name='consider_blank_line',
        input_lines=['', '1,2.0'],
        column_names=['int_feature', 'float_feature'],
        skip_blank_lines=False,
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([None, [1]], pa.large_list(pa.int64())),
                pa.array([None, [2.0]], pa.large_list(pa.float32())),
            ], ['int_feature', 'float_feature'])
        ]),
    dict(
        testcase_name='skip_blank_line_single_column',
        input_lines=['', '1'],
        column_names=['int_feature'],
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1]], pa.large_list(pa.int64())),
            ], ['int_feature'])
        ]),
    dict(
        testcase_name='consider_blank_line_single_column',
        input_lines=['', '1'],
        column_names=['int_feature'],
        skip_blank_lines=False,
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([None, [1]], pa.large_list(pa.int64())),
            ], ['int_feature'])
        ]),
    dict(
        testcase_name='empty_csv',
        input_lines=[],
        column_names=[],
        expected_result=[]),
    dict(
        testcase_name='size_2_vector_int_multivalent',
        input_lines=['12|14'],
        column_names=['int_feature'],
        multivalent_columns=['int_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays(
                [pa.array([[12, 14]], pa.large_list(pa.int64()))],
                ['int_feature'])
        ]),
    dict(
        testcase_name='multivalent_schema',
        input_lines=['1|2.3,test'],
        column_names=['multivalent_feature', 'test_feature'],
        schema=text_format.Parse(
            """
        feature { name: "multivalent_feature" type: FLOAT }
        feature { name: "test_feature" type: BYTES }""", schema_pb2.Schema()),
        multivalent_columns=['multivalent_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1, 2.3]], pa.large_list(pa.float32())),
                pa.array([[b'test']], pa.large_list(pa.large_binary()))
            ], ['multivalent_feature', 'test_feature'])
        ]),
    dict(
        testcase_name='empty_multivalent_column',
        input_lines=['|,test'],
        column_names=['empty_feature', 'test_feature'],
        multivalent_columns=['empty_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([None], pa.null()),
                pa.array([[b'test']], pa.large_list(pa.large_binary()))
            ], ['empty_feature', 'test_feature'])
        ]),
    dict(
        testcase_name='empty_string_multivalent_column',
        input_lines=['|,test', 'a|b,test'],
        column_names=['string_feature', 'test_feature'],
        multivalent_columns=['string_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'', b''], [b'a', b'b']],
                         pa.large_list(pa.large_binary())),
                pa.array([[b'test'], [b'test']], pa.large_list(
                    pa.large_binary()))
            ], ['string_feature', 'test_feature'])
        ]),
    dict(
        testcase_name='int_and_float_multivalent_column',
        input_lines=['1|2.3,test'],
        column_names=['float_feature', 'test_feature'],
        multivalent_columns=['float_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[1, 2.3]], pa.large_list(pa.float32())),
                pa.array([[b'test']], pa.large_list(pa.large_binary()))
            ], ['float_feature', 'test_feature'])
        ]),
    dict(
        testcase_name='float_and_string_multivalent_column',
        input_lines=['2.3|abc,test'],
        column_names=['string_feature', 'test_feature'],
        multivalent_columns=['string_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'2.3', b'abc']], pa.large_list(pa.large_binary())),
                pa.array([[b'test']], pa.large_list(pa.large_binary()))
            ], ['string_feature', 'test_feature'])
        ]),
    dict(
        testcase_name='int_and_string_multivalent_column_multiple_lines',
        input_lines=['1|abc,test', '2|2,test'],
        column_names=['string_feature', 'test_feature'],
        multivalent_columns=['string_feature'],
        secondary_delimiter='|',
        expected_result=[
            pa.RecordBatch.from_arrays([
                pa.array([[b'1', b'abc'], [b'2', b'2']],
                         pa.large_list(pa.large_binary())),
                pa.array([[b'test'], [b'test']], pa.large_list(
                    pa.large_binary()))
            ], ['string_feature', 'test_feature'])
        ])
]


class CSVDecoderTest(parameterized.TestCase):
  """Tests for CSV decoder."""

  @parameterized.named_parameters(_TEST_CASES)
  def test_csv_decoder(self,
                       input_lines,
                       expected_result,
                       column_names,
                       delimiter=',',
                       skip_blank_lines=True,
                       schema=None,
                       multivalent_columns=None,
                       secondary_delimiter=None):
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(
              column_names=column_names,
              delimiter=delimiter,
              skip_blank_lines=skip_blank_lines,
              schema=schema,
              multivalent_columns=multivalent_columns,
              secondary_delimiter=secondary_delimiter))
      util.assert_that(
          result,
          test_util.make_arrow_record_batches_equal_fn(self, expected_result))

  def test_csv_decoder_invalid_row(self):
    input_lines = ['1,2.0,hello', '5,12.34']
    column_names = ['int_feature', 'float_feature', 'str_feature']

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, '.*Columns do not match specified csv headers.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | csv_decoder.DecodeCSV(column_names=column_names))
        util.assert_that(
            result, test_util.make_arrow_record_batches_equal_fn(self, None))


if __name__ == '__main__':
  absltest.main()
