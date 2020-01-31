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
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation.coders import csv_decoder
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class CSVDecoderTest(absltest.TestCase):
  """Tests for CSV decoder."""

  def test_csv_decoder(self):
    input_lines = ['1,2.0,hello',
                   '5,12.34,world']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], [5]], pa.list_(pa.int64())),
            pa.array([[2.0], [12.34]], pa.list_(pa.float32())),
            pa.array([[b'hello'], [b'world']], pa.list_(pa.binary())),
        ], ['int_feature', 'float_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_schema(self):
    input_lines = ['1,1,2.0,hello',
                   '5,5,12.34,world']
    column_names = ['int_feature_parsed_as_float', 'int_feature',
                    'float_feature', 'str_feature']
    schema = text_format.Parse(
        """
        feature { name: "int_feature_parsed_as_float" type: FLOAT }
        feature { name: "int_feature" type: INT }
        feature { name: "float_feature" type: FLOAT }
        feature { name: "str_feature" type: BYTES }
        """, schema_pb2.Schema())
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], [5]], pa.list_(pa.float32())),
            pa.array([[1], [5]], pa.list_(pa.int64())),
            pa.array([[2.0], [12.34]], pa.list_(pa.float32())),
            pa.array([[b'hello'], [b'world']], pa.list_(pa.binary())),
        ], ['int_feature_parsed_as_float', 'int_feature',
            'float_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False) | csv_decoder.DecodeCSV(
              column_names=column_names,
              schema=schema,
              infer_type_from_schema=True))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_missing_values(self):
    input_lines = ['1,,hello',
                   ',12.34,']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], None], pa.list_(pa.int64())),
            pa.array([None, [12.34]], pa.list_(pa.float32())),
            pa.array([[b'hello'], None], pa.list_(pa.binary())),
        ], ['int_feature', 'float_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_int_and_float_in_same_column(self):
    input_lines = ['2,1.5',
                   '1.5,2']
    column_names = ['float_feature1', 'float_feature2']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[2.0], [1.5]], pa.list_(pa.float32())),
            pa.array([[1.5], [2.0]], pa.list_(pa.float32())),
        ], ['float_feature1', 'float_feature2'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_int_and_string_in_same_column(self):
    input_lines = ['2,abc',
                   'abc,2']
    column_names = ['str_feature1', 'str_feature2']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[b'2'], [b'abc']], pa.list_(pa.binary())),
            pa.array([[b'abc'], [b'2']], pa.list_(pa.binary())),
        ], ['str_feature1', 'str_feature2'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_float_and_string_in_same_column(self):
    input_lines = ['2.3,abc',
                   'abc,2.3']
    column_names = ['str_feature1', 'str_feature2']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[b'2.3'], [b'abc']], pa.list_(pa.binary())),
            pa.array([[b'abc'], [b'2.3']], pa.list_(pa.binary())),
        ], ['str_feature1', 'str_feature2'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_unicode(self):
    input_lines = [u'1,שקרכלשהו,22.34,text field']
    column_names = ['int_feature', 'unicode_feature',
                    'float_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1]], pa.list_(pa.int64())),
            pa.array([[22.34]], pa.list_(pa.float32())),
            pa.array([[u'שקרכלשהו'.encode('utf-8')]], pa.list_(pa.binary())),
            pa.array([[b'text field']], pa.list_(pa.binary())),
        ], ['int_feature', 'float_feature', 'unicode_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_csv_record_with_quotes(self):
    input_lines = ['1,"ab,cd,ef"',
                   '5,"wx,xy,yz"']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], [5]], pa.list_(pa.int64())),
            pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']], pa.list_(pa.binary())),
        ], ['int_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_space_delimiter(self):
    input_lines = ['1 "ab,cd,ef"',
                   '5 "wx,xy,yz"']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], [5]], pa.list_(pa.int64())),
            pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']], pa.list_(pa.binary())),
        ], ['int_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names, delimiter=' '))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_with_tab_delimiter(self):
    input_lines = ['1\t"this is a \ttext"',
                   '5\t']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1], [5]], pa.list_(pa.int64())),
            pa.array([[b'this is a \ttext'], None], pa.list_(pa.binary())),
        ], ['int_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names, delimiter='\t'))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_negative_values(self):
    input_lines = ['-34', '45']
    column_names = ['feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[-34], [45]], pa.list_(pa.int64())),
        ], ['feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_int64_max(self):
    input_lines = ['34', str(sys.maxsize)]
    column_names = ['feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[34], [sys.maxsize]], pa.list_(pa.int64())),
        ], ['feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_pos(self):
    input_lines = ['34', str(sys.maxsize+1)]
    column_names = ['feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[b'34'], [str(sys.maxsize + 1).encode('utf-8')]],
                     pa.list_(pa.binary())),
        ], ['feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_neg(self):
    input_lines = ['34', str(-(sys.maxsize+2))]
    column_names = ['feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[b'34'], [str(-(sys.maxsize + 2)).encode('utf-8')]],
                     pa.list_(pa.binary())),
        ], ['feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_pos_and_neg(self):
    input_lines = [str(sys.maxsize+1), str(-(sys.maxsize+2))]
    column_names = ['feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[str(sys.maxsize + 1).encode('utf-8')],
                      [str(-(sys.maxsize + 2)).encode('utf-8')]],
                     pa.list_(pa.binary())),
        ], ['feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_empty_row(self):
    input_lines = [',,',
                   '1,2.0,hello']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([None, [1]], pa.list_(pa.int64())),
            pa.array([None, [2.0]], pa.list_(pa.float32())),
            pa.array([None, [b'hello']], pa.list_(pa.binary())),
        ], ['int_feature', 'float_feature', 'str_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_skip_blank_line(self):
    input_lines = ['',
                   '1,2']
    column_names = ['int_feature1', 'int_feature2']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1]], pa.list_(pa.int64())),
            pa.array([[2]], pa.list_(pa.int64())),
        ], ['int_feature1', 'int_feature2'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_consider_blank_line(self):
    input_lines = ['',
                   '1,2.0']
    column_names = ['int_feature', 'float_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([None, [1]], pa.list_(pa.int64())),
            pa.array([None, [2.0]], pa.list_(pa.float32())),
        ], ['int_feature', 'float_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False) | csv_decoder.DecodeCSV(
              column_names=column_names, skip_blank_lines=False))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_skip_blank_line_single_column(self):
    input_lines = ['',
                   '1']
    column_names = ['int_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([[1]], pa.list_(pa.int64())),
        ], ['int_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_consider_blank_line_single_column(self):
    input_lines = ['',
                   '1']
    column_names = ['int_feature']
    expected_result = [
        pa.Table.from_arrays([
            pa.array([None, [1]], pa.list_(pa.int64())),
        ], ['int_feature'])
    ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False) | csv_decoder.DecodeCSV(
              column_names=column_names, skip_blank_lines=False))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_empty_csv(self):
    input_lines = []
    expected_result = []

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(input_lines, reshuffle=False)
          | csv_decoder.DecodeCSV(column_names=[]))
      util.assert_that(
          result,
          test_util.make_arrow_tables_equal_fn(self, expected_result))

  def test_csv_decoder_invalid_row(self):
    input_lines = ['1,2.0,hello',
                   '5,12.34']
    column_names = ['int_feature', 'float_feature', 'str_feature']

    with self.assertRaisesRegexp(
        ValueError, '.*Columns do not match specified csv headers.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | csv_decoder.DecodeCSV(column_names=column_names))
        util.assert_that(
            result,
            test_util.make_arrow_tables_equal_fn(self, None))


if __name__ == '__main__':
  absltest.main()
