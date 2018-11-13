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
import numpy as np
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
        {'int_feature': np.array([1], dtype=np.integer),
         'float_feature': np.array([2.0], dtype=np.floating),
         'str_feature': np.array(['hello'], dtype=np.object)},
        {'int_feature': np.array([5], dtype=np.integer),
         'float_feature': np.array([12.34], dtype=np.floating),
         'str_feature': np.array(['world'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

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
        {'int_feature_parsed_as_float': np.array([1], dtype=np.floating),
         'int_feature': np.array([1], dtype=np.integer),
         'float_feature': np.array([2.0], dtype=np.floating),
         'str_feature': np.array(['hello'], dtype=np.object)},
        {'int_feature_parsed_as_float': np.array([5], dtype=np.floating),
         'int_feature': np.array([5], dtype=np.integer),
         'float_feature': np.array([12.34], dtype=np.floating),
         'str_feature': np.array(['world'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names, schema=schema,
                                      infer_type_from_schema=True))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_missing_values(self):
    input_lines = ['1,,hello',
                   ',12.34,']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    expected_result = [
        {'int_feature': np.array([1.0], dtype=np.floating),
         'float_feature': None,
         'str_feature': np.array(['hello'], dtype=np.object)},
        {'int_feature': None,
         'float_feature': np.array([12.34], dtype=np.floating),
         'str_feature': None}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_int_and_float_in_same_column(self):
    input_lines = ['2,1.5',
                   '1.5,2']
    column_names = ['float_feature1', 'float_feature2']
    expected_result = [
        {'float_feature1': np.array([2.0], dtype=np.floating),
         'float_feature2': np.array([1.5], dtype=np.floating)},
        {'float_feature1': np.array([1.5], dtype=np.floating),
         'float_feature2': np.array([2.0], dtype=np.floating)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_int_and_string_in_same_column(self):
    input_lines = ['2,abc',
                   'abc,2']
    column_names = ['str_feature1', 'str_feature2']
    expected_result = [
        {'str_feature1': np.array(['2'], dtype=np.object),
         'str_feature2': np.array(['abc'], dtype=np.object)},
        {'str_feature1': np.array(['abc'], dtype=np.object),
         'str_feature2': np.array(['2'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_float_and_string_in_same_column(self):
    input_lines = ['2.3,abc',
                   'abc,2.3']
    column_names = ['str_feature1', 'str_feature2']
    expected_result = [
        {'str_feature1': np.array(['2.3'], dtype=np.object),
         'str_feature2': np.array(['abc'], dtype=np.object)},
        {'str_feature1': np.array(['abc'], dtype=np.object),
         'str_feature2': np.array(['2.3'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_unicode(self):
    input_lines = [u'1,שקרכלשהו,22.34,text field']
    column_names = ['int_feature', 'unicode_feature',
                    'float_feature', 'str_feature']
    expected_result = [
        {'int_feature': np.array([1], dtype=np.integer),
         'unicode_feature': np.array([u'שקרכלשהו'.encode('utf-8')],
                                     dtype=np.object),
         'float_feature': np.array([22.34], dtype=np.floating),
         'str_feature': np.array(['text field'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_csv_record_with_quotes(self):
    input_lines = ['1,"ab,cd,ef"',
                   '5,"wx,xy,yz"']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        {'int_feature': np.array([1], dtype=np.integer),
         'str_feature': np.array(['ab,cd,ef'], dtype=np.object)},
        {'int_feature': np.array([5], dtype=np.integer),
         'str_feature': np.array(['wx,xy,yz'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_space_delimiter(self):
    input_lines = ['1 "ab,cd,ef"',
                   '5 "wx,xy,yz"']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        {'int_feature': np.array([1], dtype=np.integer),
         'str_feature': np.array(['ab,cd,ef'], dtype=np.object)},
        {'int_feature': np.array([5], dtype=np.integer),
         'str_feature': np.array(['wx,xy,yz'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names,
                                      delimiter=' '))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_with_tab_delimiter(self):
    input_lines = ['1\t"this is a \ttext"',
                   '5\t']
    column_names = ['int_feature', 'str_feature']
    expected_result = [
        {'int_feature': np.array([1], dtype=np.integer),
         'str_feature': np.array(['this is a \ttext'], dtype=np.object)},
        {'int_feature': np.array([5], dtype=np.integer),
         'str_feature': None}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names,
                                      delimiter='\t'))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_negative_values(self):
    input_lines = ['-34', '45']
    column_names = ['feature']
    expected_result = [
        {'feature': np.array([-34], dtype=np.int64)},
        {'feature': np.array([45], dtype=np.int64)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_int64_max(self):
    input_lines = ['34', str(sys.maxsize)]
    column_names = ['feature']
    expected_result = [
        {'feature': np.array([34], dtype=np.int64)},
        {'feature': np.array([sys.maxsize], dtype=np.int64)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_pos(self):
    input_lines = ['34', str(sys.maxsize+1)]
    column_names = ['feature']
    expected_result = [
        {'feature': np.array(['34'], dtype=np.object)},
        {'feature': np.array([str(sys.maxsize+1)], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_neg(self):
    input_lines = ['34', str(-(sys.maxsize+2))]
    column_names = ['feature']
    expected_result = [
        {'feature': np.array(['34'], dtype=np.object)},
        {'feature': np.array([str(-(sys.maxsize+2))], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_large_int_categorical_pos_and_neg(self):
    input_lines = [str(sys.maxsize+1), str(-(sys.maxsize+2))]
    column_names = ['feature']
    expected_result = [
        {'feature': np.array([str(sys.maxsize+1)], dtype=np.object)},
        {'feature': np.array([str(-(sys.maxsize+2))], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_empty_row(self):
    input_lines = [',,',
                   '1,2.0,hello']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    expected_result = [
        {'int_feature': None,
         'float_feature': None,
         'str_feature': None},
        {'int_feature': np.array([1.0], dtype=np.floating),
         'float_feature': np.array([2.0], dtype=np.floating),
         'str_feature': np.array(['hello'], dtype=np.object)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_skip_blank_line(self):
    input_lines = ['',
                   '1,2']
    column_names = ['int_feature1', 'int_feature2']
    expected_result = [
        {'int_feature1': np.array([1], dtype=np.integer),
         'int_feature2': np.array([2], dtype=np.integer)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_consider_blank_line(self):
    input_lines = ['',
                   '1,2']
    column_names = ['float_feature1', 'float_feature2']
    expected_result = [
        {'float_feature1': None,
         'float_feature2': None},
        {'float_feature1': np.array([1.0], dtype=np.floating),
         'float_feature2': np.array([2.0], dtype=np.floating)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names,
                                      skip_blank_lines=False))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_skip_blank_line_single_column(self):
    input_lines = ['',
                   '1']
    column_names = ['int_feature']
    expected_result = [
        {'int_feature': np.array([1], dtype=np.integer)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_consider_blank_line_single_column(self):
    input_lines = ['',
                   '1']
    column_names = ['float_feature']
    expected_result = [
        {'float_feature': None},
        {'float_feature': np.array([1.0], dtype=np.floating)}]

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=column_names,
                                      skip_blank_lines=False))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_empty_csv(self):
    input_lines = []
    expected_result = []

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=[]))
      util.assert_that(
          result,
          test_util.make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_invalid_row(self):
    input_lines = ['1,2.0,hello',
                   '5,12.34']
    column_names = ['int_feature', 'float_feature', 'str_feature']

    with self.assertRaisesRegexp(
        ValueError, '.*Columns do not match specified csv headers.*'):
      with beam.Pipeline() as p:
        result = (p | beam.Create(input_lines) |
                  csv_decoder.DecodeCSV(column_names=column_names))
        util.assert_that(
            result,
            test_util.make_example_dict_equal_fn(self, None))


if __name__ == '__main__':
  absltest.main()
