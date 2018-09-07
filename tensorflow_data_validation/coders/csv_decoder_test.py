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

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.coders import csv_decoder


def _make_example_dict_equal_fn(
    test,
    expected
):
  """Makes a matcher function for comparing the example dict.

  Args:
    test: test case object.
    expected: the expected example dict.

  Returns:
    A matcher function for comparing the example dicts.
  """

  def _matcher(actual):
    """Matcher function for comparing the example dicts."""
    try:
      # Check number of examples.
      test.assertEqual(len(actual), len(expected))

      for i in range(len(actual)):
        for key in actual[i]:
          # Check each feature value.
          if isinstance(expected[i][key], np.ndarray):
            test.assertEqual(actual[i][key].dtype, expected[i][key].dtype)
            np.testing.assert_equal(actual[i][key], expected[i][key])
          else:
            test.assertEqual(actual[i][key], expected[i][key])

    except AssertionError, e:
      raise util.BeamAssertException('Failed assert: ' + str(e))

  return _matcher


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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

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
          _make_example_dict_equal_fn(self, expected_result))

  def test_csv_decoder_empty_csv(self):
    input_lines = []
    expected_result = []

    with beam.Pipeline() as p:
      result = (p | beam.Create(input_lines) |
                csv_decoder.DecodeCSV(column_names=[]))
      util.assert_that(
          result,
          _make_example_dict_equal_fn(self, expected_result))

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
            _make_example_dict_equal_fn(self, None))


if __name__ == '__main__':
  absltest.main()
