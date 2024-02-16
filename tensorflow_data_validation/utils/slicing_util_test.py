# Copyright 2019 Google LLC
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
"""Tests for the slicing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation import constants
from tensorflow_data_validation.utils import slicing_util
from tfx_bsl.public.proto import slicing_spec_pb2

from google.protobuf import text_format


class SlicingUtilTest(absltest.TestCase):

  # This should be simply self.assertCountEqual(), but
  # RecordBatch.__eq__ is not implemented.
  # TODO(zhuo): clean-up after ARROW-8277 is available.
  def _check_results(self, got, expected):
    got_dict = {g[0]: g[1] for g in got}
    expected_dict = {e[0]: e[1] for e in expected}

    self.assertCountEqual(got_dict.keys(), expected_dict.keys())
    for k, got_record_batch in got_dict.items():
      expected_record_batch = expected_dict[k]
      self.assertTrue(got_record_batch.equals(expected_record_batch))

  def test_get_feature_value_slicer(self):
    features = {'a': None, 'b': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
        pa.array([['dog'], ['cat'], ['wolf'], ['dog', 'wolf'], ['wolf']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_1_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[1], [2, 1, 1]]), pa.array([['dog'], ['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_1_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])], ['a', 'b'])
        ),
        (u'a_1_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_2_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_3_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[3], [3]]), pa.array([['wolf'], ['wolf']])],
             ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_one_feature_not_in_batch(self):
    features = {'not_an_actual_feature': None, 'a': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_1',
         pa.RecordBatch.from_arrays(
             [pa.array([[1], [2, 1]]),
              pa.array([['dog'], ['cat']])], ['a', 'b'])),
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_single_feature(self):
    features = {'a': [2]}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_no_slice(self):
    features = {'a': [3]}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = []
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_feature_not_in_record_batch(self):
    features = {'c': [0]}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = []
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_feature_not_in_record_batch_all_values(
      self):
    features = {'c': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = []
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_bytes_feature_valid_utf8(self):
    features = {'b': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([[b'dog'], [b'cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[1]]), pa.array([[b'dog']])], ['a', 'b'])
        ),
        (u'b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([[b'cat']])], ['a', 'b'])
        ),
    ]
    self._check_results(
        slicing_util.get_feature_value_slicer(features)(input_record_batch),
        expected_result)

  def test_get_feature_value_slicer_non_utf8_slice_key(self):
    features = {'a': None}
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[b'\xF0'], ['cat']]),
    ], ['a'])
    with self.assertRaisesRegex(ValueError, 'must be valid UTF-8'):
      _ = list(
          slicing_util.get_feature_value_slicer(features)(input_record_batch))

  def test_convert_slicing_config_to_fns_and_sqls(self):
    slicing_config = text_format.Parse(
        """
        slicing_specs {
          slice_keys_sql: "SELECT STRUCT(education) FROM example.education"
        }
        """, slicing_spec_pb2.SlicingConfig())

    slicing_fns, slicing_sqls = (
        slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
            slicing_config))
    self.assertEqual(slicing_fns, [])
    self.assertEqual(slicing_sqls,
                     ['SELECT STRUCT(education) FROM example.education'])

    slicing_config = text_format.Parse(
        """
        slicing_specs {}
        slicing_specs {
          feature_keys: ["country"]
        }
        slicing_specs {
          feature_keys: ["state"]
          feature_values: [{key: "age", value: "20"}]
        }
        """, slicing_spec_pb2.SlicingConfig())

    slicing_fns, slicing_sqls = (
        slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
            slicing_config))
    self.assertLen(slicing_fns, 2)
    self.assertEqual(slicing_sqls, [])

    slicing_config = text_format.Parse(
        """
        slicing_specs {
          feature_values: [{key: "a", value: "2"}]
        }
        """, slicing_spec_pb2.SlicingConfig())
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([['1'], ['2', '1']]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([['2', '1']]), pa.array([['cat']])], ['a', 'b'])),
    ]
    slicing_fns, slicing_sqls = (
        slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
            slicing_config))
    self._check_results(slicing_fns[0](input_record_batch), expected_result)

  def test_convert_slicing_config_to_fns_and_sqls_on_int_field(self):
    slicing_config = text_format.Parse(
        """
        slicing_specs {
          feature_values: [{key: "a", value: "2"}]
        }
        """, slicing_spec_pb2.SlicingConfig())
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])
    expected_result = [
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]),
              pa.array([['cat']])], ['a', 'b'])),
    ]
    slicing_fns, _ = (
        slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
            slicing_config))
    self._check_results(slicing_fns[0](input_record_batch), expected_result)

  def test_convert_slicing_config_to_fns_and_sqls_on_int_invalid(self):
    slicing_config = text_format.Parse(
        """
        slicing_specs {
          feature_values: [{key: "a", value: "2.5"}]
        }
        """, slicing_spec_pb2.SlicingConfig())
    input_record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2, 1]]),
        pa.array([['dog'], ['cat']]),
    ], ['a', 'b'])

    expected_result = [
        (u'a_2',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])),
    ]
    slicing_fns, _ = (
        slicing_util.convert_slicing_config_to_slice_functions_and_sqls(
            slicing_config))

    with self.assertRaisesRegex(
        ValueError, 'The feature to slice on has integer values but*'):
      self._check_results(slicing_fns[0](input_record_batch), expected_result)

  def test_generate_slices_sql(self):
    input_record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
            pa.array([['dog'], ['cat'], ['wolf'], ['dog', 'wolf'], ['wolf']]),
        ], ['a', 'b']),
        pa.RecordBatch.from_arrays(
            [pa.array([[1]]),
             pa.array([['dog']]),
             pa.array([[1]])], ['a', 'b', 'c']),
        pa.RecordBatch.from_arrays(
            [pa.array([[1]]),
             pa.array([['cat']]),
             pa.array([[1]])], ['a', 'b', 'd']),
        pa.RecordBatch.from_arrays(
            [pa.array([[1]]),
             pa.array([['cat']]),
             pa.array([[1]])], ['a', 'b', 'e']),
        pa.RecordBatch.from_arrays(
            [pa.array([[1]]),
             pa.array([['cat']]),
             pa.array([[1]])], ['a', 'b', 'f']),
    ]
    record_batch_with_metadata = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([['cat']])], ['a', 'b'])
    record_batch_with_metadata = pa.RecordBatch.from_arrays(
        arrays=record_batch_with_metadata.columns,
        schema=record_batch_with_metadata.schema.with_metadata({b'foo': 'bar'}))
    input_record_batches.append(record_batch_with_metadata)
    slice_sql = """
        SELECT
          STRUCT(a, b)
        FROM
          example.a, example.b
    """

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(input_record_batches, reshuffle=False)
          | 'GenerateSlicesSql' >> beam.ParDo(
              slicing_util.GenerateSlicesSqlDoFn(slice_sqls=[slice_sql])))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 18)
          expected_slice_keys = ([
              u'a_1_b_dog', u'a_1_b_cat', u'a_2_b_cat', u'a_2_b_dog',
              u'a_1_b_wolf', u'a_2_b_wolf', u'a_3_b_wolf', u'a_1_b_dog',
              u'a_1_b_cat', u'a_1_b_cat', u'a_1_b_cat', u'a_1_b_cat'] +
                                 [constants.DEFAULT_SLICE_KEY] * 6)
          actual_slice_keys = [slice_key for (slice_key, _) in got]
          self.assertCountEqual(expected_slice_keys, actual_slice_keys)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def test_generate_slices_sql_assert_record_batches(self):
    input_record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
            pa.array([['dog'], ['cat'], ['wolf'], ['dog', 'wolf'], ['wolf']]),
        ], ['a', 'b']),
    ]
    slice_sql = """
        SELECT
          STRUCT(a, b)
        FROM
          example.a, example.b
    """
    expected_result = [
        (u'a_1_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[1], [2, 1, 1]]), pa.array([['dog'], ['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_1_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_cat',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1]]), pa.array([['cat']])], ['a', 'b'])
        ),
        (u'a_2_b_dog',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])], ['a', 'b'])
        ),
        (u'a_1_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_2_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[2, 1, 1]]), pa.array([['dog', 'wolf']])],
             ['a', 'b'])
        ),
        (u'a_3_b_wolf',
         pa.RecordBatch.from_arrays(
             [pa.array([[3], [3]]), pa.array([['wolf'], ['wolf']])],
             ['a', 'b'])
        ),
        (constants.DEFAULT_SLICE_KEY, input_record_batches[0]),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(input_record_batches, reshuffle=False)
          | 'GenerateSlicesSql' >> beam.ParDo(
              slicing_util.GenerateSlicesSqlDoFn(slice_sqls=[slice_sql])))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self._check_results(got, expected_result)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def test_generate_slices_sql_invalid_slice(self):
    input_record_batches = [
        pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
                pa.array(
                    [[], [], [], [], []]
                ),
            ],
            ['a', 'b'],
        ),
    ]
    slice_sql1 = """
        SELECT
          STRUCT(a, b)
        FROM
          example.a, example.b
    """

    expected_result = [
        (constants.INVALID_SLICE_KEY, input_record_batches[0]),
        (constants.DEFAULT_SLICE_KEY, input_record_batches[0]),
    ]
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(input_record_batches, reshuffle=False)
          | 'GenerateSlicesSql'
          >> beam.ParDo(
              slicing_util.GenerateSlicesSqlDoFn(slice_sqls=[slice_sql1])
          )
      )

      def check_result(got):
        try:
          self._check_results(got, expected_result)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def test_generate_slices_sql_multiple_queries(self):
    input_record_batches = [
        pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
                pa.array(
                    [[], [], [], [], []]
                ),
            ],
            ['a', 'b'],
        ),
    ]
    slice_sql1 = """
        SELECT
          STRUCT(c)
        FROM
          example.a, example.b
    """

    slice_sql2 = """
        SELECT
          STRUCT(a)
        FROM
          example.a
    """

    expected_result = [
        (
            'a_1',
            pa.RecordBatch.from_arrays(
                [
                    pa.array([[1], [2, 1], [2, 1, 1]]),
                    pa.array([[], [], []]),
                ],
                ['a', 'b'],
            ),
        ),
        (
            'a_2',
            pa.RecordBatch.from_arrays(
                [
                    pa.array([[2, 1], [2, 1, 1]]),
                    pa.array([[], []]),
                ],
                ['a', 'b'],
            ),
        ),
        (
            'a_3',
            pa.RecordBatch.from_arrays(
                [
                    pa.array([[3], [3]]),
                    pa.array([[], []]),
                ],
                ['a', 'b'],
            ),
        ),
        (constants.INVALID_SLICE_KEY, input_record_batches[0]),
        (constants.DEFAULT_SLICE_KEY, input_record_batches[0]),
    ]
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(input_record_batches, reshuffle=False)
          | 'GenerateSlicesSql'
          >> beam.ParDo(
              slicing_util.GenerateSlicesSqlDoFn(
                  slice_sqls=[slice_sql1,
                              slice_sql2]
              )
          )
      )

      def check_result(got):
        try:
          self._check_results(got, expected_result)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)


if __name__ == '__main__':
  absltest.main()
