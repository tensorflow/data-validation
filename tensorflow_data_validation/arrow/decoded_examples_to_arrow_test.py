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
# limitations under the License

"""Tests for tensorflow_data_validation.arrow.decoded_examples_to_arrow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import six
from tensorflow_data_validation.arrow import decoded_examples_to_arrow
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa


_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="not_a_list",
        test_input=1,
        expected_error=TypeError,
        expected_error_regexp="Expected a list"),
    dict(
        testcase_name="list_of_non_dict",
        test_input=[1, 2],
        expected_error=RuntimeError,
        expected_error_regexp="Expected a dict"),
    dict(
        testcase_name="list_of_dict_of_non_bytes_key",
        test_input=[{
            1: None
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Feature names must be either bytes or unicode."),
    dict(
        testcase_name="list_of_dict_of_non_ndarray_value",
        test_input=[{
            b"a": [1]
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Expected a numpy ndarray"),
    dict(
        testcase_name="unsupported_ndarray_type",
        test_input=[{
            b"a": np.array([1j, 2j, 3j], dtype=np.complex64)
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Unsupported numpy type"),
    dict(
        testcase_name="different_ndarray_types_for_feature",
        test_input=[{
            b"a": np.array([1, 2, 3], dtype=np.int64)
        }, {
            b"a": np.array([1., 2., 3.], dtype=np.float)
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Mismatch feature numpy array types"),
    dict(
        testcase_name="ndarray_of_object_no_bytes",
        test_input=[{
            b"a": np.array([1, 2, 3], dtype=np.object)
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Expected a string or bytes object"),
]

_CONVERSION_TEST_CASES = [
    dict(
        testcase_name="unicode_feature_name",
        input_examples=[{
            u"\U0001f951": np.array([1, 2, 3], dtype=np.int64),
        }],
        expected_output={
            u"\U0001f951": pa.array([[1, 2, 3]], type=pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="supported_ndarray_types",
        input_examples=[
            {
                b"int64_feature": np.array([1, 2, 3], dtype=np.int64),
                b"uint64_feature": np.array([1, 2, 3], dtype=np.uint64),
                b"int32_feature": np.array([1, 2, 3], dtype=np.int32),
                b"uint32_feature": np.array([1, 2, 3], dtype=np.uint32),
                b"float_feature": np.array([1.], dtype=np.float32),
                b"double_feature": np.array([1.], dtype=np.float64),
                b"bytes_feature": np.array([b"abc", b"def"], dtype=np.object),
                b"unicode_feature": np.array([u"abc", u"def"], dtype=np.object),
            },
            {
                b"int64_feature": np.array([4], dtype=np.int64),
                b"int32_feature": np.array([4], dtype=np.int32),
                b"float_feature": np.array([2., 3., 4.], dtype=np.float32),
                b"double_feature": np.array([2., 3., 4.], dtype=np.float64),
                b"bytes_feature": np.array([b"ghi"], dtype=np.object),
                b"unicode_feature": np.array([u"ghi"], dtype=np.object),
            },
        ],
        expected_output={
            "int64_feature":
                pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int64())),
            "uint64_feature":
                pa.array([[1, 2, 3], None], type=pa.list_(pa.uint64())),
            "int32_feature":
                pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int32())),
            "uint32_feature":
                pa.array([[1, 2, 3], None], type=pa.list_(pa.uint32())),
            "float_feature":
                pa.array([[1.], [2., 3., 4.]], type=pa.list_(pa.float32())),
            "double_feature":
                pa.array([[1.], [2., 3., 4.]], type=pa.list_(pa.float64())),
            "bytes_feature":
                pa.array([[b"abc", b"def"], [b"ghi"]],
                         type=pa.list_(pa.binary())),
            # Note that unicode feature values are encoded in utf-8 and stored
            # in a BinaryArray.
            "unicode_feature":
                pa.array([[b"abc", b"def"], [b"ghi"]],
                         type=pa.list_(pa.binary())),
        }),
    dict(
        testcase_name="mixed_unicode_and_bytes",
        input_examples=[
            {
                b"a": np.array([b"abc"], dtype=np.object),
            },
            {
                b"a": np.array([u"def"], dtype=np.object),
            },
        ],
        expected_output={
            "a":
                pa.array([[b"abc"], [b"def"]], type=pa.list_(pa.binary()))
        }),
    dict(
        testcase_name="none_feature_value",
        input_examples=[{
            b"a": np.array([1, 2, 3], dtype=np.int64),
        }, {
            b"a": None,
        }, {
            b"a": None,
        }, {
            b"a": np.array([4], dtype=np.int64),
        }],
        expected_output={
            "a":
                pa.array([[1, 2, 3], None, None, [4]],
                         type=pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="empty_feature_value",
        input_examples=[{
            b"a": np.array([], dtype=np.int64),
        }],
        expected_output={
            "a": pa.array([[]], type=pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="missing_feature",
        input_examples=[{
            b"f1": np.array([1, 2, 3], dtype=np.int64),
        }, {
            b"f2": np.array([1., 2., 3.], dtype=np.float32),
        }, {
            b"f3": np.array([b"abc", b"def"], dtype=np.object),
        }, {
            b"f1": np.array([4, 5, 6], dtype=np.int64),
            b"f4": np.array([8], dtype=np.int64),
        }],
        expected_output={
            "f1":
                pa.array([[1, 2, 3], None, None, [4, 5, 6]],
                         pa.list_(pa.int64())),
            "f2":
                pa.array([None, [1., 2., 3.], None, None],
                         pa.list_(pa.float32())),
            "f3":
                pa.array([None, None, [b"abc", b"def"], None],
                         pa.list_(pa.binary())),
            "f4":
                pa.array([None, None, None, [8]], pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="null_array",
        input_examples=[{
            b"a": None,
        }, {
            b"a": None,
        }],
        expected_output={
            "a": pa.array([None, None], type=pa.null()),
        })
]


class DecodedExamplesToArrowTest(parameterized.TestCase):

  @parameterized.named_parameters(*_INVALID_INPUT_TEST_CASES)
  def test_invalid_input(self, test_input, expected_error,
                         expected_error_regexp):
    with self.assertRaisesRegexp(expected_error, expected_error_regexp):
      decoded_examples_to_arrow.DecodedExamplesToTable(test_input)

  @parameterized.named_parameters(*_CONVERSION_TEST_CASES)
  def test_conversion(self, input_examples, expected_output):
    table = decoded_examples_to_arrow.DecodedExamplesToTable(input_examples)
    self.assertLen(expected_output, table.num_columns)
    for feature_name, expected_arrow_array in six.iteritems(expected_output):
      self.assertLen(table.column(feature_name).data.chunks, 1)
      self.assertTrue(
          expected_arrow_array.equals(table.column(feature_name).data.chunk(0)))

  def test_conversion_empty_input(self):
    table = decoded_examples_to_arrow.DecodedExamplesToTable([])
    self.assertEqual(table.num_columns, 0)
    self.assertEqual(table.num_rows, 0)

  def test_conversion_empty_examples(self):
    input_examples = [{}] * 10
    table = decoded_examples_to_arrow.DecodedExamplesToTable(input_examples)
    self.assertEqual(table.num_rows, 10)
    self.assertEqual(table.num_columns, 0)


if __name__ == "__main__":
  absltest.main()
