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
import pyarrow as pa
import six
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.arrow import decoded_examples_to_arrow


_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="list_of_non_dict",
        test_input=[1, 2],
        expected_error=ValueError,
        expected_error_regexp="Unexpected Arrow type created from input"),
    dict(
        testcase_name="list_of_dict_of_non_str_key",
        test_input=[{
            1: None
        }],
        expected_error=pa.ArrowTypeError,
        expected_error_regexp="Expected dict key of type str or bytes"),
    dict(
        testcase_name="unsupported_ndarray_type",
        test_input=[{
            "a": np.array([1j, 2j, 3j], dtype=np.complex64)
        }],
        expected_error=RuntimeError,
        expected_error_regexp="Unsupported numpy type"),
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
                "int64_feature": np.array([1, 2, 3], dtype=np.int64),
                "uint64_feature": np.array([1, 2, 3], dtype=np.uint64),
                "int32_feature": np.array([1, 2, 3], dtype=np.int32),
                "uint32_feature": np.array([1, 2, 3], dtype=np.uint32),
                "float_feature": np.array([1.], dtype=np.float32),
                "double_feature": np.array([1.], dtype=np.float64),
                "bytes_feature": np.array([b"abc", b"def"], dtype=object),
                "unicode_feature": np.array([u"abc", u"def"], dtype=object),
            },
            {
                "int64_feature": np.array([4], dtype=np.int64),
                "int32_feature": np.array([4], dtype=np.int32),
                "float_feature": np.array([2., 3., 4.], dtype=np.float32),
                "double_feature": np.array([2., 3., 4.], dtype=np.float64),
                "bytes_feature": np.array([b"ghi"], dtype=object),
                "unicode_feature": np.array([u"ghi"], dtype=object),
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
            "unicode_feature":
                pa.array([[b"abc", b"def"], [b"ghi"]],
                         type=pa.list_(pa.string())),
        }),
    dict(
        testcase_name="mixed_unicode_and_bytes",
        input_examples=[
            {
                "a": np.array([b"abc"], dtype=object),
            },
            {
                "a": np.array([u"def"], dtype=object),
            },
        ],
        expected_output={
            "a":
                pa.array([[b"abc"], [b"def"]], type=pa.list_(pa.binary()))
        }),
    dict(
        testcase_name="none_feature_value",
        input_examples=[{
            "a": np.array([1, 2, 3], dtype=np.int64),
        }, {
            "a": None,
        }, {
            "a": None,
        }, {
            "a": np.array([4], dtype=np.int64),
        }],
        expected_output={
            "a":
                pa.array([[1, 2, 3], None, None, [4]],
                         type=pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="empty_feature_value",
        input_examples=[{
            "a": np.array([], dtype=np.int64),
        }],
        expected_output={
            "a": pa.array([[]], type=pa.list_(pa.int64())),
        }),
    dict(
        testcase_name="missing_feature",
        input_examples=[{
            "f1": np.array([1, 2, 3], dtype=np.int64),
        }, {
            "f2": np.array([1., 2., 3.], dtype=np.float32),
        }, {
            "f3": np.array([b"abc", b"def"], dtype=object),
        }, {
            "f1": np.array([4, 5, 6], dtype=np.int64),
            "f4": np.array([8], dtype=np.int64),
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
            "a": None,
        }, {
            "a": None,
        }],
        expected_output={
            "a": pa.array([None, None], type=pa.null()),
        })
]


class DecodedExamplesToArrowPyTest(parameterized.TestCase):

  @parameterized.named_parameters(*_INVALID_INPUT_TEST_CASES)
  def test_invalid_input(self, test_input, expected_error,
                         expected_error_regexp):
    with self.assertRaisesRegex(expected_error, expected_error_regexp):
      decoded_examples_to_arrow.DecodedExamplesToRecordBatch(test_input)

  @parameterized.named_parameters(*_CONVERSION_TEST_CASES)
  def test_conversion(self, input_examples, expected_output):
    record_batch = decoded_examples_to_arrow.DecodedExamplesToRecordBatch(
        input_examples)
    self.assertLen(expected_output, record_batch.num_columns)
    for feature_name, expected_arrow_array in six.iteritems(expected_output):
      actual = arrow_util.get_column(record_batch, feature_name)
      self.assertTrue(
          expected_arrow_array.equals(actual),
          "{} vs {}".format(expected_arrow_array, actual))

  def test_conversion_empty_input(self):
    record_batch = decoded_examples_to_arrow.DecodedExamplesToRecordBatch([])
    self.assertEqual(record_batch.num_columns, 0)
    self.assertEqual(record_batch.num_rows, 0)

  def test_conversion_empty_examples(self):
    input_examples = [{}] * 10
    record_batch = decoded_examples_to_arrow.DecodedExamplesToRecordBatch(
        input_examples)
    self.assertEqual(record_batch.num_rows, 10)
    self.assertEqual(record_batch.num_columns, 0)


if __name__ == "__main__":
  absltest.main()
