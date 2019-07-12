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
# limitations under the License
"""Tests for tensorflow_data_validation.arrow.merge."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_data_validation.arrow import merge
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa


class CompactTableTest(absltest.TestCase):

  def test_compact_table(self):
    b1 = pa.RecordBatch.from_arrays([
        pa.array([[1.0, 2.0], None, [3.0, 4.0, 5.0]]),
        pa.array([[1, 2, 3, 4], None, [5, 6]]),
    ], ["a", "b"])
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([[1.0], None]),
         pa.array([None, [7]])], ["a", "b"])
    table = pa.Table.from_batches([b1, b2])
    for c in table.columns:
      self.assertEqual(c.data.num_chunks, 2)
    compacted_table = merge.CompactTable(table)
    for c in compacted_table.columns:
      self.assertEqual(c.data.num_chunks, 1)
    self.assertTrue(table.equals(compacted_table))

  def test_compact_table_invalid_input(self):
    with self.assertRaisesRegexp(RuntimeError, "Could not unwrap"):
      merge.CompactTable(None)


_MERGE_TEST_CASES = [
    dict(
        testcase_name="basic_types",
        inputs=[
            {
                "bool": pa.array([False, None, True], type=pa.bool_()),
                "int64": pa.array([1, None, 3], type=pa.int64()),
                "uint64": pa.array([1, None, 3], type=pa.uint64()),
                "int32": pa.array([1, None, 3], type=pa.int32()),
                "uint32": pa.array([1, None, 3], type=pa.uint32()),
                "float": pa.array([1., None, 3.], type=pa.float32()),
                "double": pa.array([1., None, 3.], type=pa.float64()),
                "bytes": pa.array([b"abc", None, b"ghi"], type=pa.binary()),
                "unicode": pa.array([u"abc", None, u"ghi"], type=pa.utf8()),
            },
            {
                "bool": pa.array([None, False], type=pa.bool_()),
                "int64": pa.array([None, 4], type=pa.int64()),
                "uint64": pa.array([None, 4], type=pa.uint64()),
                "int32": pa.array([None, 4], type=pa.int32()),
                "uint32": pa.array([None, 4], type=pa.uint32()),
                "float": pa.array([None, 4.], type=pa.float32()),
                "double": pa.array([None, 4.], type=pa.float64()),
                "bytes": pa.array([None, b"jkl"], type=pa.binary()),
                "unicode": pa.array([None, u"jkl"], type=pa.utf8()),
            },
        ],
        expected_output={
            "bool":
                pa.array([False, None, True, None, False], type=pa.bool_()),
            "int64":
                pa.array([1, None, 3, None, 4], type=pa.int64()),
            "uint64":
                pa.array([1, None, 3, None, 4], type=pa.uint64()),
            "int32":
                pa.array([1, None, 3, None, 4], type=pa.int32()),
            "uint32":
                pa.array([1, None, 3, None, 4], type=pa.uint32()),
            "float":
                pa.array([1., None, 3., None, 4.], type=pa.float32()),
            "double":
                pa.array([1., None, 3., None, 4.], type=pa.float64()),
            "bytes":
                pa.array([b"abc", None, b"ghi", None, b"jkl"],
                         type=pa.binary()),
            "unicode":
                pa.array([u"abc", None, u"ghi", None, u"jkl"], type=pa.utf8()),
        }),
    dict(
        testcase_name="list",
        inputs=[
            {
                "list<int32>":
                    pa.array([[1, None, 3], None], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([None], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([[]], type=pa.list_(pa.int32())),
            },
        ],
        expected_output={
            "list<int32>":
                pa.array([[1, None, 3], None, None, []],
                         type=pa.list_(pa.int32()))
        }),
    dict(
        testcase_name="struct",
        inputs=[{
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"abc", None, b"def"]),
                    pa.array([[None], [1, 2], []], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }, {
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"ghi"]),
                    pa.array([[3]], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }],
        expected_output={
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"abc", None, b"def", b"ghi"]),
                    pa.array([[None], [1, 2], [], [3]],
                             type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }),
    dict(
        testcase_name="missing_or_null_column_fixed_width",
        inputs=[
            {
                "int32": pa.array([None, None], type=pa.null())
            },
            {
                "int64": pa.array([None, None], type=pa.null())
            },
            {
                "int64": pa.array([123], type=pa.int64())
            },
            {
                "int32": pa.array([456], type=pa.int32())
            },
        ],
        expected_output={
            "int32":
                pa.array([None, None, None, None, None, 456], type=pa.int32()),
            "int64":
                pa.array([None, None, None, None, 123, None], type=pa.int64()),
        }),
    dict(
        testcase_name="missing_or_null_column_list_alike",
        inputs=[
            {
                "list<int32>": pa.array([None, None], type=pa.null())
            },
            {
                "utf8": pa.array([None, None], type=pa.null())
            },
            {
                "utf8": pa.array([u"abc"], type=pa.utf8())
            },
            {
                "list<int32>":
                    pa.array([None, [123, 456]], type=pa.list_(pa.int32()))
            },
        ],
        expected_output={
            "list<int32>":
                pa.array([None, None, None, None, None, None, [123, 456]],
                         type=pa.list_(pa.int32())),
            "utf8":
                pa.array([None, None, None, None, u"abc", None, None],
                         type=pa.utf8()),
        }),
    dict(
        testcase_name="missing_or_null_column_struct",
        inputs=[{
            "struct<int32, list<int32>>": pa.array([None, None], type=pa.null())
        }, {
            "list<utf8>": pa.array([None, None], type=pa.null())
        }, {
            "struct<int32, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([1, 2, None], type=pa.int32()),
                    pa.array([[1], None, [3, 4]], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }, {
            "list<utf8>": pa.array([u"abc", None], type=pa.utf8())
        }],
        expected_output={
            "list<utf8>":
                pa.array(
                    [None, None, None, None, None, None, None, u"abc", None],
                    type=pa.utf8()),
            "struct<int32, list<int32>>":
                pa.array([
                    None, None, None, None, (1, [1]), (2, None),
                    (None, [3, 4]), None, None
                ],
                         type=pa.struct([
                             pa.field("f1", pa.int32()),
                             pa.field("f2", pa.list_(pa.int32()))
                         ])),
        }),
]

_MERGE_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="not_a_list_of_tables",
        inputs=[pa.Table.from_arrays([pa.array([1])], ["f1"]), 1],
        expected_error_regexp="Could not unwrap Table",
    ),
    dict(
        testcase_name="not_a_list",
        inputs=1,
        expected_error_regexp="expected a list",
    ),
    dict(
        testcase_name="empty_list",
        inputs=[],
        expected_error_regexp="expected a non-empty list"),
    dict(
        testcase_name="column_type_differs",
        inputs=[
            pa.Table.from_arrays([pa.array([1, 2, 3], type=pa.int32())],
                                 ["f1"]),
            pa.Table.from_arrays([pa.array([4, 5, 6], type=pa.int64())], ["f1"])
        ],
        expected_error_regexp="Trying to append a column of different type"),
]


class MergeTablesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_MERGE_INVALID_INPUT_TEST_CASES)
  def test_invalid_inputs(self, inputs, expected_error_regexp):
    with self.assertRaisesRegexp(Exception, expected_error_regexp):
      _ = merge.MergeTables(inputs)

  @parameterized.named_parameters(*_MERGE_TEST_CASES)
  def test_merge_tables(self, inputs, expected_output):
    input_tables = [
        pa.Table.from_arrays(list(in_dict.values()), list(in_dict.keys()))
        for in_dict in inputs
    ]
    merged = merge.MergeTables(input_tables)

    self.assertLen(expected_output, merged.num_columns)
    for c in merged.columns:
      self.assertEqual(c.data.num_chunks, 1)
      try:
        self.assertTrue(expected_output[c.name].equals(c.data.chunk(0)))
      except AssertionError:
        self.fail(msg="Column {}:\nexpected:{}\ngot: {}".format(
            c.name, expected_output[c.name], c))


_SLICE_TEST_CASES = [
    dict(
        testcase_name="no_index",
        row_indices=[],
        expected_output=pa.Table.from_arrays([
            pa.array([], type=pa.list_(pa.int32())),
            pa.array([], type=pa.list_(pa.binary()))
        ], ["f1", "f2"])),
    dict(
        testcase_name="one_index",
        row_indices=[1],
        expected_output=pa.Table.from_arrays([
            pa.array([None], type=pa.list_(pa.int32())),
            pa.array([["b", "c"]], type=pa.list_(pa.binary()))
        ], ["f1", "f2"])),
    dict(
        testcase_name="consecutive_first_row_included",
        row_indices=[0, 1, 2, 3],
        expected_output=pa.Table.from_arrays(
            [
                pa.array([[1, 2, 3], None, [4], []], type=pa.list_(pa.int32())),
                pa.array([["a"], ["b", "c"], None, []],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="consecutive_last_row_included",
        row_indices=[5, 6, 7, 8],
        expected_output=pa.Table.from_arrays(
            [
                pa.array([[7], [8, 9], [10], []], type=pa.list_(pa.int32())),
                pa.array([["d", "e"], ["f"], None, ["g"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="inconsecutive",
        row_indices=[1, 2, 3, 5],
        expected_output=pa.Table.from_arrays(
            [
                pa.array([None, [4], [], [7]], type=pa.list_(pa.int32())),
                pa.array([["b", "c"], None, [], ["d", "e"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="inconsecutive_last_row_included",
        row_indices=[2, 3, 4, 5, 7, 8],
        expected_output=pa.Table.from_arrays(
            [
                pa.array([[4], [], [5, 6], [7], [10], []],
                         type=pa.list_(pa.int32())),
                pa.array([None, [], None, ["d", "e"], None, ["g"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
]

_SLICE_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="row_indices_not_np",
        row_indices=[0],
        expected_error_type=TypeError,
        expected_error_regexp="expected row_indices to be a numpy array"),
    dict(
        testcase_name="row_indices_not_int32",
        row_indices=np.array([0], dtype=np.int64),
        expected_error_type=TypeError,
        expected_error_regexp="expected row_indices to be a 1-D int32"),
    dict(
        testcase_name="row_indices_not_1_d",
        row_indices=np.array([[0]], dtype=np.int32),
        expected_error_type=TypeError,
        expected_error_regexp="expected row_indices to be a 1-D"),
    dict(
        testcase_name="out_of_range",
        row_indices=np.array([1], dtype=np.int32),
        expected_error_type=RuntimeError,
        expected_error_regexp="out of range"),
]


class SliceTableByRowIndicesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_SLICE_TEST_CASES)
  def test_success(self, row_indices, expected_output):
    table = pa.Table.from_arrays([
        pa.array([[1, 2, 3], None, [4], [], [5, 6], [7], [8, 9], [10], []],
                 type=pa.list_(pa.int32())),
        pa.array(
            [["a"], ["b", "c"], None, [], None, ["d", "e"], ["f"], None, ["g"]],
            type=pa.list_(pa.binary())),
    ], ["f1", "f2"])

    sliced = merge.SliceTableByRowIndices(table,
                                          np.array(row_indices, dtype=np.int32))
    self.assertTrue(
        sliced.equals(expected_output),
        "Expected {}, got {}".format(expected_output, sliced))
    if sliced.num_rows > 0:
      for c in sliced.columns:
        self.assertEqual(c.data.num_chunks, 1)

  @parameterized.named_parameters(*_SLICE_INVALID_INPUT_TEST_CASES)
  def test_invalid_inputs(self, row_indices, expected_error_type,
                          expected_error_regexp):
    with self.assertRaisesRegexp(expected_error_type, expected_error_regexp):
      merge.SliceTableByRowIndices(
          pa.Table.from_arrays([pa.array([1])], ["f1"]), row_indices)


if __name__ == "__main__":
  absltest.main()
