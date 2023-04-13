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
"""Tests for tensorflow_data_validation.arrow.arrow_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from typing import Dict, Iterable, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl.arrow import array_util

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order


_INPUT_RECORD_BATCH = pa.RecordBatch.from_arrays([
    pa.array([[1], [2, 3]]),
    pa.array([[{
        "sf1": ["a", "b"]
    }], [{
        "sf2": [{
            "ssf1": [3]
        }, {
            "ssf1": [4]
        }]
    }]]),
    pa.array([
        {
            "sf1": [[1, 2], [3]],
            "sf2": [None],
        },
        None,
    ]),
    pa.array([[1], [2]]),
    pa.array([[2], [4]]),
    pa.array([[6], [8]]),
], ["f1", "f2", "f3", "w", "w_override1", "w_override2"])

_EXAMPLE_WEIGHT_MAP = ExampleWeightMap(
    weight_feature="w", per_feature_override={
        types.FeaturePath(["f2"]): "w_override1",
        types.FeaturePath(["f2", "sf1"]): "w_override2",
        types.FeaturePath(["f2", "sf2"]): "w_override2",
        types.FeaturePath(["f2", "sf2", "ssf1"]): "w_override1",
    })

ExpectedArray = tfx_namedtuple.namedtuple(
    "ExpectedArray", ["array", "parent_indices", "weights"])
_FEATURES_TO_ARRAYS = {
    types.FeaturePath(["f1"]): ExpectedArray(
        pa.array([[1], [2, 3]]), [0, 1], [1, 2]),
    types.FeaturePath(["w"]): ExpectedArray(
        pa.array([[1], [2]]), [0, 1], [1, 2]),
    types.FeaturePath(["w_override1"]): ExpectedArray(
        pa.array([[2], [4]]), [0, 1], [1, 2]),
    types.FeaturePath(["w_override2"]): ExpectedArray(
        pa.array([[6], [8]]), [0, 1], [1, 2]),
    types.FeaturePath(["f2"]): ExpectedArray(pa.array([[{
        "sf1": ["a", "b"]
    }], [{
        "sf2": [{
            "ssf1": [3]
        }, {
            "ssf1": [4]
        }]
    }]]), [0, 1], [2, 4]),
    types.FeaturePath(["f3"]): ExpectedArray(pa.array([{
        "sf1": [[1, 2], [3]],
        "sf2": [None],
    }, None]), [0, 1], [1, 2]),
    types.FeaturePath(["f2", "sf1"]): ExpectedArray(
        pa.array([["a", "b"], None]), [0, 1], [6, 8]),
    types.FeaturePath(["f2", "sf2"]): ExpectedArray(
        pa.array([None, [{
            "ssf1": [3]
        }, {
            "ssf1": [4]
        }]]), [0, 1], [6, 8]),
    types.FeaturePath(["f2", "sf2", "ssf1"]): ExpectedArray(
        pa.array([[3], [4]]), [1, 1], [4, 4]),
    types.FeaturePath(["f3", "sf1"]): ExpectedArray(pa.array(
        [[[1, 2], [3]], None]), [0, 1], [1, 2]),
    types.FeaturePath(["f3", "sf2"]): ExpectedArray(
        pa.array([[None], None]), [0, 1], [1, 2]),
}


class EnumerateStructNullValueTestData(NamedTuple):
  """Inputs and outputs for enumeration with pa.StructArrays with null values."""
  description: str
  """Summary of test"""
  batch: pa.RecordBatch
  """Input Record Batch"""
  expected_results: Dict[types.FeaturePath, pa.array]
  """The expected output."""


def _MakeEnumerateDataWithMissingDataAtLeaves(
    ) -> Iterable[EnumerateStructNullValueTestData]:
  """Test that having only nulls at leaf values gets translated correctly."""
  test_data_type = pa.list_(pa.struct([("f2", pa.list_(pa.float64()))]))
  struct_column_as_list_dicts = [
      [],  # first element of 'c'; note this is not counted as missing.
      [  # second element of 'c' -- a list<struct> of length 2.
          {
              "f2": [2.0],
          },
          None,  # f2 is missing
      ],
      [  # third element of 'c'
          None,  # f2 is missing
      ],
      [],  # fourth element of 'c'; note this is not counted as missing.
  ]

  array = pa.array(struct_column_as_list_dicts, type=test_data_type)

  batch = pa.RecordBatch.from_arrays([array], ["c"])

  full_expected_results = {
      types.FeaturePath(["c"]):
          pa.array([[], [{
              "f2": [2.0]
          }, None], [None], []]),
      types.FeaturePath(["c", "f2"]):
          pa.array([[2.0], None, None]),
  }
  yield "Basic", batch, full_expected_results


def _MakeEnumerateTestDataWithNullValuesAndSlicedBatches(
    ) -> Iterable[EnumerateStructNullValueTestData]:
  """Yields test data for sliced data where all slicing is consistent.

  Pyarrow slices with zero copy, sometimes subtle bugs can
  arise when processing sliced data.
  """
  test_data_type = pa.list_(pa.struct([("f2", pa.list_(pa.float64()))]))
  struct_column_as_list_dicts = [
      [],  # first element of 'c'; note this is not counted as missing.
      [  # second element of 'c' -- a list<struct> of length 2.
          {
              "f2": [2.0],
          },
          None,  # f2 is missing
      ],
      [  # third element of 'c'
          None,  # f2 is missing
      ],
      [],  # fourth element of 'c'; note this is not counted as missing.
  ]

  array = pa.array(struct_column_as_list_dicts, type=test_data_type)

  batch = pa.RecordBatch.from_arrays([array], ["c"])
  slice_start, slice_end = 1, 3
  batch = pa.RecordBatch.from_arrays([array[slice_start:slice_end]], ["c"])

  sliced_expected_results = {
      types.FeaturePath(["c"]): pa.array([[{
          "f2": [2.0]
      }, None], [None]]),
      types.FeaturePath(["c", "f2"]): pa.array([[2.0], None, None]),
  }
  # Test case 1: slicing the array.
  yield "SlicedArray", batch, sliced_expected_results

  batch = pa.RecordBatch.from_arrays([array], ["c"])[slice_start:slice_end]
  # Test case 2: slicing the RecordBatch.
  yield "SlicedRecordBatch", batch, sliced_expected_results


def _MakeEnumerateTestDataWithNullTopLevel(
    ) -> Iterable[EnumerateStructNullValueTestData]:
  """Yields test data with a top level list element is missing."""
  test_data_type = pa.list_(pa.struct([("f2", pa.list_(pa.float64()))]))
  struct_column_as_list_dicts = [
      [],  # first element of 'c'; note this is not counted as missing.
      None,  # c is missing.
      [   # third element of 'c'
          None,  # f2 is missing
      ],
      [],  # fourth element of 'c'; note this is not counted as missing.
  ]
  array = pa.array(
      struct_column_as_list_dicts, type=test_data_type)
  validity_buffer_with_null = array.buffers()[0]
  array_with_null_indicator = pa.Array.from_buffers(
      array.type,
      len(array) + array.offset,
      [validity_buffer_with_null, array.buffers()[1]],
      offset=0,
      children=[array.values])
  batch_with_missing_entry = pa.RecordBatch.from_arrays(
      [array_with_null_indicator], ["c"])
  missing_expected_results = {
      types.FeaturePath(["c"]):
          pa.array([[], None, [None], []], type=test_data_type),
      types.FeaturePath(["c", "f2"]):
          pa.array([None], type=pa.list_(pa.float64())),
  }
  yield ("ValuesPresentWithNullIndicator", batch_with_missing_entry,
         missing_expected_results)


def _MakeEnumerateTestDataWithSlicesAtDifferentOffsets(
    ) -> Iterable[EnumerateStructNullValueTestData]:
  """Yields a test cases constructed from array slices with different offsets.

  Slicing in pyarrow is zero copy, which can have subtle bugs, so ensure
  the code works under more obscure situations.
  """
  total_size = 10
  values_array = pa.array(range(total_size), type=pa.int64())
  # create 5 pyarrow.Array object each of size from the original array ([0,1],
  # [2,3], etc
  slices = [
      values_array[start:end] for (start, end)
      in zip(range(0, total_size + 1, 2), range(2, total_size + 1, 2))
  ]  # pyformat: disable
  validity = pa.array([True, False], type=pa.bool_())
  # Label fields from "0" to "5"
  new_type = pa.struct([pa.field(str(sl[0].as_py() // 2), sl.type)
                        for sl in slices])
  # Using the value buffer of validity as composed_struct's validity bitmap
  # buffer.
  composed_struct = pa.StructArray.from_buffers(
      new_type, len(slices[0]), [validity.buffers()[1]], children=slices)
  sliced_batch = pa.RecordBatch.from_arrays([composed_struct], ["c"])
  sliced_expected_results = {
      types.FeaturePath(["c"]):
          pa.array([
              [{"0": 0, "1": 2, "2": 4, "3": 6, "4": 8}],
              None,
          ]),
      types.FeaturePath(["c", "0"]): pa.array([0, None], type=pa.int64()),
      types.FeaturePath(["c", "1"]): pa.array([2, None], type=pa.int64()),
      types.FeaturePath(["c", "2"]): pa.array([4, None], type=pa.int64()),
      types.FeaturePath(["c", "3"]): pa.array([6, None], type=pa.int64()),
      types.FeaturePath(["c", "4"]): pa.array([8, None], type=pa.int64()),
  }  # pyformat: disable
  yield ("SlicedArrayWithOffests", sliced_batch, sliced_expected_results)


def _Normalize(array: pa.Array) -> pa.Array:
  """Round trips array through python objects.

  Comparing nested arrays with slices is buggy in Arrow 2.0 this method
  is useful comparing two such arrays for logical equality. The bugs
  appears to be fixed as of Arrow 5.0 this should be removable once that
  becomes the minimum version.

  Args:
    array: The array to normalize.

  Returns:
    An array that doesn't have any more zero copy slices in itself or
    it's children. Note the schema might be slightly different for
    all null arrays.
  """
  return pa.array(array.to_pylist())


class ArrowUtilTest(parameterized.TestCase):

  def testIsBinaryLike(self):
    for t in (pa.binary(), pa.large_binary(), pa.string(), pa.large_string()):
      self.assertTrue(arrow_util.is_binary_like(t))

    for t in (pa.list_(pa.binary()), pa.large_list(pa.string())):
      self.assertFalse(arrow_util.is_binary_like(t))

  def testGetWeightFeatureNotFound(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Weight column "w" not present in the input record batch\.'):
      arrow_util.get_weight_feature(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [2]]),
               pa.array([[1], [3]])], ["u", "v"]),
          weight_column="w")

  def testGetWeightFeatureNullArray(self):
    with self.assertRaisesRegex(ValueError, 'Weight column "w" cannot be '
                                r'null\.'):
      arrow_util.get_weight_feature(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [2]]),
               pa.array([None, None])], ["v", "w"]),
          weight_column="w")

  def testGetWeightFeatureMissingValue(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Weight column "w" must have exactly one value in each example\.'):
      arrow_util.get_weight_feature(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [2]]),
               pa.array([[1], []])], ["v", "w"]),
          weight_column="w")

  def testGetWeightFeatureTooManyValues(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Weight column "w" must have exactly one value in each example\.'):
      arrow_util.get_weight_feature(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [2, 3]]),
               pa.array([[1], [2, 2]])], ["v", "w"]),
          weight_column="w")

  def testEnumerateArraysStringWeight(self):
    # The arrow type of a string changes between py2 and py3 so we accept either
    with self.assertRaisesRegex(
        ValueError,
        r'Weight column "w" must be of numeric type. Found (string|binary).*'):
      for _ in arrow_util.enumerate_arrays(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [2, 3]]),
               pa.array([["a"], ["b"]])], ["v", "w"]),
          example_weight_map=ExampleWeightMap(
              weight_feature="w", per_feature_override=None),
          enumerate_leaves_only=True):
        pass

  def testEnumerateArrays(self):
    for leaves_only, has_weights, wrap_flat_struct_in_list in (
        itertools.product([True, False], [True, False], [True, False])):
      actual_results = {}
      for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
          _INPUT_RECORD_BATCH,
          _EXAMPLE_WEIGHT_MAP
          if has_weights else None, leaves_only, wrap_flat_struct_in_list):
        actual_results[feature_path] = (feature_array, weights)

      expected_results = {}
      # leaf fields
      for p in [["f1"], ["w"], ["w_override1"], ["w_override2"],
                ["f2", "sf1"], ["f2", "sf2", "ssf1"],
                ["f3", "sf1"], ["f3", "sf2"]]:
        feature_path = types.FeaturePath(p)
        expected_results[feature_path] = (
            _FEATURES_TO_ARRAYS[feature_path].array,
            _FEATURES_TO_ARRAYS[feature_path].weights if has_weights else None)
      if not leaves_only:
        for p in [["f2"], ["f2", "sf2"], ["f3"]]:
          feature_path = types.FeaturePath(p)
          expected_array = _FEATURES_TO_ARRAYS[feature_path][0]
          if wrap_flat_struct_in_list and pa.types.is_struct(
              expected_array.type):
            expected_array = array_util.ToSingletonListArray(expected_array)
          expected_results[feature_path] = (
              expected_array, _FEATURES_TO_ARRAYS[feature_path].weights
              if has_weights else None)

      self.assertLen(actual_results, len(expected_results))
      for k, v in six.iteritems(expected_results):
        self.assertIn(k, actual_results)
        actual = actual_results[k]
        self.assertTrue(
            actual[0].equals(v[0]), "leaves_only={}; has_weights={}; "
            "wrap_flat_struct_in_list={} feature={}; expected: {}; actual: {}"
            .format(leaves_only, has_weights, wrap_flat_struct_in_list, k, v,
                    actual))
        np.testing.assert_array_equal(actual[1], v[1])

  @parameterized.named_parameters(
      {
          "testcase_name": "select_column_f1",
          "col_fn": lambda x: x == "f1",
          "expected_features": [types.FeaturePath(["f1"])],
      }, {
          "testcase_name":
              "select_column_f2",
          "col_fn":
              lambda x: x == "f2",
          "expected_features": [
              types.FeaturePath(["f2", "sf1"]),
              types.FeaturePath(["f2", "sf2", "ssf1"])
          ],
      })
  def testEnumerateArraysWithColumnSelectFn(self, col_fn, expected_features):
    actual = list(
        arrow_util.enumerate_arrays(
            _INPUT_RECORD_BATCH,
            _EXAMPLE_WEIGHT_MAP,
            True,
            column_select_fn=col_fn))
    expected = list(
        (f, _FEATURES_TO_ARRAYS[f].array, _FEATURES_TO_ARRAYS[f].weights)
        for f in expected_features)
    for (actual_path, actual_col,
         actual_w), (expected_path, expected_col,
                     expected_w) in zip(actual, expected):
      self.assertEqual(expected_path, actual_path)
      self.assertEqual(expected_col, actual_col)
      self.assertEqual(pa.array(expected_w), pa.array(actual_w))

  @parameterized.named_parameters(itertools.chain(
      _MakeEnumerateDataWithMissingDataAtLeaves(),
      _MakeEnumerateTestDataWithNullValuesAndSlicedBatches(),
      _MakeEnumerateTestDataWithNullTopLevel(),
      _MakeEnumerateTestDataWithSlicesAtDifferentOffsets()))
  def testEnumerateMissingPropagatedInFlattenedStruct(self, batch,
                                                      expected_results):
    actual_results = {}
    for feature_path, feature_array, _ in arrow_util.enumerate_arrays(
        batch, example_weight_map=None, enumerate_leaves_only=False):
      actual_results[feature_path] = feature_array
    self.assertLen(actual_results, len(expected_results))
    for k, v in six.iteritems(expected_results):
      assert k in actual_results, (k, list(actual_results.keys()))
      self.assertIn(k, actual_results)
      actual = _Normalize(actual_results[k])
      v = _Normalize(v)
      self.assertTrue(
          actual.equals(v),
          "feature={}; expected: {}; actual: {}; diff: {}".format(
              k, v, actual, actual.diff(v)))

  def testGetColumn(self):
    self.assertTrue(
        arrow_util.get_column(_INPUT_RECORD_BATCH,
                              "f1").equals(pa.array([[1], [2, 3]])))
    self.assertIsNone(
        arrow_util.get_column(_INPUT_RECORD_BATCH, "xyz", missing_ok=True))
    with self.assertRaises(KeyError):
      arrow_util.get_column(_INPUT_RECORD_BATCH, "xyz")

if __name__ == "__main__":
  absltest.main()
