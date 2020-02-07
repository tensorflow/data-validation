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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util


_INPUT_TABLE = pa.Table.from_arrays([
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
    pa.array([[0], [1]])
], ["f1", "f2", "w"])

_FEATURES_TO_ARRAYS = {
    types.FeaturePath(["f1"]): (pa.array([[1], [2, 3]]), [0, 1]),
    types.FeaturePath(["w"]): (pa.array([[0], [1]]), [0, 1]),
    types.FeaturePath(["f2"]): (
        pa.array([
            [{"sf1": ["a", "b"]}],
            [{"sf2": [{"ssf1": [3]}, {"ssf1": [4]}]}]
        ]), [0, 1]),
    types.FeaturePath(["f2", "sf1"]): (
        pa.array([["a", "b"], None]), [0, 1]),
    types.FeaturePath(["f2", "sf2"]): (
        pa.array([None, [{"ssf1": [3]}, {"ssf1": [4]}]]), [0, 1]),
    types.FeaturePath(["f2", "sf2", "ssf1"]): (
        pa.array([[3], [4]]), [1, 1]),
}


class EnumerateArraysTest(parameterized.TestCase):

  def testIsListLike(self):
    for t in (pa.list_(pa.int64()), pa.large_list(pa.int64())):
      self.assertTrue(arrow_util.is_list_like(t))

    for t in (pa.binary(), pa.int64(), pa.large_string()):
      self.assertFalse(arrow_util.is_list_like(t))

  def testIsBinaryLike(self):
    for t in (pa.binary(), pa.large_binary(), pa.string(), pa.large_string()):
      self.assertTrue(arrow_util.is_binary_like(t))

    for t in (pa.list_(pa.binary()), pa.large_list(pa.string())):
      self.assertFalse(arrow_util.is_binary_like(t))

  def testGetBroadcastableColumnNotFound(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Column "w" not present in the input table\.'):
      arrow_util.get_broadcastable_column(
          pa.Table.from_arrays(
              [pa.array([[1], [2]]),
               pa.array([[1], [3]])], ["u", "v"]),
          column_name="w")

  def testGetBroadcastableColumnMissingValue(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Column "w" must have exactly one value in each example\.'):
      arrow_util.get_broadcastable_column(
          pa.Table.from_arrays(
              [pa.array([[1], [2]]),
               pa.array([[1], []])], ["v", "w"]),
          column_name="w")

  def testGetBroadcastableColumnTooManyValues(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Column "w" must have exactly one value in each example\.'):
      arrow_util.get_broadcastable_column(
          pa.Table.from_arrays(
              [pa.array([[1], [2, 3]]),
               pa.array([[1], [2, 2]])], ["v", "w"]),
          column_name="w")

  def testGetArrayEmptyPath(self):
    with self.assertRaisesRegex(
        KeyError,
        r"query_path must be non-empty.*"):
      arrow_util.get_array(
          pa.Table.from_arrays([pa.array([[1], [2, 3]])], ["v"]),
          query_path=types.FeaturePath([]),
          return_example_indices=False)

  def testGetArrayColumnMissing(self):
    with self.assertRaisesRegex(
        KeyError,
        r'query_path step 0 "x" not in table.*'):
      arrow_util.get_array(
          pa.Table.from_arrays([pa.array([[1], [2]])], ["y"]),
          query_path=types.FeaturePath(["x"]),
          return_example_indices=False)

  def testGetArrayStepMissing(self):
    with self.assertRaisesRegex(KeyError,
                                r'query_path step "ssf3" not in struct.*'):
      arrow_util.get_array(
          _INPUT_TABLE,
          query_path=types.FeaturePath(["f2", "sf2", "ssf3"]),
          return_example_indices=False)

  def testGetArrayReturnExampleIndices(self):
    table = pa.Table.from_arrays([
        pa.array([[{
            "sf": [{
                "ssf": [1]
            }, {
                "ssf": [2]
            }]
        }], [{
            "sf": [{
                "ssf": [3, 4]
            }]
        }]]),
        pa.array([["one"], ["two"]])
    ], ["f", "w"])
    feature = types.FeaturePath(["f", "sf", "ssf"])
    actual_arr, actual_indices = arrow_util.get_array(
        table, feature, return_example_indices=True)
    expected_arr = pa.array([[1], [2], [3, 4]])
    expected_indices = np.array([0, 0, 1])
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    np.testing.assert_array_equal(expected_indices, actual_indices)

  def testGetArraySubpathMissing(self):
    with self.assertRaisesRegex(
        KeyError,
        r'Cannot process .* "sssf" inside .* list<item: int64>.*'):
      arrow_util.get_array(
          _INPUT_TABLE,
          query_path=types.FeaturePath(["f2", "sf2", "ssf1", "sssf"]),
          return_example_indices=False)

  @parameterized.named_parameters(
      ((str(f), f, expected) for (f, expected) in  _FEATURES_TO_ARRAYS.items()))
  def testGetArray(self, feature, expected):
    actual_arr, actual_indices = arrow_util.get_array(
        _INPUT_TABLE, feature, return_example_indices=True)
    expected_arr, expected_indices = expected
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    np.testing.assert_array_equal(expected_indices, actual_indices)

  @parameterized.named_parameters(
      ((str(f), f, expected) for (f, expected) in  _FEATURES_TO_ARRAYS.items()))
  def testGetArrayNoBroadcast(self, feature, expected):
    actual_arr, actual_indices = arrow_util.get_array(
        _INPUT_TABLE, feature, return_example_indices=False)
    expected_arr, _ = expected
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    self.assertIsNone(actual_indices)

  def testEnumerateArraysStringWeight(self):
    # The arrow type of a string changes between py2 and py3 so we accept either
    with self.assertRaisesRegex(
        ValueError,
        r'Weight column "w" must be of numeric type. Found (string|binary).*'):
      for _ in arrow_util.enumerate_arrays(
          pa.Table.from_arrays(
              [pa.array([[1], [2, 3]]),
               pa.array([["a"], ["b"]])], ["v", "w"]),
          weight_column="w",
          enumerate_leaves_only=True):
        pass

  def testEnumerateArrays(self):
    for leaves_only, has_weights in itertools.combinations_with_replacement(
        [True, False], 2):
      actual_results = {}
      for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
          _INPUT_TABLE, "w" if has_weights else None, leaves_only):
        actual_results[feature_path] = (feature_array, weights)

      expected_results = {}
      for p in [["f1"], ["w"], ["f2", "sf1"], ["f2", "sf2", "ssf1"]]:
        feature_path = types.FeaturePath(p)
        expected_results[feature_path] = (_FEATURES_TO_ARRAYS[feature_path][0],
                                          _FEATURES_TO_ARRAYS[feature_path][1]
                                          if has_weights else None)
      if not leaves_only:
        for p in [["f2"], ["f2", "sf2"]]:
          feature_path = types.FeaturePath(p)
          expected_results[feature_path] = (
              _FEATURES_TO_ARRAYS[feature_path][0],
              _FEATURES_TO_ARRAYS[feature_path][1] if has_weights else None)

      self.assertLen(actual_results, len(expected_results))
      for k, v in six.iteritems(expected_results):
        self.assertIn(k, actual_results)
        actual = actual_results[k]
        self.assertTrue(
            actual[0].equals(v[0]), "leaves_only={}; has_weights={}; "
            "feature={}; expected: {}; actual: {}".format(
                leaves_only, has_weights, k, v, actual))
        np.testing.assert_array_equal(actual[1], v[1])


if __name__ == "__main__":
  absltest.main()
