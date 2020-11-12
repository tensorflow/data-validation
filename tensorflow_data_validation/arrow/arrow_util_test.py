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

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap
from tfx_bsl.arrow import array_util


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

ExpectedArray = collections.namedtuple("ExpectedArray",
                                       ["array", "parent_indices", "weights"])
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


class ArrowUtilTest(parameterized.TestCase):

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

  def testGetArrayEmptyPath(self):
    with self.assertRaisesRegex(
        KeyError,
        r"query_path must be non-empty.*"):
      arrow_util.get_array(
          pa.RecordBatch.from_arrays([pa.array([[1], [2, 3]])], ["v"]),
          query_path=types.FeaturePath([]),
          return_example_indices=False)

  def testGetArrayColumnMissing(self):
    with self.assertRaisesRegex(
        KeyError,
        r'query_path step 0 "x" not in record batch.*'):
      arrow_util.get_array(
          pa.RecordBatch.from_arrays([pa.array([[1], [2]])], ["y"]),
          query_path=types.FeaturePath(["x"]),
          return_example_indices=False)

  def testGetArrayStepMissing(self):
    with self.assertRaisesRegex(KeyError,
                                r'query_path step "ssf3" not in struct.*'):
      arrow_util.get_array(
          _INPUT_RECORD_BATCH,
          query_path=types.FeaturePath(["f2", "sf2", "ssf3"]),
          return_example_indices=False)

  def testGetArrayReturnExampleIndices(self):
    record_batch = pa.RecordBatch.from_arrays([
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
        record_batch, feature, return_example_indices=True)
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
          _INPUT_RECORD_BATCH,
          query_path=types.FeaturePath(["f2", "sf2", "ssf1", "sssf"]),
          return_example_indices=False)

  @parameterized.named_parameters(
      ((str(f), f, expected) for (f, expected) in  _FEATURES_TO_ARRAYS.items()))
  def testGetArray(self, feature, expected):
    actual_arr, actual_indices = arrow_util.get_array(
        _INPUT_RECORD_BATCH, feature, return_example_indices=True,
        wrap_flat_struct_in_list=False)
    expected_arr, expected_indices, _ = expected
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    np.testing.assert_array_equal(expected_indices, actual_indices)

  @parameterized.named_parameters(
      ((str(f), f, expected) for (f, expected) in  _FEATURES_TO_ARRAYS.items()))
  def testGetArrayNoBroadcast(self, feature, expected):
    actual_arr, actual_indices = arrow_util.get_array(
        _INPUT_RECORD_BATCH, feature, return_example_indices=False,
        wrap_flat_struct_in_list=False)
    expected_arr, _, _ = expected
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    self.assertIsNone(actual_indices)

  @parameterized.named_parameters(
      ((str(f), f, expected) for (f, expected) in  _FEATURES_TO_ARRAYS.items()))
  def testGetArrayWrapFlatStructArray(self, feature, expected):
    actual_arr, actual_indices = arrow_util.get_array(
        _INPUT_RECORD_BATCH, feature, return_example_indices=True,
        wrap_flat_struct_in_list=True)
    expected_arr, expected_indices, _ = expected
    if pa.types.is_struct(expected_arr.type):
      expected_arr = array_util.ToSingletonListArray(expected_arr)
    self.assertTrue(
        actual_arr.equals(expected_arr),
        "\nfeature: {};\nexpected:\n{};\nactual:\n{}".format(
            feature, expected_arr, actual_arr))
    np.testing.assert_array_equal(expected_indices, actual_indices)

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

  def testFlattenNested(self):
    input_array = pa.array([[[1, 2]], None, [None, [3]]])
    flattened, parent_indices = arrow_util.flatten_nested(
        input_array, return_parent_indices=False)
    expected = pa.array([1, 2, 3])
    expected_parent_indices = [0, 0, 2]
    self.assertIs(parent_indices, None)
    self.assertTrue(flattened.equals(expected))

    flattened, parent_indices = arrow_util.flatten_nested(
        input_array, return_parent_indices=True)
    self.assertTrue(flattened.equals(expected))
    np.testing.assert_array_equal(parent_indices, expected_parent_indices)

  def testFlattenNestedNonList(self):
    input_array = pa.array([1, 2])
    flattened, parent_indices = arrow_util.flatten_nested(
        input_array, return_parent_indices=True)
    self.assertTrue(flattened.equals(pa.array([1, 2])))
    np.testing.assert_array_equal(parent_indices, [0, 1])


if __name__ == "__main__":
  absltest.main()
