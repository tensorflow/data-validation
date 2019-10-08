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
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa


class EnumerateArraysTest(absltest.TestCase):

  def testInvalidWeightColumnMissingValue(self):
    with self.assertRaisesRegex(
        ValueError,
        'Weight feature "w" must have exactly one value.*'):
      for _ in arrow_util.enumerate_arrays(
          pa.Table.from_arrays([pa.array([[1], [2, 3]]),
                                pa.array([[1], []])], ["v", "w"]),
          weight_column="w",
          enumerate_leaves_only=False):
        pass

  def testInvalidWeightColumnTooManyValues(self):
    with self.assertRaisesRegex(
        ValueError,
        'Weight feature "w" must have exactly one value.*'):
      for _ in arrow_util.enumerate_arrays(
          pa.Table.from_arrays([pa.array([[1], [2, 3]]),
                                pa.array([[1], [2, 2]])], ["v", "w"]),
          weight_column="w",
          enumerate_leaves_only=False):
        pass

  def testInvalidWeightColumnStringValues(self):
    with self.assertRaisesRegex(
        ValueError,
        'Weight feature "w" must be of numeric type.*'):
      for _ in arrow_util.enumerate_arrays(
          pa.Table.from_arrays([pa.array([[1], [2, 3]]),
                                pa.array([["two"], ["two"]])], ["v", "w"]),
          weight_column="w",
          enumerate_leaves_only=False):
        pass

  def testEnumerate(self):
    input_table = pa.Table.from_arrays([
        pa.array([[1], [2, 3]]),
        pa.array([[{
            "sf1": [["a", "b"]]
        }], [{
            "sf2": [{
                "ssf1": [[3], [4]]
            }]
        }]]),
        pa.array([[1.0], [2.0]])
    ], ["f1", "f2", "w"])
    possible_results = {
        types.FeaturePath(["f1"]): (pa.array([[1], [2, 3]]), [1.0, 2.0]),
        types.FeaturePath(["w"]): (pa.array([[1.0], [2.0]]), [1.0, 2.0]),
        types.FeaturePath(["f2"]): (pa.array([[{
            "sf1": [["a", "b"]]
        }], [{
            "sf2": [{
                "ssf1": [[3], [4]]
            }]
        }]]), [1.0, 2.0]),
        types.FeaturePath(["f2", "sf1"]): (
            pa.array([[["a", "b"]], None]), [1.0, 2.0]),
        types.FeaturePath(["f2", "sf2"]): (
            pa.array([None, [{"ssf1": [[3], [4]]}]]), [1.0, 2.0]),
        types.FeaturePath(["f2", "sf2", "ssf1"]): (
            pa.array([[[3], [4]]]), [2.0]),
    }
    for leaves_only, has_weights in itertools.combinations_with_replacement(
        [True, False], 2):
      actual_results = {}
      for feature_path, feature_array, weights in arrow_util.enumerate_arrays(
          input_table, "w" if has_weights else None, leaves_only):
        actual_results[feature_path] = (feature_array, weights)

      expected_results = {}
      for p in [["f1"], ["w"], ["f2", "sf1"], ["f2", "sf2", "ssf1"]]:
        feature_path = types.FeaturePath(p)
        expected_results[feature_path] = (possible_results[feature_path][0],
                                          possible_results[feature_path][1]
                                          if has_weights else None)
      if not leaves_only:
        for p in [["f2"], ["f2", "sf2"]]:
          feature_path = types.FeaturePath(p)
          expected_results[feature_path] = (possible_results[feature_path][0],
                                            possible_results[feature_path][1]
                                            if has_weights else None)

      self.assertLen(actual_results, len(expected_results))
      for k, v in six.iteritems(expected_results):
        self.assertIn(k, actual_results)
        actual = actual_results[k]
        self.assertTrue(
            actual[0].equals(v[0]), "leaves_only={}; has_weights={}; "
            "feature={}; expected: {}; actual: {}".format(
                leaves_only, has_weights, k, v, actual))
        np.testing.assert_array_equal(actual[1], v[1])


class PrimitiveArrayToNumpyTest(absltest.TestCase):

  def testNumberArrayShouldShareBuffer(self):
    float_array = pa.array([1, 2, np.NaN], pa.float32())
    np_array = arrow_util.primitive_array_to_numpy(float_array)
    self.assertEqual(np_array.dtype, np.float32)
    self.assertEqual(np_array.shape, (3,))
    # Check that they share the same buffer.
    self.assertEqual(np_array.ctypes.data, float_array.buffers()[1].address)

  def testStringArray(self):
    string_array = pa.array(["a", "b"], pa.utf8())
    np_array = arrow_util.primitive_array_to_numpy(string_array)
    self.assertEqual(np_array.dtype, np.object)
    self.assertEqual(np_array.shape, (2,))
    np.testing.assert_array_equal(np_array, [u"a", u"b"])

  def testNumberArrayWithNone(self):
    float_array = pa.array([1.0, 2.0, None], pa.float64())
    np_array = arrow_util.primitive_array_to_numpy(float_array)
    self.assertEqual(np_array.dtype, np.float64)
    np.testing.assert_array_equal(np_array, [1.0, 2.0, np.NaN])


if __name__ == "__main__":
  absltest.main()
