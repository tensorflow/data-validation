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

"""Tests for example batching utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.utils import batch_util


class BatchUtilTest(absltest.TestCase):

  def test_merge_single_batch(self):
    examples = [
        {
            'a': np.array([1.0, 2.0]),
            'b': np.array(['a', 'b', 'c', 'e'])
        },
        {
            'a': np.array([3.0, 4.0, np.NaN, 5.0]),
        },
        {
            'b': np.array(['d', 'e', 'f']),
            'd': np.array([10, 20, 30]),
        },
        {
            'b': np.array(['a', 'b', 'c'])
        },
        {
            'c': np.array(['d', 'e', 'f'])
        }
    ]
    expected_batch = {
        'a': [np.array([1.0, 2.0]), np.array([3.0, 4.0, np.NaN, 5.0]),
              None, None, None],
        'b': [np.array(['a', 'b', 'c', 'e']), None, np.array(['d', 'e', 'f']),
              np.array(['a', 'b', 'c']), None],
        'c': [None, None, None, None, np.array(['d', 'e', 'f'])],
        'd': [None, None, np.array([10, 20, 30]), None, None]
    }
    actual_batch = batch_util.merge_single_batch(examples)
    # check number of features.
    self.assertLen(actual_batch, len(expected_batch))
    for feature_name in expected_batch:
      # check batch size.
      self.assertLen(actual_batch[feature_name],
                     len(expected_batch[feature_name]))
      for i in range(len(expected_batch[feature_name])):
        expected_value = expected_batch[feature_name][i]
        actual_value = actual_batch[feature_name][i]
        if expected_value is None:
          self.assertEqual(actual_value, expected_value)
        else:
          # check dtype.
          self.assertEqual(actual_value.dtype, expected_value.dtype)
          # check numpy array.
          np.testing.assert_array_equal(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
