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

"""Tests for bin_util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa

from tensorflow_data_validation.utils import bin_util


class BinArrayTest(parameterized.TestCase):
  """Tests for bin_array."""

  @parameterized.named_parameters([
      ('simple', pa.array([0.1, 0.5, 0.75]), [0.25, 0.75], [0, 1, 2],
       [0, 1, 2]),
      ('negative_values', pa.array([-0.8, -0.5, -0.1]), [0.25], [0, 1, 2],
       [0, 0, 0]),
      ('inf_values', pa.array([float('-inf'), 0.5, float('inf')]),
       [0.25, 0.75], [0, 1, 2], [0, 1, 2]),
      ('nan_values', pa.array([np.nan, 0.5]), [0.25, 0.75], [1], [1]),
      ('negative_boundaries', pa.array([-0.8, -0.5]), [-0.75, -0.25], [0, 1],
       [0, 1]),
      ('empty_array', pa.array([]), [0.25], [], []),
      ('none_value', pa.array([None, 0.5]), [0.25], [1], [1]),
      ('null_array', pa.array([None, None], type=pa.null()), [0.25], [], [])
  ])
  def test_bin_array(self, array, boundaries, expected_indices, expected_bins):
    indices, bins = bin_util.bin_array(array, boundaries)
    np.testing.assert_array_equal(expected_indices, indices)
    np.testing.assert_array_equal(expected_bins, bins)


if __name__ == '__main__':
  absltest.main()
