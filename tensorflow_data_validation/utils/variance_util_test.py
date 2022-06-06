# Copyright 2021 Google LLC
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
"""Tests for variance_util."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_data_validation.utils import variance_util


def _weighted_mean(values, weights):
  if weights is None:
    mean = np.mean(values)
  else:
    mean = np.sum(values * weights) / np.sum(weights)
  assert np.isfinite(mean)
  return mean


def _weighted_variance(values, weights):
  if weights is None:
    variance = np.var(values)
  else:
    mean = _weighted_mean(values, weights)
    variance = np.sum((values - mean)**2 * weights) / np.sum(weights)
  assert np.isfinite(variance)
  return variance


def _rel_err(est_val, true_val):
  return np.abs(est_val - true_val) / np.abs(true_val)


_MEAN_VAR_ACCUMULATOR_TEST_CASES = [
    {
        'testcase_name': 'unit_normal',
        'array_size': 1000,
        'distribution_mean': 0.0,
        'distribution_variance': 1.0,
        'use_weights': False
    },
    {
        'testcase_name': 'large_pos_shift',
        'array_size': 1000,
        'distribution_mean': 100000.0,
        'distribution_variance': 1.0,
        'use_weights': False
    },
    {
        'testcase_name': 'large_var',
        'array_size': 1000,
        'distribution_mean': 0.0,
        'distribution_variance': 10000.0,
        'use_weights': False
    },
    {
        'testcase_name': 'unit_normal_weighted',
        'array_size': 1000,
        'distribution_mean': 0.0,
        'distribution_variance': 1.0,
        'use_weights': True
    },
    {
        'testcase_name': 'large_array_large_mean_large_var',
        'array_size': 100000,
        'distribution_mean': 1000.0,
        'distribution_variance': 1000.0,
        'use_weights': False
    },
    {
        'testcase_name': 'small_array',
        'array_size': 10,
        'distribution_mean': 0.0,
        'distribution_variance': 1.0,
        'use_weights': False
    },
]

_RELATIVE_ERROR_TOLERANCE = 1e-6

_MEAN_COV_ACCUMULATOR_TEST_CASES = [
    {
        'testcase_name': 'unit_normal',
        'array_size': 10,
        'distribution_mean': 0.0,
        'distribution_variance': 1.0,
        'num_vectors': 1000
    },
    {
        'testcase_name': 'large_pos_shift',
        'array_size': 10,
        'distribution_mean': 100000.0,
        'distribution_variance': 1.0,
        'num_vectors': 1000
    },
    {
        'testcase_name': 'large_var',
        'array_size': 10,
        'distribution_mean': 0.0,
        'distribution_variance': 10000.0,
        'num_vectors': 1000
    },
    {
        'testcase_name': 'large_array_large_mean_large_var',
        'array_size': 100,
        'distribution_mean': 1000.0,
        'distribution_variance': 1000.0,
        'num_vectors': 1000
    },
    {
        'testcase_name': 'small_array',
        'array_size': 3,
        'distribution_mean': 0.0,
        'distribution_variance': 1.0,
        'num_vectors': 1000
    },
]


class MeanVarAccumulatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '1d_no_weights',
          'values': np.array([1, 2, 3, 4, 5]),
          'weights': None,
      }, {
          'testcase_name': '1d_weights',
          'values': np.array([1, 2, 3, 4, 5]),
          'weights': np.array([0.9, 23, 0.1, 0.1, 0.5]),
      }, {
          'testcase_name': '2d_no_weights',
          'values': np.array([[1, 2], [3, 4]]),
          'weights': None
      }, {
          'testcase_name': 'big_array_no_weights',
          'values': np.array([0, 1, 2] * 10000),
          'weights': None,
      })
  def test_initialize_from_array(self, values, weights):
    if weights is None:
      accumulator = variance_util.MeanVarAccumulator()
      accumulator.update(values)
    else:
      accumulator = variance_util.WeightedMeanVarAccumulator()
      accumulator.update(values, weights)
    expected_mean = _weighted_mean(values, weights)
    expected_variance = _weighted_variance(values, weights)
    self.assertAlmostEqual(expected_mean, accumulator.mean)
    self.assertAlmostEqual(expected_variance, accumulator.variance)

  @parameterized.named_parameters(*_MEAN_VAR_ACCUMULATOR_TEST_CASES)
  def test_merges_random_array(self, array_size, distribution_mean,
                               distribution_variance, use_weights):
    rng = np.random.default_rng(4444444)
    values = rng.standard_normal(array_size) * np.sqrt(
        distribution_variance) + distribution_mean
    weights = None
    if use_weights:
      weights = np.abs(rng.standard_normal(array_size))
    expected_mean = _weighted_mean(values, weights)
    expected_variance = _weighted_variance(values, weights)
    # Check a variety of splits of the data.
    for split in range(0, values.size, 1 + int(values.size / 100)):
      if weights is None:
        accumulator1 = variance_util.MeanVarAccumulator()
        accumulator1.update(values[:split])
        accumulator2 = variance_util.MeanVarAccumulator()
        accumulator2.update(values[split:])

      else:
        accumulator1 = variance_util.WeightedMeanVarAccumulator()
        accumulator1.update(values[:split], weights[:split])
        accumulator2 = variance_util.WeightedMeanVarAccumulator()
        accumulator2.update(values[split:], weights[split:])
      accumulator1.merge(accumulator2)
      self.assertLess(
          _rel_err(accumulator1.mean, expected_mean), _RELATIVE_ERROR_TOLERANCE)
      self.assertLess(
          _rel_err(accumulator1.variance, expected_variance),
          _RELATIVE_ERROR_TOLERANCE)

  @parameterized.named_parameters(*_MEAN_VAR_ACCUMULATOR_TEST_CASES)
  def test_update_random_array(self, array_size, distribution_mean,
                               distribution_variance, use_weights):
    rng = np.random.default_rng(4444444)
    values = rng.standard_normal(array_size) * np.sqrt(
        distribution_variance) + distribution_mean
    weights = None
    if use_weights:
      weights = np.abs(rng.standard_normal(array_size))
    expected_mean = _weighted_mean(values, weights)
    expected_variance = _weighted_variance(values, weights)
    if weights is None:
      accumulator = variance_util.MeanVarAccumulator()
      accumulator.update(values)
    else:
      accumulator = variance_util.WeightedMeanVarAccumulator()
      accumulator.update(values, weights)

    # Iterate over chunks updating - array_size should be divisible by 10.
    batch_size = 10
    for idx in range(0, values.size, batch_size):
      if weights is None:
        accumulator.update(values[idx:idx + batch_size])
      else:
        accumulator.update(values[idx:idx + batch_size],
                           weights[idx:idx + batch_size])
    self.assertLess(
        _rel_err(accumulator.mean, expected_mean), _RELATIVE_ERROR_TOLERANCE)
    self.assertLess(
        _rel_err(accumulator.variance, expected_variance),
        _RELATIVE_ERROR_TOLERANCE)

  def test_combines_empty_non_empty(self):
    accumulator1 = variance_util.MeanVarAccumulator()
    accumulator2 = variance_util.MeanVarAccumulator()
    accumulator2.update(np.array([1, 1, 1]))
    accumulator1.merge(accumulator2)
    self.assertEqual(accumulator1.mean, 1)
    self.assertEqual(accumulator1.variance, 0)

  def test_combines_non_empty_empty(self):
    accumulator1 = variance_util.MeanVarAccumulator()
    accumulator2 = variance_util.MeanVarAccumulator()
    accumulator2.update(np.array([1, 1, 1]))
    accumulator2.merge(accumulator1)
    self.assertEqual(accumulator2.mean, 1)
    self.assertEqual(accumulator2.variance, 0)

  def test_combines_two_empty(self):
    accumulator1 = variance_util.MeanVarAccumulator()
    accumulator2 = variance_util.MeanVarAccumulator()
    accumulator1.merge(accumulator2)
    self.assertEqual(accumulator1.mean, 0)
    self.assertEqual(accumulator1.variance, 0)


class MeanCovAccumulatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '1d x 3',
          'vectors': np.array([[1],
                               [-6],
                               [15]]),
      }, {
          'testcase_name': '5d x 5',
          'vectors': np.array([[1, 2.4e-9, -3, 43333, 5.1],
                               [-1, 6.99, 8e12, 9, 250],
                               [15, -391746.2, -7.3, 30, 14],
                               [1000, 0.1, -1e6, 12, 49],
                               [88, -3e10, 7e-9, 0.2, 983]]),
      })
  def test_initialize_from_array(self, vectors):
    accumulator = variance_util.MeanCovAccumulator()
    accumulator.update(vectors)
    expected_mean = np.mean(vectors, axis=0)
    expected_covariance = np.cov(vectors, rowvar=False).ravel()
    actual_mean = accumulator.mean
    actual_covariance = accumulator.covariance.ravel()

    self.assertEqual(expected_mean.size, actual_mean.size)
    self.assertEqual(expected_covariance.size, actual_covariance.size)
    for expected, actual in zip(expected_mean, actual_mean):
      self.assertAlmostEqual(expected, actual)
    for expected, actual in zip(expected_covariance, actual_covariance):
      self.assertAlmostEqual(expected, actual)

  @parameterized.named_parameters(*_MEAN_COV_ACCUMULATOR_TEST_CASES)
  def test_merges_random_array(self, array_size, distribution_mean,
                               distribution_variance, num_vectors):
    rng = np.random.default_rng(4444444)
    vectors = []
    for _ in range(num_vectors):
      vector = rng.standard_normal(array_size) * np.sqrt(
          distribution_variance) + distribution_mean
      vectors.append(vector)
    vectors = np.asarray(vectors)

    expected_mean = np.mean(vectors, axis=0)
    expected_covariance = np.cov(vectors, rowvar=False).ravel()

    # Check a variety of splits of the data.
    for split in range(0, vectors.size, 1 + int(vectors.size / 100)):
      accumulator1 = variance_util.MeanCovAccumulator()
      accumulator1.update(vectors[:split])
      accumulator2 = variance_util.MeanCovAccumulator()
      accumulator2.update(vectors[split:])
      accumulator1.merge(accumulator2)
      actual_mean = accumulator1.mean
      actual_covariance = accumulator1.covariance.ravel()

      self.assertEqual(expected_mean.size, actual_mean.size)
      self.assertEqual(expected_covariance.size, actual_covariance.size)
      for expected, actual in zip(expected_mean, actual_mean):
        self.assertAlmostEqual(expected, actual)
      for expected, actual in zip(expected_covariance, actual_covariance):
        self.assertAlmostEqual(expected, actual)

  @parameterized.named_parameters(*_MEAN_COV_ACCUMULATOR_TEST_CASES)
  def test_update_random_array(self, array_size, distribution_mean,
                               distribution_variance, num_vectors):

    rng = np.random.default_rng(4444444)
    vectors = []
    for _ in range(num_vectors):
      vector = rng.standard_normal(array_size) * np.sqrt(
          distribution_variance) + distribution_mean
      vectors.append(vector)
    vectors = np.asarray(vectors)
    accumulator = variance_util.MeanCovAccumulator()

    # Iterate over chunks updating - array_size should be divisible by 10.
    batch_size = 10
    for idx in range(0, vectors.size, batch_size):
      accumulator.update(vectors[idx:idx + batch_size])

    expected_mean = np.mean(vectors, axis=0)
    expected_covariance = np.cov(vectors, rowvar=False).ravel()
    actual_mean = accumulator.mean
    actual_covariance = accumulator.covariance.ravel()

    self.assertEqual(expected_mean.size, actual_mean.size)
    self.assertEqual(expected_covariance.size, actual_covariance.size)
    for expected, actual in zip(expected_mean, actual_mean):
      self.assertAlmostEqual(expected, actual)
    for expected, actual in zip(expected_covariance, actual_covariance):
      self.assertAlmostEqual(expected, actual)

  # Checks handling for division by zero when computing covariance
  def test_single_observations(self):
    vectors1 = np.array([[1, 2, 3]])
    vectors2 = np.array([[4, 5, 6]])
    vectors3 = np.array([[7, 8, 9]])
    accumulator1 = variance_util.MeanCovAccumulator()
    accumulator2 = variance_util.MeanCovAccumulator()
    accumulator3 = variance_util.MeanCovAccumulator()

    accumulator1.update(vectors1)
    self.assertListEqual([1, 2, 3], list(accumulator1.mean))
    self.assertListEqual([0, 0, 0, 0, 0, 0, 0, 0, 0],
                         list(accumulator1.covariance.ravel()))

    accumulator1.update(vectors2)
    self.assertListEqual([2.5, 3.5, 4.5], list(accumulator1.mean))
    self.assertListEqual([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                         list(accumulator1.covariance.ravel()))

    accumulator3.update(vectors3)
    accumulator2.merge(accumulator3)
    self.assertListEqual([7, 8, 9], list(accumulator2.mean))
    self.assertListEqual([0, 0, 0, 0, 0, 0, 0, 0, 0],
                         list(accumulator2.covariance.ravel()))

    accumulator1.merge(accumulator2)
    self.assertListEqual([4, 5, 6], list(accumulator1.mean))
    self.assertListEqual([9, 9, 9, 9, 9, 9, 9, 9, 9],
                         list(accumulator1.covariance.ravel()))

  def test_combines_empty_non_empty(self):
    vectors = np.array([[-1, 3, 6],
                        [2, -5, 8],
                        [4, 7, -9]])
    accumulator1 = variance_util.MeanCovAccumulator()
    accumulator2 = variance_util.MeanCovAccumulator()
    accumulator2.update(vectors)
    accumulator1.merge(accumulator2)
    expected_mean = list(np.mean(vectors, axis=0))
    expected_covariance = list(np.cov(vectors, rowvar=False).ravel())
    actual_mean = list(accumulator1.mean)
    actual_covariance = list(accumulator1.covariance.ravel())

    self.assertListEqual(expected_mean, actual_mean)
    self.assertListEqual(expected_covariance, actual_covariance)

  def test_combines_non_empty_empty(self):
    vectors = np.array([[-1, 3, 6],
                        [2, -5, 8],
                        [4, 7, -9]])
    accumulator1 = variance_util.MeanCovAccumulator()
    accumulator2 = variance_util.MeanCovAccumulator()
    accumulator2.update(vectors)
    accumulator2.merge(accumulator1)
    expected_mean = list(np.mean(vectors, axis=0))
    expected_covariance = list(np.cov(vectors, rowvar=False).ravel())
    actual_mean = list(accumulator2.mean)
    actual_covariance = list(accumulator2.covariance.ravel())

    self.assertListEqual(expected_mean, actual_mean)
    self.assertListEqual(expected_covariance, actual_covariance)

  def test_combines_two_empty(self):
    accumulator1 = variance_util.MeanCovAccumulator()
    accumulator2 = variance_util.MeanCovAccumulator()
    accumulator1.merge(accumulator2)

    self.assertIsNone(accumulator1.mean)
    self.assertIsNone(accumulator1.covariance)


if __name__ == '__main__':
  absltest.main()
