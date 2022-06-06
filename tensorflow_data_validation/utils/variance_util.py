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
"""Utilities for calculating numerically stable variance."""

from typing import Optional

import numpy as np


class WeightedMeanVarAccumulator(object):
  """Tracks quantities for numerically stable mean and variance calculation."""
  __slots__ = ['count', 'mean', 'variance', 'weights_mean']

  def __init__(self):
    self.count = 0
    self.mean = 0.0
    self.variance = 0.0
    self.weights_mean = 0.0

  def update(self, array: np.ndarray, weights: np.ndarray):
    """Updates a WeightedMeanVarAccumulator with a batch of values and weights.

    Args:
      array: An ndarray with numeric type.
      weights: An weight array. It must have the same shape as `array`.

    Raises:
      ValueError: If weights and values have incompatible shapes, or if called
      on an unweighted accumulator.
    """
    array = array.astype(np.float64)
    combined_count = array.size
    if combined_count == 0:
      return
    if not np.array_equal(array.shape, weights.shape):
      raise ValueError('incompatible weights shape')
    weights = weights.astype(np.float64)
    weights_mean = np.mean(weights)
    if weights_mean == 0:
      self.count += combined_count
      return

    mean = np.sum(weights * array) / (combined_count * weights_mean)
    variance = np.sum(weights * (array - mean)**2) / (
        combined_count * weights_mean)
    self._combine(combined_count, mean, variance, weights_mean)

  def merge(self, other: 'WeightedMeanVarAccumulator'):
    """Combines two WeightedMeanVarAccumulators, updating in place.

    Args:
      other: A MeanVarAccumulator to merge with self.
    """
    self._combine(other.count, other.mean, other.variance, other.weights_mean)

  def _combine(self, b_count: int, b_mean: float, b_variance: float,
               b_weights_mean: float):
    """Combine weighted mean and variance parameters, updating in place."""

    a_count = self.count
    a_mean = self.mean
    a_variance = self.variance
    a_weights_mean = self.weights_mean

    new_count = a_count + b_count
    new_weight_sum = a_count * a_weights_mean + b_count * b_weights_mean
    if new_count == 0 or new_weight_sum == 0:
      return
    # In the case of very inbalanced sizes we prefer ratio ~= 0
    if b_count * b_weights_mean > a_count * a_weights_mean:
      a_count, b_count = b_count, a_count
      a_mean, b_mean = b_mean, a_mean
      a_variance, b_variance = b_variance, a_variance
      a_weights_mean, b_weights_mean = b_weights_mean, a_weights_mean
    ratio = b_count * b_weights_mean / new_weight_sum
    new_weights_mean = a_weights_mean + b_count / new_count * (
        b_weights_mean - a_weights_mean)
    new_mean = a_mean + ratio * (b_mean - a_mean)
    var = a_variance + ratio * (
        b_variance - a_variance + (b_mean - new_mean) *
        (b_mean - a_mean))
    self.count = new_count
    self.mean = new_mean
    self.variance = var
    self.weights_mean = new_weights_mean


class MeanVarAccumulator(object):
  """Tracks quantities for numerically stable mean and variance calculation."""
  __slots__ = ['count', 'mean', 'variance']

  def __init__(self):
    self.count = 0
    self.mean = 0.0
    self.variance = 0.0

  def update(self, array: np.ndarray):
    """Updates a MeanVarAccumulator with a batch of values.

    Args:
      array: An ndarray with numeric type.

    Raises:
      ValueError: If called on a weighted accumulator.
    """
    array = array.astype(np.float64)
    count = array.size
    if count == 0:
      return
    mean = np.mean(array)
    variance = np.var(array)
    self._combine(count, mean, variance)

  def merge(self, other: 'MeanVarAccumulator'):
    """Combines two MeanVarAccumulator, updating in place.

    Args:
      other: A MeanVarAccumulator to merge with self.
    """
    self._combine(other.count, other.mean, other.variance)

  def _combine(self, b_count: int, b_mean: float,
               b_variance: float):
    """Combine unweighted mean and variance parameters, updating accumulator."""
    # In the case of very imbalanced sizes we prefer ratio ~= 0
    a_count, a_mean, a_variance = self.count, self.mean, self.variance
    if b_count > a_count:
      a_count, b_count = b_count, a_count
      a_mean, b_mean = b_mean, a_mean
      a_variance, b_variance = b_variance, a_variance
    new_count = a_count + b_count
    if new_count == 0:
      return
    ratio = b_count / new_count
    new_mean = a_mean + ratio * (b_mean - a_mean)
    new_variance = a_variance + ratio * (
        b_variance - a_variance + (b_mean - new_mean) *
        (b_mean - a_mean))
    self.count = new_count
    self.mean = new_mean
    self.variance = new_variance


class MeanCovAccumulator(object):
  """Tracks values for numerically stable mean and covariance calculation."""
  __slots__ = ['count', 'mean', 'covariance']

  def __init__(self):
    self.count = 0
    self.mean = None
    self.covariance = None

  def update(self, array: np.ndarray):
    """Updates a MeanCovAccumulator with a batch of values.

    Args:
      array: An ndarray with numeric type.
    """
    count = len(array)
    if count == 0:
      return
    elif count == 1:
      dim = array[0].size
      covariance = np.zeros((dim, dim), dtype=np.float64)
    else:
      covariance = np.cov(array, rowvar=False)
    mean = np.mean(array, axis=0)
    self._combine(count, mean, covariance)

  def merge(self, other: 'MeanCovAccumulator'):
    """Combines two MeanCovAccumulator, updating in place.

    Args:
      other: A MeanCovAccumulator to merge with self.
    """
    self._combine(other.count, other.mean, other.covariance)

  def _combine(self, b_count: int, b_mean: Optional[np.ndarray],
               b_covariance: Optional[np.ndarray]):
    """Combine unweighted mean and covariance parameters, updating accumulator."""
    a_count, a_mean, a_covariance = self.count, self.mean, self.covariance
    new_count = a_count + b_count
    if new_count == a_count:
      return
    elif new_count == b_count:
      # Avoid division by zero, which would happen otherwise if b_count=1
      new_mean = b_mean
      new_covariance = b_covariance
    else:
      if a_mean is None:
        a_mean = np.zeros((np.shape(b_mean)), dtype=np.float64)
      if a_covariance is None:
        a_covariance = np.zeros((np.shape(b_covariance)), dtype=np.float64)
      ratio = b_count / new_count
      new_mean = a_mean + ratio * (b_mean - a_mean)
      new_covariance = (a_covariance * (a_count - 1) +
                        b_covariance * (b_count - 1) +
                        (np.outer(
                            (a_mean - new_mean),
                            (a_mean - new_mean) * a_count)) +
                        (np.outer(
                            (b_mean - new_mean),
                            (b_mean - new_mean) * b_count))) / (new_count - 1)
    self.count = new_count
    self.mean = new_mean
    self.covariance = new_covariance
