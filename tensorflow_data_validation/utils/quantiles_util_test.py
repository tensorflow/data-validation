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

"""Tests for quantile utilities."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.utils import quantiles_util
from typing import List, Tuple

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


def _run_quantiles_combiner_test(test: absltest.TestCase,
                                 q_combiner: quantiles_util.QuantilesCombiner,
                                 batches: List[List[np.ndarray]],
                                 expected_result: np.ndarray):
  """Tests quantiles combiner."""
  summaries = [q_combiner.add_input(q_combiner.create_accumulator(),
                                    batch) for batch in batches]
  result = q_combiner.extract_output(q_combiner.merge_accumulators(summaries))
  test.assertEqual(result.dtype, expected_result.dtype)
  test.assertEqual(result.size, expected_result.size)
  for i in range(expected_result.size):
    test.assertAlmostEqual(result[i], expected_result[i])


def _assert_buckets_almost_equal(test: absltest.TestCase,
                                 a: List[Tuple[float, float, float]],
                                 b: List[Tuple[float, float, float]]):
  """Check if the histogram buckets are almost equal."""
  test.assertEqual(len(a), len(b))
  for i in range(len(a)):
    test.assertAlmostEqual(a[i].low_value, b[i].low_value)
    test.assertAlmostEqual(a[i].high_value, b[i].high_value)
    test.assertAlmostEqual(a[i].sample_count, b[i].sample_count)


class QuantilesUtilTest(absltest.TestCase):

  def test_quantiles_combiner(self):
    batches = [[np.linspace(1, 100, 100)],
               [np.linspace(101, 200, 100)],
               [np.linspace(201, 300, 100)]]
    expected_result = np.array([61.0, 121.0, 181.0, 241.0], dtype=np.float32)
    q_combiner = quantiles_util.QuantilesCombiner(5, 0.00001)
    _run_quantiles_combiner_test(self, q_combiner, batches, expected_result)

  def test_quantiles_combiner_with_weights(self):
    batches = [[np.linspace(1, 100, 100), [1] * 100],
               [np.linspace(101, 200, 100), [2] * 100],
               [np.linspace(201, 300, 100), [3] * 100]]
    expected_result = np.array([111.0, 171.0, 221.0, 261.0], dtype=np.float32)
    q_combiner = quantiles_util.QuantilesCombiner(5, 0.00001, has_weights=True)
    _run_quantiles_combiner_test(self, q_combiner, batches, expected_result)

  def test_generate_quantiles_histogram(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([61.0, 121.0, 181.0, 241.0], dtype=np.float32),
        min_val=1.0, max_val=300.0, total_count=300.0, num_buckets=5)
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 61.0
          sample_count: 60.0
        }
        buckets {
          low_value: 61.0
          high_value: 121.0
          sample_count: 60.0
        }
        buckets {
          low_value: 121.0
          high_value: 181.0
          sample_count: 60.0
        }
        buckets {
          low_value: 181.0
          high_value: 241.0
          sample_count: 60.0
        }
        buckets {
          low_value: 241.0
          high_value: 300.0
          sample_count: 60.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    self.assertEqual(result, expected_result)

  def test_generate_quantiles_histogram_diff_num_buckets_multiple(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([61.0, 121.0, 181.0, 241.0, 301.0],
                           dtype=np.float32),
        min_val=1.0, max_val=360.0, total_count=360.0, num_buckets=3)
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 121.0
          sample_count: 120.0
        }
        buckets {
          low_value: 121.0
          high_value: 241.0
          sample_count: 120.0
        }
        buckets {
          low_value: 241.0
          high_value: 360.0
          sample_count: 120.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    self.assertEqual(result, expected_result)

  def test_generate_quantiles_histogram_diff_num_buckets_non_multiple(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([61.0, 121.0, 181.0, 241.0],
                           dtype=np.float32),
        min_val=1.0, max_val=300.0, total_count=300.0, num_buckets=4)
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 76.0
          sample_count: 75.0
        }
        buckets {
          low_value: 76.0
          high_value: 151.0
          sample_count: 75.0
        }
        buckets {
          low_value: 151.0
          high_value: 226.0
          sample_count: 75.0
        }
        buckets {
          low_value: 226.0
          high_value: 300.0
          sample_count: 75.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    self.assertEqual(result, expected_result)

  def test_generate_equi_width_histogram(self):
    result = quantiles_util.generate_equi_width_histogram(
        quantiles=np.array([1, 5, 10, 15, 20], dtype=np.float32),
        min_val=0, max_val=24.0, total_count=18, num_buckets=3)
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 0
          high_value: 8.0
          sample_count: 7.8
        }
        buckets {
          low_value: 8.0
          high_value: 16.0
          sample_count: 4.8
        }
        buckets {
          low_value: 16.0
          high_value: 24.0
          sample_count: 5.4
        }
        type: STANDARD
        """, statistics_pb2.Histogram())
    self.assertEqual(result, expected_result)

  def test_generate_equi_width_buckets(self):
    _assert_buckets_almost_equal(
        self, quantiles_util.generate_equi_width_buckets(
            quantiles=[1.0, 5.0, 10.0, 15.0, 20.0],
            min_val=0, max_val=24.0, total_count=18, num_buckets=3),
        [quantiles_util.Bucket(0, 8.0, 7.8),
         quantiles_util.Bucket(8.0, 16.0, 4.8),
         quantiles_util.Bucket(16.0, 24.0, 5.4)])

    _assert_buckets_almost_equal(
        self, quantiles_util.generate_equi_width_buckets(
            quantiles=[1.0, 2.0, 3.0, 4.0, 5.0],
            min_val=1.0, max_val=5.0, total_count=6, num_buckets=3),
        [quantiles_util.Bucket(1.0, 2.33333333, 2.33333333),
         quantiles_util.Bucket(2.33333333, 3.66666666, 1.33333333),
         quantiles_util.Bucket(3.66666666, 5, 2.33333333)])

    _assert_buckets_almost_equal(
        self, quantiles_util.generate_equi_width_buckets(
            quantiles=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            min_val=1.0, max_val=1.0, total_count=100, num_buckets=3),
        [quantiles_util.Bucket(1.0, 1.0, 100.0)])

  def test_find_median(self):
    self.assertEqual(quantiles_util.find_median([5.0]), 5.0)
    self.assertEqual(quantiles_util.find_median([3.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0, 6.0]), 4.5)


if __name__ == '__main__':
  absltest.main()
