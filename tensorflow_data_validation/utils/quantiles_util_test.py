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
from absl.testing import parameterized
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


def _assert_buckets_almost_equal(test: parameterized.TestCase,
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
    expected_result = np.array(
        [1.0, 61.0, 121.0, 181.0, 241.0, 300.0], dtype=np.float32)
    q_combiner = quantiles_util.QuantilesCombiner(5, 0.00001)
    _run_quantiles_combiner_test(self, q_combiner, batches, expected_result)

  def test_quantiles_combiner_with_weights(self):
    batches = [[np.linspace(1, 100, 100), [1] * 100],
               [np.linspace(101, 200, 100), [2] * 100],
               [np.linspace(201, 300, 100), [3] * 100]]
    expected_result = np.array(
        [1.0, 111.0, 171.0, 221.0, 261.0, 300.0], dtype=np.float32)
    q_combiner = quantiles_util.QuantilesCombiner(5, 0.00001, has_weights=True)
    _run_quantiles_combiner_test(self, q_combiner, batches, expected_result)

  def test_generate_quantiles_histogram(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array(
            [1.0, 61.0, 121.0, 181.0, 241.0, 300.0], dtype=np.float32),
        total_count=300.0, num_buckets=5)
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
        quantiles=np.array([1.0, 61.0, 121.0, 181.0, 241.0, 301.0, 360.0],
                           dtype=np.float32),
        total_count=360.0, num_buckets=3)
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

  def test_generate_equi_width_histogram(self):
    result = quantiles_util.generate_equi_width_histogram(
        quantiles=np.array([0, 1, 5, 10, 15, 20, 24], dtype=np.float32),
        finite_min=0, finite_max=24, total_count=18, num_buckets=3)
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

  def test_find_median(self):
    self.assertEqual(quantiles_util.find_median([5.0]), 5.0)
    self.assertEqual(quantiles_util.find_median([3.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0, 6.0]), 4.5)


_EQUI_WIDTH_BUCKETS_TESTS = [
    {
        'testcase_name': 'finite_values_integer_boundaries',
        'quantiles': [0, 1.0, 5.0, 10.0, 15.0, 20.0, 24.0],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 18,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(0, 8.0, 7.8),
                             quantiles_util.Bucket(8.0, 16.0, 4.8),
                             quantiles_util.Bucket(16.0, 24.0, 5.4)],
    },
    {
        'testcase_name': 'finite_values_float_boundaries',
        'quantiles': [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
        'finite_min': 1,
        'finite_max': 5,
        'total_count': 6,
        'num_buckets': 3,
        'expected_buckets': [
            quantiles_util.Bucket(1.0, 2.33333333, 2.33333333),
            quantiles_util.Bucket(2.33333333, 3.66666666, 1.33333333),
            quantiles_util.Bucket(3.66666666, 5, 2.33333333)],
    },
    {
        'testcase_name': 'same_min_max',
        'quantiles': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'finite_min': 1,
        'finite_max': 1,
        'total_count': 100,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(1.0, 1.0, 100.0)],
    },
    {
        'testcase_name': 'only_neg_inf',
        'quantiles': [float('-inf')] * 10,
        'finite_min': float('inf'),
        'finite_max': float('-inf'),
        'total_count': 100,
        'num_buckets': 3,
        'expected_buckets': [
            quantiles_util.Bucket(float('-inf'), float('-inf'), 100.0)],
    },
    {
        'testcase_name': 'only_pos_inf',
        'quantiles': [float('inf')] * 10,
        'finite_min': float('inf'),
        'finite_max': float('-inf'),
        'total_count': 100,
        'num_buckets': 3,
        'expected_buckets': [
            quantiles_util.Bucket(float('inf'), float('inf'), 100.0)],
    },
    {
        'testcase_name': 'only_neg_and_pos_inf',
        'quantiles': [float('-inf')] * 5 + [float('inf')] * 5,
        'finite_min': float('inf'),
        'finite_max': float('-inf'),
        'total_count': 100,
        'num_buckets': 3,
        'expected_buckets': [
            quantiles_util.Bucket(float('-inf'), float('-inf'), 50.0),
            quantiles_util.Bucket(float('inf'), float('inf'), 50.0)],
    },
    {
        'testcase_name': 'finite_min_max_in_quantile_boundaries',
        'quantiles': [
            float('-inf'), 0, 1.0, 5.0, 10.0, 15.0, 20.0, 24.0, float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 18,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 8.0, 8.1),
                             quantiles_util.Bucket(8.0, 16.0, 3.6),
                             quantiles_util.Bucket(16.0, float('inf'), 6.3)],
    },
    {
        'testcase_name': 'finite_min_max_in_quantile_boundaries_multiple_inf',
        'quantiles': [
            float('-inf'), float('-inf'), float('-inf'), 0, 1.0, 5.0, 10.0,
            15.0, 20.0, 24.0, float('inf'), float('inf'), float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 27,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 8.0, 12.6),
                             quantiles_util.Bucket(8.0, 16.0, 3.6),
                             quantiles_util.Bucket(16.0, float('inf'), 10.8)],
    },
    {
        'testcase_name': 'no_finite_min_max_in_quantile_boundaries',
        'quantiles': [
            float('-inf'), 1.0, 5.0, 10.0, 15.0, 20.0, float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 18,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 8.0, 7.8),
                             quantiles_util.Bucket(8.0, 16.0, 4.8),
                             quantiles_util.Bucket(16.0, float('inf'), 5.4)],
    },
    {
        'testcase_name':
            'no_finite_min_max_in_quantile_boundaries_multiple_inf',
        'quantiles': [
            float('-inf'), float('-inf'), float('-inf'), 1.0, 5.0, 10.0,
            15.0, 20.0, float('inf'), float('inf'), float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 27,
        'num_buckets': 3,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 8.0, 12.42),
                             quantiles_util.Bucket(8.0, 16.0, 4.32),
                             quantiles_util.Bucket(16.0, float('inf'), 10.26)],
    },
    {
        'testcase_name': 'fewer_finite_boundaries_than_buckets',
        'quantiles': [
            float('-inf'), float('-inf'), float('-inf'), 0, 12.0, 18.0, 24.0,
            float('inf'), float('inf'), float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 27,
        'num_buckets': 6,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 4.0, 10.0),
                             quantiles_util.Bucket(4.0, 8.0, 1.0),
                             quantiles_util.Bucket(8.0, 12.0, 1.0),
                             quantiles_util.Bucket(12.0, 16.0, 2.0),
                             quantiles_util.Bucket(16.0, 20.0, 2.0),
                             quantiles_util.Bucket(20.0, float('inf'), 11.0)],
    },
    {
        'testcase_name': 'fewer_finite_boundaries_than_buckets_single_bucket',
        'quantiles': [
            float('-inf'), float('-inf'), float('-inf'), 0, 24.0,
            float('inf'), float('inf'), float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 42,
        'num_buckets': 6,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 4.0, 19.0),
                             quantiles_util.Bucket(4.0, 8.0, 1.0),
                             quantiles_util.Bucket(8.0, 12.0, 1.0),
                             quantiles_util.Bucket(12.0, 16.0, 1.0),
                             quantiles_util.Bucket(16.0, 20.0, 1.0),
                             quantiles_util.Bucket(20.0, float('inf'), 19.0)],
    },
    {
        'testcase_name': 'same_finite_boundaries_as_buckets',
        'quantiles': [
            float('-inf'), float('-inf'), float('-inf'), 0, 6.0, 12.0, 18.0,
            24.0, float('inf'), float('inf'), float('inf')],
        'finite_min': 0,
        'finite_max': 24,
        'total_count': 30,
        'num_buckets': 4,
        'expected_buckets': [quantiles_util.Bucket(float('-inf'), 6.0, 12.0),
                             quantiles_util.Bucket(6.0, 12.0, 3.0),
                             quantiles_util.Bucket(12.0, 18.0, 3.0),
                             quantiles_util.Bucket(18.0, float('inf'), 12.0)],
    },
]


class GenerateEquiWidthBucketsTest(parameterized.TestCase):

  @parameterized.named_parameters(*_EQUI_WIDTH_BUCKETS_TESTS)
  def test_generate_equi_width_buckets(
      self, quantiles, finite_min, finite_max, total_count, num_buckets,
      expected_buckets):
    _assert_buckets_almost_equal(
        self, quantiles_util.generate_equi_width_buckets(
            quantiles, finite_min, finite_max, total_count, num_buckets),
        expected_buckets)

  def test_generate_equi_width_buckets_unsorted_quantiles(self):
    with self.assertRaisesRegexp(AssertionError,
                                 'Quantiles output not sorted.*'):
      quantiles_util.generate_equi_width_buckets([1, 2, 1, 3], 1, 3, 10, 2)

if __name__ == '__main__':
  absltest.main()
