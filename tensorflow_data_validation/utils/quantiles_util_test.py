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
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_data_validation.utils import quantiles_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


def _assert_buckets_almost_equal(test: parameterized.TestCase,
                                 a: List[statistics_pb2.Histogram.Bucket],
                                 b: List[statistics_pb2.Histogram.Bucket]):
  """Check if the histogram buckets are almost equal."""
  test.assertEqual(len(a), len(b))
  for i in range(len(a)):
    test.assertEqual(a[i], b[i])
    test.assertAlmostEqual(a[i].low_value, b[i].low_value)
    test.assertAlmostEqual(a[i].high_value, b[i].high_value)
    test.assertAlmostEqual(a[i].sample_count, b[i].sample_count)


class QuantilesUtilTest(absltest.TestCase):

  def test_generate_quantiles_histogram(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([1.0, 60.0, 120.0, 180.0, 240.0, 300.0],
                           dtype=np.float32),
        cumulative_counts=np.array([1, 60, 120, 180, 240, 300]))
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 60.0
          sample_count: 60.0
        }
        buckets {
          low_value: 60.0
          high_value: 120.0
          sample_count: 60.0
        }
        buckets {
          low_value: 120.0
          high_value: 180.0
          sample_count: 60.0
        }
        buckets {
          low_value: 180.0
          high_value: 240.0
          sample_count: 60.0
        }
        buckets {
          low_value: 240.0
          high_value: 300.0
          sample_count: 60.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    _assert_buckets_almost_equal(self, result.buckets, expected_result.buckets)

  def test_all_duplicates(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([1, 1, 1], dtype=np.float32),
        cumulative_counts=np.array([2, 2, 2]))
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 1.0
          sample_count: 1.0
        }
        buckets {
          low_value: 1.0
          high_value: 1.0
          sample_count: 1.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    _assert_buckets_almost_equal(self, result.buckets, expected_result.buckets)

  def test_generate_quantiles_histogram_low_bucket_partial_duplicate(self):
    # This test documents an edge case. If we generate 2 quantiles of the input
    # [1, 2] we get bin boundaries [1, 2, 2].
    # Because bins include their upper bound *and* the first bin includes its
    # lower bound, the first bin includes 1, 2 while the second includes 2.
    # So we split the count for 2 across the overlapping bins.
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([1, 2, 2], dtype=np.float32),
        cumulative_counts=np.array([1, 2, 2]))
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 1.5
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 0.5
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    _assert_buckets_almost_equal(self, result.buckets, expected_result.buckets)

  def test_generate_quantiles_histogram_duplicate_buckets(self):
    result = quantiles_util.generate_quantiles_histogram(
        quantiles=np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0],
                           dtype=np.float32),
        cumulative_counts=np.array([1, 34, 34, 34, 51, 51, 60]))
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 12.0
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 11.0
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 11.0
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 8.5
        }
        buckets {
          low_value: 3.0
          high_value: 3.0
          sample_count: 8.5
        }
        buckets {
          low_value: 3.0
          high_value: 4.0
          sample_count: 9.0
        }
        type: QUANTILES
        """, statistics_pb2.Histogram())
    _assert_buckets_almost_equal(self, result.buckets, expected_result.buckets)

  def test_generate_equi_width_histogram(self):
    expected_result = text_format.Parse(
        """
        buckets {
          low_value: 1.0
          high_value: 2.5
          sample_count: 38.25
        }
        buckets {
          low_value: 2.5
          high_value: 4.0
          sample_count: 21.75
        }
        type: STANDARD
        """, statistics_pb2.Histogram())
    result = quantiles_util.generate_equi_width_histogram(
        quantiles=np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0],
                           dtype=np.float32),
        cumulative_counts=np.array([1, 34, 34, 34, 51, 51, 60]),
        finite_min=1,
        finite_max=4,
        num_buckets=2,
        num_pos_inf=0)
    self.assertEqual(result, expected_result)

  def test_find_median(self):
    self.assertEqual(quantiles_util.find_median([5.0]), 5.0)
    self.assertEqual(quantiles_util.find_median([3.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0]), 4.0)
    self.assertEqual(quantiles_util.find_median([3.0, 4.0, 5.0, 6.0]), 4.5)


def _bucket(low, high, sample) -> statistics_pb2.Histogram.Bucket:
  return statistics_pb2.Histogram.Bucket(
      low_value=low, high_value=high, sample_count=sample)


_EQUI_WIDTH_BUCKETS_TESTS = [
    {
        'testcase_name': 'finite_values_integer_boundaries',
        'quantiles': [1, 2, 3, 4, 5, 7],
        'cumulative_counts': [2, 5, 7, 10, 12, 15],
        'finite_min': 1,
        'finite_max': 7,
        'num_buckets': 2,
        'num_pos_inf': 0,
        'expected_buckets': [
            _bucket(1, 4, 10),
            _bucket(4, 7, 5),
        ],
    },
    {
        'testcase_name':
            'finite_values_fractional_boundaries',
        'quantiles': [1, 2, 3, 4, 5, 7],
        'cumulative_counts': [2, 5, 7, 10, 12, 15],
        'finite_min':
            1,
        'finite_max':
            7,
        'num_buckets':
            4,
        'num_pos_inf':
            0,
        'expected_buckets': [
            _bucket(1.0, 2.5, 6.0),
            _bucket(2.5, 4.0, 4.0),
            _bucket(4.0, 5.5, 2.75),
            _bucket(5.5, 7.0, 2.25),
        ],
    },
    {
        'testcase_name': 'finite_values_one_bucket',
        'quantiles': [1, 2, 3, 4, 5, 7],
        'cumulative_counts': [2, 5, 7, 10, 12, 15],
        'finite_min': 1,
        'finite_max': 7,
        'num_buckets': 1,
        'num_pos_inf': 0,
        'expected_buckets': [_bucket(1.0, 7.0, 15.0),],
    },
    {
        'testcase_name': 'single_finite_value',
        'quantiles': [5, 5, 5, 5, 5],
        'cumulative_counts': [3, 3, 3, 3, 3],
        'finite_min': 5,
        'finite_max': 5,
        'num_buckets': 1,
        'num_pos_inf': 0,
        'expected_buckets': [_bucket(5.0, 5.0, 3.0),],
    },
    {
        'testcase_name':
            'leading_negative_inf',
        'quantiles': [float('-inf'), float('-inf'), 1, 2, 3],
        'cumulative_counts': [5, 7, 10, 12, 15],
        'finite_min':
            1,
        'finite_max':
            3,
        'num_buckets':
            4,
        'num_pos_inf':
            0,
        'expected_buckets': [
            _bucket(float('-inf'), float('-inf'), 7),
            _bucket(1, 1.5, 2.5),
            _bucket(1.5, 2, 2.5),
            _bucket(2, 2.5, 1.5),
            _bucket(2.5, 3, 1.5),
        ],
    },
    {
        'testcase_name':
            'trailing_inf',
        'quantiles': [1, 2, 3, float('inf'),
                      float('inf')],
        'cumulative_counts': [3, 5, 6, 7, 8],
        'finite_min':
            1,
        'finite_max':
            4,
        'num_buckets':
            2,
        'num_pos_inf':
            0.5,
        'expected_buckets': [
            _bucket(1, 2.5, 5.5),
            _bucket(2.5, 4, 2),
            _bucket(float('inf'), float('inf'), 0.5),
        ],
    },
    {
        'testcase_name':
            'single_finite_between_inf',
        'quantiles': [float('-inf'), 1, float('inf')],
        'cumulative_counts': [3, 5, 9],
        'finite_min':
            1,
        'finite_max':
            1,
        'num_buckets':
            99,
        'num_pos_inf':
            4,
        'expected_buckets': [
            _bucket(float('-inf'), float('-inf'), 3),
            _bucket(1, 1, 2),
            _bucket(float('inf'), float('inf'), 4),
        ],
    },
    {
        'testcase_name':
            'leading_and_trailing_inf',
        'quantiles': [float('-inf'), 1, 2, 3,
                      float('inf')],
        'cumulative_counts': [3, 5, 6, 7, 8],
        'finite_min':
            1,
        'finite_max':
            4,
        'num_buckets':
            2,
        'num_pos_inf':
            0.5,
        'expected_buckets': [
            _bucket(float('-inf'), float('-inf'), 3),
            _bucket(1, 2.5, 3.5),
            _bucket(2.5, 4, 1),
            _bucket(float('inf'), float('inf'), 0.5),
        ],
    },
    {
        'testcase_name': 'all_inf',
        'quantiles': [float('-inf'), float('inf')],
        'cumulative_counts': [1, 5],
        'finite_min': float('-inf'),
        'finite_max': float('inf'),
        'num_buckets': 99,
        'num_pos_inf': 0.5,
        'expected_buckets': [_bucket(float('-inf'), float('inf'), 5),],
    },
    {
        'testcase_name':
            'float32_overflow',
        'quantiles': [-3.4e+38, 1, 3.4e+38],
        'cumulative_counts': [1, 3, 5],
        'finite_min':
            -3.4e+38,
        'finite_max':
            3.4e+38,
        'num_buckets':
            3,
        'num_pos_inf':
            0,
        'expected_buckets': [
            _bucket(-3.4e+38, -1.1333333333333332e+38, 2),
            _bucket(-1.1333333333333332e+38, 1.1333333333333336e+38,
                    1.666666666666667),
            _bucket(1.1333333333333336e+38, 3.4e+38, 1.3333333333333333)
        ],
    },
    {
        'testcase_name': 'float64_overflow',
        'quantiles': [-1.7976931348623157E+308, 0, 1.7976931348623157E+308],
        'cumulative_counts': [1, 3, 5],
        'finite_min': -1.7976931348623157E+308,
        'finite_max': 1.7976931348623157E+308,
        'num_buckets': 3,
        'num_pos_inf': 0,
        'expected_buckets': [],
    },
]


def _total_sample_count(h):
  acc = 0
  for b in h.buckets:
    acc += b.sample_count
  return acc


def _random_cdf(size):
  boundaries = np.cumsum(np.random.randint(0, 2, size=size + 1))
  counts = np.cumsum(np.random.random_sample(size=size + 1))
  return boundaries, counts


class GenerateEquiWidthBucketsTest(parameterized.TestCase):

  @parameterized.named_parameters(*_EQUI_WIDTH_BUCKETS_TESTS)
  def test_generate_equi_width_buckets(self, quantiles, cumulative_counts,
                                       finite_min, finite_max, num_buckets,
                                       num_pos_inf, expected_buckets):
    quantiles = np.array(quantiles).astype(float)
    cumulative_counts = np.array(cumulative_counts).astype(float)
    standard_hist = quantiles_util.generate_equi_width_histogram(
        quantiles, cumulative_counts, finite_min, finite_max, num_buckets,
        num_pos_inf)
    _assert_buckets_almost_equal(self, standard_hist.buckets, expected_buckets)

  def test_generate_equi_width_buckets_unsorted_quantiles(self):
    with self.assertRaises(AssertionError):
      quantiles_util.generate_equi_width_histogram(
          np.array([5, 1]), np.array([1, 2]), 1, 5, 10, 0)

  def test_total_weight_preserved_fuzz(self):
    for _ in range(5):
      for size in range(1, 20):
        bounds, counts = _random_cdf(size)
        for num_bins in range(1, 40):
          standard_hist = quantiles_util.generate_equi_width_histogram(
              bounds, counts, bounds.min(), bounds.max(), num_bins, 0.0)
          np.testing.assert_almost_equal(
              _total_sample_count(standard_hist), counts[-1])


class TestRebinQuantiles(absltest.TestCase):

  def test_rebin_factor_divides(self):
    quantiles = np.array([0, 1, 2, 3, 4])
    cum_counts = np.array([0, 1, 2, 3, 4])
    rebinned_quantiles, rebinned_counts = quantiles_util.rebin_quantiles(
        quantiles, cum_counts, 2)
    np.testing.assert_equal(rebinned_quantiles, np.array([0, 2, 4]))
    np.testing.assert_equal(rebinned_counts, np.array([0, 2, 4]))

  def test_rebin_factor_does_not_divide(self):
    quantiles = np.array([0, 1, 2, 3, 4])
    cum_counts = np.array([0, 1, 2, 3, 4])
    with self.assertRaises(ValueError):
      _ = quantiles_util.rebin_quantiles(quantiles, cum_counts, 3)


if __name__ == '__main__':
  absltest.main()
