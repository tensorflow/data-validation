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
"""Tests for StatsOptions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_data_validation.statistics import stats_options

INVALID_STATS_OPTIONS = [
    {
        'testcase_name': 'invalid_generators',
        'stats_options_kwargs': {
            'generators': {}
        },
        'exception_type': TypeError,
        'error_message': 'generators is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_generator',
        'stats_options_kwargs': {
            'generators': [{}]
        },
        'exception_type': TypeError,
        'error_message': 'Statistics generator must extend one of '
                         'CombinerStatsGenerator, TransformStatsGenerator, '
                         'or CombinerFeatureStatsGenerator '
                         'found object of type dict.'
    },
    {
        'testcase_name': 'invalid_feature_whitelist',
        'stats_options_kwargs': {
            'feature_whitelist': {}
        },
        'exception_type': TypeError,
        'error_message': 'feature_whitelist is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_schema',
        'stats_options_kwargs': {
            'schema': {}
        },
        'exception_type': TypeError,
        'error_message': 'schema is of type dict, should be a Schema proto.'
    },
    {
        'testcase_name': 'invalid_slice_functions_list',
        'stats_options_kwargs': {
            'slice_functions': {}
        },
        'exception_type': TypeError,
        'error_message': 'slice_functions is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_slice_function_type',
        'stats_options_kwargs': {
            'slice_functions': [1]
        },
        'exception_type': TypeError,
        'error_message': 'slice_functions must contain functions only.'
    },
    {
        'testcase_name': 'sample_count_zero',
        'stats_options_kwargs': {
            'sample_count': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_count 0'
    },
    {
        'testcase_name': 'sample_count_negative',
        'stats_options_kwargs': {
            'sample_count': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_count -1'
    },
    {
        'testcase_name': 'both_sample_count_and_sample_rate',
        'stats_options_kwargs': {
            'sample_count': 100,
            'sample_rate': 0.5
        },
        'exception_type': ValueError,
        'error_message': 'Only one of sample_count or sample_rate can be '
                         'specified.'
    },
    {
        'testcase_name': 'sample_rate_zero',
        'stats_options_kwargs': {
            'sample_rate': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate 0'
    },
    {
        'testcase_name': 'sample_rate_negative',
        'stats_options_kwargs': {
            'sample_rate': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate -1'
    },
    {
        'testcase_name': 'sample_rate_above_one',
        'stats_options_kwargs': {
            'sample_rate': 2
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate 2'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_one',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': 1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets 1'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_zero',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets 0'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets -1'
    },
    {
        'testcase_name': 'num_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_histogram_buckets -1'
    },
    {
        'testcase_name': 'num_quantiles_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_quantiles_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_quantiles_histogram_buckets -1'
    },
    {
        'testcase_name': 'desired_batch_size_zero',
        'stats_options_kwargs': {
            'desired_batch_size': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid desired_batch_size 0'
    },
    {
        'testcase_name': 'desired_batch_size_negative',
        'stats_options_kwargs': {
            'desired_batch_size': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid desired_batch_size -1'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_zero',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate 0'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_negative',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate -1'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_above_one',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': 2
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate 2'
    },
]


class StatsOptionsTest(parameterized.TestCase):

  @parameterized.named_parameters(*INVALID_STATS_OPTIONS)
  def test_stats_options(self, stats_options_kwargs, exception_type,
                         error_message):
    with self.assertRaisesRegexp(exception_type, error_message):
      stats_options.StatsOptions(**stats_options_kwargs)


if __name__ == '__main__':
  absltest.main()
