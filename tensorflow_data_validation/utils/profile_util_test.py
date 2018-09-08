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
"""Tests for profile utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation.utils import profile_util


class ProfileUtilTest(absltest.TestCase):

  def test_profile_input(self):
    examples = [
        {
            'a': np.array([1.0, 2.0], dtype=np.floating),
            'b': np.array(['a', 'b', 'c', 'e'], dtype=np.object),
        },
        {
            'a': np.array([3.0, 4.0, np.NaN, 5.0], dtype=np.floating),
        },
        {
            'b': np.array(['d', 'e', 'f'], dtype=np.object),
            'd': np.array([10, 20, 30], dtype=np.integer),
        },
        {
            'b': np.array(['a', 'b', 'c'], dtype=np.object),
        },
        {
            'c': np.array(['d', 'e', 'f'], dtype=np.object),
        },
    ]

    expected_distributions = {
        'int_feature_values_count': [3L, 3L, 3L, 1L],
        'float_feature_values_count': [2L, 4L, 6L, 2L],
        'string_feature_values_count': [3L, 4L, 13L, 4L],
    }
    p = beam.Pipeline()
    _ = (
        p
        | 'Create' >> beam.Create(examples)
        | 'Profile' >> profile_util.Profile())

    runner = p.run()
    runner.wait_until_finish()
    result_metrics = runner.metrics()

    num_metrics = len(
        result_metrics.query(beam.metrics.metric.MetricsFilter().with_namespace(
            profile_util.METRICS_NAMESPACE))['counters'])
    self.assertEqual(num_metrics, 1)

    counter = result_metrics.query(beam.metrics.metric.MetricsFilter()
                                   .with_name('num_instances'))['counters']
    self.assertEqual(len(counter), 1)
    self.assertEqual(counter[0].committed, 5L)

    for distribution_name, expected_value in six.iteritems(
        expected_distributions):
      metric_filter = beam.metrics.metric.MetricsFilter().with_name(
          distribution_name)
      distribution = result_metrics.query(metric_filter)['distributions']
      self.assertEqual(len(distribution), 1)
      self.assertEqual([
          distribution[0].committed.min, distribution[0].committed.max,
          distribution[0].committed.sum, distribution[0].committed.count
      ], expected_value)


if __name__ == '__main__':
  absltest.main()
