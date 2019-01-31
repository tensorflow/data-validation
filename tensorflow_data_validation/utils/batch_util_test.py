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
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_data_validation.utils import batch_util


class BatchUtilTest(absltest.TestCase):

  def test_batch_examples(self):
    examples = [{
        'a': np.array([1.0, 2.0], dtype=np.floating),
        'b': np.array(['a', 'b', 'c', 'e'], dtype=np.object)
    }, {
        'a': np.array([3.0, 4.0, np.NaN, 5.0], dtype=np.floating),
    }, {
        'b': np.array(['d', 'e', 'f'], dtype=np.object),
        'd': np.array([10, 20, 30], dtype=np.integer),
    }, {
        'b': np.array(['a', 'b', 'c'], dtype=np.object)
    }, {
        'c': np.array(['d', 'e', 'f'], dtype=np.object)
    }]

    expected_batched_examples = [{
        'a': np.array([np.array([1.0, 2.0]), np.array([3.0, 4.0, np.NaN, 5.0]),
                       None], dtype=np.object),
        'b': np.array([np.array(['a', 'b', 'c', 'e']), None,
                       np.array(['d', 'e', 'f'])], dtype=np.object),
        'd': np.array([np.NaN, np.NaN, np.array([10, 20, 30])], dtype=np.object)
    }, {
        'b': np.array([np.array(['a', 'b', 'c']), None], dtype=np.object),
        'c': np.array([None, np.array(['d', 'e', 'f'])], dtype=np.object)
    }]

    def _batched_example_equal_fn(expected_batched_examples):
      """Makes a matcher function for comparing batched examples."""
      # TODO(pachristopher): Find out the right way to compare the outcome with
      # the expected output.
      def _matcher(actual_batched_examples):
        self.assertEqual(
            len(actual_batched_examples), len(expected_batched_examples))
        for idx, batched_example in enumerate(actual_batched_examples):
          self.assertCountEqual(batched_example, expected_batched_examples[idx])

      return _matcher

    with beam.Pipeline() as p:
      result = (p
                | beam.Create(examples)
                | batch_util.BatchExamples(desired_batch_size=3))
      util.assert_that(
          result, _batched_example_equal_fn(expected_batched_examples))


if __name__ == '__main__':
  absltest.main()
