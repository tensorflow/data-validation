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
import pyarrow as pa
from tensorflow_data_validation.utils import batch_util
from tensorflow_data_validation.utils import test_util


class BatchUtilTest(absltest.TestCase):

  def test_batch_examples(self):
    examples = [
        {
            'a': np.array([1.0, 2.0], dtype=np.float32),
            'b': np.array(['a', 'b', 'c', 'e'])
        },
        {
            'a': np.array([3.0, 4.0, 5.0], dtype=np.float32),
        },
        {
            'b': np.array(['d', 'e', 'f']),
            'd': np.array([10, 20, 30], dtype=np.int64),
        },
        {
            'b': np.array(['a', 'b', 'c'])
        },
        {
            'c': np.array(['d', 'e', 'f'])
        }
    ]
    expected_record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]], type=pa.list_(
                pa.float32())),
            pa.array([['a', 'b', 'c', 'e'], None])
        ], ['a', 'b']),
        pa.RecordBatch.from_arrays([
            pa.array([['d', 'e', 'f'], ['a', 'b', 'c']]),
            pa.array([[10, 20, 30], None], type=pa.list_(pa.int64()))
        ], ['b', 'd']),
        pa.RecordBatch.from_arrays([pa.array([['d', 'e', 'f']])], ['c']),
    ]

    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(examples, reshuffle=False)
          | batch_util.BatchExamplesToArrowRecordBatches(desired_batch_size=2))
      util.assert_that(
          result,
          test_util.make_arrow_record_batches_equal_fn(self,
                                                       expected_record_batches))


if __name__ == '__main__':
  absltest.main()
