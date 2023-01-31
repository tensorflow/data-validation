# Copyright 2022 Google LLC
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
# limitations under the License
"""Tests for artifacts_io_impl."""
import tempfile

from absl.testing import absltest
import apache_beam as beam
from tensorflow_data_validation.utils import artifacts_io_impl
from tensorflow_metadata.proto.v0 import statistics_pb2


class RecordSinkAndSourceTest(absltest.TestCase):

  def test_write_and_read_records(self):
    datasets = [
        statistics_pb2.DatasetFeatureStatisticsList(
            datasets=[statistics_pb2.DatasetFeatureStatistics(name='d1')]),
        statistics_pb2.DatasetFeatureStatisticsList(
            datasets=[statistics_pb2.DatasetFeatureStatistics(name='d2')])
    ]
    output_prefix = tempfile.mkdtemp() + '/statistics'

    with beam.Pipeline() as p:
      provider = artifacts_io_impl.get_io_provider('tfrecords')
      _ = (p | beam.Create(datasets) | provider.record_sink_impl(output_prefix))

    got = provider.record_iterator_impl(provider.glob(output_prefix))
    self.assertCountEqual(datasets, got)


if __name__ == '__main__':
  absltest.main()
