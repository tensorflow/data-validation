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
"""Record sink support."""

from typing import Callable, Iterable, Iterator, Optional, Type, TypeVar

import apache_beam as beam
import tensorflow as tf
from google.protobuf import message

from tensorflow_metadata.proto.v0 import statistics_pb2


class StatisticsIOProvider(object):
  """Provides access to read and write statistics proto to record files."""

  def record_sink_impl(self,
                       output_path_prefix: str) -> beam.PTransform:
    """Gets a beam IO sink for writing sharded statistics protos."""
    raise NotImplementedError

  def record_iterator_impl(
      self,
      paths: Optional[Iterable[str]] = None,
  ) -> Iterator[statistics_pb2.DatasetFeatureStatisticsList]:
    """Get a file-backed iterator over sharded statistics protos.

    Args:
      paths: A list of file paths containing statistics records.
    """
    raise NotImplementedError

  def glob(self, output_path_prefix: str) -> Iterator[str]:
    """Return files matching the pattern produced by record_sink_impl."""
    raise NotImplementedError

  def file_suffix(self) -> str:
    """Returns a file suffix (e.g., .tfrecords)."""
    raise NotImplementedError


def get_io_provider(
    file_format: Optional[str] = None) -> StatisticsIOProvider:
  """Get a StatisticsIOProvider for writing and reading sharded stats.

  Args:
    file_format: Optional file format. Supports only tfrecords. If unset,
      defaults to tfrecords.

  Returns:
    A StatisticsIOProvider.
  """

  if file_format is None:
    file_format = 'tfrecords'
  if file_format not in ('tfrecords',):
    raise ValueError('Unrecognized file_format %s' % file_format)
  return _TFRecordProviderImpl()


class _TFRecordProviderImpl(StatisticsIOProvider):
  """TFRecord backed impl."""

  def record_sink_impl(self, output_path_prefix: str) -> beam.PTransform:
    return beam.io.WriteToTFRecord(
        output_path_prefix,
        coder=beam.coders.ProtoCoder(
            statistics_pb2.DatasetFeatureStatisticsList
        ),
    )

  def glob(self, output_path_prefix) -> Iterator[str]:
    """Returns filenames matching the output pattern of record_sink_impl."""
    return tf.io.gfile.glob(output_path_prefix + '-*-of-*')

  def record_iterator_impl(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      paths: Iterable[str],
  ) -> Iterator[statistics_pb2.DatasetFeatureStatisticsList]:
    """Provides iterators over tfrecord backed statistics."""
    for path in paths:
      for record in tf.compat.v1.io.tf_record_iterator(path):
        stats_shard = statistics_pb2.DatasetFeatureStatisticsList()
        stats_shard.ParseFromString(record)
        yield stats_shard

  def file_suffix(self) -> str:
    """Returns a file suffix (e.g., .tfrecords)."""
    return '.tfrecords'


def get_default_columnar_provider() -> Optional[StatisticsIOProvider]:
  return None


def should_write_sharded():
  return False


def feature_skew_sink(
    output_path_prefix: str, proto: Type[message.Message]
) -> beam.PTransform:
  """Sink for writing feature skew results."""
  return beam.io.WriteToTFRecord(
      output_path_prefix, coder=beam.coders.ProtoCoder(proto)
  )


_MESSAGE_TYPE = TypeVar('_MESSAGE_TYPE')  # pylint: disable=invalid-name


def default_record_reader(
    input_pattern: str,
    message_factory: Callable[[], _MESSAGE_TYPE]) -> Iterator[_MESSAGE_TYPE]:
  """TFRecord based record iterator."""
  for path in tf.io.gfile.glob(input_pattern):
    for record in tf.compat.v1.io.tf_record_iterator(path):
      m = message_factory()
      m.ParseFromString(record)
      yield m
