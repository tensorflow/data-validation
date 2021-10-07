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
# limitations under the License
"""IO utilities."""

import os
import pickle
from typing import Any, Iterator, List, Text, Union
import uuid

import apache_beam as beam
import tensorflow as tf


def write_string_to_file(filename: Text, file_content: Text) -> None:
  """Writes a string to a given file.

  Args:
    filename: path to a file.
    file_content: contents that need to be written to the file.
  """
  with tf.io.gfile.GFile(filename, mode="w") as f:
    f.write(file_content)


def read_file_to_string(filename: Text,
                        binary_mode: bool = False) -> Union[Text, bytes]:
  """Reads the entire contents of a file to a string.

  Args:
    filename: path to a file
    binary_mode: whether to open the file in binary mode or not. This changes
      the type of the object returned.

  Returns:
    contents of the file as a string or bytes.
  """
  if binary_mode:
    f = tf.io.gfile.GFile(filename, mode="rb")
  else:
    f = tf.io.gfile.GFile(filename, mode="r")
  return f.read()


@beam.ptransform_fn
def _serialize_and_write_fn(pcoll, output_path):
  _ = pcoll | beam.Map(pickle.dumps) | beam.io.WriteToTFRecord(output_path)


class Materializer(object):
  """Helper to allow materialization of PCollection contents.

  Materializer is intended to simplify retrieving PCollection contents into
  memory. Internally it is backed by tmp files written to the provided
  directory, which must already exist. To use a Materializer:

  m = Materializer(my_path)
  with beam.Pipeline() as p:
    p | SomeOperation(...) | m.writer()

  Then, once the pipeline is run

  for item in m.reader():
    ...

  m.cleanup()

  Or to use as a context manager with automated cleanup:

  with Materializer(my_path) as m:
    with beam.Pipeline() as p:
      p | SomeOperation(...) | m.writer()
    for item in m.reader():
      ...

  The contents of the PCollection passed to writer() must be serializable with
  pickle.
  """

  def __init__(self, output_dir: str):
    self._output_path = os.path.join(
        output_dir, "%s_tmp_materialized.tfrecords" % uuid.uuid4())
    self._deleted = False

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    self.cleanup()
    return False

  def writer(self) -> beam.PTransform:
    """Retrieve a PSink writing to a temporary file path."""
    if self._deleted:
      raise ValueError("Materializer must not be used after cleanup.")

    # TODO(b/68154497): Relint
    # pylint: disable=no-value-for-parameter
    return _serialize_and_write_fn(self._output_path)
    # pylint: enable=no-value-for-parameter

  def _output_files(self) -> List[Union[bytes, str]]:
    return tf.io.gfile.glob(self._output_path + "-*-of-*")

  def reader(self) -> Iterator[Any]:
    """Get an iterator over output written to writer().

    This function depends on the pipeline being run.

    Returns:
      An iterator yielding:
        Contents of the PCollection passed to writer().
    """
    if self._deleted:
      raise ValueError("Materializer must not be used after cleanup.")
    def _iter():
      for path in self._output_files():
        for record in tf.compat.v1.io.tf_record_iterator(path):
          yield pickle.loads(record)
    return _iter()

  def cleanup(self):
    """Deletes files backing this Materializer."""
    if self._deleted:
      raise ValueError("Materializer must not be used after cleanup.")
    for path in self._output_files():
      tf.io.gfile.remove(path)
    self._deleted = True
