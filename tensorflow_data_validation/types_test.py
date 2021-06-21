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
# limitations under the License.

"""Tests for types."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
import pyarrow as pa
from tensorflow_data_validation import types  # pylint: disable=unused-import


def _make_record_batch(num_cols, num_rows):
  columns = [
      pa.array([[b"kk"]] * num_rows, type=pa.large_list(pa.large_binary()))
      for _ in range(num_cols)
  ]
  column_names = ["col%d" % c for c in range(num_cols)]
  return pa.record_batch(columns, column_names)


class _Tracker(object):
  """A singleton to track whether _TrackedCoder.encode/decode is called."""

  _instance = None

  def reset(self):
    self.encode_called = False
    self.decode_called = False

  def __new__(cls):
    if cls._instance is None:
      cls._instance = object.__new__(cls)
      cls._instance.reset()
    return cls._instance


class _TrackedCoder(types._ArrowRecordBatchCoder):

  def encode(self, value):
    _Tracker().encode_called = True
    return super().encode(value)

  def decode(self, encoded):
    _Tracker().decode_called = True
    return super().decode(encoded)


class TypesTest(absltest.TestCase):

  def test_coder(self):
    rb = _make_record_batch(10, 10)
    coder = types._ArrowRecordBatchCoder()
    self.assertTrue(coder.decode(coder.encode(rb)).equals(rb))

  def test_coder_end_to_end(self):
    # First check that the registration is done.
    self.assertIsInstance(
        beam.coders.typecoders.registry.get_coder(pa.RecordBatch),
        types._ArrowRecordBatchCoder)
    # Then replace the registered coder with our patched one to track whether
    # encode() / decode() are called.
    beam.coders.typecoders.registry.register_coder(pa.RecordBatch,
                                                   _TrackedCoder)
    rb = _make_record_batch(1000, 1)
    def pipeline(root):
      sample = (
          root
          | beam.Create([rb] * 20)
          | beam.combiners.Sample.FixedSizeGlobally(5))

      def matcher(actual):
        self.assertLen(actual, 1)
        actual = actual[0]
        self.assertLen(actual, 5)
        for actual_rb in actual:
          self.assertTrue(actual_rb.equals(rb))

      util.assert_that(sample, matcher)

    _Tracker().reset()
    beam.runners.DirectRunner().run(pipeline)
    self.assertTrue(_Tracker().encode_called)
    self.assertTrue(_Tracker().decode_called)
    beam.coders.typecoders.registry.register_coder(pa.RecordBatch,
                                                   types._ArrowRecordBatchCoder)


if __name__ == "__main__":
  absltest.main()
