# Copyright 2019 Google LLC
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
"""Tests for empty_value_counter_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation.statistics.generators import empty_value_counter_generator
from tensorflow_data_validation.utils import test_util

from tensorflow_metadata.proto.v0 import statistics_pb2


class EmptyValueCounterGeneratorTest(
    test_util.CombinerFeatureStatsGeneratorTest
):

  def test_empty_value_counter_generator_for_string(self):
    input_batches = [
        pa.array([["abc"], [""]]),
        pa.array([[""], ["def"]]),
        pa.array([[""], None]),
    ]
    generator = empty_value_counter_generator.EmptyValueCounterGenerator()
    self.assertCombinerOutputEqual(
        input_batches,
        generator,
        statistics_pb2.FeatureNameStatistics(
            custom_stats=[
                statistics_pb2.CustomStatistic(name="str_empty", num=3),
            ]
        ),
    )

  def test_empty_value_counter_generator_for_ints(self):
    input_batches = [
        pa.array([[0], [-1], [10]]),
        pa.array([[0], [-1], None]),
        pa.array([[2], [-1], [-1], [100]]),
    ]
    generator = empty_value_counter_generator.EmptyValueCounterGenerator()
    self.assertCombinerOutputEqual(
        input_batches,
        generator,
        statistics_pb2.FeatureNameStatistics(
            custom_stats=[
                statistics_pb2.CustomStatistic(name="int_-1", num=4),
            ]
        ),
    )

  def test_empty_value_counter_generator_for_lists(self):
    input_batches = [
        pa.array([[[]], None, [["abc", "foo"]]]),
        pa.array([[["foo"]], None, [[]], [[]], [[]], [["", "jk", "tst"]]]),
    ]
    generator = empty_value_counter_generator.EmptyValueCounterGenerator()
    self.assertCombinerOutputEqual(
        input_batches,
        generator,
        statistics_pb2.FeatureNameStatistics(
            custom_stats=[
                statistics_pb2.CustomStatistic(name="list_empty", num=4),
            ]
        ),
    )


if __name__ == "__main__":
  absltest.main()
