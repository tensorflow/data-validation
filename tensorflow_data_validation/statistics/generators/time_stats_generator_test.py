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
"""Tests for time_stats_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import time_stats_generator
from tensorflow_data_validation.utils import test_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


VALID_FORMATS_TESTS = [
    {
        'testcase_name': 'time_only_formats',
        'input_batch': pa.array([['23:59', '23:59:58', '23:59:58.123456']]),
        'expected_matching_formats': {
            '%H:%M': 1,
            '%H:%M:%S': 1,
            '%H:%M:%S.%f': 1,
        },
    },
    {
        'testcase_name':
            'date_only_formats',
        'input_batch':
            pa.array([[
                '2018-11-30',
                '2018/11/30',
                '20181130',
                '18-11-30',  # Will be identified as '%y-%m-%d' and '%d-%m-%y'.
                '18/11/30',  # Will be identified as '%y/%m/%d' and '%d/%m/%y'.
                '30-November-2018',
            ]]),
        'expected_matching_formats': {
            '%Y-%m-%d': 1,
            '%Y/%m/%d': 1,
            '%Y%m%d': 1,
            '%y-%m-%d': 1,
            '%d-%m-%y': 1,
            '%y/%m/%d': 1,
            '%d/%m/%y': 1,
            '%d-%B-%Y': 1,
        },
    },
    {
        'testcase_name': 'combined_formats',
        'input_batch': pa.array([[
            '2018-11-30T23:59',
            '2018/11/30 23:59',
            'Fri Nov 30 10:47:02 2018'
        ]]),
        'expected_matching_formats': {
            '%Y-%m-%dT%H:%M': 1,
            '%Y/%m/%d %H:%M': 1,
            '%a %b %d %H:%M:%S %Y': 1
        },
    },
]


class TimeStatsGeneratorValidFormatsTest(parameterized.TestCase):

  @parameterized.named_parameters(*VALID_FORMATS_TESTS)
  def test_time_stats_generator_valid_formats(self, input_batch,
                                              expected_matching_formats):
    """Tests that generator's add_input method properly counts valid formats."""
    generator = time_stats_generator.TimeStatsGenerator(values_threshold=1)
    accumulator = generator.add_input(generator.create_accumulator(),
                                      types.FeaturePath(['']),
                                      input_batch)
    self.assertDictEqual(expected_matching_formats,
                         accumulator.matching_formats)


class TimeStatsGeneratorTest(test_util.CombinerFeatureStatsGeneratorTest):

  def test_time_stats_generator_invalid_initialization_values(self):
    """Tests bad initialization values."""
    with self.assertRaises(ValueError) as context:
      time_stats_generator.TimeStatsGenerator(values_threshold=0)
      self.assertIn('TimeStatsGenerator expects a values_threshold > 0, got 0.',
                    str(context.exception))

      time_stats_generator.TimeStatsGenerator(match_ratio=1.1)
      self.assertIn('TimeStatsGenerator expects a match_ratio in (0, 1].',
                    str(context.exception))

      time_stats_generator.TimeStatsGenerator(match_ratio=0)
      self.assertIn('TimeStatsGenerator expects a match_ratio in (0, 1].',
                    str(context.exception))

  def test_time_stats_generator_empty_input(self):
    """Tests generator on empty input."""
    generator = time_stats_generator.TimeStatsGenerator()
    self.assertCombinerOutputEqual([], generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_time_stats_generator_values_threshold_check(self):
    """Tests generator values threshold."""
    # Expected to give 6 matches with the same format.
    input_batches = [
        pa.array([['2018-11-30', '2018-11-30', '2018-11-30'], ['2018-11-30']]),
        pa.array([['2018-11-30', '2018-11-30']]),
        pa.array([None, None]),
    ]
    # Try generator with values_threshold=7 (should not create stats).
    generator = time_stats_generator.TimeStatsGenerator(values_threshold=7)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

    # Try generator with values_threshold=6 (should create stats).
    generator = time_stats_generator.TimeStatsGenerator(values_threshold=6)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info',
                str="time_domain {string_format: '%Y-%m-%d'}"),
            statistics_pb2.CustomStatistic(name='time_match_ratio', num=1.0),
        ]))

  def test_time_stats_generator_utf8_check(self):
    """Tests that generator invalidates stats if there is a non-utf8 string."""
    # Absent invalidation, this is expected to give 6 matches.
    input_batches = [
        pa.array([['2018-11-30', '2018-11-30', '2018-11-30'], ['2018-11-30']]),
        pa.array([['2018-11-30', '2018-11-30']]),
        # Non utf-8 string that will invalidate the accumulator.
        pa.array([[b'\xF0']]),
    ]
    # No domain_info should be generated as the non-utf8 string should
    # invalidate the stats. Absent this type issue, these examples would
    # satisfy the specified match_ratio and values_threshold.
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.5, values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_time_stats_generator_inconsistent_type_invalidation_check(self):
    """Tests that generator invalidates stats if inconsistent types are used."""
    # Absent invalidation, this is expected to give 6 matches.
    input_batches = [
        pa.array([['2018-11-30', '2018-11-30', '2018-11-30'], ['2018-11-30']]),
        pa.array([['2018-11-30', '2018-11-30']]),
        pa.array([[1.0]]),
    ]
    # No domain_info should be generated as the incorrect type of the 1.0 value
    # should invalidate the stats. Absent this type issue, these examples would
    # satisfy the specified match_ratio and values_threshold.
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.5, values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  @mock.patch.object(time_stats_generator._PartialTimeStats, 'update')
  def test_time_stats_generator_invalidated_exits_add_input_early(
      self, mock_update):
    input_batch = pa.array([['2018-11-30']])
    generator = time_stats_generator.TimeStatsGenerator()
    accumulator = generator.create_accumulator()

    # When an accumulator is invalidated is True, it is not updated when an
    # input batch is added.
    accumulator.invalidated = True
    generator.add_input(accumulator, types.FeaturePath(['']), input_batch)
    self.assertFalse(mock_update.called)

    # When an accumulator is not invalidated, it is updated when an input batch
    # is added.
    accumulator.invalidated = False
    generator.add_input(accumulator, types.FeaturePath(['']), input_batch)
    self.assertTrue(mock_update.called)

  @mock.patch.object(time_stats_generator._PartialTimeStats, 'update')
  def test_time_stats_generator_no_values_exits_add_input_early(
      self, mock_update):
    generator = time_stats_generator.TimeStatsGenerator()
    accumulator = generator.create_accumulator()

    # The accumulator is not updated when the values list in an input batch is
    # None.
    input_batch = pa.array([None])
    generator.add_input(accumulator, types.FeaturePath(['']), input_batch)
    self.assertFalse(mock_update.called)

    # The accumulator is not updated when the values list in an input batch is
    # empty.
    input_batch = pa.array([])
    generator.add_input(accumulator, types.FeaturePath(['']), input_batch)
    self.assertFalse(mock_update.called)

    # The accumulator is updated when a non-empty input_batch is added.
    input_batch = pa.array([['2018-11-30']])
    generator.add_input(accumulator, types.FeaturePath(['']), input_batch)
    self.assertTrue(mock_update.called)

  def test_time_stats_generator_match_ratio_with_same_valid_format(self):
    """Tests match ratio where all valid values have the same format."""
    input_batches = [
        pa.array([['2018-11-30', '2018-11-30', '2018-11-30'],
                  ['2018-11-30', '2018-11-30']]),
        pa.array([['not-valid', 'not-valid', 'not-valid'],
                  ['not-valid', 'not-valid']]),
    ]
    # Try generator with match_ratio 0.51 (should not create stats).
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.51, values_threshold=5)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())
    # Try generator with match_ratio 0.49 (should create stats).
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.49, values_threshold=5)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info',
                str="time_domain {string_format: '%Y-%m-%d'}"),
            statistics_pb2.CustomStatistic(name='time_match_ratio', num=0.50),
        ]))

  def test_time_stats_generator_match_ratio_with_different_valid_formats(self):
    """Tests match ratio where valid values have different formats."""
    input_batches = [
        pa.array(
            [['2018-11-30', '2018/11/30', '20181130', '18-11-30', '18/11/30'],
             ['11-30-2018', '11/30/2018', '11302018', '11/30/18', '11/30/18']]),
    ]
    # Any single format could satisfy the match_ratio, but this should identify
    # only the most common as the time format.
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.05, values_threshold=1)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info',
                str="time_domain {string_format: '%m/%d/%y'}"),
            statistics_pb2.CustomStatistic(name='time_match_ratio', num=0.2),
        ]))

    # No single valid format satisfies the specified match_ratio, so this should
    # not create stats.
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.3, values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_time_stats_generator_no_valid_formats(self):
    """Tests that the generator handles batches that contain no valid values."""
    # None of these values is a valid format.
    input_batches = [
        pa.array([['', '2018-Nov-30', '20183011']]),
        pa.array([['all/invalid', '2018-11-30invalid']]),
        pa.array([['invalid2018-11-30', 'invalid\n2018-11-30']])
    ]
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.1, values_threshold=1)
    self.assertCombinerOutputEqual(input_batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_time_stats_generator_combined_string_formats(self):
    """Tests that the generator handles combined string formats."""
    # The combined format is the most common, since the generator should count
    # it only as the combined format and not its component parts.
    input_batches = [
        pa.array([['2018/11/30 23:59', '2018/12/01 23:59']]),
        pa.array([['2018/11/30 23:59', '23:59']]),
        pa.array([['2018/11/30', '2018/11/30']]),
    ]
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.1, values_threshold=1)
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info',
                str="time_domain {string_format: '%Y/%m/%d %H:%M'}"),
            statistics_pb2.CustomStatistic(name='time_match_ratio', num=0.5),
        ]))

  def test_time_stats_generator_integer_formats(self):
    """Tests that the generator handles integer formats."""
    # Three of values are within the valid range for Unix seconds, one is within
    # the valid range for Unix milliseconds, and the other two are not within
    # the valid range for any integer time formats.
    input_batches = [
        pa.array([[631152001, 631152002]]),
        pa.array([[631152003, 631152000001]]),
        pa.array([[1, 2]])
    ]
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.1, values_threshold=1)
    assert schema_pb2.TimeDomain.UNIX_SECONDS == 1
    self.assertCombinerOutputEqual(
        input_batches, generator,
        statistics_pb2.FeatureNameStatistics(custom_stats=[
            statistics_pb2.CustomStatistic(
                name='domain_info',
                str=('time_domain {integer_format: 1}')
            ),
            statistics_pb2.CustomStatistic(name='time_match_ratio', num=0.5),
        ]))

  def test_time_stats_generator_non_time_integers(self):
    """Tests that the generator handles integers that are not times."""
    # None of these numbers are valid times.
    input_batches = [
        pa.array([[1, 2]]),
    ]
    generator = time_stats_generator.TimeStatsGenerator(
        match_ratio=0.1, values_threshold=1)
    self.assertCombinerOutputEqual(
        input_batches, generator, statistics_pb2.FeatureNameStatistics())


if __name__ == '__main__':
  absltest.main()
