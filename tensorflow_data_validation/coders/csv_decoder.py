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
"""Decode CSV records into in-memory representation for tf data validation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import csv
import apache_beam as beam
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import Dict, List, Optional, Text, Union

from tensorflow_metadata.proto.v0 import statistics_pb2

# Type for representing a CSV record and a field value.
# pylint: disable=invalid-name
CSVRecord = Union[bytes, Text]
CSVCell = Union[bytes, Text]
# pylint: enable=invalid-name

# Named tuple with column name and its type.
ColumnInfo = collections.namedtuple('ColumnInfo', ['name', 'type'])


@beam.typehints.with_input_types(CSVRecord)
@beam.typehints.with_output_types(types.ExampleBatch)
class DecodeCSV(beam.PTransform):
  """Decodes CSV records into an in-memory dict representation.

  Currently we assume each column has only a single value.
  """

  def __init__(self,
               column_names,
               delimiter = ',',
               skip_blank_lines = True):
    """Initializes the CSV decoder.

    Args:
      column_names: List of feature names. Order must match the order in the
          CSV file.
      delimiter: A one-character string used to separate fields.
      skip_blank_lines: A boolean to indicate whether to skip over blank lines
          rather than interpreting them as missing values.
    """
    if not isinstance(column_names, list):
      raise TypeError('column_names is of type %s, should be a list' %
                      type(column_names).__name__)
    self._column_names = column_names
    self._delimiter = delimiter
    self._skip_blank_lines = skip_blank_lines

  def expand(self, lines):
    """Decodes the input CSV records into an in-memory dict representation.

    Args:
      lines: A PCollection of strings representing the lines in the CSV file.

    Returns:
      A PCollection of dicts representing the CSV records.
    """
    input_rows = (
        lines | 'ParseCSVRecords' >> beam.Map(
            CSVParser(delimiter=self._delimiter).parse))

    column_info = (
        input_rows | 'InferFeatureTypes' >> beam.CombineGlobally(
            _FeatureTypeInferrer(
                column_names=self._column_names,
                skip_blank_lines=self._skip_blank_lines)))

    # Do second pass to generate the in-memory dict representation.
    return (input_rows | 'CreateInMemoryDict' >> beam.FlatMap(
        _make_example_dict,
        skip_blank_lines=self._skip_blank_lines,
        column_info=beam.pvalue.AsSingleton(column_info)))


class _LineGenerator(object):
  """A csv line generator that allows feeding lines to a csv.DictReader."""

  def __init__(self):
    self._lines = []

  def push_line(self, line):
    # This API currently supports only one line at a time.
    assert not self._lines
    self._lines.append(line)

  def __iter__(self):
    return self

  def next(self):
    """Gets the next line to process."""
    # This API currently supports only one line at a time.
    num_lines = len(self._lines)
    if num_lines == 0:
      raise ValueError('No line was found.')

    assert num_lines == 1, 'Unexpected number of lines %d' % num_lines
    # This doesn't maintain insertion order to the list, which is fine
    # because the list has only 1 element. If there were more and we wanted
    # to maintain order and timecomplexity we would switch to deque.popleft.
    return self._lines.pop()


# This is in agreement with Tensorflow conversions for Unicode values for both
# Python 2 and 3 (and also works for non-Unicode objects).
def _to_utf8_string(s):
  """Encodes the input csv line as a utf-8 string when applicable."""
  return s if isinstance(s, bytes) else s.encode('utf-8')


class CSVParser(object):
  """A parser to parse CSV formatted data."""

  class _ReaderWrapper(object):
    """A wrapper for csv.reader to make it picklable."""

    def __init__(self, delimiter):
      self._state = (delimiter)
      self._line_generator = _LineGenerator()
      self._reader = csv.reader(self._line_generator, delimiter=delimiter)

    def read_record(self, csv_string):
      self._line_generator.push_line(_to_utf8_string(csv_string))
      return self._reader.next()

    def __getstate__(self):
      return self._state

    def __setstate__(self, state):
      self.__init__(*state)

  def __init__(self, delimiter):
    """Initializes CSVParser.

    Args:
      delimiter: A one-character string used to separate fields.
    """
    self._delimiter = delimiter
    self._reader = self._ReaderWrapper(delimiter)

  def __reduce__(self):
    return CSVParser, (self._delimiter,)

  def parse(self, csv_string):
    """Parse a CSV record into a list of strings."""
    return self._reader.read_record(csv_string)


def _make_example_dict(
    row, skip_blank_lines,
    column_info
):
  """Create the in-memory representation from the CSV record.

  Args:
    row: List of cell values in a CSV record.
    skip_blank_lines: A boolean to indicate whether to skip over blank lines
      rather than interpreting them as missing values.
    column_info: List of tuples specifying column name and its type.

  Returns:
    A list containing the in-memory dict representation of the input CSV row.
  """
  if not row and skip_blank_lines:
    return []

  result = {}
  for index, field in enumerate(row):
    feature_name, feature_type = column_info[index]
    if not field:
      # If the field is an empty string, add the feature key with value as None.
      result[feature_name] = None
    elif feature_type == statistics_pb2.FeatureNameStatistics.INT:
      result[feature_name] = np.asarray([int(field)], dtype=np.integer)
    elif feature_type == statistics_pb2.FeatureNameStatistics.FLOAT:
      result[feature_name] = np.asarray([float(field)], dtype=np.floating)
    elif feature_type == statistics_pb2.FeatureNameStatistics.STRING:
      result[feature_name] = np.asarray([field], dtype=np.object)
    else:
      raise TypeError('Cannot determine the type of column %s.' % feature_name)
  return [result]


def _infer_value_type(
    value):
  """Infer feature type from the input value."""
  # If the value is an empty string, we can set the feature type to be
  # either FLOAT or STRING. We conservatively set it to be FLOAT.
  if not value:
    return statistics_pb2.FeatureNameStatistics.FLOAT

  # If all the characters in the value are digits, consider the
  # type to be INT.
  if value.isdigit():
    return statistics_pb2.FeatureNameStatistics.INT
  else:
    # If the type is not INT, we next check for FLOAT type (according to our
    # type hierarchy). If we can convert the string to a float value, we
    # fix the type to be FLOAT. Else we resort to STRING type.
    try:
      float(value)
    except ValueError:
      return statistics_pb2.FeatureNameStatistics.STRING

    return statistics_pb2.FeatureNameStatistics.FLOAT


def _type_hierarchy_level(
    feature_type):
  """Get level of the input type in the type hierarchy.

  Our type hierarchy is as follows,
      INT (level 0) --> FLOAT (level 1) --> STRING (level 2)

  Args:
    feature_type: A statistics_pb2.FeatureNameStatistics.Type value.

  Returns:
    The hierarchy level of the input type.
  """
  if feature_type == statistics_pb2.FeatureNameStatistics.INT:
    return 0
  elif feature_type == statistics_pb2.FeatureNameStatistics.FLOAT:
    return 1
  elif feature_type == statistics_pb2.FeatureNameStatistics.STRING:
    return 2
  else:
    raise TypeError('Unknown feature type %s.' % feature_type)


@beam.typehints.with_input_types(List[CSVCell])
@beam.typehints.with_output_types(beam.typehints.List[ColumnInfo])
class _FeatureTypeInferrer(beam.CombineFn):
  """Class to infer feature types as a beam.CombineFn."""

  def __init__(self, column_names,
               skip_blank_lines):
    """Initializes a feature type inferrer combiner."""
    self._column_names = column_names
    self._skip_blank_lines = skip_blank_lines

  def create_accumulator(
      self):  # pytype: disable=invalid-annotation
    """Creates an empty accumulator to keep track of the feature types."""
    return {}

  def add_input(
      self, accumulator,
      input_row
  ):
    """Updates the feature types in the accumulator using the input row.

    Args:
      accumulator: A dict containing the already inferred feature types.
      input_row: A list containing feature values of a CSV record.


    Returns:
      A dict containing the updated feature types based on input row.

    Raises:
      ValueError: If the columns do not match the specified csv headers.
    """
    # If the row is empty and we don't want to skip blank lines,
    # add an empty string to each column.
    if not input_row and not self._skip_blank_lines:
      input_row = ['' for _ in range(len(self._column_names))]
    elif input_row and len(input_row) != len(self._column_names):
      raise ValueError('Columns do not match specified csv headers: %s -> %s'
                       % (self._column_names, input_row))

    # Iterate over each feature value and update the type.
    for index, field in enumerate(input_row):
      feature_name = self._column_names[index]

      # Get the already inferred type of the feature.
      previous_type = accumulator.get(feature_name, None)
      # Infer the type from the current feature value.
      current_type = _infer_value_type(field)

      # If the type inferred from the current value is higher in the type
      # hierarchy compared to the already inferred type, we update the type.
      # The type hierarchy is,
      #   INT (level 0) --> FLOAT (level 1) --> STRING (level 2)
      if (previous_type is None or (_type_hierarchy_level(current_type) >
                                    _type_hierarchy_level(previous_type))):
        accumulator[feature_name] = current_type
    return accumulator

  def merge_accumulators(
      self, accumulators
  ):
    """Merge the feature types inferred from the different partitions.

    Args:
      accumulators: A list of dicts containing the feature types inferred from
          the different partitions of the data.

    Returns:
      A dict containing the merged feature types.
    """
    result = {}
    for shard_types in accumulators:
      # Merge the types inferred in each partition using the type hierarchy.
      # Specifically, whenever we observe a type higher in the type hierarchy
      # we update the type.
      for feature_name, feature_type in shard_types.items():
        if (feature_name not in result or
            (_type_hierarchy_level(feature_type) > _type_hierarchy_level(
                result[feature_name]))):
          result[feature_name] = feature_type
    return result

  def extract_output(self, accumulator
                    ):
    """Return a list of tuples containing the column info."""
    return [
        ColumnInfo(col_name, accumulator.get(col_name, None))
        for col_name in self._column_names
    ]
