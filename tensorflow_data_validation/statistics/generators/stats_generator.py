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
"""Base class for statistics generators.

A statistics generator is used to compute the statistics of features in
parallel. We support two types of generators:

1) CombinerStatsGenerator
   This generator computes statistics using a combiner function. It emits
   partial states processing a batch of examples at a time,
   merges the partial states, and finally computes the statistics from the
   merged partial state at the end. Specifically, the generator
   must implement the following four methods,

   Initializes an accumulator to store the partial state and returns it.
       create_accumulator()

   Incorporates a batch of input examples (represented as an arrow table) into
   the current accumulator
   and returns the updated accumulator.
       add_input(accumulator, input_table)

   Merge the partial states in the accumulators and returns the accumulator
   containing the merged state.
       merge_accumulators(accumulators)

   Compute statistics from the partial state in the accumulator and
   return the result as a DatasetFeatureStatistics proto.
       extract_output(accumulator)

2) TransformStatsGenerator
   This generator computes statistics using a user-provided Beam PTransform.
   The PTransform must accept a Beam PCollection where each element is a tuple
   containing a slice key and an Arrow table representing a batch of examples.
   It must return a PCollection where each element is a tuple containing a
   slice key and a DatasetFeatureStatistics proto representing the statistics
   of a slice.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from tensorflow_data_validation.pyarrow_tf import pyarrow as pa
from typing import Iterable, Optional, Text, TypeVar
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsGenerator(object):
  """Generate statistics."""

  def __init__(self, name: Text,
               schema: Optional[schema_pb2.Schema] = None) -> None:
    """Initializes a statistics generator.

    Args:
      name: A unique name associated with the statistics generator.
      schema: An optional schema for the dataset.
    """
    self._name = name
    self._schema = schema

  @property
  def name(self):
    return self._name

  @property
  def schema(self):
    return self._schema

# Have a type variable to represent the type of the accumulator
# in a combiner stats generator.
ACCTYPE = TypeVar('ACCTYPE')


class CombinerStatsGenerator(StatsGenerator):
  """Generate statistics using combiner function.

  This object mirrors a beam.CombineFn except for the add_input interface, which
  is expected to be defined by its sub-classes.
  """

  def create_accumulator(self) -> ACCTYPE:  # pytype: disable=invalid-annotation
    """Returns a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    raise NotImplementedError

  def add_input(
      self, accumulator: ACCTYPE, input_table: pa.Table) -> ACCTYPE:
    """Returns result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      input_table: An Arrow Table whose columns are features rows are examples.
        The columns are of type List<primitive> or Null (If a feature's value
        is None across all the examples in the batch, its corresponding column
        is of Null type).

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    raise NotImplementedError

  def merge_accumulators(self, accumulators: Iterable[ACCTYPE]) -> ACCTYPE:
    """Merges several accumulators to a single accumulator value.

    Note: mutating any element in `accumulators` is not allowed and will result
    in undefined behavior.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    raise NotImplementedError

  def extract_output(
      self, accumulator: ACCTYPE
  ) -> statistics_pb2.DatasetFeatureStatistics:  # pytype: disable=invalid-annotation
    """Returns result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    raise NotImplementedError


class CombinerFeatureStatsGenerator(StatsGenerator):
  """Generate feature level statistics using combiner function.

  This interface is a simplification of CombinerStatsGenerator for the special
  case of statistics that do not require cross-feature computations. It mirrors
  a beam.CombineFn for the values of a specific feature.
  """

  def create_accumulator(self) -> ACCTYPE:  # pytype: disable=invalid-annotation
    """Returns a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    raise NotImplementedError

  def add_input(self, accumulator: ACCTYPE, input_column: pa.Column
               ) -> ACCTYPE:
    """Returns result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      input_column: An arrow column representing a batch of feature values
        which should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    raise NotImplementedError

  def merge_accumulators(self, accumulators: Iterable[ACCTYPE]) -> ACCTYPE:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    raise NotImplementedError

  def extract_output(
      self, accumulator: ACCTYPE) -> statistics_pb2.FeatureNameStatistics:  # pytype: disable=invalid-annotation
    """Returns result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    raise NotImplementedError


class TransformStatsGenerator(StatsGenerator):
  """Generate statistics using a Beam PTransform.

  Note that the input PTransform must take a PCollection of sliced
  examples (tuple of (slice_key, example)) as input and output a
  PCollection of sliced protos
  (tuple of (slice_key, DatasetFeatureStatistics proto)).
  """

  def __init__(self,
               name: Text,
               ptransform: beam.PTransform,
               schema: Optional[schema_pb2.Schema] = None) -> None:
    self._ptransform = ptransform
    super(TransformStatsGenerator, self).__init__(name, schema)

  @property
  def ptransform(self):
    return self._ptransform
