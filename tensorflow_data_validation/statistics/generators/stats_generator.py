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

import abc
import apache_beam as beam
import pyarrow as pa
import six

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import input_batch
from typing import Any, Dict, Hashable, Iterable, Optional, Text, Tuple, TypeVar
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
      input_table: An Arrow Table whose columns are features and rows are
        examples. The columns are of type List<primitive> or Null (If a
        feature's value is None across all the examples in the batch, its
        corresponding column is of Null type).

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

  def add_input(self, accumulator: ACCTYPE, feature_path: types.FeaturePath,
                feature_array: pa.Array) -> ACCTYPE:
    """Returns result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      feature_path: The path of the feature.
      feature_array: An arrow Array representing a batch of feature values
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


CONSTITUENT_ACCTYPE = TypeVar('CONSTITUENT_ACCTYPE')


@six.add_metaclass(abc.ABCMeta)
class ConstituentStatsGenerator(object):
  """A stats generator meant to be used as a part of a composite generator.

  A constituent stats generator facilitates sharing logic between several stats
  generators. It is functionally identical to a beam.CombineFn, but it expects
  add_input to be called with instances of InputBatch.
  """

  @classmethod
  @abc.abstractmethod
  def key(cls) -> Hashable:
    """A class method which returns an ID for instances of this stats generator.

    This method should take all the arguments to the __init__ method so that the
    result of ConstituentStatsGenerator.key(*init_args) is identical to
    ConstituentStatsGenerator(*init_args).key(). This allows a
    CompositeStatsGenerator to construct a specific constituent generator in its
    __init__, and then recover the corresonding output value in its
    extract_composite_output method.

    Returns:
      A unique ID for instances of this stats generator class.
    """

  @abc.abstractmethod
  def get_key(self) -> Hashable:
    """Returns the ID of this specific generator.

    Returns:
      A unique ID for this stats generator class instance.
    """

  @abc.abstractmethod
  def create_accumulator(self) -> CONSTITUENT_ACCTYPE:  # pytype: disable=invalid-annotation
    """Returns a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """

  @abc.abstractmethod
  def add_input(
      self, accumulator: CONSTITUENT_ACCTYPE, batch: input_batch.InputBatch
  ) -> CONSTITUENT_ACCTYPE:  # pytype: disable=invalid-annotation
    """Returns result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      batch: An InputBatch which wraps an Arrow Table whose columns are features
        and rows are examples. The columns are of type List<primitive> or Null
        (If a feature's value is None across all the examples in the batch, its
        corresponding column is of Null type).

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """

  @abc.abstractmethod
  def merge_accumulators(
      self, accumulators: Iterable[CONSTITUENT_ACCTYPE]) -> CONSTITUENT_ACCTYPE:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """

  @abc.abstractmethod
  def extract_output(self, accumulator: CONSTITUENT_ACCTYPE) -> Any:  # pytype: disable=invalid-annotation
    """Returns result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      The final output value which should be used by composite generators which
      use this constituent generator.
    """


class CompositeStatsGenerator(CombinerStatsGenerator):
  """A combiner generator built from ConstituentStatsGenerators.

  Typical usage involves overriding the __init__, to provide a set of
  constituent generators, and extract_composite_output, to process the outputs
  of those constituent generators. As a toy example, consider:

      class ExampleCompositeStatsGenerator(
          stats_generator.CompositeStatsGenerator):

        def __init__(self,
                     schema: schema_pb2.Schema,
                     name: Text = 'ExampleCompositeStatsGenerator'
                     ) -> None:
          # custom logic to build the set of relevant constituents
          self._paths = [types.FeaturePath(['f1']), types.FeaturePath(['f2'])]
          constituents = [CountMissingCombiner(p) for p in self._paths]

          # call super class init with constituents
          super(ExampleCompositeStatsGenerator, self).__init__(
             name, constituents, schema)

        def extract_composite_outputs(self, accumulator):
          # custom logic to convert constituent outputs to stats proto
          stats = statistics_pb2.DatasetFeatureStatistics()
          for path in self._paths:
            # lookup output from a particular combiner using the key() function,
            # which typically takes the same args as __init__.
            num_missing = accumulator[CountMissingCombiner.key(path)]
            stats.features.add(path=path).custom_stats.add(
                name='num_missing', num=count_missing)

  This class is very similar to the SingleInputTupleCombineFn and adds two small
  features:
    1) The input value passed to add_inputs is wrapped in an InputBatch object
       before being passed on to the constituent generators.
    2) The API for providing constituents and retrieving their outputs is a dict
       rather than a tuple, which makes it easier to keep track of which output
       came from which constituent generator.
  """

  def __init__(self, name: Text,
               constituents: Iterable[ConstituentStatsGenerator],
               schema: Optional[schema_pb2.Schema]) -> None:
    super(CompositeStatsGenerator, self).__init__(name, schema)
    self._keys, self._constituents = zip(*(
        (c.get_key(), c) for c in constituents))

  def create_accumulator(self):
    return [c.create_accumulator() for c in self._constituents]

  def add_input(
      self, accumulator: Iterable[CONSTITUENT_ACCTYPE],
      input_arrow_table: pa.Table
  ) -> Iterable[CONSTITUENT_ACCTYPE]:  # pytype: disable=invalid-annotation
    batch = input_batch.InputBatch(input_arrow_table)
    return [
        c.add_input(a, batch) for c, a in zip(self._constituents, accumulator)
    ]

  def merge_accumulators(
      self, accumulators: Iterable[Iterable[CONSTITUENT_ACCTYPE]]
  ) -> Iterable[CONSTITUENT_ACCTYPE]:  # pytype: disable=invalid-annotation
    return [
        c.merge_accumulators(a)
        for c, a in zip(self._constituents, zip(*accumulators))
    ]

  def extract_output(
      self, accumulator: Tuple[CONSTITUENT_ACCTYPE]
  ) -> statistics_pb2.FeatureNameStatistics:  # pytype: disable=invalid-annotation
    return self.extract_composite_output(
        dict(
            zip(self._keys,
                (c.extract_output(a)
                 for c, a in zip(self._constituents, accumulator)))))

  def extract_composite_output(
      self, accumulator: Dict[Text,
                              Any]) -> statistics_pb2.FeatureNameStatistics:
    """Extracts output from a dict of outputs for each constituent combiner.

    Args:
      accumulator: A dict mapping from combiner keys to the corresponding output
        for that combiner.

    Returns:
      A proto representing the result of this stats generator.
    """
    raise NotImplementedError()


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
