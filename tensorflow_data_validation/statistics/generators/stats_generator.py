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

   Incorporates a batch of input examples into the current accumulator
   and returns the updated accumulator.
       add_input(accumulator, input_batch)

   Merge the partial states in the accumulators and returns the accumulator
   containing the merged state.
       merge_accumulators(accumulators)

   Compute statistics from the partial state in the accumulator and
   return the result as a DatasetFeatureStatistics proto.
       extract_output(accumulator)

2) TransformStatsGenerator
   This generator computes statistics using a user-provided Beam PTransform.
   The PTransform must accept a Beam PCollection where each element is a Python
   dict whose keys are feature names and values are numpy arrays representing a
   batch of examples. It must return a PCollection containing a single element
   which is a DatasetFeatureStatistics proto.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import List, Optional, TypeVar
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsGenerator(object):
  """Generate statistics."""

  def __init__(self, name,
               schema = None):
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



class CombinerStatsGenerator(StatsGenerator):
  """Generate statistics using combiner function.

  This object mirrors a beam.CombineFn.
  """

  def create_accumulator(self):  # pytype: disable=invalid-annotation
    """Return a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    raise NotImplementedError

  def add_input(self, accumulator,
                input_batch):
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      input_batch: A Python dict whose keys are strings denoting feature
          names and values are numpy arrays representing a batch of examples,
          which should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    raise NotImplementedError

  def merge_accumulators(self, accumulators):
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    raise NotImplementedError

  def extract_output(
      self, accumulator
  ):  # pytype: disable=invalid-annotation
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    raise NotImplementedError


class TransformStatsGenerator(StatsGenerator):
  """Generate statistics using a Beam PTransform."""

  def __init__(self,
               name,
               ptransform,
               schema = None):
    self._ptransform = ptransform
    super(TransformStatsGenerator, self).__init__(name, schema)

  @property
  def ptransform(self):
    return self._ptransform
