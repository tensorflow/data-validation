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

"""Beam implementation of statistics generators."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.types_compat import List, TypeVar

from tensorflow_metadata.proto.v0 import statistics_pb2


@beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(statistics_pb2.DatasetFeatureStatisticsList)
class GenerateStatisticsImpl(beam.PTransform):
  """PTransform that applies a set of generators."""

  def __init__(
      self,
      generators):
    self._generators = generators

  def expand(self, pcoll):
    result_protos = []
    # Iterate over the stats generators. For each generator,
    #   a) if it is a CombinerStatsGenerator, wrap it as a beam.CombineFn
    #      and run it.
    #   b) if it is a TransformStatsGenerator, wrap it as a beam.PTransform
    #      and run it.
    for generator in self._generators:
      if isinstance(generator, stats_generator.CombinerStatsGenerator):
        result_protos.append(
            pcoll |
            generator.name >> beam.CombineGlobally(
                _CombineFnWrapper(generator)))
      elif isinstance(generator, stats_generator.TransformStatsGenerator):
        result_protos.append(
            pcoll |
            generator.name >> generator.ptransform)
      else:
        raise TypeError('Statistics generator must extend one of '
                        'CombinerStatsGenerator or TransformStatsGenerator, '
                        'found object of type %s' %
                        type(generator).__class__.__name__)

    # Each stats generator will output a PCollection of DatasetFeatureStatistics
    # protos. We now flatten the list of PCollections into a single PCollection,
    # then merge the DatasetFeatureStatistics protos in the PCollection into a
    # single DatasetFeatureStatisticsList proto.
    return (result_protos | 'FlattenFeatureStatistics' >> beam.Flatten()
            | 'MergeDatasetFeatureStatisticsProtos' >>
            beam.CombineGlobally(_merge_dataset_feature_stats_protos)
            | 'MakeDatasetFeatureStatisticsListProto' >>
            beam.Map(_make_dataset_feature_statistics_list_proto))


def _merge_dataset_feature_stats_protos(
    stats_protos
):
  """Merge together a list of DatasetFeatureStatistics protos.

  Args:
    stats_protos: A list of DatasetFeatureStatistics protos to merge.

  Returns:
    The merged DatasetFeatureStatistics proto.
  """
  stats_per_feature = {}
  # Iterate over each DatasetFeatureStatistics proto and merge the
  # FeatureNameStatistics protos per feature.
  for stats_proto in stats_protos:
    for feature_stats_proto in stats_proto.features:
      if feature_stats_proto.name not in stats_per_feature:
        stats_per_feature[feature_stats_proto.name] = feature_stats_proto
      else:
        stats_per_feature[feature_stats_proto.name].MergeFrom(
            feature_stats_proto)

  # Create a new DatasetFeatureStatistics proto.
  result = statistics_pb2.DatasetFeatureStatistics()
  num_examples = None
  for feature_stats_proto in stats_per_feature.values():
    # Add the merged FeatureNameStatistics proto for the feature
    # into the DatasetFeatureStatistics proto.
    new_feature_stats_proto = result.features.add()
    new_feature_stats_proto.CopyFrom(feature_stats_proto)

    # Get the number of examples from one of the features that
    # has common stats.
    if num_examples is None:
      stats_type = feature_stats_proto.WhichOneof('stats')
      stats_proto = None
      if stats_type == 'num_stats':
        stats_proto = feature_stats_proto.num_stats
      else:
        stats_proto = feature_stats_proto.string_stats

      if stats_proto.HasField('common_stats'):
        num_examples = (stats_proto.common_stats.num_non_missing +
                        stats_proto.common_stats.num_missing)

  # Set the num_examples field.
  if num_examples is not None:
    result.num_examples = num_examples
  return result


def _make_dataset_feature_statistics_list_proto(
    stats_proto
):
  """Constructs a DatasetFeatureStatisticsList proto.

  Args:
    stats_proto: The input DatasetFeatureStatistics proto.

  Returns:
    The DatasetFeatureStatisticsList proto containing the input stats proto.
  """
  # Create a new DatasetFeatureStatisticsList proto.
  result = statistics_pb2.DatasetFeatureStatisticsList()

  # Add the input DatasetFeatureStatistics proto.
  dataset_stats_proto = result.datasets.add()
  dataset_stats_proto.CopyFrom(stats_proto)
  return result




@beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(
    statistics_pb2.DatasetFeatureStatistics)
class _CombineFnWrapper(beam.CombineFn):
  """Class to wrap a CombinerStatsGenerator as a beam.CombineFn."""

  def __init__(
      self,
      generator):
    self._generator = generator

  def __reduce__(self):
    return _CombineFnWrapper, (self._generator,)

  def create_accumulator(self
                        ):  # pytype: disable=invalid-annotation
    return self._generator.create_accumulator()

  def add_input(self, accumulator,
                input_batch):
    return self._generator.add_input(accumulator, input_batch)

  def merge_accumulators(self, accumulators):
    return self._generator.merge_accumulators(accumulators)

  def extract_output(
      self,
      accumulator
  ):  # pytype: disable=invalid-annotation
    return self._generator.extract_output(accumulator)
