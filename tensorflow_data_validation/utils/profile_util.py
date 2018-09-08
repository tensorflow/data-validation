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
"""Utilities for profiling data flowing through TF.DV."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Generator

from tensorflow_metadata.proto.v0 import statistics_pb2

# Namespace for all TFDV metrics.
METRICS_NAMESPACE = 'tfx.DataValidation'


@beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(types.ExampleBatch)
class _ProfileFn(beam.DoFn):
  """See docstring of Profile for details."""

  def __init__(self):
    # Counter for number of examples processed.
    self._num_instances = beam.metrics.Metrics.counter(METRICS_NAMESPACE,
                                                       'num_instances')

    # Distribution metrics to track the distribution of feature value lengths
    # for each type.
    self._int_feature_values_count = beam.metrics.Metrics.distribution(
        METRICS_NAMESPACE, 'int_feature_values_count')
    self._float_feature_values_count = beam.metrics.Metrics.distribution(
        METRICS_NAMESPACE, 'float_feature_values_count')
    self._string_feature_values_count = beam.metrics.Metrics.distribution(
        METRICS_NAMESPACE, 'string_feature_values_count')
    self._unknown_feature_values_count = beam.metrics.Metrics.distribution(
        METRICS_NAMESPACE, 'unknown_feature_values_count')

  def process(self,
              element):
    self._num_instances.inc(1)
    for _, value in six.iteritems(element):
      if not isinstance(value, np.ndarray):
        self._unknown_feature_values_count.update(1)
        continue
      feature_type = stats_util.make_feature_type(value.dtype)
      if feature_type == statistics_pb2.FeatureNameStatistics.INT:
        self._int_feature_values_count.update(len(value))
      elif feature_type == statistics_pb2.FeatureNameStatistics.FLOAT:
        self._float_feature_values_count.update(len(value))
      elif feature_type == statistics_pb2.FeatureNameStatistics.STRING:
        self._string_feature_values_count.update(len(value))
      else:
        self._unknown_feature_values_count.update(len(value))
    yield element


@beam.typehints.with_input_types(types.ExampleBatch)
@beam.typehints.with_output_types(types.ExampleBatch)
class Profile(beam.PTransform):
  """Profiles the input examples by emitting counters for different properties.

  Each input example is a dict of feature name to np.ndarray of feature values.
  The functor passes through each example to the output while emitting certain
  counters that track the number of examples and number of features of each
  type.

  Args:
    examples: PCollection of examples. Each example should be a dict of feature
      name to a numpy array of values (OK to be empty).

  Returns:
    PCollection of examples (same as input).
  """

  def expand(self, pcoll):
    return pcoll | beam.ParDo(_ProfileFn())
