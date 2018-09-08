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
# ==============================================================================

"""API for schema inference and statistics validation.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import logging
from tensorflow_data_validation.anomalies import pywrap_tensorflow_data_validation
from tensorflow_data_validation.types_compat import Optional
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def infer_schema(statistics,
                 infer_feature_shape = True,
                 max_string_domain_size = 100
                ):
  """Infer schema from the input statistics.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer. Schema
        inference is currently only supported for lists with a single
        DatasetFeatureStatistics proto.
    infer_feature_shape: A boolean to indicate if shape of the features need
        to be inferred from the statistics.
    max_string_domain_size: Maximum size of the domain of a string feature in
        order to be interpreted as a categorical feature.

  Returns:
    A Schema protocol buffer.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics proto does not have only one dataset.
  """
  if not isinstance(statistics, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(statistics).__name__)

  if len(statistics.datasets) != 1:
    raise ValueError('Only statistics proto with one dataset is currently '
                     'supported for inferring schema.')

  _check_for_unsupported_stats_fields(statistics.datasets[0], 'statistics')

  schema_proto_string = pywrap_tensorflow_data_validation.InferSchema(
      statistics.datasets[0].SerializeToString(), max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  if infer_feature_shape:
    _infer_shape(result)

  return result


def _infer_shape(schema):
  """Infer shapes of the features."""
  for feature in schema.feature:
    # Currently we infer shape only for features with valency 1.
    if (feature.presence.min_fraction == 1 and
        feature.value_count.min == feature.value_count.max == 1):
      feature.shape.dim.add().size = 1


def validate_statistics(
    statistics,
    schema,
    environment = None,
    previous_statistics = None,
    serving_statistics = None,
):
  """Validate the input statistics against the provided input schema.

  This method validates the `statistics` against the `schema`. If an optional
  `environment` is specified, the `schema` is filtered using the
  `environment` and the `statistics` is validated against the filtered schema.
  The optional `previous_statistics` and `serving_statistics` are the statistics
  computed over the treatment data for drift- and skew-detection, respectively.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer denoting the
        statistics computed over the current data. Validation is currently only
        supported for lists with a single DatasetFeatureStatistics proto.
    schema: A Schema protocol buffer.
    environment: An optional string denoting the validation environment.
        Must be one of the default environments specified in the schema.
        By default, validation assumes that all Examples in a pipeline adhere
        to a single schema. In some cases introducing slight schema variations
        is necessary, for instance features used as labels are required during
        training (and should be validated), but are missing during serving.
        Environments can be used to express such requirements. For example,
        assume a feature named 'LABEL' is required for training, but is expected
        to be missing from serving. This can be expressed by defining two
        distinct environments in schema: ["SERVING", "TRAINING"] and
        associating 'LABEL' only with environment "TRAINING".
    previous_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over an earlier data (for
        example, previous day's data). If provided, the `validate_statistics`
        method will detect if there exists drift between current data and
        previous data. Configuration for drift detection can be done by
        specifying a `drift_comparator` in the schema. For now drift detection
        is only supported for categorical features.
    serving_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over the serving data. If
        provided, the `validate_statistics` method will identify if there exists
        distribution skew between current data and serving data. Configuration
        for skew detection can be done by specifying a `skew_comparator` in the
        schema. For now skew detection is only supported for categorical
        features.

  Returns:
    An Anomalies protocol buffer.

  Raises:
    TypeError: If any of the input arguments is not of the expected type.
    ValueError: If the input statistics proto does not have only one dataset.
  """
  if not isinstance(statistics, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(statistics).__name__)

  if len(statistics.datasets) != 1:
    raise ValueError('statistics proto contains multiple datasets. Only '
                     'one dataset is currently supported for validation.')

  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  if environment is not None:
    if environment not in schema.default_environment:
      raise ValueError('Environment %s not found in the schema.' % environment)
  else:
    environment = ''

  _check_for_unsupported_stats_fields(statistics.datasets[0], 'statistics')
  _check_for_unsupported_schema_fields(schema)

  if previous_statistics is not None:
    if not isinstance(
        previous_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'previous_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(previous_statistics).__name__)

    if len(previous_statistics.datasets) != 1:
      raise ValueError(
          'previous_statistics proto contains multiple datasets. '
          'Only one dataset is currently supported for validation.')

    _check_for_unsupported_stats_fields(previous_statistics.datasets[0],
                                        'previous_statistics')

  if serving_statistics is not None:
    if not isinstance(
        serving_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'serving_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(serving_statistics).__name__)

    if len(serving_statistics.datasets) != 1:
      raise ValueError(
          'serving_statistics proto contains multiple datasets. '
          'Only one dataset is currently supported for validation.')

    _check_for_unsupported_stats_fields(serving_statistics.datasets[0],
                                        'serving_statistics')

  # Serialize the input protos.
  serialized_schema = schema.SerializeToString()
  serialized_stats = statistics.datasets[0].SerializeToString()
  serialized_previous_stats = (
      previous_statistics.datasets[0].SerializeToString()
      if previous_statistics is not None else '')
  serialized_serving_stats = (
      serving_statistics.datasets[0].SerializeToString()
      if serving_statistics is not None else '')

  anomalies_proto_string = (
      pywrap_tensorflow_data_validation.ValidateFeatureStatistics(
          serialized_stats, serialized_schema, environment,
          serialized_previous_stats, serialized_serving_stats))

  # Parse the serialized Anomalies proto.
  result = anomalies_pb2.Anomalies()
  result.ParseFromString(anomalies_proto_string)
  return result


def _check_for_unsupported_schema_fields(schema):
  """Log warnings when we encounter unsupported fields in the schema."""
  if schema.sparse_feature:
    logging.warning('The input schema has sparse features which'
                    ' are currently not supported.')


def _check_for_unsupported_stats_fields(
    stats,
    stats_type):
  """Log warnings when we encounter unsupported fields in the statistics."""
  for feature in stats.features:
    if feature.HasField('struct_stats'):
      logging.warning('Feature "%s" in the %s has a struct_stats field which '
                      'is currently not supported.', feature.name, stats_type)
