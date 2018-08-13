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

from tensorflow_data_validation.anomalies import pywrap_tensorflow_data_validation
from tensorflow_data_validation.types_compat import Optional
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def infer_schema(statistics,
                 max_string_domain_size = 100
                ):
  """Infer schema from the input statistics.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer. Schema
        inference is currently only supported for lists with a single
        DatasetFeatureStatistics proto.
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

  schema_proto_string = pywrap_tensorflow_data_validation.InferSchema(
      statistics.datasets[0].SerializeToString(), max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)
  return result


def validate_statistics(statistics,
                        schema):
  """Validate the input statistics using the provided input schema.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer. Validation
        is currently only supported for lists with a single
        DatasetFeatureStatistics proto.
    schema: A Schema protocol buffer.

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

  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  if len(statistics.datasets) != 1:
    raise ValueError('Only statistics proto with one dataset is currently '
                     'supported for validation.')

  anomalies_proto_string = (
      pywrap_tensorflow_data_validation.ValidateFeatureStatistics(
          statistics.datasets[0].SerializeToString(),
          schema.SerializeToString()))

  # Parse the serialized Anomalies proto.
  result = anomalies_pb2.Anomalies()
  result.ParseFromString(anomalies_proto_string)
  return result
