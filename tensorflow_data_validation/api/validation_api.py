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
import apache_beam as beam
import tensorflow as tf
from tensorflow_data_validation import types
from tensorflow_data_validation.anomalies import pywrap_tensorflow_data_validation
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import anomalies_util
from tensorflow_data_validation.utils import slicing_util
from tensorflow_data_validation.types_compat import Optional
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# Set of anomaly types that do not apply on a per-example basis.
_GLOBAL_ONLY_ANOMALY_TYPES = set([
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_FRACTION_PRESENT,
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_NOT_PRESENT,
    anomalies_pb2.AnomalyInfo.SCHEMA_TRAINING_SERVING_SKEW,
    anomalies_pb2.AnomalyInfo.COMPARATOR_CONTROL_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.COMPARATOR_TREATMENT_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.COMPARATOR_L_INFTY_HIGH,
    anomalies_pb2.AnomalyInfo.NO_DATA_IN_SPAN,
])


def infer_schema(statistics,
                 infer_feature_shape = True,
                 max_string_domain_size = 100
                ):
  """Infers schema from the input statistics.

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
      tf.compat.as_bytes(statistics.datasets[0].SerializeToString()),
      max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  # TODO(b/113605666): Push this shape inference logic into example validation
  # code.
  if infer_feature_shape:
    _infer_shape(result)

  return result


def _infer_shape(schema):
  """Infers shapes of the features."""
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
  """Validates the input statistics against the provided input schema.

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
          tf.compat.as_bytes(serialized_stats),
          tf.compat.as_bytes(serialized_schema),
          tf.compat.as_bytes(environment),
          tf.compat.as_bytes(serialized_previous_stats),
          tf.compat.as_bytes(serialized_serving_stats)))

  # Parse the serialized Anomalies proto.
  result = anomalies_pb2.Anomalies()
  result.ParseFromString(anomalies_proto_string)
  return result


def _check_for_unsupported_schema_fields(schema):
  """Logs warnings when we encounter unsupported fields in the schema."""
  if schema.sparse_feature:
    logging.warning('The input schema has sparse features which'
                    ' are currently not supported.')


def _check_for_unsupported_stats_fields(
    stats,
    stats_type):
  """Logs warnings when we encounter unsupported fields in the statistics."""
  for feature in stats.features:
    if feature.HasField('struct_stats'):
      logging.warning('Feature "%s" in the %s has a struct_stats field which '
                      'is currently not supported.', feature.name, stats_type)


def validate_instance(
    instance,
    options,
    environment = None
):
  """Validates a single example against the schema provided in `options`.

  If an optional `environment` is specified, the schema is filtered using the
  `environment` and the `instance` is validated against the filtered schema.

  Args:
    instance: A single example in the form of a dict mapping a feature name to a
      numpy array.
    options: Options for generating data statistics. This must contain a
      schema.
    environment: An optional string denoting the validation environment. Must be
      one of the default environments specified in the schema. In some cases
      introducing slight schema variations is necessary, for instance features
      used as labels are required during training (and should be validated), but
      are missing during serving. Environments can be used to express such
      requirements. For example, assume a feature named 'LABEL' is required for
      training, but is expected to be missing from serving. This can be
      expressed by defining two distinct environments in the schema: ["SERVING",
      "TRAINING"] and associating 'LABEL' only with environment "TRAINING".

  Returns:
    An Anomalies protocol buffer.

  Raises:
    ValueError: If `options` is not a StatsOptions object.
    ValueError: If `options` does not contain a schema.
  """
  if not isinstance(options, stats_options.StatsOptions):
    raise ValueError('options must be a StatsOptions object.')
  if options.schema is None:
    raise ValueError('options must include a schema.')
  feature_statistics_list = (
      stats_impl.generate_statistics_in_memory([instance], options))
  anomalies = validate_statistics(feature_statistics_list, options.schema,
                                  environment)
  if anomalies.anomaly_info:
    # If anomalies were found, remove anomaly types that do not apply on a
    # per-example basis from the Anomalies proto.
    anomalies_util.remove_anomaly_types(anomalies, _GLOBAL_ONLY_ANOMALY_TYPES)
  return anomalies


def _detect_anomalies_in_example(example,
                                 options):
  """Validates the example against the schema provided in `options`."""
  return (example, validate_instance(example, options))


@beam.typehints.with_input_types(
    beam.typehints.Tuple[types.BeamExample, anomalies_pb2.Anomalies])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[types.BeamSliceKey, types.BeamExample])
class _GenerateAnomalyReasonSliceKeys(beam.DoFn):
  """Yields a slice key for each anomaly reason in the Anomalies proto."""

  def process(self, element):
    example, anomalies_proto = element
    for slice_key_and_example in slicing_util.generate_slices(
        example, [anomalies_util.anomalies_slicer],
        anomaly_proto=anomalies_proto):
      yield slice_key_and_example


@beam.typehints.with_input_types(types.BeamExample)
@beam.typehints.with_output_types(
    beam.typehints.KV[types.BeamSliceKey, beam.typehints
                      .Iterable[types.BeamExample]])
class IdentifyAnomalousExamples(beam.PTransform):
  """API for identifying anomalous examples.

  Validates each input example against the schema provided in `options` and
  outputs a sample of the anomalous examples found per anomaly reason.
  """

  def __init__(
      self,
      options,
      max_examples_per_anomaly = 10):
    """Initializes pipeline that identifies anomalous examples.

    Args:
      options: Options for generating data statistics. This must contain a
        schema.
      max_examples_per_anomaly: The maximum number of anomalous examples to
        output per anomaly reason.
    """

    self.options = options
    self.max_examples_per_anomaly = max_examples_per_anomaly

  @property
  def options(self):
    return self._options

  @options.setter
  def options(self, options):
    if not isinstance(options, stats_options.StatsOptions):
      raise ValueError('options must be a `StatsOptions` object.')
    if options.schema is None:
      raise ValueError('options must include a schema.')
    self._options = options

  @property
  def max_examples_per_anomaly(self):
    return self._max_examples_per_anomaly

  @max_examples_per_anomaly.setter
  def max_examples_per_anomaly(self, max_examples_per_anomaly):
    if not isinstance(max_examples_per_anomaly, int):
      raise TypeError('max_examples_per_anomaly must be an integer.')
    if max_examples_per_anomaly < 1:
      raise ValueError(
          'Invalid max_examples_per_anomaly %d.' % max_examples_per_anomaly)
    self._max_examples_per_anomaly = max_examples_per_anomaly

  def expand(self, dataset):
    dataset = (
        dataset
        | 'DetectAnomaliesInExamples' >> beam.Map(
            _detect_anomalies_in_example, options=self.options)
        | 'GenerateAnomalyReasonKeys' >> beam.ParDo(
            _GenerateAnomalyReasonSliceKeys()))
    # TODO(b/118835367): Add option to generate summary statistics for anomalous
    # examples on a per-anomaly-reason basis.
    return (
        dataset | 'SampleExamplesPerAnomalyReason' >>
        beam.combiners.Sample.FixedSizePerKey(self.max_examples_per_anomaly))
