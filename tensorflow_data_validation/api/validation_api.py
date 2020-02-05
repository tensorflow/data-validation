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
"""API for schema inference and statistics validation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import logging
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.anomalies.proto import validation_config_pb2
from tensorflow_data_validation.anomalies.proto import validation_metadata_pb2
from tensorflow_data_validation.api import validation_options as vo
from tensorflow_data_validation.pywrap import pywrap_tensorflow_data_validation
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import anomalies_util
from tensorflow_data_validation.utils import slicing_util
from typing import Callable, List, Optional, Text

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
    anomalies_pb2.AnomalyInfo.COMPARATOR_LOW_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.COMPARATOR_HIGH_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.NO_DATA_IN_SPAN,
])


def infer_schema(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    infer_feature_shape: bool = True,
    max_string_domain_size: int = 100,
    schema_transformations: Optional[List[
        Callable[[schema_pb2.Schema, statistics_pb2.DatasetFeatureStatistics],
                 schema_pb2.Schema]]] = None
) -> schema_pb2.Schema:
  """Infers schema from the input statistics.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer. Schema inference
      is currently supported only for lists with a single
      DatasetFeatureStatistics proto or lists with multiple
      DatasetFeatureStatistics protos corresponding to data slices that include
      the default slice (i.e., the slice with all examples). If a list with
      multiple DatasetFeatureStatistics protos is used, this function will infer
      the schema from the statistics corresponding to the default slice.
    infer_feature_shape: A boolean to indicate if shape of the features need to
      be inferred from the statistics.
    max_string_domain_size: Maximum size of the domain of a string feature in
        order to be interpreted as a categorical feature.
    schema_transformations: List of transformation functions to apply to the
        auto-inferred schema. Each transformation function should take the
        schema and statistics as input and should return the transformed schema.
        The transformations are applied in the order provided in the list.

  Returns:
    A Schema protocol buffer.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """
  if not isinstance(statistics, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(statistics).__name__)

  # This will raise an exception if there are multiple datasets, none of which
  # corresponds to the default slice.
  dataset_statistics = _get_default_dataset_statistics(statistics)

  # dataset_statistics may include stats for composite features like
  # SparseFeatures and WeightedFeatures. We cannot infer a useful schema from
  # these stats, so we remove them at the start.
  dataset_statistics = _remove_features_missing_common_stats(dataset_statistics)

  _check_for_unsupported_stats_fields(dataset_statistics, 'statistics')

  schema_proto_string = pywrap_tensorflow_data_validation.InferSchema(
      tf.compat.as_bytes(dataset_statistics.SerializeToString()),
      max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  _may_be_set_legacy_flag(result)

  # TODO(b/113605666): Push this shape inference logic into example validation
  # code.
  if infer_feature_shape:
    _infer_shape(result)

  if schema_transformations is not None:
    for transformation_fn in schema_transformations:
      result = transformation_fn(result, statistics.datasets[0])
  return result


# Note that this flag is legacy code.
def _may_be_set_legacy_flag(schema: schema_pb2.Schema):
  """Sets legacy flag to False if it exists."""
  if getattr(schema, 'generate_legacy_feature_spec', None) is not None:
    schema.generate_legacy_feature_spec = False


def _infer_shape(schema: schema_pb2.Schema):
  """Infers shapes of the features."""
  for feature in schema.feature:
    # Currently we infer shape only for required features.
    if (feature.presence.min_fraction == 1 and
        feature.value_count.min == feature.value_count.max):
      feature.shape.dim.add().size = feature.value_count.min


# TODO(pachristopher): Add support for updating only a subset of features.
def update_schema(schema: schema_pb2.Schema,
                  statistics: statistics_pb2.DatasetFeatureStatisticsList,
                  infer_feature_shape: Optional[bool] = True,
                  max_string_domain_size: Optional[int] = 100
                 ) -> schema_pb2.Schema:
  """Updates input schema to conform to the input statistics.

  Args:
    schema: A Schema protocol buffer.
    statistics: A DatasetFeatureStatisticsList protocol buffer. Schema inference
      is currently supported only for lists with a single
      DatasetFeatureStatistics proto or lists with multiple
      DatasetFeatureStatistics protos corresponding to data slices that include
      the default slice (i.e., the slice with all examples). If a list with
      multiple DatasetFeatureStatistics protos is used, this function will
      update the schema to conform to the statistics corresponding to the
      default slice.
    infer_feature_shape: A boolean to indicate if shape of the features need to
      be inferred from the statistics.
    max_string_domain_size: Maximum size of the domain of a string feature in
      order to be interpreted as a categorical feature.

  Returns:
    A Schema protocol buffer.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)
  if not isinstance(statistics, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(statistics).__name__)

  # This will raise an exception if there are multiple datasets, none of which
  # corresponds to the default slice.
  dataset_statistics = _get_default_dataset_statistics(statistics)

  _check_for_unsupported_stats_fields(dataset_statistics, 'statistics')

  schema_proto_string = pywrap_tensorflow_data_validation.UpdateSchema(
      tf.compat.as_bytes(schema.SerializeToString()),
      tf.compat.as_bytes(dataset_statistics.SerializeToString()),
      max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  # TODO(b/113605666): Push this shape inference logic into example validation
  # code.
  if infer_feature_shape:
    _infer_shape(result)
  return result


def validate_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    schema: schema_pb2.Schema,
    environment: Optional[Text] = None,
    previous_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    serving_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
) -> anomalies_pb2.Anomalies:
  """Validates the input statistics against the provided input schema.

  This method validates the `statistics` against the `schema`. If an optional
  `environment` is specified, the `schema` is filtered using the
  `environment` and the `statistics` is validated against the filtered schema.
  The optional `previous_statistics` and `serving_statistics` are the statistics
  computed over the control data for drift- and skew-detection, respectively.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer denoting the
       statistics computed over the current data. Validation is currently
       supported only for lists with a single DatasetFeatureStatistics proto or
       lists with multiple DatasetFeatureStatistics protos corresponding to data
       slices that include the default slice (i.e., the slice with all
       examples). If a list with multiple DatasetFeatureStatistics protos is
       used, this function will validate the statistics corresponding to the
       default slice.
    schema: A Schema protocol buffer.
       Note that TFDV does not currently support validation of the following
       messages/fields in the Schema protocol buffer:
       - FeaturePresenceWithinGroup
       - Schema-level FloatDomain and IntDomain (validation is supported for
         Feature-level FloatDomain and IntDomain)
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
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """

  # This check is added here because the arguments name for previous_statistics
  # is different in TFX::OSS and TFX internal. It is preferred to report the
  # error with the name used in the API.
  if previous_statistics is not None:
    if not isinstance(
        previous_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'previous_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(previous_statistics).__name__)

  return validate_statistics_internal(statistics, schema, environment,
                                      previous_statistics, serving_statistics)


def validate_statistics_internal(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    schema: schema_pb2.Schema,
    environment: Optional[Text] = None,
    previous_span_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    serving_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    previous_version_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    validation_options: Optional[vo.ValidationOptions] = None,
    enable_diff_regions: bool = False
) -> anomalies_pb2.Anomalies:
  """Validates the input statistics against the provided input schema.

  This method validates the `statistics` against the `schema`. If an optional
  `environment` is specified, the `schema` is filtered using the
  `environment` and the `statistics` is validated against the filtered schema.
  The optional `previous_span_statistics`, `serving_statistics`, and
  `previous_version_statistics` are the statistics computed over the control
  data for drift detection, skew detection, and dataset-level anomaly detection
  across versions, respectively.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer denoting the
       statistics computed over the current data. Validation is currently
       supported only for lists with a single DatasetFeatureStatistics proto or
       lists with multiple DatasetFeatureStatistics protos corresponding to data
       slices that include the default slice (i.e., the slice with all
       examples). If a list with multiple DatasetFeatureStatistics protos is
       used, this function will validate the statistics corresponding to the
       default slice.
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
    previous_span_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over an earlier data (for
        example, previous day's data). If provided, the
        `validate_statistics_internal` method will detect if there exists drift
        between current data and previous data. Configuration for drift
        detection can be done by specifying a `drift_comparator` in the schema.
        For now drift detection is only supported for categorical features.
    serving_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over the serving data. If
        provided, the `validate_statistics_internal` method will identify if
        there exists distribution skew between current data and serving data.
        Configuration for skew detection can be done by specifying a
        `skew_comparator` in the schema. For now skew detection is only
        supported for categorical features.
    previous_version_statistics: An optional DatasetFeatureStatisticsList
        protocol buffer denoting the statistics computed over an earlier data
        (typically, previous run's data within the same day). If provided,
        the `validate_statistics_internal` method will detect if there exists a
        change in the number of examples between current data and previous
        version data. Configuration for such dataset-wide anomaly detection can
        be done by specifying a `num_examples_version_comparator` in the schema.
    validation_options: Optional input used to specify the options of this
        validation.
    enable_diff_regions: Specifies whether to include a comparison between the
        existing schema and the fixed schema in the Anomalies protocol buffer
        output.

  Returns:
    An Anomalies protocol buffer.

  Raises:
    TypeError: If any of the input arguments is not of the expected type.
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """
  if not isinstance(statistics, statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(statistics).__name__)

  # This will raise an exception if there are multiple datasets, none of which
  # corresponds to the default slice.
  dataset_statistics = _get_default_dataset_statistics(statistics)

  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  if environment is not None:
    if environment not in schema.default_environment:
      raise ValueError('Environment %s not found in the schema.' % environment)
  else:
    environment = ''

  _check_for_unsupported_stats_fields(dataset_statistics, 'statistics')

  if previous_span_statistics is not None:
    if not isinstance(
        previous_span_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'previous_span_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(previous_span_statistics).__name__)

    previous_dataset_statistics = _get_default_dataset_statistics(
        previous_span_statistics)
    _check_for_unsupported_stats_fields(previous_dataset_statistics,
                                        'previous_statistics')

  if serving_statistics is not None:
    if not isinstance(
        serving_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'serving_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(serving_statistics).__name__)

    serving_dataset_statistics = _get_default_dataset_statistics(
        serving_statistics)
    _check_for_unsupported_stats_fields(serving_dataset_statistics,
                                        'serving_statistics')

  if previous_version_statistics is not None:
    if not isinstance(previous_version_statistics,
                      statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError('previous_version_statistics is of type %s, should be '
                      'a DatasetFeatureStatisticsList proto.' %
                      type(previous_version_statistics).__name__)

    previous_version_dataset_statistics = _get_default_dataset_statistics(
        previous_version_statistics)
    _check_for_unsupported_stats_fields(previous_version_dataset_statistics,
                                        'previous_version_statistics')

  # Serialize the input protos.
  serialized_schema = schema.SerializeToString()
  serialized_stats = dataset_statistics.SerializeToString()
  serialized_previous_span_stats = (
      previous_dataset_statistics.SerializeToString()
      if previous_span_statistics is not None else '')
  serialized_serving_stats = (
      serving_dataset_statistics.SerializeToString()
      if serving_statistics is not None else '')
  serialized_previous_version_stats = (
      previous_version_dataset_statistics.SerializeToString()
      if previous_version_statistics is not None else '')

  features_needed_pb = validation_metadata_pb2.FeaturesNeededProto()
  if validation_options is not None and validation_options.features_needed:
    for path, reason_list in validation_options.features_needed.items():
      path_and_reason_feature_need = (
          features_needed_pb.path_and_reason_feature_need.add())
      path_and_reason_feature_need.path.CopyFrom(path.to_proto())
      for reason in reason_list:
        r = path_and_reason_feature_need.reason_feature_needed.add()
        r.comment = reason.comment

  serialized_features_needed = features_needed_pb.SerializeToString()

  validation_config = validation_config_pb2.ValidationConfig()
  if validation_options is not None:
    validation_config.new_features_are_warnings = (
        validation_options.new_features_are_warnings)
    for override in validation_options.severity_overrides:
      validation_config.severity_overrides.append(override)
  serialized_validation_config = validation_config.SerializeToString()

  anomalies_proto_string = (
      pywrap_tensorflow_data_validation.ValidateFeatureStatistics(
          tf.compat.as_bytes(serialized_stats),
          tf.compat.as_bytes(serialized_schema),
          tf.compat.as_bytes(environment),
          tf.compat.as_bytes(serialized_previous_span_stats),
          tf.compat.as_bytes(serialized_serving_stats),
          tf.compat.as_bytes(serialized_previous_version_stats),
          tf.compat.as_bytes(serialized_features_needed),
          tf.compat.as_bytes(serialized_validation_config),
          enable_diff_regions))

  # Parse the serialized Anomalies proto.
  result = anomalies_pb2.Anomalies()
  result.ParseFromString(anomalies_proto_string)
  return result


def _remove_features_missing_common_stats(
    stats: statistics_pb2.DatasetFeatureStatistics
) -> statistics_pb2.DatasetFeatureStatistics:
  """Remove FeatureNameStatistics for feature paths missing common stats.

  Args:
    stats: The stats from which to remove features

  Returns:
    A version of the input stats with the feature paths removed.
  """
  valid_features = []
  for feature in stats.features:
    if (feature.HasField('num_stats') or feature.HasField('string_stats') or
        feature.HasField('bytes_stats') or
        feature.HasField('struct_stats')):
      valid_features.append(feature)
  del stats.features[:]
  stats.features.extend(valid_features)
  return stats


def _check_for_unsupported_stats_fields(
    stats: statistics_pb2.DatasetFeatureStatistics,
    stats_type: str):
  """Logs warnings when we encounter unsupported fields in the statistics."""
  for feature in stats.features:
    if feature.HasField('struct_stats'):
      logging.warning('Feature "%s" in the %s has a struct_stats field which '
                      'is currently not supported.', feature.name, stats_type)


def validate_instance(
    instance: pa.Table,
    options: stats_options.StatsOptions,
    environment: Optional[str] = None
) -> anomalies_pb2.Anomalies:
  """Validates a batch of examples against the schema provided in `options`.

  If an optional `environment` is specified, the schema is filtered using the
  `environment` and the `instance` is validated against the filtered schema.

  Args:
    instance: A batch of examples in the form of an Arrow table.
    options: `tfdv.StatsOptions` for generating data statistics. This must
      contain a schema.
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
      stats_impl.generate_statistics_in_memory(instance, options))
  anomalies = validate_statistics(feature_statistics_list, options.schema,
                                  environment)
  if anomalies.anomaly_info:
    # If anomalies were found, remove anomaly types that do not apply on a
    # per-example basis from the Anomalies proto.
    anomalies_util.remove_anomaly_types(anomalies, _GLOBAL_ONLY_ANOMALY_TYPES)
  return anomalies


def _detect_anomalies_in_example(table: pa.Table,
                                 options: stats_options.StatsOptions):
  """Validates the example against the schema provided in `options`."""
  # Verify that we have a single row.
  assert table.num_rows == 1
  return (table, validate_instance(table, options))


def _get_default_dataset_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> statistics_pb2.DatasetFeatureStatistics:
  """Gets the DatasetFeatureStatistics to use for validation.

  If there is a single DatasetFeatureStatistics, this function returns that. If
  there are multiple DatasetFeatureStatistics, this function attempts to find
  the one that corresponds to the default slice. If found, this function returns
  that. If not found, this function raises an error.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer.

  Returns:
    A DatasetFeatureStatistics protocol buffer to use for validation.

  Raises:
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """
  if len(statistics.datasets) == 1:
    return statistics.datasets[0]
  # If there are multiple datasets, attempt to find the dataset for the
  # default slice (i.e., slice for all examples) from among the datasets.
  for dataset in statistics.datasets:
    if dataset.name == constants.DEFAULT_SLICE_KEY:
      logging.warning('Multiple datasets found in statistics. Using the '
                      'default slice dataset.')
      return dataset
  # If there are multiple datasets, but the default slice is not found, raise an
  # error.
  raise ValueError('Only statistics proto with one dataset or the default '
                   'slice (i.e., "All Examples" slice) is currently supported.')


@beam.typehints.with_input_types(
    beam.typehints.Tuple[pa.Table, anomalies_pb2.Anomalies])
@beam.typehints.with_output_types(types.BeamSlicedTable)
class _GenerateAnomalyReasonSliceKeys(beam.DoFn):
  """Yields a slice key for each anomaly reason in the Anomalies proto."""

  def process(self, element):
    table, anomalies_proto = element
    for slice_key in slicing_util.generate_slices(
        table, [anomalies_util.anomalies_slicer], anomalies=anomalies_proto):
      yield slice_key, table


@beam.typehints.with_input_types(pa.Table)
@beam.typehints.with_output_types(types.BeamSlicedTable)
class IdentifyAnomalousExamples(beam.PTransform):
  """API for identifying anomalous examples.

  Validates each input example against the schema provided in `options` and
  outputs (anomaly reason, anomalous example) tuples.

  Note: This transform requires that the input PCollection consist of pyarrow
  Tables that have a single row (i.e., batch size == 1).
  """

  def __init__(
      self,
      options: stats_options.StatsOptions):
    """Initializes pipeline that identifies anomalous examples.

    Args:
      options: Options for generating data statistics. This must contain a
        schema.
    """

    self.options = options

  @property
  def options(self) -> stats_options.StatsOptions:
    return self._options

  @options.setter
  def options(self, options) -> None:
    if not isinstance(options, stats_options.StatsOptions):
      raise ValueError('options must be a `StatsOptions` object.')
    if options.schema is None:
      raise ValueError('options must include a schema.')
    self._options = options

  def expand(self, dataset: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    return (
        dataset
        | 'DetectAnomaliesInExamples' >> beam.Map(
            _detect_anomalies_in_example, options=self.options)
        | 'GenerateAnomalyReasonKeys' >> beam.ParDo(
            _GenerateAnomalyReasonSliceKeys()))
