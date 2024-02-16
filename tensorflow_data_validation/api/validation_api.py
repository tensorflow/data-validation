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

import itertools
import logging
from typing import Callable, Iterable, List, Optional, Text, Tuple, Set
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.anomalies.proto import custom_validation_config_pb2
from tensorflow_data_validation.anomalies.proto import validation_config_pb2
from tensorflow_data_validation.anomalies.proto import validation_metadata_pb2
from tensorflow_data_validation.api import validation_options as vo
from tensorflow_data_validation.pywrap.tensorflow_data_validation_extension import validation as pywrap_tensorflow_data_validation
from tensorflow_data_validation.skew import feature_skew_detector
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import anomalies_util
from tensorflow_data_validation.utils import slicing_util
from tensorflow_data_validation.utils import stats_util

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2
# Set of anomaly types that do not apply on a per-example basis.
_GLOBAL_ONLY_ANOMALY_TYPES = frozenset([
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_FRACTION_PRESENT,
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
    anomalies_pb2.AnomalyInfo.FEATURE_TYPE_NOT_PRESENT,
    anomalies_pb2.AnomalyInfo.SCHEMA_TRAINING_SERVING_SKEW,
    anomalies_pb2.AnomalyInfo.COMPARATOR_CONTROL_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.COMPARATOR_TREATMENT_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.COMPARATOR_L_INFTY_HIGH,
    anomalies_pb2.AnomalyInfo.COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH,
    anomalies_pb2.AnomalyInfo.COMPARATOR_LOW_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.COMPARATOR_HIGH_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.NO_DATA_IN_SPAN,
    anomalies_pb2.AnomalyInfo.DATASET_LOW_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.DATASET_HIGH_NUM_EXAMPLES,
])

_MULTIPLE_ERRORS = 'Multiple errors'


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

  schema_proto_string = pywrap_tensorflow_data_validation.InferSchema(
      tf.compat.as_bytes(dataset_statistics.SerializeToString()),
      max_string_domain_size, infer_feature_shape)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  _may_be_set_legacy_flag(result)

  if schema_transformations is not None:
    for transformation_fn in schema_transformations:
      result = transformation_fn(result, statistics.datasets[0])
  return result


# Note that this flag is legacy code.
def _may_be_set_legacy_flag(schema: schema_pb2.Schema):
  """Sets legacy flag to False if it exists."""
  if getattr(schema, 'generate_legacy_feature_spec', None) is not None:
    schema.generate_legacy_feature_spec = False


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
    infer_feature_shape: DEPRECATED, do not use. If a feature specifies
      a shape, the shape will always be validated. If the feature does not
      specify a shape, this function will not try inferring a shape from the
      given statistics.
    max_string_domain_size: Maximum size of the domain of a string feature in
      order to be interpreted as a categorical feature.

  Returns:
    A Schema protocol buffer.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics proto contains multiple datasets, none
        of which corresponds to the default slice.
  """
  del infer_feature_shape

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

  schema_proto_string = pywrap_tensorflow_data_validation.UpdateSchema(
      tf.compat.as_bytes(schema.SerializeToString()),
      tf.compat.as_bytes(dataset_statistics.SerializeToString()),
      max_string_domain_size)

  # Parse the serialized Schema proto.
  result = schema_pb2.Schema()
  result.ParseFromString(schema_proto_string)

  return result


def _merge_descriptions(
    anomaly_info: anomalies_pb2.AnomalyInfo,
    other_anomaly_info: Optional[anomalies_pb2.AnomalyInfo]) -> str:
  """Merges anomaly descriptions."""
  descriptions = []
  if other_anomaly_info is not None:
    for reason in itertools.chain(anomaly_info.reason,
                                  other_anomaly_info.reason):
      descriptions.append(reason.description)
  else:
    descriptions = [reason.description for reason in anomaly_info.reason]
  return ' '.join(descriptions)


def _merge_custom_anomalies(
    anomalies: anomalies_pb2.Anomalies,
    custom_anomalies: anomalies_pb2.Anomalies) -> anomalies_pb2.Anomalies:
  """Merges custom_anomalies with anomalies."""
  for key, custom_anomaly_info in custom_anomalies.anomaly_info.items():
    if key in anomalies.anomaly_info:
      # If the key is found in in both inputs, we know it has multiple errors.
      anomalies.anomaly_info[key].short_description = _MULTIPLE_ERRORS
      anomalies.anomaly_info[key].description = _merge_descriptions(
          anomalies.anomaly_info[key], custom_anomaly_info)
      anomalies.anomaly_info[key].severity = max(
          anomalies.anomaly_info[key].severity, custom_anomaly_info.severity)
      anomalies.anomaly_info[key].reason.extend(custom_anomaly_info.reason)
    else:
      anomalies.anomaly_info[key].CopyFrom(custom_anomaly_info)
      # Also populate top-level descriptions.
      anomalies.anomaly_info[key].description = _merge_descriptions(
          custom_anomaly_info, None)
      if len(anomalies.anomaly_info[key].reason) > 1:
        anomalies.anomaly_info[key].short_description = _MULTIPLE_ERRORS
      else:
        anomalies.anomaly_info[
            key].short_description = custom_anomaly_info.reason[
                0].short_description
  return anomalies


def validate_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    schema: schema_pb2.Schema,
    environment: Optional[Text] = None,
    previous_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    serving_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    custom_validation_config: Optional[
        custom_validation_config_pb2.CustomValidationConfig] = None
) -> anomalies_pb2.Anomalies:
  """Validates the input statistics against the provided input schema.

  This method validates the `statistics` against the `schema`. If an optional
  `environment` is specified, the `schema` is filtered using the
  `environment` and the `statistics` is validated against the filtered schema.
  The optional `previous_statistics` and `serving_statistics` are the statistics
  computed over the control data for drift- and skew-detection, respectively.

  If drift- or skew-detection is conducted, then the raw skew/drift measurements
  for each feature that is compared will be recorded in the `drift_skew_info`
  field in the returned `Anomalies` proto.

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
        specifying a `drift_comparator` in the schema.
    serving_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over the serving data. If
        provided, the `validate_statistics` method will identify if there exists
        distribution skew between current data and serving data. Configuration
        for skew detection can be done by specifying a `skew_comparator` in the
        schema.
    custom_validation_config: An optional config that can be used to specify
        custom validations to perform. If doing single-feature validations,
        the test feature will come from `statistics` and will be mapped to
        `feature` in the SQL query. If doing feature pair validations, the test
        feature will come from `statistics` and will be mapped to `feature_test`
        in the SQL query, and the base feature will come from
        `previous_statistics` and will be mapped to `feature_base` in the SQL
        query.

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
                                      previous_statistics, serving_statistics,
                                      None, None, False,
                                      custom_validation_config)


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
    enable_diff_regions: bool = False,
    custom_validation_config: Optional[
        custom_validation_config_pb2.CustomValidationConfig] = None
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
    serving_statistics: An optional DatasetFeatureStatisticsList protocol
        buffer denoting the statistics computed over the serving data. If
        provided, the `validate_statistics_internal` method will identify if
        there exists distribution skew between current data and serving data.
        Configuration for skew detection can be done by specifying a
        `skew_comparator` in the schema.
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
    custom_validation_config: An optional config that can be used to specify
        custom validations to perform. If doing single-feature validations,
        the test feature will come from `statistics` and will be mapped to
        `feature` in the SQL query. If doing feature pair validations, the test
        feature will come from `statistics` and will be mapped to `feature_test`
        in the SQL query, and the base feature will come from
        `previous_statistics` and will be mapped to `feature_base` in the SQL
        query.

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

  if previous_span_statistics is not None:
    if not isinstance(
        previous_span_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'previous_span_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(previous_span_statistics).__name__)

    previous_dataset_statistics = _get_default_dataset_statistics(
        previous_span_statistics)

  if serving_statistics is not None:
    if not isinstance(
        serving_statistics, statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError(
          'serving_statistics is of type %s, should be '
          'a DatasetFeatureStatisticsList proto.'
          % type(serving_statistics).__name__)

    serving_dataset_statistics = _get_default_dataset_statistics(
        serving_statistics)

  if previous_version_statistics is not None:
    if not isinstance(previous_version_statistics,
                      statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError('previous_version_statistics is of type %s, should be '
                      'a DatasetFeatureStatisticsList proto.' %
                      type(previous_version_statistics).__name__)

    previous_version_dataset_statistics = _get_default_dataset_statistics(
        previous_version_statistics)

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

  if custom_validation_config is not None:
    serialized_previous_statistics = previous_span_statistics.SerializeToString(
    ) if previous_span_statistics is not None else ''
    custom_anomalies_string = (
        pywrap_tensorflow_data_validation.CustomValidateStatistics(
            tf.compat.as_bytes(statistics.SerializeToString()),
            tf.compat.as_bytes(serialized_previous_statistics),
            tf.compat.as_bytes(custom_validation_config.SerializeToString()),
            tf.compat.as_bytes(environment)))
    custom_anomalies = anomalies_pb2.Anomalies()
    custom_anomalies.ParseFromString(custom_anomalies_string)
    result = _merge_custom_anomalies(result, custom_anomalies)

  return result


def custom_validate_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    validations: custom_validation_config_pb2.CustomValidationConfig,
    baseline_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    environment: Optional[str] = None) -> anomalies_pb2.Anomalies:
  """Validates the input statistics with the user-supplied SQL queries.

  If the SQL query from a user-supplied validation returns False, TFDV will
  return an anomaly for that validation. In single feature valdiations, the test
  feature will be mapped to `feature` in the SQL query. In two feature
  validations, the test feature will be mapped to `feature_test` in the SQL
  query, and the base feature will be mapped to `feature_base`.

  If an optional `environment` is supplied, TFDV will run validations with
  that environment specified and validations with no environment specified.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer that holds the
      statistics to validate.
    validations: Configuration that specifies the dataset(s) and feature(s) to
      validate and the SQL query to use for the validation. The SQL query must
      return a boolean value.
    baseline_statistics: An optional DatasetFeatureStatisticsList protocol
      buffer that holds the baseline statistics used when validating feature
      pairs.
    environment: If supplied, TFDV will run validations with that
      environment specified and validations with no environment specified. If
      not supplied, TFDV will run all validations.
  Returns:
    An Anomalies protocol buffer.
  """
  serialized_statistics = statistics.SerializeToString()
  serialized_baseline_statistics = (
      baseline_statistics.SerializeToString()
      if baseline_statistics is not None else '')
  serialized_validations = validations.SerializeToString()
  environment = '' if environment is None else environment
  serialized_anomalies = (
      pywrap_tensorflow_data_validation.CustomValidateStatistics(
          tf.compat.as_bytes(serialized_statistics),
          tf.compat.as_bytes(serialized_baseline_statistics),
          tf.compat.as_bytes(serialized_validations),
          tf.compat.as_bytes(environment)))
  result = anomalies_pb2.Anomalies()
  result.ParseFromString(serialized_anomalies)
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


def validate_instance(
    instance: pa.RecordBatch,
    options: stats_options.StatsOptions,
    environment: Optional[str] = None
) -> anomalies_pb2.Anomalies:
  """Validates a batch of examples against the schema provided in `options`.

  If an optional `environment` is specified, the schema is filtered using the
  `environment` and the `instance` is validated against the filtered schema.

  Args:
    instance: A batch of examples in the form of an Arrow RecordBatch.
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


def _detect_anomalies_in_example(record_batch: pa.RecordBatch,
                                 options: stats_options.StatsOptions):
  """Validates the example against the schema provided in `options`."""
  # Verify that we have a single row.
  assert record_batch.num_rows == 1
  return (record_batch, validate_instance(record_batch, options))


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


class _GenerateAnomalyReasonSliceKeys(beam.DoFn):
  """Yields a slice key for each anomaly reason in the Anomalies proto."""

  def process(
      self, element: Tuple[pa.RecordBatch, anomalies_pb2.Anomalies]
  ) -> Iterable[types.SlicedRecordBatch]:
    record_batch, anomalies_proto = element
    for sliced_record_batch in slicing_util.generate_slices(
        record_batch, [anomalies_util.get_anomalies_slicer(anomalies_proto)]):
      yield sliced_record_batch


class IdentifyAnomalousExamples(beam.PTransform):
  """API for identifying anomalous examples.

  Validates each input example against the schema provided in `options` and
  outputs (anomaly reason, anomalous example) tuples.

  Note: This transform requires that the input PCollection consist of pyarrow
  RecordBatches that have a single row (i.e., batch size == 1).
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

  def expand(
      self, dataset: beam.PCollection[pa.RecordBatch]
  ) -> beam.PCollection[types.SlicedRecordBatch]:
    return (
        dataset
        | 'DetectAnomaliesInExamples' >> beam.Map(
            _detect_anomalies_in_example, options=self.options)
        | 'GenerateAnomalyReasonKeys' >> beam.ParDo(
            _GenerateAnomalyReasonSliceKeys()))


class DetectFeatureSkew(beam.PTransform):
  """API for detecting feature skew between training and serving examples.

  Example:

  ```python
    with beam.Pipeline(runner=...) as p:
       training_examples = p | 'ReadTrainingData' >>
         beam.io.ReadFromTFRecord(
            training_filepaths, coder=beam.coders.ProtoCoder(tf.train.Example))
       serving_examples = p | 'ReadServingData' >>
         beam.io.ReadFromTFRecord(
            serving_filepaths, coder=beam.coders.ProtoCoder(tf.train.Example))
       _ = ((training_examples, serving_examples) | 'DetectFeatureSkew' >>
         DetectFeatureSkew(identifier_features=['id1'], sample_size=5)
       | 'WriteFeatureSkewResultsOutput' >>
         tfdv.WriteFeatureSkewResultsToTFRecord(output_path)
       | 'WriteFeatureSkwePairsOutput' >>
       tfdv.WriteFeatureSkewPairsToTFRecord(output_path))
  ```

  See the documentation for DetectFeatureSkewImpl for more detail about feature
  skew detection.
  """

  def __init__(
      self,
      identifier_features: List[types.FeatureName],
      features_to_ignore: Optional[List[types.FeatureName]] = None,
      sample_size: int = 0,
      float_round_ndigits: Optional[int] = None,
      allow_duplicate_identifiers: bool = False) -> None:
    """Initializes the feature skew detection PTransform.

    Args:
      identifier_features: Names of features to use as identifiers.
      features_to_ignore: Names of features for which no feature skew detection
        is done.
      sample_size: Size of the sample of training-serving example pairs that
        exhibit skew to include in the skew results.
      float_round_ndigits: Number of digits precision after the decimal point to
        which to round float values before comparing them.
      allow_duplicate_identifiers: If set, skew detection will be done on
        examples for which there are duplicate identifier feature values. In
        this case, the counts in the FeatureSkew result are based on each
        training-serving example pair analyzed. Examples with given identifier
        feature values must all fit in memory.
    """
    self._identifier_features = identifier_features
    self._features_to_ignore = features_to_ignore
    self._sample_size = sample_size
    self._float_round_ndigits = float_round_ndigits
    self._allow_duplicate_identifiers = allow_duplicate_identifiers

  def expand(
      self, datasets: Tuple[beam.pvalue.PCollection, beam.pvalue.PCollection]
  ) -> Tuple[beam.pvalue.PCollection, beam.pvalue.PCollection]:

    result = (
        datasets
        | 'DetectFeatureSkew' >> feature_skew_detector.DetectFeatureSkewImpl(
            self._identifier_features, self._features_to_ignore,
            self._sample_size, self._float_round_ndigits,
            self._allow_duplicate_identifiers))
    return result[feature_skew_detector.SKEW_RESULTS_KEY], result[
        feature_skew_detector.SKEW_PAIRS_KEY]


@beam.typehints.with_input_types(feature_skew_results_pb2.FeatureSkew)
class WriteFeatureSkewResultsToTFRecord(beam.PTransform):
  """API for writing serialized feature skew results to a TFRecord file."""

  def __init__(self, output_path: str) -> None:
    """Initializes the transform.

    Args:
      output_path: Output path for writing feature skew results.
    """
    self._output_path = output_path

  def expand(self, feature_skew_results: beam.PCollection) -> beam.pvalue.PDone:
    return (feature_skew_results
            | 'WriteFeatureSkewResults' >> beam.io.WriteToTFRecord(
                self._output_path,
                shard_name_template='',
                coder=beam.coders.ProtoCoder(
                    feature_skew_results_pb2.FeatureSkew)))


@beam.typehints.with_input_types(feature_skew_results_pb2.SkewPair)
class WriteSkewPairsToTFRecord(beam.PTransform):
  """API for writing serialized skew pairs to a TFRecord file."""

  def __init__(self, output_path: str) -> None:
    """Initializes the transform.

    Args:
      output_path: Output path for writing skew pairs.
    """
    self._output_path = output_path

  def expand(self, skew_pairs: beam.PCollection) -> beam.pvalue.PDone:
    return (skew_pairs
            | 'WriteSkewPairs' >> beam.io.WriteToTFRecord(
                self._output_path,
                shard_name_template='',
                coder=beam.coders.ProtoCoder(
                    feature_skew_results_pb2.SkewPair)))


def _prepend_slice_path(slice_name: str,
                        path: types.FeaturePath) -> types.FeaturePath:
  steps = path.steps()
  return types.FeaturePath(('slice(%s)::' % slice_name + steps[0],) + steps[1:])


def _prepend_slice_name(slice_name: str, name: str) -> str:
  return 'slice(%s)::' % slice_name + name


def _flatten_statistics_for_sliced_validation(
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> Tuple[statistics_pb2.DatasetFeatureStatisticsList, Set[str]]:
  """Flattens sliced stats into unsliced stats with prepended slice keys."""
  result = statistics_pb2.DatasetFeatureStatisticsList()
  dataset_flat = result.datasets.add()
  # Copy top level metadata from the default (overall) slice.
  default_slice = stats_util.DatasetListView(statistics).get_default_slice()
  if default_slice is None:
    raise ValueError('Missing default slice')
  dataset_flat.CopyFrom(default_slice.proto())
  dataset_flat.ClearField('features')
  dataset_flat.ClearField('cross_features')
  slice_names = set()
  for dataset in statistics.datasets:
    slice_names.add(dataset.name)
    for feature in dataset.features:
      copied_feature = dataset_flat.features.add()
      copied_feature.CopyFrom(feature)
      copied_feature.path.CopyFrom(
          _prepend_slice_path(dataset.name,
                              types.FeaturePath.from_proto(
                                  copied_feature.path)).to_proto())
    for cross_feature in dataset.cross_features:
      copied_cross_feature = dataset_flat.cross_features.add()
      copied_cross_feature.CopyFrom(cross_feature)
      copied_cross_feature.path_x.CopyFrom(
          _prepend_slice_path(
              dataset.name,
              types.FeaturePath.from_proto(
                  copied_cross_feature.path_x)).to_proto())
      copied_cross_feature.path_y.CopyFrom(
          _prepend_slice_path(
              dataset.name,
              types.FeaturePath.from_proto(
                  copied_cross_feature.path_y)).to_proto())
  return result, slice_names


def _replicate_schema_for_sliced_validation(
    schema: schema_pb2.Schema, slice_names: Set[str]) -> schema_pb2.Schema:
  """Replicates features in a schema with prepended slice names."""
  if schema.HasField('dataset_constraints') is not None:
    logging.error('DatasetConstraints will not be validated per-slice.')
  result = schema_pb2.Schema()
  result.string_domain.extend(schema.string_domain)
  result.float_domain.extend(schema.float_domain)
  result.int_domain.extend(schema.int_domain)
  for slice_name in slice_names:
    for feature in schema.feature:
      new_feature = result.feature.add()
      new_feature.CopyFrom(feature)
      new_feature.name = _prepend_slice_name(slice_name, feature.name)
    for sparse_feature in schema.sparse_feature:
      new_sparse_feature = result.sparse_feature.add()
      new_sparse_feature.CopyFrom(sparse_feature)
      new_sparse_feature.name = _prepend_slice_name(slice_name,
                                                    sparse_feature.name)
    for weighted_feature in schema.weighted_feature:
      new_weighted_feature = result.weighted_feature.add()
      new_weighted_feature.CopyFrom(weighted_feature)
      new_weighted_feature.name = _prepend_slice_name(slice_name,
                                                      weighted_feature.name)
  return result


def validate_corresponding_slices(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    schema: schema_pb2.Schema,
    environment: Optional[Text] = None,
    previous_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    serving_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
) -> anomalies_pb2.Anomalies:
  """Validates corresponding sliced statistics.

  Sliced statistics are flattened into a single unsliced stats input prior to
  validation. If multiple statistics are provided, validation is performed on
  corresponding slices. DatasetConstraints, if present, are applied to the
  overall slice.

  Note: This API is experimental and subject to change.

  Args:
    statistics: See validate_statistics.
    schema: See validate_statistics.
    environment: See validate_statistics.
    previous_statistics: See validate_statistics.
    serving_statistics: See validate_statistics.

  Returns:
    An Anomalies protocol buffer.

  Raises:
    TypeError: If any of the input arguments is not of the expected type.
  """
  all_slice_keys = set()
  statistics, keys = _flatten_statistics_for_sliced_validation(statistics)
  all_slice_keys.update(keys)
  if previous_statistics:
    previous_statistics, keys = _flatten_statistics_for_sliced_validation(
        previous_statistics)
    all_slice_keys.update(keys)
  if serving_statistics:
    serving_statistics, keys = _flatten_statistics_for_sliced_validation(
        serving_statistics)
    all_slice_keys.update(keys)
  schema = _replicate_schema_for_sliced_validation(schema, all_slice_keys)
  return validate_statistics(statistics, schema, environment,
                             previous_statistics, serving_statistics)
