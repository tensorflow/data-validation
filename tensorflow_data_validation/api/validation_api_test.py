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

"""Tests for Validation API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation import types
from tensorflow_data_validation.anomalies.proto import custom_validation_config_pb2
from tensorflow_data_validation.api import validation_api
from tensorflow_data_validation.api import validation_options
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.types import FeaturePath
from tensorflow_data_validation.utils import schema_util
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format

from tensorflow.python.util.protobuf import compare  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


IDENTIFY_ANOMALOUS_EXAMPLES_VALID_INPUTS = [{
    'testcase_name':
        'no_anomalies',
    'examples': [
        pa.RecordBatch.from_arrays([pa.array([['A']])], ['annotated_enum']),
        pa.RecordBatch.from_arrays([pa.array([['C']])], ['annotated_enum']),
    ],
    'schema_text':
        """
              string_domain {
                name: "MyAloneEnum"
                value: "A"
                value: "B"
                value: "C"
              }
              feature {
                name: "annotated_enum"
                value_count {
                  min:1
                  max:1
                }
                presence {
                  min_count: 1
                }
                type: BYTES
                domain: "MyAloneEnum"
              }
              feature {
                name: "ignore_this"
                lifecycle_stage: DEPRECATED
                value_count {
                  min:1
                }
                presence {
                  min_count: 1
                }
                type: BYTES
              }
              """,
    'expected_result': []
}, {
    'testcase_name':
        'same_anomaly_reason',
    'examples': [
        pa.RecordBatch.from_arrays([pa.array([['D']])], ['annotated_enum']),
        pa.RecordBatch.from_arrays([pa.array([['D']])], ['annotated_enum']),
        pa.RecordBatch.from_arrays([pa.array([['C']])], ['annotated_enum']),
    ],
    'schema_text':
        """
              string_domain {
                name: "MyAloneEnum"
                value: "A"
                value: "B"
                value: "C"
              }
              feature {
                name: "annotated_enum"
                value_count {
                  min:1
                  max:1
                }
                presence {
                  min_count: 1
                }
                type: BYTES
                domain: "MyAloneEnum"
              }
              """,
    'expected_result': [('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                         pa.RecordBatch.from_arrays([pa.array([['D']])],
                                                    ['annotated_enum'])),
                        ('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                         pa.RecordBatch.from_arrays([pa.array([['D']])],
                                                    ['annotated_enum']))]
}, {
    'testcase_name':
        'different_anomaly_reasons',
    'examples': [
        pa.RecordBatch.from_arrays([pa.array([['D']])], ['annotated_enum']),
        pa.RecordBatch.from_arrays([pa.array([['C']])], ['annotated_enum']),
        pa.RecordBatch.from_arrays([pa.array([[1]])],
                                   ['feature_not_in_schema']),
    ],
    'schema_text':
        """
              string_domain {
                name: "MyAloneEnum"
                value: "A"
                value: "B"
                value: "C"
              }
              feature {
                name: "annotated_enum"
                value_count {
                  min:1
                  max:1
                }
                presence {
                  min_count: 0
                }
                type: BYTES
                domain: "MyAloneEnum"
              }
              """,
    'expected_result': [('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                         pa.RecordBatch.from_arrays([pa.array([['D']])],
                                                    ['annotated_enum'])),
                        ('feature_not_in_schema_SCHEMA_NEW_COLUMN',
                         pa.RecordBatch.from_arrays([pa.array([[1]])],
                                                    ['feature_not_in_schema']))]
}]


class ValidationTestCase(parameterized.TestCase):

  def _assert_equal_anomalies(self, actual_anomalies, expected_anomalies):
    # Check if the actual anomalies matches with the expected anomalies.
    for feature_name in expected_anomalies:
      self.assertIn(feature_name, actual_anomalies.anomaly_info)
      # Doesn't compare the diff_regions.
      actual_anomalies.anomaly_info[feature_name].ClearField('diff_regions')

      self.assertEqual(actual_anomalies.anomaly_info[feature_name],
                       expected_anomalies[feature_name])
    self.assertEqual(
        len(actual_anomalies.anomaly_info), len(expected_anomalies))


class ValidationApiTest(ValidationTestCase):

  def test_infer_schema(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                infer_feature_shape=False)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_with_string_domain(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 6
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 3
                min_num_values: 1
                max_num_values: 1
              }
              unique: 2
              rank_histogram {
                buckets {
                  low_rank: 0
                  high_rank: 0
                  label: "a"
                  sample_count: 2.0
                }
                buckets {
                  low_rank: 1
                  high_rank: 1
                  label: "b"
                  sample_count: 1.0
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_count: 1
          }
          type: BYTES
          domain: "feature1"
        }
        string_domain {
          name: "feature1"
          value: "a"
          value: "b"
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_without_string_domain(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 6
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 3
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  low_rank: 0
                  high_rank: 0
                  label: "a"
                  sample_count: 2.0
                }
                buckets {
                  low_rank: 1
                  high_rank: 1
                  label: "b"
                  sample_count: 1.0
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                max_string_domain_size=1)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_with_infer_shape(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 0
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
            }
          }
          features: {
            path { step: 'feature2' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 2
                num_non_missing: 5
                min_num_values: 1
                max_num_values: 1
              }
              unique: 5
            }
          }
          features: {
            path { step: 'feature3' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 0
                num_non_missing: 7
                min_num_values: 0
                max_num_values: 1
              }
              unique: 5
            }
          }
          features: {
            path { step: 'nested_feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 0
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 7
                  min_num_values: 1
                  max_num_values: 1
                }
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 7
                  min_num_values: 1
                  max_num_values: 1
                }
              }
              unique: 3
            }
          }
          features: {
            path { step: 'nested_feature2' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 2
                num_non_missing: 5
                min_num_values: 1
                max_num_values: 1
                presence_and_valency_stats {
                  num_missing: 2
                  num_non_missing: 5
                  min_num_values: 1
                  max_num_values: 1
                }
                presence_and_valency_stats {
                  num_missing: 2
                  num_non_missing: 5
                  min_num_values: 1
                  max_num_values: 1
                }
              }
              unique: 5
            }
          }
          features: {
            path { step: 'nested_feature3' }
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 0
                num_non_missing: 7
                min_num_values: 0
                max_num_values: 1
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 7
                  min_num_values: 0
                  max_num_values: 1
                }
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 7
                  min_num_values: 0
                  max_num_values: 1
                }
              }
              unique: 5
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          shape { dim { size: 1 } }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "feature2"
          value_count: { min: 1 max: 1}
          presence: {
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "feature3"
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "nested_feature1"
          shape: {
            dim { size: 1 }
            dim { size: 1 }
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "nested_feature2"
          value_counts: {
              value_count { min: 1 max: 1 }
              value_count { min: 1 max: 1 }
          }
          presence: {
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "nested_feature3"
          value_counts: {
              value_count { }
              value_count { }
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                infer_feature_shape=True)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_with_transformations(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            path { step: 'foo' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
            }
          }
          features: {
            path { step: 'xyz_query' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    def _semantic_type_transformation_fn(schema, unused_stats):
      for feature in schema.feature:
        if 'query' in feature.name:
          feature.natural_language_domain.CopyFrom(
              schema_pb2.NaturalLanguageDomain())
      return schema

    expected_schema = text_format.Parse(
        """
        feature {
          name: "foo"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "xyz_query"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
          natural_language_domain {}
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(
        statistics, infer_feature_shape=False,
        schema_transformations=[_semantic_type_transformation_fn])
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_multiple_datasets_with_default_slice(self):
    statistics = text_format.Parse(
        """
        datasets {
          name: 'All Examples'
          num_examples: 7
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 2
              }
              unique: 3
            }
          }
        }
        datasets {
          name: 'feature1_testvalue'
          num_examples: 4
          features: {
            path { step: 'feature1' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 1
              }
              unique: 1
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "feature1"
          value_count: {
            min: 1
          }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                infer_feature_shape=False)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_invalid_statistics_input(self):
    with self.assertRaisesRegexp(
        TypeError, '.*should be a DatasetFeatureStatisticsList proto.*'):
      _ = validation_api.infer_schema({})

  def test_infer_schema_invalid_multiple_datasets_no_default_slice(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    with self.assertRaisesRegexp(ValueError,
                                 '.*statistics proto with one dataset.*'):
      _ = validation_api.infer_schema(statistics)

  def test_infer_schema_composite_feature_stats(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 4
          features: {
            path { step: 'value' }
            type: STRING
            string_stats: {
              common_stats: {
                num_non_missing: 2
                num_missing: 2
                min_num_values: 1
                max_num_values: 1
              }
            }
          }
          features: {
            path { step: 'weight' }
            type: FLOAT
            num_stats: {
              common_stats: {
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 1
              }
            }
          }
          features: {
            path { step: 'index' }
            type: INT
            num_stats: {
              common_stats: {
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 1
              }
            }
          }
          features: {
            path { step: 'weighted_feature' }
            custom_stats {
              name: 'missing_value'
              num: 2
            }
          }
          features: {
            path { step: 'sparse_feature' }
            custom_stats {
              name: 'missing_value'
              num: 2
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    expected_schema = text_format.Parse(
        """
        feature {
          name: "value"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "weight"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_fraction: 1
            min_count: 1
          }
          type: FLOAT
        }
        feature {
          name: "index"
          value_count: {
            min: 1
            max: 1
          }
          presence: {
            min_fraction: 1
            min_count: 1
          }
          type: INT
        }
        """, schema_pb2.Schema())
    validation_api._may_be_set_legacy_flag(expected_schema)

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                infer_feature_shape=False)
    self.assertEqual(actual_schema, expected_schema)

  def _assert_drift_skew_info(
      self, actual_drift_skew_infos, expected_drift_skew_infos):
    self.assertLen(actual_drift_skew_infos, len(expected_drift_skew_infos))
    expected_drift_skew_infos = [
        text_format.Parse(e, anomalies_pb2.DriftSkewInfo())
        for e in expected_drift_skew_infos
    ]
    path_to_expected = {
        tuple(e.path.step): e for e in expected_drift_skew_infos
    }
    def check_measurements(actual_measurements, expected_measurements):
      for actual_measurement, expected_measurement in zip(
          actual_measurements, expected_measurements):
        self.assertEqual(actual_measurement.type, expected_measurement.type)
        self.assertAlmostEqual(actual_measurement.value,
                               expected_measurement.value)
        self.assertAlmostEqual(actual_measurement.threshold,
                               expected_measurement.threshold)

    for actual in actual_drift_skew_infos:
      expected = path_to_expected[tuple(actual.path.step)]
      self.assertIsNotNone(
          expected, 'Did not expect a DriftSkewInfo for {}'.format(
              tuple(actual.path.step)))

      check_measurements(actual.drift_measurements, expected.drift_measurements)
      check_measurements(actual.skew_measurements, expected.skew_measurements)

  def test_update_schema(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
        }
        feature {
          name: "annotated_enum"
          value_count {
            min:1
            max:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
      path {
        step: "annotated_enum"
      }
      description: "Examples contain values missing from the schema: D (?). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D (?). "
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

    # Verify the updated schema.
    actual_updated_schema = validation_api.update_schema(schema, statistics)
    expected_updated_schema = schema
    schema_util.get_domain(
        expected_updated_schema,
        types.FeaturePath(['annotated_enum'])).value.append('D')
    self.assertEqual(actual_updated_schema, expected_updated_schema)

    # Verify that there are no anomalies with the updated schema.
    actual_updated_anomalies = validation_api.validate_statistics(
        statistics, actual_updated_schema)
    self._assert_equal_anomalies(actual_updated_anomalies, {})

  def test_update_schema_multiple_datasets_with_default_slice(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
        }
        feature {
          name: "annotated_enum"
          value_count {
            min:1
            max:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets{
          name: 'All Examples'
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        datasets{
          name: 'other dataset'
          num_examples: 5
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 2
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "E"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
      path {
        step: "annotated_enum"
      }
      description: "Examples contain values missing from the schema: D (?). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D (?). "
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

    # Verify the updated schema.
    actual_updated_schema = validation_api.update_schema(schema, statistics)
    expected_updated_schema = schema
    schema_util.get_domain(
        expected_updated_schema,
        types.FeaturePath(['annotated_enum'])).value.append('D')
    self.assertEqual(actual_updated_schema, expected_updated_schema)

    # Verify that there are no anomalies with the updated schema.
    actual_updated_anomalies = validation_api.validate_statistics(
        statistics, actual_updated_schema)
    self._assert_equal_anomalies(actual_updated_anomalies, {})

  def test_update_schema_invalid_schema_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    with self.assertRaisesRegexp(
        TypeError, 'schema is of type.*'):
      _ = validation_api.update_schema({}, statistics)

  def test_update_schema_invalid_statistics_input(self):
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        TypeError, 'statistics is of type.*'):
      _ = validation_api.update_schema(schema, {})

  def test_update_schema_invalid_multiple_datasets_no_default_slice(self):
    schema = schema_pb2.Schema()
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    with self.assertRaisesRegexp(ValueError,
                                 '.*statistics proto with one dataset.*'):
      _ = validation_api.update_schema(schema, statistics)

  # See b/179197768.
  def test_update_schema_remove_inferred_shape(self):
    stats1 = text_format.Parse("""
    datasets {
      num_examples: 10000
      features {
        path {
          step: ["f1"]
        }
        num_stats {
          common_stats {
            num_non_missing: 10000
            num_missing: 0
            min_num_values: 1
            max_num_values: 1
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    stats2 = text_format.Parse("""
    datasets {
      num_examples: 10000
      features {
        path {
          step: ["f1"]
        }
        num_stats {
          common_stats {
            num_non_missing: 9999
            num_missing: 1
            min_num_values: 1
            max_num_values: 1
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    # Scenario 1: shape is inferred from stats1, then should be removed
    # when schema is updated against stats2.
    schema = validation_api.infer_schema(stats1, infer_feature_shape=True)
    self.assertLen(schema.feature, 1)
    self.assertTrue(schema.feature[0].HasField('shape'))

    updated_schema = validation_api.update_schema(schema, stats2)
    self.assertLen(updated_schema.feature, 1)
    self.assertFalse(updated_schema.feature[0].HasField('shape'))

    # once shape is dropped, it should not be added back, even if the stats
    # provided support a fixed shape.
    updated_schema = validation_api.update_schema(updated_schema, stats1)
    self.assertLen(updated_schema.feature, 1)
    self.assertFalse(updated_schema.feature[0].HasField('shape'))

    # Scenario 2: shape is not inferred from stats2, then should not be
    # added when schema is updated against stat1.
    schema = validation_api.infer_schema(stats2, infer_feature_shape=True)
    self.assertLen(schema.feature, 1)
    self.assertFalse(schema.feature[0].HasField('shape'))

    updated_schema = validation_api.update_schema(schema, stats1)
    self.assertLen(updated_schema.feature, 1)
    self.assertFalse(updated_schema.feature[0].HasField('shape'))

    # Scenario 3: shape is inferred from stats1, then should not be removed
    # when schema is updated against (again) stats1.
    schema = validation_api.infer_schema(stats1, infer_feature_shape=True)
    self.assertLen(schema.feature, 1)
    self.assertTrue(schema.feature[0].HasField('shape'))

    updated_schema = validation_api.update_schema(schema, stats1)
    self.assertLen(updated_schema.feature, 1)
    self.assertTrue(updated_schema.feature[0].HasField('shape'))

  def test_validate_stats(self):
    schema = text_format.Parse(
        """
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
        }
        feature {
          name: "annotated_enum"
          value_count {
            min:1
            max:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          value_count {
            min:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
      path {
        step: "annotated_enum"
      }
      description: "Examples contain values missing from the schema: D (?). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D (?). "
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_stats_weighted_feature(self):
    schema = text_format.Parse(
        """
        feature {
          name: "value"
        }
        feature {
          name: "weight"
        }
        weighted_feature {
          name: "weighted_feature"
          feature {
            step: "value"
          }
          weight_feature {
            step: "weight"
          }
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'weighted_feature' }
            custom_stats {
              name: 'missing_weight'
              num: 1.0
            }
            custom_stats {
              name: 'missing_value'
              num: 2.0
            }
            custom_stats {
              name: 'min_weight_length_diff'
              num: 3.0
            }
            custom_stats {
              name: 'max_weight_length_diff'
              num: 4.0
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'weighted_feature':
            text_format.Parse(
                """
      path {
        step: "weighted_feature"
      }
      description: "Found 1 examples missing weight feature. Found 2 examples missing value feature. Mismatch between weight and value feature with min_weight_length_diff = 3 and max_weight_length_diff = 4."
      severity: ERROR
      short_description: "Multiple errors"
      reason {
        type: WEIGHTED_FEATURE_MISSING_WEIGHT
        short_description: "Missing weight feature"
        description: "Found 1 examples missing weight feature."
      }
      reason {
        type: WEIGHTED_FEATURE_MISSING_VALUE
        short_description: "Missing value feature"
        description: "Found 2 examples missing value feature."
      }
      reason {
        type: WEIGHTED_FEATURE_LENGTH_MISMATCH
        short_description: "Length mismatch between value and weight feature"
        description: "Mismatch between weight and value feature with min_weight_length_diff = 3 and max_weight_length_diff = 4."
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_stats_weighted_feature_name_collision(self):
    schema = text_format.Parse(
        """
        feature {
          name: "value"
        }
        feature {
          name: "weight"
        }
        feature {
          name: "colliding_feature"
        }
        weighted_feature {
          name: "colliding_feature"
          feature {
            step: "value"
          }
          weight_feature {
            step: "weight"
          }
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'colliding_feature' }
            custom_stats {
              name: 'missing_weight'
              num: 1.0
            }
            custom_stats {
              name: 'missing_value'
              num: 2.0
            }
            custom_stats {
              name: 'min_weight_length_diff'
              num: 3.0
            }
            custom_stats {
              name: 'max_weight_length_diff'
              num: 4.0
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'colliding_feature':
            text_format.Parse(
                """
      path {
        step: "colliding_feature"
      }
      description: "Weighted feature name collision."
      severity: ERROR
      short_description: "Weighted feature name collision"
      reason {
        type: WEIGHTED_FEATURE_NAME_COLLISION
        short_description: "Weighted feature name collision"
        description: "Weighted feature name collision."
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_stats_weighted_feature_sparse_feature_name_collision(self):
    schema = text_format.Parse(
        """
        feature {
          name: "value"
        }
        feature {
          name: "weight"
        }
        feature {
          name: "index"
        }
        weighted_feature {
          name: "colliding_feature"
          feature {
            step: "value"
          }
          weight_feature {
            step: "weight"
          }
        }
        sparse_feature {
          name: "colliding_feature"
          value_feature {
            name: "value"
          }
          index_feature {
            name: "index"
          }
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'colliding_feature' }
            custom_stats {
              name: 'missing_weight'
              num: 1.0
            }
            custom_stats {
              name: 'missing_index'
              num: 1.0
            }
            custom_stats {
              name: 'missing_value'
              num: 2.0
            }
            custom_stats {
              name: 'missing_value'
              num: 2.0
            }
            custom_stats {
              name: 'min_length_diff'
              num: 3.0
            }
            custom_stats {
              name: 'min_weight_length_diff'
              num: 3.0
            }
            custom_stats {
              name: 'max_length_diff'
              num: 4.0
            }
            custom_stats {
              name: 'max_weight_length_diff'
              num: 4.0
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    expected_anomalies = {
        'colliding_feature':
            text_format.Parse(
                """
      path {
        step: "colliding_feature"
      }
      description: "Weighted feature name collision."
      severity: ERROR
      short_description: "Weighted feature name collision"
      reason {
        type: WEIGHTED_FEATURE_NAME_COLLISION
        short_description: "Weighted feature name collision"
        description: "Weighted feature name collision."
      }
            """, anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  # pylint: disable=line-too-long
  _annotated_enum_anomaly_info = """
            path {
              step: "annotated_enum"
            }
            description: "Examples contain values missing from the schema: b (?).  The Linfty distance between current and previous is 0.25 (up to six significant digits), above the threshold 0.01. The feature value with maximum difference is: b"
            severity: ERROR
            short_description: "Multiple errors"
            reason {
              type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
              short_description: "Unexpected string values"
              description: "Examples contain values missing from the schema: b (?). "
            }
            reason {
              type: COMPARATOR_L_INFTY_HIGH
              short_description: "High Linfty distance between current and previous"
              description: "The Linfty distance between current and previous is 0.25 (up to six significant digits), above the threshold 0.01. The feature value with maximum difference is: b"
            }"""

  _bar_anomaly_info = """
            path {
              step: "bar"
            }
            short_description: "High Linfty distance between training and serving"
            description: "The Linfty distance between training and serving is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
            severity: ERROR
            reason {
              type: COMPARATOR_L_INFTY_HIGH
              short_description: "High Linfty distance between training and serving"
              description: "The Linfty distance between training and serving is 0.2 (up to six significant digits), above the threshold 0.1. The feature value with maximum difference is: a"
            }"""

  def test_validate_stats_with_previous_stats(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 2
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats { num_non_missing: 2 num_missing: 0 max_num_values: 1 }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    previous_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 4
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats { num_non_missing: 4 num_missing: 0 max_num_values: 1 }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    schema = text_format.Parse(
        """
        feature {
          name: "annotated_enum"
          type: BYTES
          domain: "annotated_enum"
          drift_comparator { infinity_norm { threshold: 0.01 } }
        }
        string_domain { name: "annotated_enum" value: "a" }
        """, schema_pb2.Schema())

    expected_anomalies = {
        'annotated_enum': text_format.Parse(self._annotated_enum_anomaly_info,
                                            anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(
        statistics, schema, previous_statistics=previous_statistics)
    self._assert_drift_skew_info(anomalies.drift_skew_info, [
        """
        path { step: ["annotated_enum"] }
        drift_measurements {
          type: L_INFTY
          value: 0.25
          threshold: 0.01
        }
        """,
    ])
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  @parameterized.named_parameters(*[
      dict(testcase_name='no_skew',
           has_skew=False),
      dict(testcase_name='with_skew',
           has_skew=True),
  ])
  def test_validate_stats_with_serving_stats(self, has_skew):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 2 }
                buckets { label: "c" sample_count: 7 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    serving_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            name: 'bar'
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    threshold = 0.1 if has_skew else 1.0
    schema = text_format.Parse(
        """
        feature {
          name: 'bar'
          type: BYTES
          skew_comparator {
            infinity_norm { threshold: %f }
          }
        }""" % threshold, schema_pb2.Schema())

    expected_anomalies = {}
    if has_skew:
      expected_anomalies['bar'] = text_format.Parse(
          self._bar_anomaly_info, anomalies_pb2.AnomalyInfo())
    # Validate the stats.
    anomalies = validation_api.validate_statistics(
        statistics, schema, serving_statistics=serving_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)
    self._assert_drift_skew_info(anomalies.drift_skew_info, [
        """
        path { step: ["bar"] }
        skew_measurements {
          type: L_INFTY
          value: 0.2
          threshold: %f
        }
        """ % threshold,
    ])

  def test_validate_stats_with_environment(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 1000
          features {
            path { step: 'feature' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 1000
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    schema = text_format.Parse(
        """
        default_environment: "TRAINING"
        default_environment: "SERVING"
        feature {
          name: "label"
          not_in_environment: "SERVING"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        feature {
          name: "feature"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        """, schema_pb2.Schema())

    expected_anomalies_training = {
        'label':
            text_format.Parse(
                """
            path {
              step: "label"
            }
            description: "Column is completely missing"
            severity: ERROR
            short_description: "Column dropped"
            reason {
              type: SCHEMA_MISSING_COLUMN
              short_description: "Column dropped"
              description: "Column is completely missing"
            }
            """, anomalies_pb2.AnomalyInfo())
    }
    # Validate the stats in TRAINING environment.
    anomalies_training = validation_api.validate_statistics(
        statistics, schema, environment='TRAINING')
    self._assert_equal_anomalies(anomalies_training,
                                 expected_anomalies_training)

    # Validate the stats in SERVING environment.
    anomalies_serving = validation_api.validate_statistics(
        statistics, schema, environment='SERVING')
    self._assert_equal_anomalies(anomalies_serving, {})

  def test_validate_stats_with_previous_and_serving_stats(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 2 }
                buckets { label: "c" sample_count: 7 }
              }
            }
          }
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    previous_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    serving_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    schema = text_format.Parse(
        """
        feature {
          name: 'bar'
          type: BYTES
          skew_comparator { infinity_norm { threshold: 0.1 } }
        }
        feature {
          name: "annotated_enum"
          type: BYTES
          domain: "annotated_enum"
          drift_comparator { infinity_norm { threshold: 0.01 } }
        }
        string_domain { name: "annotated_enum" value: "a" }
        """, schema_pb2.Schema())

    expected_anomalies = {
        'bar': text_format.Parse(self._bar_anomaly_info,
                                 anomalies_pb2.AnomalyInfo()),
        'annotated_enum': text_format.Parse(self._annotated_enum_anomaly_info,
                                            anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(
        statistics,
        schema,
        previous_statistics=previous_statistics,
        serving_statistics=serving_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)
    self._assert_drift_skew_info(anomalies.drift_skew_info, [
        """
        path { step: ["bar"] }
        skew_measurements {
          type: L_INFTY
          value: 0.2
          threshold: 0.1
        }
        """,
        """
        path { step: ["annotated_enum"] }
        drift_measurements {
          type: L_INFTY
          value: 0.25
          threshold: 0.01
        }
        """,
    ])

  # pylint: enable=line-too-long

  def test_validate_stats_with_previous_and_serving_stats_with_default_slices(
      self):
    # All input statistics protos have multiple datasets, one of which
    # corresponds to the default slice.
    statistics = text_format.Parse(
        """
        datasets {
          name: 'All Examples'
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    previous_statistics = text_format.Parse(
        """
        datasets {
          name: 'All Examples'
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }
        datasets {
          name: "annotated_enum_b"
          num_examples: 1
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 1
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    serving_statistics = text_format.Parse(
        """
        datasets {
          name: 'All Examples'
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }
        datasets {
          name: "annotated_enum_a"
          num_examples: 3
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 3
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    schema = text_format.Parse(
        """
        feature {
          name: "annotated_enum"
          type: BYTES
          domain: "annotated_enum"
          drift_comparator { infinity_norm { threshold: 0.01 } }
        }
        string_domain { name: "annotated_enum" value: "a" }
        """, schema_pb2.Schema())

    expected_anomalies = {
        'annotated_enum': text_format.Parse(self._annotated_enum_anomaly_info,
                                            anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics(
        statistics,
        schema,
        previous_statistics=previous_statistics,
        serving_statistics=serving_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)
  # pylint: enable=line-too-long

  def test_validate_stats_invalid_statistics_input(self):
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        TypeError, 'statistics is of type.*'):
      _ = validation_api.validate_statistics({}, schema)

  def test_validate_stats_invalid_previous_statistics_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        TypeError, 'previous_statistics is of type.*'):
      _ = validation_api.validate_statistics(statistics, schema,
                                             previous_statistics='test')

  def test_validate_stats_internal_invalid_previous_span_statistics_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(TypeError,
                                 'previous_span_statistics is of type.*'):
      _ = validation_api.validate_statistics_internal(
          statistics, schema, previous_span_statistics='test')

  def test_validate_stats_invalid_serving_statistics_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        TypeError, 'serving_statistics is of type.*'):
      _ = validation_api.validate_statistics(statistics, schema,
                                             serving_statistics='test')

  def test_validate_stats_invalid_previous_version_statistics_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(TypeError,
                                 'previous_version_statistics is of type.*'):
      _ = validation_api.validate_statistics_internal(
          statistics, schema, previous_version_statistics='test')

  def test_validate_stats_invalid_schema_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    with self.assertRaisesRegexp(TypeError, '.*should be a Schema proto.*'):
      _ = validation_api.validate_statistics(statistics, {})

  def test_validate_stats_invalid_environment(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([statistics_pb2.DatasetFeatureStatistics()])
    schema = text_format.Parse(
        """
        default_environment: "TRAINING"
        default_environment: "SERVING"
        feature {
          name: "label"
          not_in_environment: "SERVING"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        """, schema_pb2.Schema())
    with self.assertRaisesRegexp(
        ValueError, 'Environment.*not found in the schema.*'):
      _ = validation_api.validate_statistics(statistics, schema,
                                             environment='INVALID')

  def test_validate_stats_invalid_statistics_multiple_datasets_no_default_slice(
      self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        ValueError, 'Only statistics proto with one dataset or the default.*'):
      _ = validation_api.validate_statistics(statistics, schema)

  def test_validate_stats_invalid_previous_statistics_multiple_datasets(self):
    current_stats = statistics_pb2.DatasetFeatureStatisticsList()
    current_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics()
    ])
    previous_stats = statistics_pb2.DatasetFeatureStatisticsList()
    previous_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        ValueError, 'Only statistics proto with one dataset or the default.*'):
      _ = validation_api.validate_statistics(current_stats, schema,
                                             previous_statistics=previous_stats)

  def test_validate_stats_invalid_serving_statistics_multiple_datasets(self):
    current_stats = statistics_pb2.DatasetFeatureStatisticsList()
    current_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics()
    ])
    serving_stats = statistics_pb2.DatasetFeatureStatisticsList()
    serving_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        ValueError, 'Only statistics proto with one dataset or the default.*'):
      _ = validation_api.validate_statistics(current_stats, schema,
                                             serving_statistics=serving_stats)

  def test_validate_stats_invalid_previous_version_stats_multiple_datasets(
      self):
    current_stats = statistics_pb2.DatasetFeatureStatisticsList()
    current_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics()
    ])
    previous_version_stats = statistics_pb2.DatasetFeatureStatisticsList()
    previous_version_stats.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        ValueError, 'Only statistics proto with one dataset or the default.*'):
      _ = validation_api.validate_statistics_internal(
          current_stats,
          schema,
          previous_version_statistics=previous_version_stats)

  def test_validate_stats_with_custom_validations(self):
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    schema = text_format.Parse(
        """
        feature {
          name: 'annotated_enum'
          type: BYTES
          unique_constraints {
            min: 4
            max: 4
          }
        }
        """, schema_pb2.Schema())
    validation_config = text_format.Parse("""
      feature_validations {
       feature_path { step: 'annotated_enum' }
       validations {
         sql_expression: 'feature.string_stats.common_stats.num_missing < 3'
         severity: WARNING
         description: 'Feature has too many missing.'
       }
     }
    """, custom_validation_config_pb2.CustomValidationConfig())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
               path { step: 'annotated_enum' }
               short_description: 'Multiple errors'
               description: 'Expected at least 4 unique values but found only 3. Custom validation triggered anomaly. Query: feature.string_stats.common_stats.num_missing < 3 Test dataset: default slice'
               severity: ERROR
               reason {
                 type: FEATURE_TYPE_LOW_UNIQUE
                 short_description: 'Low number of unique values'
                 description: 'Expected at least 4 unique values but found only 3.'
               }
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Feature has too many missing.'
                 description: 'Custom validation triggered anomaly. Query: feature.string_stats.common_stats.num_missing < 3 Test dataset: default slice'
               }
    """, anomalies_pb2.AnomalyInfo())
    }
    anomalies = validation_api.validate_statistics(statistics, schema, None,
                                                   None, None,
                                                   validation_config)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_stats_internal_with_previous_version_stats(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 2 }
                buckets { label: "c" sample_count: 7 }
              }
            }
          }
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    previous_span_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    serving_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    previous_version_statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 10
                num_missing: 0
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 3 }
                buckets { label: "b" sample_count: 1 }
                buckets { label: "c" sample_count: 6 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    schema = text_format.Parse(
        """
        feature {
          name: 'bar'
          type: BYTES
          skew_comparator { infinity_norm { threshold: 0.1 } }
        }
        feature {
          name: "annotated_enum"
          type: BYTES
          domain: "annotated_enum"
          drift_comparator { infinity_norm { threshold: 0.01 } }
        }
        string_domain { name: "annotated_enum" value: "a" }
        """, schema_pb2.Schema())

    expected_anomalies = {
        'bar': text_format.Parse(self._bar_anomaly_info,
                                 anomalies_pb2.AnomalyInfo()),
        'annotated_enum': text_format.Parse(self._annotated_enum_anomaly_info,
                                            anomalies_pb2.AnomalyInfo())
    }

    # Validate the stats.
    anomalies = validation_api.validate_statistics_internal(
        statistics,
        schema,
        previous_span_statistics=previous_span_statistics,
        serving_statistics=serving_statistics,
        previous_version_statistics=previous_version_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)
  # pylint: enable=line-too-long

  def test_validate_stats_internal_with_validation_options_set(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 10
          features {
            path { step: 'bar' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 2 }
                buckets { label: "c" sample_count: 7 }
              }
            }
          }
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 0
                num_non_missing: 10
                max_num_values: 1
              }
              rank_histogram {
                buckets { label: "a" sample_count: 1 }
                buckets { label: "b" sample_count: 1 }
              }
            }
          }
        }""", statistics_pb2.DatasetFeatureStatisticsList())

    empty_schema = schema_pb2.Schema()

    # In this test case, both `bar` and `annotated_enum` are not defined in
    # schema. But since only `bar` is in features_needed path, the expected
    # anomalies only reports it. Besides, since new_features_are_warnings is
    # set to true, the severity in the report is WARNING.
    expected_anomalies = {
        'bar': text_format.Parse("""
         description: "New column (column in data but not in schema)"
         severity: WARNING
         short_description: "New column"
         reason {
           type: SCHEMA_NEW_COLUMN
           short_description: "New column"
           description: "New column (column in data but not in schema)"
         }
         path {
           step: "bar"
         }""", anomalies_pb2.AnomalyInfo())
    }

    features_needed = {
        FeaturePath(['bar']): [
            validation_options.ReasonFeatureNeeded(comment='reason1'),
            validation_options.ReasonFeatureNeeded(comment='reason2')
        ]
    }
    new_features_are_warnings = True
    vo = validation_options.ValidationOptions(
        features_needed, new_features_are_warnings)

    # Validate the stats.
    anomalies = validation_api.validate_statistics_internal(
        statistics,
        empty_schema,
        validation_options=vo)
    self._assert_equal_anomalies(anomalies, expected_anomalies)
  # pylint: enable=line-too-long

  def test_custom_validate_statistics_single_feature(self):
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    config = text_format.Parse("""
      feature_validations {
       feature_path { step: 'annotated_enum' }
       validations {
         sql_expression: 'feature.string_stats.common_stats.num_missing < 3'
         severity: ERROR
         description: 'Feature has too many missing.'
       }
     }
    """, custom_validation_config_pb2.CustomValidationConfig())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
               path { step: 'annotated_enum' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Feature has too many missing.'
                 description: 'Custom validation triggered anomaly. Query: feature.string_stats.common_stats.num_missing < 3 Test dataset: default slice'
               }
    """, anomalies_pb2.AnomalyInfo())
    }
    anomalies = validation_api.custom_validate_statistics(statistics, config)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_custom_validate_statistics_two_features(self):
    test_statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 10
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    base_statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'annotated_enum' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 5
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    config = text_format.Parse("""
      feature_pair_validations {
       feature_test_path { step: 'annotated_enum' }
       feature_base_path { step: 'annotated_enum' }
       validations {
         sql_expression: 'feature_test.string_stats.unique = feature_base.string_stats.unique'
         severity: ERROR
         description: 'Test and base do not have same number of uniques.'
       }
     }
    """, custom_validation_config_pb2.CustomValidationConfig())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
               path { step: 'annotated_enum' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Test and base do not have same number of uniques.'
                 description: 'Custom validation triggered anomaly. Query: feature_test.string_stats.unique = feature_base.string_stats.unique Test dataset: default slice Base dataset:  Base path: annotated_enum'
               }
    """, anomalies_pb2.AnomalyInfo())
    }
    anomalies = validation_api.custom_validate_statistics(
        test_statistics, config, base_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_custom_validate_statistics_environment(self):
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'some_feature' }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
              unique: 10
              rank_histogram {
                buckets {
                  label: "D"
                  sample_count: 1
                }
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    config = text_format.Parse("""
      feature_validations {
       feature_path { step: 'some_feature' }
       validations {
         sql_expression: 'feature.string_stats.common_stats.num_missing < 1'
         severity: ERROR
         description: 'Too many missing'
         in_environment: 'TRAINING'
       }
       validations {
         sql_expression: 'feature.string_stats.common_stats.num_missing > 5'
         severity: ERROR
         description: 'Too few missing'
         in_environment: 'SERVING'
       }
     }
    """, custom_validation_config_pb2.CustomValidationConfig())
    expected_anomalies = {
        'some_feature':
            text_format.Parse(
                """
               path { step: 'some_feature' }
               severity: ERROR
               reason {
                 type: CUSTOM_VALIDATION
                 short_description: 'Too many missing'
                 description: 'Custom validation triggered anomaly. Query: feature.string_stats.common_stats.num_missing < 1 Test dataset: default slice'
               }
    """, anomalies_pb2.AnomalyInfo())
    }
    anomalies = validation_api.custom_validate_statistics(
        statistics, config, None, 'TRAINING')
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_instance(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['D']])],
                                          ['annotated_enum'])
    schema = text_format.Parse(
        """
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
        }
        feature {
          name: "annotated_enum"
          value_count {
            min:1
            max:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          value_count {
            min:1
          }
          presence {
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
      path {
        step: "annotated_enum"
      }
      description: "Examples contain values missing from the schema: D "
        "(~100%). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D "
          "(~100%). "
      }
            """, anomalies_pb2.AnomalyInfo())
    }
    options = stats_options.StatsOptions(schema=schema)
    anomalies = validation_api.validate_instance(instance, options)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_instance_global_only_anomaly_type(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['D']])],
                                          ['annotated_enum'])
    # This schema has a presence.min_count > 1, which will generate an anomaly
    # of type FEATURE_TYPE_LOW_NUMBER_PRESENT when any single example is
    # validated using this schema. This test checks that this anomaly type
    # (which is not meaningful in per-example validation) is not included in the
    # Anomalies proto that validate_instance returns.
    schema = text_format.Parse(
        """
        string_domain {
          name: "MyAloneEnum"
          value: "A"
          value: "B"
          value: "C"
        }
        feature {
          name: "annotated_enum"
          value_count {
            min:1
            max:1
          }
          presence {
            min_count: 5
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        """, schema_pb2.Schema())
    expected_anomalies = {
        'annotated_enum':
            text_format.Parse(
                """
      path {
        step: "annotated_enum"
      }
      description: "Examples contain values missing from the schema: D "
        "(~100%). "
      severity: ERROR
      short_description: "Unexpected string values"
      reason {
        type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
        short_description: "Unexpected string values"
        description: "Examples contain values missing from the schema: D "
          "(~100%). "
      }
            """, anomalies_pb2.AnomalyInfo())
    }
    options = stats_options.StatsOptions(schema=schema)
    anomalies = validation_api.validate_instance(instance, options)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_instance_environment(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['A']])], ['feature'])
    schema = text_format.Parse(
        """
        default_environment: "TRAINING"
        default_environment: "SERVING"
        feature {
          name: "label"
          not_in_environment: "SERVING"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        feature {
          name: "feature"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        """, schema_pb2.Schema())
    options = stats_options.StatsOptions(schema=schema)

    # Validate the instance in TRAINING environment.
    expected_anomalies_training = {
        'label':
            text_format.Parse(
                """
            path {
              step: "label"
            }
            description: "Column is completely missing"
            severity: ERROR
            short_description: "Column dropped"
            reason {
              type: SCHEMA_MISSING_COLUMN
              short_description: "Column dropped"
              description: "Column is completely missing"
            }
            """, anomalies_pb2.AnomalyInfo())
    }
    anomalies_training = validation_api.validate_instance(
        instance, options, environment='TRAINING')
    self._assert_equal_anomalies(anomalies_training,
                                 expected_anomalies_training)

    # Validate the instance in SERVING environment.
    anomalies_serving = validation_api.validate_instance(
        instance, options, environment='SERVING')
    self._assert_equal_anomalies(anomalies_serving, {})

  def test_validate_instance_invalid_environment(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['A']])], ['feature'])
    schema = text_format.Parse(
        """
        default_environment: "TRAINING"
        default_environment: "SERVING"
        feature {
          name: "label"
          not_in_environment: "SERVING"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        feature {
          name: "feature"
          value_count { min: 1 max: 1 }
          presence { min_count: 1 }
          type: BYTES
        }
        """, schema_pb2.Schema())
    options = stats_options.StatsOptions(schema=schema)

    with self.assertRaisesRegexp(
        ValueError, 'Environment.*not found in the schema.*'):
      _ = validation_api.validate_instance(
          instance, options, environment='INVALID')

  def test_validate_instance_invalid_options(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['A']])], ['feature'])
    with self.assertRaisesRegexp(ValueError,
                                 'options must be a StatsOptions object.'):
      _ = validation_api.validate_instance(instance, {})

  def test_validate_instance_stats_options_without_schema(self):
    instance = pa.RecordBatch.from_arrays([pa.array([['A']])], ['feature'])
    # This instance of StatsOptions has no schema.
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(ValueError, 'options must include a schema.'):
      _ = validation_api.validate_instance(instance, options)


class NLValidationTest(ValidationTestCase):

  @parameterized.named_parameters(*[
      dict(
          testcase_name='no_coverage',
          min_coverage=None,
          feature_coverage=None,
          min_avg_token_length=None,
          feature_avg_token_length=None,
          expected_anomaly_types=set(),
          expected_min_coverage=None,
          expected_min_avg_token_length=None),
      dict(
          testcase_name='missing_stats',
          min_coverage=0.4,
          feature_coverage=None,
          min_avg_token_length=None,
          feature_avg_token_length=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.STATS_NOT_AVAILABLE]),
          expected_min_coverage=None,
          expected_min_avg_token_length=None,
      ),
      dict(
          testcase_name='low_min_coverage',
          min_coverage=0.4,
          feature_coverage=0.5,
          min_avg_token_length=None,
          feature_avg_token_length=None,
          expected_anomaly_types=set(),
          expected_min_coverage=0.4,
          expected_min_avg_token_length=None),
      dict(
          testcase_name='high_min_coverage',
          min_coverage=0.5,
          feature_coverage=0.4,
          min_avg_token_length=None,
          feature_avg_token_length=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.FEATURE_COVERAGE_TOO_LOW]),
          expected_min_coverage=0.4,
          expected_min_avg_token_length=None,
      ),
      dict(
          testcase_name='low_min_avg_token_length',
          min_coverage=None,
          feature_coverage=None,
          min_avg_token_length=4,
          feature_avg_token_length=5,
          expected_anomaly_types=set(),
          expected_min_coverage=None,
          expected_min_avg_token_length=4,
      ),
      dict(
          testcase_name='high_min_avg_token_length',
          min_coverage=None,
          feature_coverage=None,
          min_avg_token_length=5,
          feature_avg_token_length=4,
          expected_anomaly_types=set([
              anomalies_pb2.AnomalyInfo
              .FEATURE_COVERAGE_TOO_SHORT_AVG_TOKEN_LENGTH
          ]),
          expected_min_coverage=None,
          expected_min_avg_token_length=4,
      ),
  ])
  def test_validate_nl_domain_coverage(self, min_coverage, feature_coverage,
                                       min_avg_token_length,
                                       feature_avg_token_length,
                                       expected_anomaly_types,
                                       expected_min_coverage,
                                       expected_min_avg_token_length):
    schema = text_format.Parse(
        """
        feature {
          name: "nl_feature"
          natural_language_domain {
          }
          type: INT
        }
        """, schema_pb2.Schema())
    if min_coverage is not None:
      schema.feature[
          0].natural_language_domain.coverage.min_coverage = min_coverage
    if min_avg_token_length is not None:
      schema.feature[
          0].natural_language_domain.coverage.min_avg_token_length = min_avg_token_length

    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'nl_feature' }
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    if feature_coverage is not None or feature_avg_token_length is not None:
      nl_stats = statistics_pb2.NaturalLanguageStatistics()
      if feature_coverage is not None:
        nl_stats.feature_coverage = feature_coverage
      if feature_avg_token_length is not None:
        nl_stats.avg_token_length = feature_avg_token_length

      custom_stat = statistics.datasets[0].features[0].custom_stats.add()
      custom_stat.name = 'nl_statistics'
      custom_stat.any.Pack(nl_stats)

    # Validate the stats and update schema.
    anomalies = validation_api.validate_statistics(statistics, schema)
    schema = validation_api.update_schema(schema, statistics)
    anomaly_types = set(
        [r.type for r in anomalies.anomaly_info['nl_feature'].reason])
    self.assertSetEqual(expected_anomaly_types, anomaly_types)

    for field, str_field in [(expected_min_coverage, 'min_coverage'),
                             (expected_min_avg_token_length,
                              'min_avg_token_length')]:
      if field is None:
        self.assertFalse(
            schema.feature[0].natural_language_domain.coverage.HasField(
                str_field))
      else:
        self.assertAlmostEqual(
            getattr(schema.feature[0].natural_language_domain.coverage,
                    str_field), field)

  @parameterized.named_parameters(*[
      dict(
          testcase_name='missing_stats',
          token_name=100,
          fraction_values=(None, 0.4, 0.6),
          sequence_values=(None, None, None, 1, 3),
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.STATS_NOT_AVAILABLE]),
          expected_fraction_values=None,
          expected_sequence_values=None),
      dict(
          testcase_name='all_fraction_constraints_satisfied',
          token_name=100,
          fraction_values=(0.5, 0.4, 0.6),
          sequence_values=None,
          expected_anomaly_types=set(),
          expected_fraction_values=(0.4, 0.6),
          expected_sequence_values=None),
      dict(
          testcase_name='int_token_min_fraction_constraint_too_high',
          token_name=100,
          fraction_values=(0.5, 0.6, 0.6),
          sequence_values=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_SMALL_FRACTION]),
          expected_fraction_values=(0.5, 0.6),
          expected_sequence_values=None),
      dict(
          testcase_name='string_token_min_fraction_constraint_too_high',
          token_name='str',
          fraction_values=(0.5, 0.6, 0.6),
          sequence_values=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_SMALL_FRACTION]),
          expected_fraction_values=(0.5, 0.6),
          expected_sequence_values=None),
      dict(
          testcase_name='int_token_max_fraction_constraint_too_low',
          token_name=100,
          fraction_values=(0.5, 0.4, 0.4),
          sequence_values=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_LARGE_FRACTION]),
          expected_fraction_values=(0.4, 0.5),
          expected_sequence_values=None),
      dict(
          testcase_name='string_token_max_fraction_constraint_too_low',
          token_name='str',
          fraction_values=(0.5, 0.4, 0.4),
          sequence_values=None,
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_LARGE_FRACTION]),
          expected_fraction_values=(0.4, 0.5),
          expected_sequence_values=None),
      dict(
          testcase_name='all_sequence_constraints_satisfied',
          token_name=100,
          fraction_values=None,
          sequence_values=(2, 2, 2, 1, 3),
          expected_anomaly_types=set(),
          expected_fraction_values=None,
          expected_sequence_values=(1, 3),
      ),
      dict(
          testcase_name='int_token_min_sequence_constraint_too_high',
          token_name=100,
          fraction_values=None,
          sequence_values=(0, 2, 1, 1, 3),
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_FEW_OCCURRENCES]),
          expected_fraction_values=None,
          expected_sequence_values=(0, 3),
      ),
      dict(
          testcase_name='string_token_min_sequence_constraint_too_high',
          token_name='str',
          fraction_values=None,
          sequence_values=(0, 2, 1, 1, 3),
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_FEW_OCCURRENCES]),
          expected_fraction_values=None,
          expected_sequence_values=(0, 3),
      ),
      dict(
          testcase_name='int_token_max_sequence_constraint_too_low',
          token_name=100,
          fraction_values=None,
          sequence_values=(2, 4, 3, 1, 3),
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_MANY_OCCURRENCES]),
          expected_fraction_values=None,
          expected_sequence_values=(1, 4),
      ),
      dict(
          testcase_name='string_token_max_sequence_constraint_too_low',
          token_name='str',
          fraction_values=None,
          sequence_values=(2, 4, 3, 1, 3),
          expected_anomaly_types=set(
              [anomalies_pb2.AnomalyInfo.SEQUENCE_VALUE_TOO_MANY_OCCURRENCES]),
          expected_fraction_values=None,
          expected_sequence_values=(1, 4),
      ),
  ])
  def test_validate_nl_domain_token_constraints(self, token_name,
                                                fraction_values,
                                                sequence_values,
                                                expected_anomaly_types,
                                                expected_fraction_values,
                                                expected_sequence_values):
    fraction, min_fraction, max_fraction = (
        fraction_values if fraction_values else (None, None, None))
    expected_min_fraction, expected_max_fraction = (
        expected_fraction_values if expected_fraction_values else (None, None))

    min_sequence_stat, max_sequence_stat, avg_sequence_stat, min_sequence, max_sequence = (
        sequence_values if sequence_values else (None, None, None, None, None))
    expected_min_sequence, expected_max_sequence = (
        expected_sequence_values if expected_sequence_values else (None, None))

    schema = text_format.Parse(
        """
        feature {
          name: "nl_feature"
          natural_language_domain {
            token_constraints {
              int_value: 200
              min_per_sequence: 1
              max_per_sequence: 3
              min_fraction_of_sequences: 0.1
              max_fraction_of_sequences: 0.3
            }
          }
          type: INT
        }
        """, schema_pb2.Schema())
    if (min_fraction is not None or max_fraction is not None or
        min_sequence is not None or max_sequence is not None):
      token_constraint = (
          schema.feature[0].natural_language_domain.token_constraints.add())
      if isinstance(token_name, int):
        token_constraint.int_value = token_name
      else:
        token_constraint.string_value = token_name
      if min_fraction is not None:
        token_constraint.min_fraction_of_sequences = min_fraction
      if max_fraction is not None:
        token_constraint.max_fraction_of_sequences = max_fraction
      if min_sequence is not None:
        token_constraint.min_per_sequence = min_sequence
      if max_sequence is not None:
        token_constraint.max_per_sequence = max_sequence

    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 10
          features {
            path { step: 'nl_feature' }
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 7
                min_num_values: 1
                max_num_values: 1
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())
    nl_stats = statistics_pb2.NaturalLanguageStatistics()
    token_stats = nl_stats.token_statistics.add()
    token_stats.int_token = 200
    token_stats.fraction_of_sequences = 0.2
    token_stats.per_sequence_min_frequency = 2
    token_stats.per_sequence_max_frequency = 2
    token_stats.per_sequence_avg_frequency = 2
    if (fraction is not None or min_sequence_stat is not None or
        max_sequence_stat is not None):
      token_stats = nl_stats.token_statistics.add()
      if isinstance(token_name, int):
        token_stats.int_token = token_name
      else:
        token_stats.string_token = token_name
      if fraction is not None:
        token_stats.fraction_of_sequences = fraction
      if min_sequence_stat is not None:
        token_stats.per_sequence_min_frequency = min_sequence_stat
      if max_sequence_stat is not None:
        token_stats.per_sequence_max_frequency = max_sequence_stat
      if avg_sequence_stat is not None:
        token_stats.per_sequence_avg_frequency = avg_sequence_stat
    custom_stat = statistics.datasets[0].features[0].custom_stats.add()
    custom_stat.name = 'nl_statistics'
    custom_stat.any.Pack(nl_stats)

    # Validate the stats.
    anomalies = validation_api.validate_statistics(statistics, schema)
    anomaly_types = set(
        [r.type for r in anomalies.anomaly_info['nl_feature'].reason])
    self.assertSetEqual(anomaly_types, expected_anomaly_types)

    schema = validation_api.update_schema(schema, statistics)
    for field, str_field in [
        (expected_min_fraction, 'min_fraction_of_sequences'),
        (expected_max_fraction, 'max_fraction_of_sequences'),
        (expected_min_sequence, 'min_per_sequence'),
        (expected_max_sequence, 'max_per_sequence')
    ]:
      if field is None:
        self.assertFalse(
            len(schema.feature[0].natural_language_domain.token_constraints) and
            schema.feature[0].natural_language_domain.token_constraints[1]
            .HasField(str_field))
      else:
        self.assertAlmostEqual(
            getattr(
                schema.feature[0].natural_language_domain.token_constraints[1],
                str_field), field)


class IdentifyAnomalousExamplesTest(parameterized.TestCase):

  @parameterized.named_parameters(*IDENTIFY_ANOMALOUS_EXAMPLES_VALID_INPUTS)
  def test_identify_anomalous_examples(self, examples, schema_text,
                                       expected_result):
    schema = text_format.Parse(schema_text, schema_pb2.Schema())
    options = stats_options.StatsOptions(schema=schema)

    def _assert_fn(got):

      # TODO(zhuo): clean-up after ARROW-8277 is available.
      class _RecordBatchEqualityWrapper(object):
        __hash__ = None

        def __init__(self, record_batch):
          self._batch = record_batch

        def __eq__(self, other):
          return self._batch.equals(other._batch)  # pylint: disable=protected-access

      wrapped_got = [(k, _RecordBatchEqualityWrapper(v)) for k, v in got]
      wrapped_expected = [
          (k, _RecordBatchEqualityWrapper(v)) for k, v in expected_result]
      self.assertCountEqual(wrapped_got, wrapped_expected)

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(examples)
          | validation_api.IdentifyAnomalousExamples(options))
      util.assert_that(result, _assert_fn)

  def test_identify_anomalous_examples_options_of_wrong_type(self):
    examples = [{'annotated_enum': np.array(['D'], dtype=object)}]
    options = 1
    with self.assertRaisesRegexp(ValueError, 'options must be a `StatsOptions` '
                                 'object.'):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(examples)
            | validation_api.IdentifyAnomalousExamples(options))

  def test_identify_anomalous_examples_options_without_schema(self):
    examples = [{'annotated_enum': np.array(['D'], dtype=object)}]
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(ValueError, 'options must include a schema'):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(examples)
            | validation_api.IdentifyAnomalousExamples(options))


class DetectFeatureSkewTest(absltest.TestCase):

  def _assert_feature_skew_results_protos_equal(self, actual, expected) -> None:
    self.assertLen(actual, len(expected))
    sorted_actual = sorted(actual, key=lambda t: t.feature_name)
    sorted_expected = sorted(expected, key=lambda e: e.feature_name)
    for i in range(len(sorted_actual)):
      compare.assertProtoEqual(self, sorted_actual[i], sorted_expected[i])

  def _assert_skew_pairs_equal(self, actual, expected) -> None:
    self.assertLen(actual, len(expected))
    for each in actual:
      self.assertIn(each, expected)

  def test_detect_feature_skew(self):
    training_data = [
        text_format.Parse("""
            features {
              feature {
                key: 'id'
                value { bytes_list { value: [ 'first_feature' ] } }
              }
            feature {
                key: 'feature_a'
                value { int64_list { value: [ 12, 24 ] } }
              }
            feature {
                key: 'feature_b'
                value { float_list { value: [ 10.0 ] } }
              }
           }
       """, tf.train.Example()),
        text_format.Parse("""
            features {
              feature {
                key: 'id'
                value { bytes_list { value: [ 'second_feature' ] } }
              }
            feature {
                key: 'feature_a'
                value { int64_list { value: [ 5 ] } }
              }
            feature {
                key: 'feature_b'
                value { float_list { value: [ 15.0 ] } }
              }
           }
       """, tf.train.Example())
    ]
    serving_data = [
        text_format.Parse("""
            features {
              feature {
                key: 'id'
                value { bytes_list { value: [ 'first_feature' ] } }
              }
            feature {
                key: 'feature_b'
                value { float_list { value: [ 10.0 ] } }
              }
           }
       """, tf.train.Example()),
        text_format.Parse("""
            features {
              feature {
                key: 'id'
                value { bytes_list { value: [ 'second_feature' ] } }
              }
            feature {
                key: 'feature_a'
                value { int64_list { value: [ 5 ] } }
              }
            feature {
                key: 'feature_b'
                value { float_list { value: [ 20.0 ] } }
              }
           }
       """, tf.train.Example())
    ]

    expected_feature_skew_result = [
        text_format.Parse(
            """
        feature_name: 'feature_a'
        base_count: 2
        test_count: 1
        match_count: 1
        base_only: 1
        diff_count: 1""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'feature_b'
        base_count: 2
        test_count: 2
        match_count: 1
        mismatch_count: 1
        diff_count: 1""", feature_skew_results_pb2.FeatureSkew())
    ]

    with beam.Pipeline() as p:
      training_data = p | 'CreateTraining' >> beam.Create(training_data)
      serving_data = p | 'CreateServing' >> beam.Create(serving_data)
      feature_skew, skew_sample = (
          (training_data, serving_data)
          | 'DetectSkew' >> validation_api.DetectFeatureSkew(
              identifier_features=['id'], sample_size=1))
      util.assert_that(
          feature_skew,
          test_util.make_skew_result_equal_fn(self,
                                              expected_feature_skew_result),
          'CheckFeatureSkew')
      util.assert_that(skew_sample, util.is_not_empty(), 'CheckSkewSample')

  def test_write_feature_skew_results_to_tf_record(self):
    feature_skew_results = [
        text_format.Parse(
            """
        feature_name: 'skewed'
        base_count: 2
        test_count: 2
        mismatch_count: 2
        diff_count: 2""", feature_skew_results_pb2.FeatureSkew()),
        text_format.Parse(
            """
        feature_name: 'no_skew'
        base_count: 2
        test_count: 2
        match_count: 2""", feature_skew_results_pb2.FeatureSkew())
    ]
    output_path = os.path.join(tempfile.mkdtemp(), 'feature_skew')
    with beam.Pipeline() as p:
      _ = (
          p | beam.Create(feature_skew_results)
          | validation_api.WriteFeatureSkewResultsToTFRecord(output_path))

    skew_results_from_file = []
    for record in tf.compat.v1.io.tf_record_iterator(output_path):
      skew_results_from_file.append(
          feature_skew_results_pb2.FeatureSkew.FromString(record))
    self._assert_feature_skew_results_protos_equal(skew_results_from_file,
                                                   feature_skew_results)

  def test_write_skew_pairs_to_tf_record(self):
    base_example = text_format.Parse(
        """
                                     features {
                feature {
                  key: 'id'
                  value { bytes_list { value: [ 'id_feature' ] } }
                }
              feature {
                  key: 'feature_a'
                  value { float_list { value: [ 10.0 ] } }
              }
             }""",
        tf.train.Example(),
    )
    test_example = text_format.Parse(
        """features {
                feature {
                  key: 'id'
                  value { bytes_list { value: [ 'id_feature' ] } }
                }
              feature {
                  key: 'feature_a'
                  value { float_list { value: [ 11.0 ] } }
                }
             }""",
        tf.train.Example(),
    )
    skew_pair = feature_skew_results_pb2.SkewPair(
        base=base_example.SerializeToString(),
        test=test_example.SerializeToString(),
        mismatched_features=['feature_a'],
    )
    skew_pairs = [skew_pair, skew_pair]
    output_path = os.path.join(tempfile.mkdtemp(), 'skew_pairs')
    with beam.Pipeline() as p:
      _ = (
          p | beam.Create(skew_pairs)
          | validation_api.WriteSkewPairsToTFRecord(output_path))

    skew_pairs_from_file = []
    for record in tf.compat.v1.io.tf_record_iterator(output_path):
      skew_pairs_from_file.append(
          feature_skew_results_pb2.SkewPair.FromString(record))
    self._assert_skew_pairs_equal(skew_pairs_from_file, skew_pairs)


def _construct_sliced_statistics(
    values_slice1,
    values_slice2) -> statistics_pb2.DatasetFeatureStatisticsList:
  values_overall = values_slice1 + values_slice2
  datasets = []

  stats_slice1 = tfdv.generate_statistics_from_dataframe(
      pd.DataFrame.from_dict({'foo': values_slice1}))
  stats_slice1.datasets[0].name = 'slice1'
  datasets.append(stats_slice1.datasets[0])

  if values_slice2:
    stats_slice2 = tfdv.generate_statistics_from_dataframe(
        pd.DataFrame.from_dict({'foo': values_slice2}))
    stats_slice2.datasets[0].name = 'slice2'
    datasets.append(stats_slice2.datasets[0])

  stats_overall = tfdv.generate_statistics_from_dataframe(
      pd.DataFrame.from_dict({'foo': values_overall}))
  stats_overall.datasets[0].name = tfdv.constants.DEFAULT_SLICE_KEY
  datasets.append(stats_overall.datasets[0])

  statistics = statistics_pb2.DatasetFeatureStatisticsList(datasets=datasets)
  return statistics


def _test_schema():
  return text_format.Parse(
      """
    feature {
      name: "foo"
      type: BYTES
      string_domain {
        name: "feature_foo"
        value: "1"
        value: "2"
        value: "3"
        value: "4"
      }
      distribution_constraints: {min_domain_mass: 0.5}
      presence: {min_fraction: 1.0}
    }
    """, schema_pb2.Schema())


class ValidateCorrespondingSlicesTest(ValidationTestCase):

  def test_no_anomalies(self):
    sliced_stats = _construct_sliced_statistics(['1', '2', '3', '4'],
                                                ['2', '2', '3'])
    schema = _test_schema()
    anomalies = validation_api.validate_corresponding_slices(
        sliced_stats, schema)
    self._assert_equal_anomalies(anomalies, {})

  def test_missing_slice_in_previous_stats_is_not_error(self):
    sliced_stats1 = _construct_sliced_statistics(['1', '2'], ['3', '4'])
    sliced_stats2 = _construct_sliced_statistics(['1', '2', '3', '4'], [])

    schema = _test_schema()
    anomalies = validation_api.validate_corresponding_slices(
        sliced_stats1, schema, previous_statistics=sliced_stats2)
    self._assert_equal_anomalies(anomalies, {})

  def test_missing_slice_in_current_stats_is_error(self):
    sliced_stats1 = _construct_sliced_statistics(['1', '2', '3', '4'], [])
    sliced_stats2 = _construct_sliced_statistics(['1', '2'], ['3', '4'])

    schema = _test_schema()
    anomalies = validation_api.validate_corresponding_slices(
        sliced_stats1, schema, previous_statistics=sliced_stats2)
    self._assert_equal_anomalies(
        anomalies, {
            "\'slice(slice2)::foo\'":
                text_format.Parse("""
        description: "Column is completely missing"
        severity: ERROR
        short_description: "Column dropped"
        reason {
          type: SCHEMA_MISSING_COLUMN
          short_description: "Column dropped"
          description: "Column is completely missing"
        }
        path {
          step: "slice(slice2)::foo"
        }
        """, anomalies_pb2.AnomalyInfo())
        })

  def test_anomaly_in_one_slice(self):
    sliced_stats = _construct_sliced_statistics(['1', '2', '3', '4'], ['5'])
    schema = _test_schema()
    anomalies = validation_api.validate_corresponding_slices(
        sliced_stats, schema)
    self._assert_equal_anomalies(
        anomalies, {
            "\'slice(slice2)::foo\'":
                text_format.Parse(
                    """
            description: "Examples contain values missing from the schema: 5 (~100%). "
            severity: ERROR
            short_description: "Unexpected string values"
            reason {
              type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
              short_description: "Unexpected string values"
              description: "Examples contain values missing from the schema: 5 (~100%). "
            }
            path {
              step: "slice(slice2)::foo"
            }
        """, anomalies_pb2.AnomalyInfo())
        })

  def test_distributional_anomaly_between_slices(self):
    sliced_stats1 = _construct_sliced_statistics(['1', '2'], ['3', '4'])
    sliced_stats2 = _construct_sliced_statistics(['1', '2'], ['1', '2'])
    schema = _test_schema()
    schema_util.get_feature(
        schema, 'foo').drift_comparator.infinity_norm.threshold = 0.3
    anomalies = validation_api.validate_corresponding_slices(
        sliced_stats1, schema, previous_statistics=sliced_stats2)
    self._assert_equal_anomalies(
        anomalies, {
            "\'slice(slice2)::foo\'":
                text_format.Parse(
                    """
            description: "The Linfty distance between current and previous is 0.5 (up to six significant digits), above the threshold 0.3. The feature value with maximum difference is: 4"
            severity: ERROR
            short_description: "High Linfty distance between current and previous"
            reason {
              type: COMPARATOR_L_INFTY_HIGH
              short_description: "High Linfty distance between current and previous"
              description: "The Linfty distance between current and previous is 0.5 (up to six significant digits), above the threshold 0.3. The feature value with maximum difference is: 4"
            }
            path {
              step: "slice(slice2)::foo"
            }
        """, anomalies_pb2.AnomalyInfo())
        })


if __name__ == '__main__':
  absltest.main()
