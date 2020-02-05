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

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.api import validation_api
from tensorflow_data_validation.api import validation_options
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.types import FeaturePath
from tensorflow_data_validation.utils import schema_util

from google.protobuf import text_format

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


IDENTIFY_ANOMALOUS_EXAMPLES_VALID_INPUTS = [
    {
        'testcase_name':
            'no_anomalies',
        'examples': [
            pa.Table.from_arrays([pa.array([['A']])], ['annotated_enum']),
            pa.Table.from_arrays([pa.array([['C']])], ['annotated_enum']),
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
    },
    {
        'testcase_name':
            'same_anomaly_reason',
        'examples': [
            pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum']),
            pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum']),
            pa.Table.from_arrays([pa.array([['C']])], ['annotated_enum']),
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
        'expected_result':
            [
                ('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                 pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum'])),
                ('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                 pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum']))
            ]
    },
    {
        'testcase_name':
            'different_anomaly_reasons',
        'examples': [
            pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum']),
            pa.Table.from_arrays([pa.array([['C']])], ['annotated_enum']),
            pa.Table.from_arrays([pa.array([[1]])], ['feature_not_in_schema']),
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
        'expected_result':
            [
                ('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
                 pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum'])),
                ('feature_not_in_schema_SCHEMA_NEW_COLUMN',
                 pa.Table.from_arrays([pa.array([[1]])],
                                      ['feature_not_in_schema']))
            ]
    }
]


class ValidationApiTest(absltest.TestCase):

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
                num_missing: 0
                num_non_missing: 7
                min_num_values: 5
                max_num_values: 5
              }
              unique: 5
            }
          }
          features: {
            path { step: 'feature3' }
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
          shape { dim { size: 5 } }
          presence: {
            min_fraction: 1.0
            min_count: 1
          }
          type: BYTES
        }
        feature {
          name: "feature3"
          value_count: { min: 1 max: 1}
          presence: {
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

  def _assert_equal_anomalies(self,
                              actual_anomalies,
                              expected_anomalies):
    # Check if the actual anomalies matches with the expected anomalies.
    for feature_name in expected_anomalies:
      self.assertIn(feature_name, actual_anomalies.anomaly_info)
      # Doesn't compare the diff_regions.
      actual_anomalies.anomaly_info[feature_name].ClearField('diff_regions')

      self.assertEqual(actual_anomalies.anomaly_info[feature_name],
                       expected_anomalies[feature_name])
    self.assertEqual(
        len(actual_anomalies.anomaly_info), len(expected_anomalies))

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
    self._assert_equal_anomalies(anomalies, expected_anomalies)

  def test_validate_stats_with_serving_stats(self):
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

    schema = text_format.Parse(
        """
        feature {
          name: 'bar'
          type: BYTES
          skew_comparator {
            infinity_norm { threshold: 0.1}
          }
        }""", schema_pb2.Schema())

    expected_anomalies = {
        'bar': text_format.Parse(self._bar_anomaly_info,
                                 anomalies_pb2.AnomalyInfo())
    }
    # Validate the stats.
    anomalies = validation_api.validate_statistics(
        statistics, schema, serving_statistics=serving_statistics)
    self._assert_equal_anomalies(anomalies, expected_anomalies)

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

  def test_validate_instance(self):
    instance = pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum'])
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
    instance = pa.Table.from_arrays([pa.array([['D']])], ['annotated_enum'])
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
    instance = pa.Table.from_arrays([pa.array([['A']])], ['feature'])
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
    instance = pa.Table.from_arrays([pa.array([['A']])], ['feature'])
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
    instance = pa.Table.from_arrays([pa.array([['A']])], ['feature'])
    with self.assertRaisesRegexp(ValueError,
                                 'options must be a StatsOptions object.'):
      _ = validation_api.validate_instance(instance, {})

  def test_validate_instance_stats_options_without_schema(self):
    instance = pa.Table.from_arrays([pa.array([['A']])], ['feature'])
    # This instance of StatsOptions has no schema.
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(ValueError, 'options must include a schema.'):
      _ = validation_api.validate_instance(instance, options)


class IdentifyAnomalousExamplesTest(parameterized.TestCase):

  @parameterized.named_parameters(*IDENTIFY_ANOMALOUS_EXAMPLES_VALID_INPUTS)
  def test_identify_anomalous_examples(self, examples, schema_text,
                                       expected_result):
    schema = text_format.Parse(schema_text, schema_pb2.Schema())
    options = stats_options.StatsOptions(schema=schema)
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(examples)
          | validation_api.IdentifyAnomalousExamples(options))
      util.assert_that(result, util.equal_to(expected_result))

  def test_identify_anomalous_examples_options_of_wrong_type(self):
    examples = [{'annotated_enum': np.array(['D'], dtype=np.object)}]
    options = 1
    with self.assertRaisesRegexp(ValueError, 'options must be a `StatsOptions` '
                                 'object.'):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(examples)
            | validation_api.IdentifyAnomalousExamples(options))

  def test_identify_anomalous_examples_options_without_schema(self):
    examples = [{'annotated_enum': np.array(['D'], dtype=np.object)}]
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(ValueError, 'options must include a schema'):
      with beam.Pipeline() as p:
        _ = (
            p | beam.Create(examples)
            | validation_api.IdentifyAnomalousExamples(options))


if __name__ == '__main__':
  absltest.main()
