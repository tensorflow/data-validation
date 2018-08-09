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
from tensorflow_data_validation.api import validation_api
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class ValidationApiTest(absltest.TestCase):

  def test_infer_schema(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            name: 'feature1'
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 4
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
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_with_string_domain(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            name: 'feature1'
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 4
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
                buckets {
                  low_rank: 2
                  high_rank: 2
                  label: "c"
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
          value: "c"
        }
        """, schema_pb2.Schema())

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_without_string_domain(self):
    statistics = text_format.Parse(
        """
        datasets {
          num_examples: 7
          features: {
            name: 'feature1'
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 4
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
                buckets {
                  low_rank: 2
                  high_rank: 2
                  label: "c"
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

    # Infer the schema from the stats.
    actual_schema = validation_api.infer_schema(statistics,
                                                max_string_domain_size=2)
    self.assertEqual(actual_schema, expected_schema)

  def test_infer_schema_invalid_statistics_input(self):
    with self.assertRaisesRegexp(
        TypeError, '.*should be a DatasetFeatureStatisticsList proto.*'):
      _ = validation_api.infer_schema({})

  def test_infer_schema_invalid_multiple_datasets(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    with self.assertRaisesRegexp(ValueError,
                                 '.*statistics proto with one dataset.*'):
      _ = validation_api.infer_schema(statistics)

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
          value_count: {
            min:1
            max:1
          }
          presence: {
            min_count: 1
          }
          type: BYTES
          domain: "MyAloneEnum"
        }
        feature {
          name: "ignore_this"
          lifecycle_stage: DEPRECATED
          value_count: {
            min:1
          }
          presence: {
            min_count: 1
          }
          type: BYTES
        }
        """, schema_pb2.Schema())
    statistics = text_format.Parse(
        """
        datasets{
          num_examples: 1000
          features: {
            name: 'annotated_enum'
            type: STRING
            string_stats: {
              common_stats: {
                num_missing: 3
                num_non_missing: 4
                min_num_values: 1
                max_num_values: 1
              }
              unique: 3
              rank_histogram: {
                buckets: {
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
    # Check if the actual anomalies matches with the expected anomalies.
    # Doesn't compare the diff_regions.
    for feature_name in expected_anomalies:
      self.assertIn(feature_name, anomalies.anomaly_info)
      self.assertEqual(anomalies.anomaly_info[feature_name],
                       expected_anomalies[feature_name])
    self.assertEqual(len(anomalies.anomaly_info), len(expected_anomalies))

  def test_validate_stats_invalid_statistics_input(self):
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(
        TypeError, '.*should be a DatasetFeatureStatisticsList proto.*'):
      _ = validation_api.validate_statistics({}, schema)

  def test_validate_stats_invalid_schema_input(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    with self.assertRaisesRegexp(TypeError, '.*should be a Schema proto.*'):
      _ = validation_api.validate_statistics(statistics, {})

  def test_validate_stats_invalid_multiple_datasets(self):
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    statistics.datasets.extend([
        statistics_pb2.DatasetFeatureStatistics(),
        statistics_pb2.DatasetFeatureStatistics()
    ])
    schema = schema_pb2.Schema()
    with self.assertRaisesRegexp(ValueError,
                                 '.*statistics proto with one dataset.*'):
      _ = validation_api.validate_statistics(statistics, schema)


if __name__ == '__main__':
  absltest.main()
