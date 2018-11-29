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
"""Tests for anomalies_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_data_validation.utils import anomalies_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import anomalies_pb2

SET_REMOVE_ANOMALY_TYPES_CHANGES_PROTO_TESTS = [
    {
        'testcase_name':
            'single_reason_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
                anomalies_pb2.AnomalyInfo.ENUM_TYPE_UNEXPECTED_STRING_VALUES
            ]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Examples contain values missing from the "
                    "schema."
                  severity: ERROR
                  short_description: "Unexpected string values"
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing from the "
                      "schema."
                  }
                }
              }""",
        'expected_anomalies_proto_text': ''
    },
    {
        'testcase_name':
            'multiple_reasons_some_removed',
        'anomaly_types_to_remove':
            set([anomalies_pb2.AnomalyInfo.ENUM_TYPE_BYTES_NOT_STRING]),
        'input_anomalies_proto_text':
            """
            anomaly_info {
              key: "example_1"
              value {
                description: "Expected bytes but got string. Examples "
                   "contain values missing from the schema."
                 severity: ERROR
                 short_description: "Multiple errors"
                 reason {
                   type: ENUM_TYPE_BYTES_NOT_STRING
                   short_description: "Bytes not string"
                   description: "Expected bytes but got string."
                 }
                 reason {
                   type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                   short_description: "Unexpected string values"
                   description: "Examples contain values missing from the "
                     "schema."
                 }
               }
             }""",
        'expected_anomalies_proto_text':
            """
            anomaly_info {
              key: "example_1"
              value {
                 description: "Examples contain values missing from the "
                   "schema."
                 severity: ERROR
                 short_description: "Unexpected string values"
                 reason {
                   type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                   short_description: "Unexpected string values"
                   description: "Examples contain values missing from the "
                     "schema."
                 }
               }
             }"""
    },
    {
        'testcase_name':
            'multiple_reasons_all_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.ENUM_TYPE_BYTES_NOT_STRING,
                anomalies_pb2.AnomalyInfo.ENUM_TYPE_UNEXPECTED_STRING_VALUES,
            ]),
        'input_anomalies_proto_text':
            """
            anomaly_info {
              key: "example_1"
              value {
                 description: "Expected bytes but got string. Examples "
                   "contain values missing from the schema."
                 severity: ERROR
                 short_description: "Multiple errors"
                 reason {
                   type: ENUM_TYPE_BYTES_NOT_STRING
                   short_description: "Bytes not string"
                   description: "Expected bytes but got string."
                 }
                 reason {
                   type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                   short_description: "Unexpected string values"
                   description: "Examples contain values missing from the "
                     "schema."
                 }
               }
             }""",
        'expected_anomalies_proto_text': ''
    },
    {
        'testcase_name':
            'multiple_features_some_reasons_removed',
        'anomaly_types_to_remove':
            set(
                [anomalies_pb2.AnomalyInfo.ENUM_TYPE_UNEXPECTED_STRING_VALUES]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Expected bytes but got string. Examples "
                    "contain values missing from the schema."
                  severity: ERROR
                  short_description: "Multiple errors"
                  reason {
                    type: ENUM_TYPE_BYTES_NOT_STRING
                    short_description: "Bytes not string"
                    description: "Expected bytes but got string."
                  }
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing from the "
                      "schema."
                  }
                }
              }
            anomaly_info {
              key: "example_2"
              value {
                description: "Examples contain values missing from the schema."
                severity: ERROR
                short_description: "Unexpected string values"
                reason {
                  type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                  short_description: "Unexpected string values"
                  description: "Examples contain values missing from the "
                    "schema."
                }
              }
            }""",
        'expected_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                 description: "Expected bytes but got string."
                 severity: ERROR
                 short_description: "Bytes not string"
                 reason {
                    type: ENUM_TYPE_BYTES_NOT_STRING
                    short_description: "Bytes not string"
                    description: "Expected bytes but got string."
                 }
              }
            }"""
    },
    {
        'testcase_name':
            'multiple_features_all_reasons_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.ENUM_TYPE_BYTES_NOT_STRING,
                anomalies_pb2.AnomalyInfo.ENUM_TYPE_UNEXPECTED_STRING_VALUES
            ]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Expected bytes but got string. Examples "
                    "contain values missing from the schema."
                  severity: ERROR
                  short_description: "Multiple errors"
                  reason {
                    type: ENUM_TYPE_BYTES_NOT_STRING
                    short_description: "Bytes not string"
                    description: "Expected bytes but got string."
                  }
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing from the "
                      "schema."
                  }
                }
              }
            anomaly_info {
              key: "example_2"
              value {
                description: "Examples contain values missing from the schema."
                severity: ERROR
                short_description: "Unexpected string values"
                reason {
                  type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                  short_description: "Unexpected string values"
                  description: "Examples contain values missing from the "
                    "schema."
                }
              }
            }""",
        'expected_anomalies_proto_text': ''
    }
]

SET_REMOVE_ANOMALY_TYPES_DOES_NOT_CHANGE_PROTO_TESTS = [
    {
        'testcase_name':
            'single_reason_not_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_NOT_PRESENT
            ]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Examples contain values missing from the "
                    "schema."
                  severity: ERROR
                  short_description: "Unexpected string values"
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing from the "
                      "schema."
                  }
                }
              }"""
    },
    {
        'testcase_name':
            'multiple_reasons_not_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_NOT_PRESENT
            ]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Expected bytes but got string. Examples "
                    "contain values missing from the schema."
                  severity: ERROR
                  short_description: "Multiple errors"
                  reason {
                    type: ENUM_TYPE_BYTES_NOT_STRING
                    short_description: "Bytes not string"
                    description: "Expected bytes but got string."
                  }
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing from the "
                      "schema."
                  }
                }
              }"""
    },
    {
        'testcase_name':
            'multiple_features_no_reasons_removed',
        'anomaly_types_to_remove':
            set([
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_LOW_NUMBER_PRESENT,
                anomalies_pb2.AnomalyInfo.FEATURE_TYPE_NOT_PRESENT
            ]),
        'input_anomalies_proto_text':
            """
             anomaly_info {
               key: "example_1"
               value {
                  description: "Expected bytes but got string."
                  severity: ERROR
                  short_description: "Bytes not string"
                  reason {
                    type: ENUM_TYPE_BYTES_NOT_STRING
                    short_description: "Bytes not string"
                    description: "Expected bytes but got string."
                  }
                }
              }
              anomaly_info {
                key: "example_2"
                value {
                   description: "Examples contain values missing from the "
                     "schema."
                   severity: ERROR
                   short_description: "Unexpected string values"
                   reason {
                     type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                     short_description: "Unexpected string values"
                     description: "Examples contain values missing from the "
                       "schema."
                   }
                }
              }"""
    }
]


class AnomaliesUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(*SET_REMOVE_ANOMALY_TYPES_CHANGES_PROTO_TESTS)
  def test_remove_anomaly_types_changes_proto(self, anomaly_types_to_remove,
                                              input_anomalies_proto_text,
                                              expected_anomalies_proto_text):
    """Tests where remove_anomaly_types modifies the Anomalies proto."""
    input_anomalies_proto = text_format.Parse(input_anomalies_proto_text,
                                              anomalies_pb2.Anomalies())
    expected_anomalies_proto = text_format.Parse(expected_anomalies_proto_text,
                                                 anomalies_pb2.Anomalies())
    anomalies_util.remove_anomaly_types(input_anomalies_proto,
                                        anomaly_types_to_remove)
    compare.assertProtoEqual(self, input_anomalies_proto,
                             expected_anomalies_proto)

  @parameterized.named_parameters(
      *SET_REMOVE_ANOMALY_TYPES_DOES_NOT_CHANGE_PROTO_TESTS)
  def test_remove_anomaly_types_does_not_change_proto(
      self, anomaly_types_to_remove, input_anomalies_proto_text):
    """Tests where remove_anomaly_types does not modify the Anomalies proto."""
    input_anomalies_proto = text_format.Parse(input_anomalies_proto_text,
                                              anomalies_pb2.Anomalies())
    expected_anomalies_proto = anomalies_pb2.Anomalies()
    expected_anomalies_proto.CopyFrom(input_anomalies_proto)
    anomalies_util.remove_anomaly_types(input_anomalies_proto,
                                        anomaly_types_to_remove)
    compare.assertProtoEqual(self, input_anomalies_proto,
                             expected_anomalies_proto)



if __name__ == '__main__':
  absltest.main()
