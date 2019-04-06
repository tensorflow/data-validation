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
"""Tests for validation_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import absltest
import tensorflow as tf

from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils import validation_lib

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class ValidationLibTest(absltest.TestCase):

  def test_validate_tfexamples_in_tfrecord(self):
    input_examples = [
        # This example is anomalous because its feature contains a value that is
        # not in the string_domain specified in the schema.
        """
          features {
              feature {
                key: 'annotated_enum'
                value { bytes_list { value: [ 'D' ] } }
              }
          }
        """,
        # This example is anomalous because it contains a feature that is not
        # in the schema.
        """
          features {
              feature {
                key: 'annotated_enum'
                value { bytes_list { value: [ 'A' ] } }
              }
              feature {
                key: 'unknown_feature'
                value { bytes_list { value: [ 'A' ] } }
              }
          }
        """,
    ]
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
    options = stats_options.StatsOptions(
        schema=schema,
        num_top_values=2,
        num_rank_histogram_buckets=2,
        num_values_histogram_buckets=2,
        num_histogram_buckets=2,
        num_quantiles_histogram_buckets=2)

    temp_dir_path = self.create_tempdir().full_path
    input_data_path = os.path.join(temp_dir_path, 'input_data.tfrecord')
    with tf.python_io.TFRecordWriter(input_data_path) as writer:
      for example in input_examples:
        example = text_format.Parse(example, tf.train.Example())
        writer.write(example.SerializeToString())

    expected_result = text_format.Parse(
        """
    datasets {
      name: 'annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES'
      num_examples: 1
      features {
        name: 'annotated_enum'
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            min_num_values: 1
            max_num_values: 1
            avg_num_values: 1.0
            tot_num_values: 1
            num_values_histogram {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              type: QUANTILES
            }
          }
          unique: 1
          top_values {
            value: "D"
            frequency: 1.0
          }
          avg_length: 1.0
          rank_histogram {
            buckets {
              label: "D"
              sample_count: 1.0
            }
          }
        }
      }
    }
    datasets {
      name: 'unknown_feature_SCHEMA_NEW_COLUMN'
      num_examples: 1
      features {
        name: 'unknown_feature'
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            min_num_values: 1
            max_num_values: 1
            avg_num_values: 1.0
            tot_num_values: 1
            num_values_histogram {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              type: QUANTILES
            }
          }
          unique: 1
          top_values {
            value: "A"
            frequency: 1.0
          }
          avg_length: 1.0
          rank_histogram {
            buckets {
              label: "A"
              sample_count: 1.0
            }
          }
        }
      }
      features {
        name: 'annotated_enum'
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 1
            num_missing: 0
            min_num_values: 1
            max_num_values: 1
            avg_num_values: 1.0
            tot_num_values: 1
            num_values_histogram {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 0.5
              }
              type: QUANTILES
            }
          }
          unique: 1
          top_values {
            value: "A"
            frequency: 1.0
          }
          avg_length: 1.0
          rank_histogram {
            buckets {
              label: "A"
              sample_count: 1.0
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    actual_result = validation_lib.validate_tfexamples_in_tfrecord(
        data_location=input_data_path, stats_options=options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([actual_result])

  def test_validate_tfexamples_in_tfrecord_no_schema(self):
    temp_dir_path = self.create_tempdir().full_path
    input_data_path = os.path.join(temp_dir_path, 'input_data.tfrecord')
    # By default, StatsOptions does not include a schema.
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(
        ValueError, 'The specified stats_options must include a schema.'):
      validation_lib.validate_tfexamples_in_tfrecord(
          data_location=input_data_path, stats_options=options)


if __name__ == '__main__':
  absltest.main()
