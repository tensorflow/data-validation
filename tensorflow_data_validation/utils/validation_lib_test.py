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
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils import validation_lib

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class ValidationLibTest(parameterized.TestCase):

  @parameterized.named_parameters(('no_sampled_examples', 0),
                                  ('sampled_examples', 99))
  def test_validate_examples_in_tfrecord(self, num_sampled_examples):
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
    with tf.io.TFRecordWriter(input_data_path) as writer:
      for example in input_examples:
        example = text_format.Parse(example, tf.train.Example())
        writer.write(example.SerializeToString())

    expected_result = text_format.Parse(
        """
    datasets {
      name: 'annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES'
      num_examples: 1
      features {
        path: {
          step: 'annotated_enum'
        }
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
        path: {
          step: 'unknown_feature'
        }
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
        path: {
          step: 'annotated_enum'
        }
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

    actual_result = validation_lib.validate_examples_in_tfrecord(
        data_location=input_data_path,
        stats_options=options,
        num_sampled_examples=num_sampled_examples)
    if num_sampled_examples:
      actual_result, sampled_examples = actual_result
      self.assertCountEqual(
          [('annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
            [text_format.Parse(input_examples[0], tf.train.Example())]),
           ('unknown_feature_SCHEMA_NEW_COLUMN',
            [text_format.Parse(input_examples[1], tf.train.Example())])],
          sampled_examples.items())
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([actual_result])

  def test_validate_examples_in_tfrecord_no_schema(self):
    temp_dir_path = self.create_tempdir().full_path
    input_data_path = os.path.join(temp_dir_path, 'input_data.tfrecord')
    # By default, StatsOptions does not include a schema.
    options = stats_options.StatsOptions()
    with self.assertRaisesRegexp(
        ValueError, 'The specified stats_options must include a schema.'):
      validation_lib.validate_examples_in_tfrecord(
          data_location=input_data_path, stats_options=options)

  def _get_anomalous_csv_test(self, delimiter, output_column_names,
                              generate_single_file, has_schema):
    """Creates test CSV(s) and returns a tuple containing information re same.

    This is used to test validate_examples_in_csv. The function creates test CSV
    file(s) and returns a tuple consisting of the location of those file(s), the
    column names (if not provided as part of the CSV file), the stats options,
    and a proto containing the anomalies that should be detected in the examples
    in the test CSV(s).

    Args:
      delimiter: The one-character string used to separate fields in the
        generated CSV file(s).
      output_column_names: Whether to output a list of column names. If True,
        this function uses the first record as the column_names value returned
        in the tuple. If False, this function returns None as the column_names
        value.
      generate_single_file: If True, generates a single test CSV file. If false,
        generates multiple test CSV files.
      has_schema: If True, includes the schema in the output options.

    Returns:
      A tuple consisting of the following values:
        data_location: The location of the test CSV file(s).
        column_names: A list of column names to be treated as the CSV header, or
          None if the first line in the test CSV should be used as the
          header.
        options: `tfdv.StatsOptions` for generating data statistics.
        expected_result: The anomalies that should be detected in the examples
          in the CSV(s).
    """
    fields = [['annotated_enum', 'other_feature'], ['D', '1'], ['A', '2']]
    column_names = None
    if output_column_names:
      column_names = fields[0]
      fields = fields[1:]
    records = []
    for row in fields:
      records.append(delimiter.join(row))

    temp_dir = self.create_tempdir().full_path
    if not generate_single_file:
      records_per_file = [records[0:1], records[1:]]
    else:
      records_per_file = [records]
    for i, records in enumerate(records_per_file):
      filepath = os.path.join(temp_dir, 'input_data_%s.csv' % i)
      with open(filepath, 'w+') as writer:
        for record in records:
          writer.write(record + '\n')
    data_location = os.path.join(temp_dir, 'input_data_*.csv')

    if has_schema:
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
                  name: "other_feature"
                  value_count {
                    min:1
                    max:1
                  }
                  presence {
                    min_count: 1
                  }
                  type: INT
                }
                """, schema_pb2.Schema())
    else:
      schema = None
    options = stats_options.StatsOptions(
        schema=schema,
        num_top_values=2,
        num_rank_histogram_buckets=2,
        num_values_histogram_buckets=2,
        num_histogram_buckets=2,
        num_quantiles_histogram_buckets=2)

    expected_result = text_format.Parse(
        """
    datasets {
      name: 'annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES'
      num_examples: 1
      features {
        path: {
          step: 'annotated_enum'
        }
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
      features {
        path: {
          step: 'other_feature'
        }
        type: INT
        num_stats {
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
          mean: 1.0
          min: 1.0
          median: 1.0
          max: 1.0
          histograms {
            buckets {
              low_value: 1.0
              high_value: 1.0
              sample_count: 1.0
            }
          }
          histograms {
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
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    return (data_location, column_names, options, expected_result)

  def test_validate_examples_in_csv(self):
    data_location, _, options, expected_result = (
        self._get_anomalous_csv_test(
            delimiter=',',
            output_column_names=False,
            generate_single_file=True,
            has_schema=True))

    result = validation_lib.validate_examples_in_csv(
        data_location=data_location,
        stats_options=options,
        column_names=None,
        delimiter=',')
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([result])

  def test_validate_examples_in_csv_with_examples(self):
    data_location, _, options, expected_result = (
        self._get_anomalous_csv_test(
            delimiter=',',
            output_column_names=False,
            generate_single_file=True,
            has_schema=True))

    result, sampled_examples = validation_lib.validate_examples_in_csv(
        data_location=data_location,
        stats_options=options,
        column_names=None,
        delimiter=',',
        num_sampled_examples=99)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([result])
    self.assertCountEqual([
        'annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES',
    ], sampled_examples.keys())
    got_df = sampled_examples[
        'annotated_enum_ENUM_TYPE_UNEXPECTED_STRING_VALUES']
    expected_df = pd.DataFrame.from_records(
        [['D', 1]], columns=['annotated_enum', 'other_feature'])
    expected_df['annotated_enum'] = expected_df['annotated_enum'].astype(bytes)
    # We can't be too picky about dtypes; try to coerce to expected types.
    for col in got_df.columns:
      if col in expected_df.columns:
        got_df[col] = got_df[col].astype(expected_df[col].dtype)
    self.assertTrue(expected_df.equals(got_df))

  def test_validate_examples_in_csv_no_header_in_file(self):
    data_location, column_names, options, expected_result = (
        self._get_anomalous_csv_test(
            delimiter=',',
            output_column_names=True,
            generate_single_file=True,
            has_schema=True))

    assert column_names is not None
    result = validation_lib.validate_examples_in_csv(
        data_location=data_location,
        stats_options=options,
        column_names=column_names,
        delimiter=',')
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([result])

  def test_validate_examples_in_csv_no_schema(self):
    data_location, _, options, _ = (
        self._get_anomalous_csv_test(
            delimiter=',',
            output_column_names=False,
            generate_single_file=True,
            has_schema=False))

    assert options.schema is None
    with self.assertRaisesRegexp(ValueError, 'The specified stats_options.*'):
      validation_lib.validate_examples_in_csv(
          data_location=data_location,
          stats_options=options,
          column_names=None,
          delimiter=',')

  def test_validate_examples_in_csv_tab_delimiter(self):
    data_location, _, options, expected_result = (
        self._get_anomalous_csv_test(
            delimiter='\t',
            output_column_names=False,
            generate_single_file=True,
            has_schema=True))

    result = validation_lib.validate_examples_in_csv(
        data_location=data_location,
        stats_options=options,
        column_names=None,
        delimiter='\t')
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([result])

  def test_validate_examples_in_csv_multiple_files(self):
    data_location, column_names, options, expected_result = (
        self._get_anomalous_csv_test(
            delimiter=',',
            output_column_names=True,
            generate_single_file=False,
            has_schema=True))

    result = validation_lib.validate_examples_in_csv(
        data_location=data_location,
        stats_options=options,
        column_names=column_names,
        delimiter=',')
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result)
    compare_fn([result])


if __name__ == '__main__':
  absltest.main()
