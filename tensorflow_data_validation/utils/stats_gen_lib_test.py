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
"""Tests for stat_gen_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.io.filesystem import CompressionTypes
import pandas as pd
import tensorflow as tf

from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.utils import stats_gen_lib
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class StatsGenTest(parameterized.TestCase):

  def setUp(self):
    super(StatsGenTest, self).setUp()
    self._default_stats_options = stats_options.StatsOptions(
        num_top_values=2,
        num_rank_histogram_buckets=2,
        num_values_histogram_buckets=2,
        num_histogram_buckets=2,
        num_quantiles_histogram_buckets=2)

  def _get_temp_dir(self):
    return tempfile.mkdtemp()

  def _make_example(self, feature_name_to_type_values_tuple_map):
    """Makes a tensorflow example.

    Args:
      feature_name_to_type_values_tuple_map: A map of feature name to
        [feature_type, feature_value_list] tuples. The feature type is one of
        'bytes'/'float'/'int'.

    Raises:
      ValueError: input feature type is invalid.

    Returns:
      A tf.Example.
    """
    result = tf.train.Example()
    for feature_name in feature_name_to_type_values_tuple_map:
      (feature_type, feature_values) = (
          feature_name_to_type_values_tuple_map[feature_name])
      if feature_type == 'bytes':
        result.features.feature[
            feature_name].bytes_list.value[:] = feature_values
      elif feature_type == 'float':
        result.features.feature[
            feature_name].float_list.value[:] = feature_values
      elif feature_type == 'int':
        result.features.feature[
            feature_name].int64_list.value[:] = feature_values
      else:
        raise ValueError('Invalid feature type: ' + feature_type)
    return result

  def _write_tfexamples_to_tfrecords(self, examples, compression_type):
    filename = 'input_data.tfrecord'
    if compression_type == tf.compat.v1.python_io.TFRecordCompressionType.GZIP:
      filename += '.gz'
    data_location = os.path.join(self._get_temp_dir(), filename)
    with tf.io.TFRecordWriter(
        data_location, options=compression_type) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    return data_location

  _BEAM_COMPRESSION_TYPES = [
      {
          'testcase_name': 'no_compression',
          'compression_type': CompressionTypes.AUTO
      },
      {
          'testcase_name': 'gzip_compression',
          'compression_type': CompressionTypes.GZIP
      },
  ]

  @parameterized.named_parameters(*_BEAM_COMPRESSION_TYPES)
  def test_stats_gen_with_tfrecords_of_tfexamples(self, compression_type):
    examples = [
        self._make_example({
            'a': ('float', [1.0, 2.0]),
            'b': ('bytes', [b'a', b'b', b'c', b'e'])
        }),
        self._make_example({
            'a': ('float', [3.0, 4.0, float('nan'), 5.0]),
            'b': ('bytes', [b'a', b'c', b'd', b'a'])
        }),
        self._make_example({
            'a': ('float', [1.0]),
            'b': ('bytes', [b'a', b'b', b'c', b'd'])
        })
    ]
    tf_compression_lookup = {
        CompressionTypes.AUTO:
            tf.compat.v1.python_io.TFRecordCompressionType.NONE,
        CompressionTypes.GZIP:
            tf.compat.v1.python_io.TFRecordCompressionType.GZIP
    }
    input_data_path = self._write_tfexamples_to_tfrecords(
        examples, tf_compression_lookup[compression_type])

    expected_result = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        path {
          step: "a"
        }
        type: FLOAT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 0
            min_num_values: 1
            max_num_values: 4
            avg_num_values: 2.33333333
            tot_num_values: 7
          }
          mean: 2.66666666
          std_dev: 1.49071198
          num_zeros: 0
          min: 1.0
          max: 5.0
          median: 3.0
        }
      }
      features {
        path {
          step: "b"
        }
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 3
            min_num_values: 4
            max_num_values: 4
            avg_num_values: 4.0
            tot_num_values: 12
          }
          unique: 5
          top_values {
            value: "a"
            frequency: 4.0
          }
          top_values {
            value: "c"
            frequency: 3.0
          }
          avg_length: 1.0
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 4.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "c"
              sample_count: 3.0
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    result = stats_gen_lib.generate_statistics_from_tfrecord(
        data_location=input_data_path,
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def _write_records_to_csv(self, records, tmp_dir, filename,
                            compression_type=''):
    data_location = os.path.join(tmp_dir, filename)
    if compression_type == 'gzip':
      with gzip.GzipFile(data_location, 'wb') as writer:
        writer.write('\n'.join(records).encode('utf-8'))
    else:
      with open(data_location, 'w') as writer:
        writer.write('\n'.join(records))
    return data_location

  def _get_csv_test(self, delimiter=',', with_header=False):
    fields = [['feature1', 'feature2'], ['1.0', 'aa'], ['2.0', 'bb'],
              ['3.0', 'cc'], ['4.0', 'dd'], ['5.0', 'ee'], ['6.0', 'ff'],
              ['7.0', 'gg'], ['', '']]
    records = []
    for row in fields:
      records.append(delimiter.join(row))

    expected_result = text_format.Parse(
        """
    datasets {
  num_examples: 8
  features {
    path {
      step: "feature1"
    }
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 7
        num_missing: 1
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        tot_num_values: 7
      }
      mean: 4.0
      std_dev: 2.0
      min: 1.0
      max: 7.0
      median: 4.0
    }
  }
  features {
    path {
      step: "feature2"
    }
    type: STRING
    string_stats {
      common_stats {
        num_non_missing: 7
        num_missing: 1
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        tot_num_values: 7
      }
      unique: 7
      top_values {
        value: "gg"
        frequency: 1.0
      }
      top_values {
        value: "ff"
        frequency: 1.0
      }
      avg_length: 2.0
      rank_histogram {
        buckets {
          label: "gg"
          sample_count: 1.0
        }
        buckets {
          low_rank: 1
          high_rank: 1
          label: "ff"
          sample_count: 1.0
        }
      }
    }
  }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    if with_header:
      return (records, None, expected_result)
    return (records[1:], records[0].split(delimiter), expected_result)

  @parameterized.named_parameters(*_BEAM_COMPRESSION_TYPES)
  def test_stats_gen_with_csv_no_header_in_file(self, compression_type):
    records, header, expected_result = self._get_csv_test(delimiter=',',
                                                          with_header=False)
    compression_type_lookup = {
        CompressionTypes.AUTO: '',
        CompressionTypes.GZIP: 'gzip'
    }
    input_data_path = self._write_records_to_csv(
        records, self._get_temp_dir(), 'input_data.csv',
        compression_type=compression_type_lookup[compression_type])

    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        column_names=header,
        delimiter=',',
        stats_options=self._default_stats_options,
        compression_type=compression_type)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_csv_header_in_file(self):
    records, header, expected_result = self._get_csv_test(delimiter=',',
                                                          with_header=True)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')

    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        column_names=header,
        delimiter=',',
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_csv_tab_delimiter_no_header_in_file(self):
    records, header, expected_result = self._get_csv_test(delimiter='\t',
                                                          with_header=False)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.tsv')

    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        column_names=header,
        delimiter='\t',
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_csv_header_in_multiple_files(self):
    records, _, expected_result = self._get_csv_test(delimiter=',',
                                                     with_header=True)
    header = records.pop(0)
    # Split the records into two subsets and write to separate files.
    records1 = [header] + records[0:3]
    records2 = [header] + records[3:]
    tmp_dir = self._get_temp_dir()
    self._write_records_to_csv(records1, tmp_dir, 'input_data1.csv')
    self._write_records_to_csv(records2, tmp_dir, 'input_data2.csv')
    input_data_path = os.path.join(tmp_dir, 'input_data*')

    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        column_names=None,
        delimiter=',',
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_csv_with_schema(self):
    records = ['feature1', '1']
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')
    schema = text_format.Parse(
        """
        feature { name: "feature1" type: BYTES }
        """, schema_pb2.Schema())

    expected_result = text_format.Parse(
        """
    datasets {
  num_examples: 1
  features {
    path {
      step: "feature1"
    }
    type: STRING
    string_stats {
      common_stats {
        num_non_missing: 1
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        tot_num_values: 1
      }
      unique: 1
      top_values {
        value: "1"
        frequency: 1.0
      }
      avg_length: 1.0
      rank_histogram {
        buckets {
          label: "1"
          sample_count: 1.0
        }
      }
    }
  }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    self._default_stats_options.schema = schema
    self._default_stats_options.infer_type_from_schema = True
    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        delimiter=',',
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_invalid_csv_header_in_multiple_files(self):
    records, _, _ = self._get_csv_test(delimiter=',',
                                       with_header=True)
    header = records.pop(0)
    # Split the records into two subsets and write to separate files.
    records1 = [header] + records[0:3]
    records2 = ['random,header'] + records[3:]
    tmp_dir = self._get_temp_dir()
    self._write_records_to_csv(records1, tmp_dir, 'input_data1.csv')
    self._write_records_to_csv(records2, tmp_dir, 'input_data2.csv')
    input_data_path = os.path.join(tmp_dir, 'input_data*')

    with self.assertRaisesRegexp(
        ValueError, 'Files have different headers.'):
      _ = stats_gen_lib.generate_statistics_from_csv(
          data_location=input_data_path, column_names=None, delimiter=',')

  def test_stats_gen_with_csv_missing_column(self):
    records = [',', ',']
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')
    expected_result = text_format.Parse(
        """
        datasets {
          num_examples: 2
          features {
            path {
              step: "feature1"
            }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 2
              }
            }
          }
          features {
            path {
              step: "feature2"
            }
            type: STRING
            string_stats {
              common_stats {
                num_missing: 2
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    result = stats_gen_lib.generate_statistics_from_csv(
        data_location=input_data_path,
        column_names=['feature1', 'feature2'],
        delimiter=',',
        stats_options=self._default_stats_options)
    compare_fn = test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self, expected_result, check_histograms=False)
    compare_fn([result])

  def test_stats_gen_with_header_in_empty_csv_file(self):
    input_data_path = self._write_records_to_csv([], self._get_temp_dir(),
                                                 'input_data.csv')

    with self.assertRaisesRegexp(
        ValueError, 'Found empty file when reading the header.*'):
      _ = stats_gen_lib.generate_statistics_from_csv(
          data_location=input_data_path, column_names=None, delimiter=',')

  def test_stats_gen_with_dataframe(self):
    records, _, expected_result = self._get_csv_test(delimiter=',',
                                                     with_header=True)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')

    dataframe = pd.read_csv(input_data_path)
    result = stats_gen_lib.generate_statistics_from_dataframe(
        dataframe=dataframe,
        stats_options=self._default_stats_options, n_jobs=1)
    self.assertLen(result.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self,
        result.datasets[0],
        expected_result.datasets[0],
        check_histograms=False)

  def test_stats_gen_with_dataframe_feature_allowlist(self):
    records, _, expected_result = self._get_csv_test(delimiter=',',
                                                     with_header=True)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')

    dataframe = pd.read_csv(input_data_path)
    stats_options_allowlist = self._default_stats_options
    stats_options_allowlist.feature_allowlist = list(dataframe.columns)
    dataframe['to_be_removed_column'] = [
        [1, 2], [], None, [1], None, [3, 4], [], None]
    result = stats_gen_lib.generate_statistics_from_dataframe(
        dataframe=dataframe, stats_options=stats_options_allowlist, n_jobs=1)
    self.assertLen(result.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self,
        result.datasets[0],
        expected_result.datasets[0],
        check_histograms=False)

  def test_stats_gen_with_dataframe_invalid_njobs_zero(self):
    records, _, _ = self._get_csv_test(delimiter=',', with_header=True)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')
    dataframe = pd.read_csv(input_data_path)
    with self.assertRaisesRegexp(
        ValueError, 'Invalid n_jobs parameter.*'):
      _ = stats_gen_lib.generate_statistics_from_dataframe(
          dataframe=dataframe,
          stats_options=self._default_stats_options, n_jobs=0)

  def test_stats_gen_with_dataframe_invalid_njobs_negative(self):
    records, _, _ = self._get_csv_test(delimiter=',', with_header=True)
    input_data_path = self._write_records_to_csv(records, self._get_temp_dir(),
                                                 'input_data.csv')
    dataframe = pd.read_csv(input_data_path)
    with self.assertRaisesRegexp(
        ValueError, 'Invalid n_jobs parameter.*'):
      _ = stats_gen_lib.generate_statistics_from_dataframe(
          dataframe=dataframe,
          stats_options=self._default_stats_options, n_jobs=-2)

  def test_get_csv_header(self):
    temp_directory = self._get_temp_dir()
    delimiter = ','
    records = ['feature1,feature2', '1.0,aa']
    expected_header = ['feature1', 'feature2']
    self._write_records_to_csv(records, temp_directory, 'input_data_1.csv')
    self._write_records_to_csv(records, temp_directory, 'input_data_2.csv')
    data_location = os.path.join(temp_directory, 'input_data_*.csv')
    header = stats_gen_lib.get_csv_header(data_location, delimiter)
    self.assertEqual(header, expected_header)

  def test_get_csv_header_no_file(self):
    data_location = os.path.join(self._get_temp_dir(), 'fileA.csv')
    delimiter = ','
    with self.assertRaisesRegexp(ValueError, 'No file found.*'):
      _ = stats_gen_lib.get_csv_header(data_location, delimiter)

  def test_get_csv_header_empty_file(self):
    empty_file = os.path.join(self._get_temp_dir(), 'empty.csv')
    open(empty_file, 'w+').close()
    delimiter = ','
    with self.assertRaisesRegexp(ValueError, 'Found empty file.*'):
      _ = stats_gen_lib.get_csv_header(empty_file, delimiter)

  def test_get_csv_header_different_headers(self):
    temp_directory = self._get_temp_dir()
    delimiter = ','
    records_1 = ['feature1,feature2', '1.0,aa']
    records_2 = ['feature1,feature2_different', '2.0,bb']
    self._write_records_to_csv(records_1, temp_directory, 'input_data_1.csv')
    self._write_records_to_csv(records_2, temp_directory, 'input_data_2.csv')
    data_location = os.path.join(temp_directory, 'input_data_*.csv')
    with self.assertRaisesRegexp(ValueError, 'Files have different headers.'):
      _ = stats_gen_lib.get_csv_header(data_location, delimiter)

  def test_get_csv_header_gzip(self):
    temp_directory = self._get_temp_dir()
    delimiter = ','
    records = ['feature1,feature2', '1.0,aa']
    expected_header = ['feature1', 'feature2']
    self._write_records_to_csv(
        records, temp_directory, 'input_data_1.csv.gz', compression_type='gzip')
    self._write_records_to_csv(
        records, temp_directory, 'input_data_2.csv.gz', compression_type='gzip')

    data_location = os.path.join(temp_directory, 'input_data_*.csv.gz')
    header = stats_gen_lib.get_csv_header(data_location, delimiter)
    self.assertEqual(header, expected_header)

  def test_get_csv_header_new_line(self):
    temp_directory = self._get_temp_dir()
    delimiter = ','
    records = ['"\n","feature2"', '1.0,aa']
    expected_header = ['\n', 'feature2']
    self._write_records_to_csv(
        records, temp_directory, 'input_data_1.csv.gz', compression_type='gzip')
    self._write_records_to_csv(
        records, temp_directory, 'input_data_2.csv.gz', compression_type='gzip')

    data_location = os.path.join(temp_directory, 'input_data_*.csv.gz')
    header = stats_gen_lib.get_csv_header(data_location, delimiter)
    self.assertEqual(header, expected_header)


if __name__ == '__main__':
  absltest.main()
