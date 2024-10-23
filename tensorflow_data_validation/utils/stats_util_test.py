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

"""Tests for utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl.testing import absltest
import numpy as np
import tensorflow as tf
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import artifacts_io_impl
from tensorflow_data_validation.utils import stats_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import statistics_pb2

FLAGS = flags.FLAGS


class StatsUtilTest(absltest.TestCase):

  def test_get_feature_type_get_int(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int8')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int16')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int32')),
        statistics_pb2.FeatureNameStatistics.INT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('int64')),
        statistics_pb2.FeatureNameStatistics.INT)

  def test_get_feature_type_get_float(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float16')),
        statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float32')),
        statistics_pb2.FeatureNameStatistics.FLOAT)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('float64')),
        statistics_pb2.FeatureNameStatistics.FLOAT)

  def test_get_feature_type_get_string(self):
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('S')),
        statistics_pb2.FeatureNameStatistics.STRING)
    self.assertEqual(
        stats_util.get_feature_type(np.dtype('U')),
        statistics_pb2.FeatureNameStatistics.STRING)

  def test_get_feature_type_get_none(self):
    self.assertIsNone(stats_util.get_feature_type(np.dtype('complex64')))

  def test_make_dataset_feature_stats_proto(self):
    stats = {
        types.FeaturePath(['feature_1']): {
            'Mutual Information': 0.5,
            'Correlation': 0.1
        },
        types.FeaturePath(['feature_2']): {
            'Mutual Information': 0.8,
            'Correlation': 0.6
        }
    }
    expected = {
        types.FeaturePath(['feature_1']):
            text_format.Parse(
                """
            path {
              step: 'feature_1'
            }
            custom_stats {
              name: 'Correlation'
              num: 0.1
            }
            custom_stats {
              name: 'Mutual Information'
              num: 0.5
            }
           """, statistics_pb2.FeatureNameStatistics()),
        types.FeaturePath(['feature_2']):
            text_format.Parse(
                """
            path {
              step: 'feature_2'
            }
            custom_stats {
              name: 'Correlation'
              num: 0.6
            }
            custom_stats {
              name: 'Mutual Information'
              num: 0.8
            }
           """, statistics_pb2.FeatureNameStatistics())
    }
    actual = stats_util.make_dataset_feature_stats_proto(stats)
    self.assertEqual(len(actual.features), len(expected))
    for actual_feature_stats in actual.features:
      compare.assertProtoEqual(
          self,
          actual_feature_stats,
          expected[types.FeaturePath.from_proto(actual_feature_stats.path)],
          normalize_numbers=True)

  def test_get_utf8(self):
    self.assertEqual(u'This is valid.',
                     stats_util.maybe_get_utf8(b'This is valid.'))
    self.assertIsNone(stats_util.maybe_get_utf8(b'\xF0'))

  def test_write_load_stats_text(self):
    stats = text_format.Parse("""
      datasets { name: 'abc' }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    stats_path = os.path.join(FLAGS.test_tmpdir, 'stats.pbtxt')
    stats_util.write_stats_text(stats=stats, output_path=stats_path)
    self.assertEqual(stats, stats_util.load_stats_text(input_path=stats_path))
    self.assertEqual(stats, stats_util.load_statistics(input_path=stats_path))

  def test_load_stats_tfrecord(self):
    stats = text_format.Parse("""
      datasets { name: 'abc' }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    stats_path = os.path.join(FLAGS.test_tmpdir, 'stats.tfrecord')
    with tf.io.TFRecordWriter(stats_path) as writer:
      writer.write(stats.SerializeToString())
    self.assertEqual(stats,
                     stats_util.load_stats_tfrecord(input_path=stats_path))
    self.assertEqual(stats, stats_util.load_statistics(input_path=stats_path))

  def test_load_stats_binary(self):
    stats = text_format.Parse("""
      datasets { name: 'abc' }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    stats_path = os.path.join(FLAGS.test_tmpdir, 'stats.binpb')
    with open(stats_path, 'w+b') as f:
      f.write(stats.SerializeToString())
    self.assertEqual(stats, stats_util.load_stats_binary(input_path=stats_path))

  def test_write_stats_text_invalid_stats_input(self):
    with self.assertRaisesRegex(
        TypeError, '.*should be a DatasetFeatureStatisticsList proto.'):
      _ = stats_util.write_stats_text({}, 'stats.pbtxt')

  def test_get_custom_stats_numeric(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              num: 100.0
            }
        """, statistics_pb2.FeatureNameStatistics())
    self.assertEqual(stats_util.get_custom_stats(stats, 'abc'), 100.0)

  def test_get_custom_stats_string(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              str: 'xyz'
            }
        """, statistics_pb2.FeatureNameStatistics())
    self.assertEqual(stats_util.get_custom_stats(stats, 'abc'), 'xyz')

  def test_get_custom_stats_not_found(self):
    stats = text_format.Parse(
        """
            name: 'feature'
            custom_stats {
              name: 'abc'
              num: 100.0
            }
        """, statistics_pb2.FeatureNameStatistics())
    with self.assertRaisesRegex(ValueError, 'Custom statistics.*not found'):
      stats_util.get_custom_stats(stats, 'xyz')

  def test_get_slice_stats(self):
    statistics = text_format.Parse("""
    datasets {
      name: "slice1"
      num_examples: 100
    }
    datasets {
      name: "slice2"
      num_examples: 200
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    for slice_key in ['slice1', 'slice2']:
      actual_stats = stats_util.get_slice_stats(statistics, slice_key)
      self.assertLen(actual_stats.datasets, 1)
      self.assertEqual(actual_stats.datasets[0].name, slice_key)
    with self.assertRaisesRegex(ValueError, 'Invalid slice key'):
      stats_util.get_slice_stats(statistics, 'slice3')


_STATS_PROTO = """
datasets: {
  name: "slice0"
}
datasets: {
  name: "slice1"
}
datasets: {
  name: 'All Examples'
  features: {
    path: {
      step: "f0_step1"
      step: "f0_step2"
    }
  }
  features: {
    path: {
      step: "f1"
    }
  }
  features: {
    path: {
      step: "f3_derived"
    }
    validation_derived_source: {
       deriver_name: "my_deriver_name",
       source_path: {
           step: "f0_step1"
           step: "f0_step2"
       }
       source_path: {
           step: "f1"
       }
    }
  }
  cross_features: {
    path_x: {
      step: "f1x"
    }
    path_y: {
      step: "f1y"
    }
  }
  cross_features: {
    path_x: {
      step: "f2x"
    }
    path_y: {
      step: "f2y"
    }
  }
}
"""


class DatasetListViewTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
    text_format.Parse(_STATS_PROTO, self._stats_proto)

  def test_list_slices(self):
    view = stats_util.DatasetListView(self._stats_proto)
    self.assertCountEqual(['slice0', 'slice1', 'All Examples'],
                          view.list_slices())

  def test_get_slice0(self):
    view = stats_util.DatasetListView(self._stats_proto)
    slice0 = view.get_slice('slice0')
    self.assertEqual(self._stats_proto.datasets[0], slice0.proto())

  def test_get_slice1(self):
    view = stats_util.DatasetListView(self._stats_proto)
    slice1 = view.get_slice('slice1')
    self.assertEqual(self._stats_proto.datasets[1], slice1.proto())

  def test_get_default(self):
    view = stats_util.DatasetListView(self._stats_proto)
    default_slice = view.get_default_slice()
    self.assertEqual(self._stats_proto.datasets[2], default_slice.proto())

  def test_get_missing_slice(self):
    view = stats_util.DatasetListView(self._stats_proto)
    slice99 = view.get_slice('slice99')
    self.assertIsNone(slice99)

  def test_get_feature_by_name(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    feature1 = view.get_feature('f1')
    self.assertEqual(self._stats_proto.datasets[2].features[1],
                     feature1.proto())

  def test_get_feature_by_path(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    feature1 = view.get_feature(types.FeaturePath(['f0_step1', 'f0_step2']))
    self.assertEqual(self._stats_proto.datasets[2].features[0],
                     feature1.proto())

  def test_get_feature_by_path_steps(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    feature1 = view.get_feature(['f0_step1', 'f0_step2'])
    self.assertEqual(self._stats_proto.datasets[2].features[0],
                     feature1.proto())

  def test_get_derived_feature(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    feature1 = view.get_derived_feature('my_deriver_name', [
        types.FeaturePath(['f0_step1', 'f0_step2']),
        types.FeaturePath(['f1'])
    ])
    self.assertEqual(self._stats_proto.datasets[2].features[2],
                     feature1.proto())

  def test_get_derived_feature_ambiguous(self):
    stats_proto = statistics_pb2.DatasetFeatureStatisticsList.FromString(
        self._stats_proto.SerializeToString())
    # Duplicate the derived feature.
    stats_proto.datasets[2].features.append(stats_proto.datasets[2].features[2])
    view = stats_util.DatasetListView(stats_proto).get_default_slice()
    with self.assertRaisesRegex(ValueError,
                                'Ambiguous result, 2 features matched'):
      view.get_derived_feature('my_deriver_name', [
          types.FeaturePath(['f0_step1', 'f0_step2']),
          types.FeaturePath(['f1'])
      ])

  def test_get_derived_feature_missing(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    self.assertIsNone(
        view.get_derived_feature('mismatched_name', [
            types.FeaturePath(['f0_step1', 'f0_step2']),
            types.FeaturePath(['f1'])
        ]))
    self.assertIsNone(
        view.get_derived_feature('my_deriver_name', [
            types.FeaturePath(['f0_step1', 'f0_step2', 'mismatched_step']),
            types.FeaturePath(['f1'])
        ]))
    self.assertIsNone(view.get_derived_feature('my_deriver_name', []))

  def test_get_missing_feature(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    self.assertIsNone(view.get_feature(types.FeaturePath(['not', 'a', 'path'])))

  def test_list_features(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    self.assertCountEqual(view.list_features(), [
        types.FeaturePath(['f0_step1', 'f0_step2']),
        types.FeaturePath(['f1']),
        types.FeaturePath(['f3_derived'])
    ])

  def test_get_cross_feature(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    cross_feature = view.get_cross_feature(
        types.FeaturePath(['f1x']), types.FeaturePath(['f1y']))
    self.assertEqual(self._stats_proto.datasets[2].cross_features[0],
                     cross_feature.proto())

  def test_list_cross_features(self):
    view = stats_util.DatasetListView(self._stats_proto).get_default_slice()
    self.assertCountEqual(
        view.list_cross_features(),
        [(types.FeaturePath(['f1x']), types.FeaturePath(['f1y'])),
         (types.FeaturePath(['f2x']), types.FeaturePath(['f2y']))])

  def test_get_feature_defined_by_name(self):
    stats = statistics_pb2.DatasetFeatureStatisticsList()
    text_format.Parse(
        """
    datasets: {
      name: 'All Examples'
      features: {
        name: "f0"
      }
      features: {
        name: "f1"
      }
    }
    """, stats)
    view = stats_util.DatasetListView(stats).get_default_slice()
    self.assertEqual(stats.datasets[0].features[1],
                     view.get_feature(types.FeaturePath(['f1'])).proto())

  def test_mixed_path_and_name_is_an_error(self):
    stats = statistics_pb2.DatasetFeatureStatisticsList()
    text_format.Parse(
        """
    datasets: {
      name: 'All Examples'
      features: {
        path: {
          step: "f0_step1"
          step: "f0_step2"
        }
      }
      features: {
        name: "f1"
      }
    }
    """, stats)
    view = stats_util.DatasetListView(stats).get_default_slice()
    with self.assertRaisesRegex(ValueError,
                                ('Features must be specified with '
                                 'either path or name within a Dataset')):
      view.get_feature(types.FeaturePath('f1'))


class LoadShardedStatisticsTest(absltest.TestCase):

  def test_load_sharded_paths(self):
    full_stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
    text_format.Parse(_STATS_PROTO, full_stats_proto)
    tmp_dir = self.create_tempdir()
    tmp_path = os.path.join(tmp_dir, 'statistics-0-of-1')
    writer = tf.compat.v1.io.TFRecordWriter(tmp_path)
    for dataset in full_stats_proto.datasets:
      shard = statistics_pb2.DatasetFeatureStatisticsList()
      shard.datasets.append(dataset)
      writer.write(shard.SerializeToString())
    writer.close()
    view = stats_util.load_sharded_statistics(
        input_paths=[tmp_path],
        io_provider=artifacts_io_impl.get_io_provider('tfrecords'))
    compare.assertProtoEqual(self, view.proto(), full_stats_proto)

  def test_load_sharded_pattern(self):
    full_stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
    text_format.Parse(_STATS_PROTO, full_stats_proto)
    tmp_dir = self.create_tempdir()
    tmp_path = os.path.join(tmp_dir, 'statistics-0-of-1')
    writer = tf.compat.v1.io.TFRecordWriter(tmp_path)
    for dataset in full_stats_proto.datasets:
      shard = statistics_pb2.DatasetFeatureStatisticsList()
      shard.datasets.append(dataset)
      writer.write(shard.SerializeToString())
    writer.close()
    view = stats_util.load_sharded_statistics(
        input_path_prefix=tmp_path.rstrip('-0-of-1'),
        io_provider=artifacts_io_impl.get_io_provider('tfrecords'))
    compare.assertProtoEqual(self, view.proto(), full_stats_proto)


class FeatureViewTest(absltest.TestCase):

  def test_num_stats(self):
    feature = statistics_pb2.FeatureNameStatistics()
    text_format.Parse(
        """
      num_stats: {
        common_stats: {
          num_non_missing: 1
        }
      }
    """, feature)
    view = stats_util.FeatureView(feature)
    self.assertEqual(view.numeric_statistics(), feature.num_stats)
    self.assertIsNone(view.string_statistics())
    self.assertIsNone(view.bytes_statistics())
    self.assertIsNone(view.struct_statistics())

    self.assertEqual(view.common_statistics(), feature.num_stats.common_stats)

  def test_string_stats(self):
    feature = statistics_pb2.FeatureNameStatistics()
    text_format.Parse(
        """
      string_stats: {
        common_stats: {
          num_non_missing: 1
        }
      }
    """, feature)
    view = stats_util.FeatureView(feature)
    self.assertIsNone(view.numeric_statistics())
    self.assertEqual(view.string_statistics(), feature.string_stats)
    self.assertIsNone(view.bytes_statistics())
    self.assertIsNone(view.struct_statistics())

    self.assertEqual(view.common_statistics(),
                     feature.string_stats.common_stats)

  def test_bytes_stats(self):
    feature = statistics_pb2.FeatureNameStatistics()
    text_format.Parse(
        """
      bytes_stats: {
        common_stats: {
          num_non_missing: 1
        }
      }
    """, feature)
    view = stats_util.FeatureView(feature)
    self.assertIsNone(view.numeric_statistics())
    self.assertIsNone(view.string_statistics())
    self.assertEqual(view.bytes_statistics(), feature.bytes_stats)
    self.assertIsNone(view.struct_statistics())

    self.assertEqual(view.common_statistics(), feature.bytes_stats.common_stats)

  def test_struct_stats(self):
    feature = statistics_pb2.FeatureNameStatistics()
    text_format.Parse(
        """
      struct_stats: {
        common_stats: {
          num_non_missing: 1
        }
      }
    """, feature)
    view = stats_util.FeatureView(feature)
    self.assertIsNone(view.numeric_statistics())
    self.assertIsNone(view.string_statistics())
    self.assertIsNone(view.bytes_statistics())
    self.assertEqual(view.struct_statistics(), feature.struct_stats)

    self.assertEqual(view.common_statistics(),
                     feature.struct_stats.common_stats)

  def test_custom_stats(self):
    feature = statistics_pb2.FeatureNameStatistics()
    text_format.Parse(
        """
      custom_stats: {
        name: "stat1",
        str: "val1"
      }
      custom_stats: {
        name: "stat2",
        str: "val2"
      }
    """, feature)
    view = stats_util.FeatureView(feature)
    self.assertEqual(view.custom_statistic('stat1').str, 'val1')
    self.assertEqual(view.custom_statistic('stat2').str, 'val2')
    self.assertIsNone(view.custom_statistic('stat3'))

if __name__ == '__main__':
  absltest.main()
