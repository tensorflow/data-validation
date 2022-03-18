# Copyright 2021 Google LLC
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
"""Tests for mutual_information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import mutual_information
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.utils import test_util
from tfx_bsl.arrow import table_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

TEST_SEED = 10
TEST_MAX_ENCODING_LENGTH = 3
EMPTY_SET = set()


class EncodeExamplesTest(absltest.TestCase):
  """Tests for _encode_examples."""

  def assert_encoder_output_equal(self,
                                  batch,
                                  expected,
                                  multivalent_features,
                                  categorical_features,
                                  excluded_features=None):
    self.assertEqual(
        mutual_information._encode_examples(batch, multivalent_features,
                                            categorical_features,
                                            excluded_features or [],
                                            TEST_MAX_ENCODING_LENGTH), expected)

  def test_encoder_two_features(self):
    batch = pa.RecordBatch.from_arrays([
        pa.array([["a", "b", "a", "a"], None, ["b"]]),
        pa.array([[1], [2], None])
    ], ["fa", "fb"])
    expected = {
        types.FeaturePath(["fa"]): [[3, 1], [None, None], [0, 1]],
        types.FeaturePath(["fb"]): [[1], [2], [None]]
    }
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     set([types.FeaturePath(["fa"])]))

  def test_encoder_feature_excluded(self):
    batch = pa.RecordBatch.from_arrays([
        pa.array([["a", "b", "a", "a"], None, ["b"]]),
        pa.array([[1], [2], None])
    ], ["fa", "fb"])
    expected = {
        types.FeaturePath(["fa"]): [[3, 1], [None, None], [0, 1]],
    }
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     set([types.FeaturePath(["fa"])]),
                                     set([types.FeaturePath(["fb"])]))

  def test_encoder_multivalent_numerical_with_nulls(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1.0, 1.0, np.NaN], None, [2.0, 2.0, 1.0], []])], ["fa"])
    expected = {
        types.FeaturePath(["fa"]): [[2, 0, 0], [None, None, None], [1, 0, 2],
                                    [None, None, None]]
    }
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     EMPTY_SET)

  def test_encoder_univalent_with_nulls(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([None, [2.0], [], [None], [np.NaN]])], ["fa"])
    expected = {
        types.FeaturePath(["fa"]): [[None], [2], [None], [None], [None]]
    }
    self.assert_encoder_output_equal(batch, expected, EMPTY_SET, EMPTY_SET)

  def test_encoder_univalent(self):
    batch = pa.RecordBatch.from_arrays([pa.array([None, [1], [2], [3], [4]])],
                                       ["fa"])
    expected = {types.FeaturePath(["fa"]): [[None], [1], [2], [3], [4]]}
    self.assert_encoder_output_equal(batch, expected, EMPTY_SET, EMPTY_SET)

  def test_encoder_multivalent_categorical(self):
    batch = pa.RecordBatch.from_arrays([
        pa.array(
            [None, ["4", "3", "2", "1"], ["4", "3", "2"], ["4", "3"], ["4"]])
    ], ["fa"])
    expected = {
        types.FeaturePath(["fa"]): [[None, None, None], [1, 1, 2], [1, 1, 1],
                                    [1, 1, 0], [1, 0, 0]]
    }
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     set([types.FeaturePath(["fa"])]))

  def test_encoder_multivalent_categorical_missing(self):
    batch = pa.RecordBatch.from_arrays([pa.array([None, None])], ["fa"])
    expected = {types.FeaturePath(["fa"]): []}
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     set([types.FeaturePath(["fa"])]))

  def test_encoder_multivalent_numeric(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([None, [0, 5, 9], [9], [3, 5], [2, 8, 8, 8]])], ["fa"])
    expected = {
        types.FeaturePath(["fa"]): [[None, None, None], [1, 1, 1], [0, 0, 1],
                                    [1, 1, 0], [1, 3, 0]]
    }
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     EMPTY_SET)

  def test_encoder_multivalent_categorical_all_empty(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7]])
    empty_feat_array = pa.array([[], [], [], []])
    batch = pa.RecordBatch.from_arrays([label_array, empty_feat_array],
                                       ["label_key", "empty_feature"])
    expected = {
        types.FeaturePath(["empty_feature"]): [[None, None, None],
                                               [None, None, None],
                                               [None, None, None],
                                               [None, None, None]],
        types.FeaturePath(["label_key"]): [[0.1], [0.2], [0.7], [0.7]]
    }
    self.assert_encoder_output_equal(
        batch, expected, set([types.FeaturePath(["empty_feature"])]),
        set([types.FeaturePath(["empty_feature"])]))

  def test_encoder_multivalent_numerical_all_empty(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7]])
    empty_feat_array = pa.array([[], [], [], []])
    batch = pa.RecordBatch.from_arrays([label_array, empty_feat_array],
                                       ["label_key", "empty_feature"])
    expected = {
        types.FeaturePath(["empty_feature"]): [[None, None, None],
                                               [None, None, None],
                                               [None, None, None],
                                               [None, None, None]],
        types.FeaturePath(["label_key"]): [[0.1], [0.2], [0.7], [0.7]]
    }
    self.assert_encoder_output_equal(
        batch, expected, set([types.FeaturePath(["empty_feature"])]), EMPTY_SET)

  def test_encoder_multivalent_numeric_missing(self):
    batch = pa.RecordBatch.from_arrays([pa.array([None, None])], ["fa"])
    expected = {types.FeaturePath(["fa"]): []}
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     EMPTY_SET)

  def test_encoder_multivalent_numeric_too_large(self):
    batch = pa.RecordBatch.from_arrays([pa.array([2**53 + 1])], ["fa"])
    expected = {}
    self.assert_encoder_output_equal(batch, expected,
                                     set([types.FeaturePath(["fa"])]),
                                     EMPTY_SET)


class MutualInformationTest(absltest.TestCase):
  """Tests that MutualInformation returns the correct AMI value."""

  def _assert_ami_output_equal(self,
                               batch,
                               expected,
                               schema,
                               label_feature,
                               normalize_by_max=False,
                               allow_invalid_partitions=False):
    """Checks that AMI computation is correct."""
    actual = mutual_information.MutualInformation(
        label_feature,
        schema,
        TEST_SEED,
        TEST_MAX_ENCODING_LENGTH,
        normalize_by_max=normalize_by_max,
        allow_invalid_partitions=allow_invalid_partitions).compute(batch)
    test_util.assert_dataset_feature_stats_proto_equal(self, actual, expected)

  def test_mi_with_univalent_features(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.2], None, [0.9], [0.4],
                            [0.8]])
    # Random floats that do not map onto the label
    terrible_feat_array = pa.array([[0.4], [0.1], [0.4], [np.NaN], [0.8], [0.2],
                                    [0.5], [0.1]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, label_array, terrible_feat_array],
        ["label_key", "perfect_feature", "terrible_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "perfect_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "terrible_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "perfect_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 1.0957612
          }
        }
        features {
          path {
            step: "terrible_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_batch_smaller_than_k(self):
    label_array = pa.array([[0.1], [0.2]])
    feat_array_1 = pa.array([[0.4], [0.1]])
    feat_array_2 = pa.array([[0.2], [0.4]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, feat_array_1, feat_array_2],
        ["label_key", "feat_array_1", "feat_array_2"])

    schema = text_format.Parse(
        """
        feature {
          name: "feat_array_1"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "feat_array_2"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    # Data is invalid (partition size of 2, but k of 3) but since
    # allow_invalid_partitions is True, the output for this partition will
    # simply be empty, rather than raising an exception.
    expected = statistics_pb2.DatasetFeatureStatistics()
    self._assert_ami_output_equal(
        batch,
        expected,
        schema,
        types.FeaturePath(["label_key"]),
        allow_invalid_partitions=True)

  def test_mi_normalized(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.2], None, [0.9], [0.4],
                            [0.8]])
    terrible_feat_array = pa.array([[0.4], [0.1], [0.4], [np.NaN], [0.8], [0.2],
                                    [0.5], [0.1]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, label_array, terrible_feat_array],
        ["label_key", "perfect_feature", "terrible_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "perfect_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "terrible_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "perfect_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 1.0
          }
        }
        features {
          path {
            step: "terrible_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(
        batch,
        expected,
        schema,
        types.FeaturePath(["label_key"]),
        normalize_by_max=True)

  def test_mi_with_univalent_feature_empty(self):
    label_array = pa.array([], type=pa.float32())
    null_feat_array = pa.array([], type=pa.float32())
    batch = pa.RecordBatch.from_arrays([label_array, null_feat_array],
                                       ["label_key", "null_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "null_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "null_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_unicode_labels(self):
    label_array = pa.array([["•"], ["•"], [b"\xc5\x8cmura"]])
    null_feat_array = pa.array([[3.1], [2.1], [1.1]])
    batch = pa.RecordBatch.from_arrays([label_array, null_feat_array],
                                       ["label_key", "null_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "null_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "null_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_univalent_feature_all_null(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7]])
    null_feat_array = pa.array([[np.NaN], [np.NaN], [np.NaN], [np.NaN]])
    batch = pa.RecordBatch.from_arrays([label_array, null_feat_array],
                                       ["label_key", "null_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "null_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "null_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_feature_all_null(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7]])
    null_feat_array = pa.array([[np.NaN], [np.NaN], [np.NaN], [np.NaN]])
    batch = pa.RecordBatch.from_arrays([label_array, null_feat_array],
                                       ["label_key", "null_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "null_feature"
          type: FLOAT
          value_count {
              min: 0
              max: 3
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "null_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_feature_all_empty(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7]])
    empty_feat_array = pa.array([[np.NaN], [], [], []])
    batch = pa.RecordBatch.from_arrays([label_array, empty_feat_array],
                                       ["label_key", "empty_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "empty_feature"
          type: FLOAT
          value_count {
              min: 0
              max: 3
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "empty_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_feature_univalent_label(self):
    label_array = pa.array([[0.1], [0.2], [0.7], [0.7], [0.2], [0.7], [0.7]])
    feat_array = pa.array([[3.1], None, [4.0], [None], [1.2, 8.5], [2.3],
                           [1.2, 3.2, 3.9]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "feature"
          type: FLOAT
          value_count {
              min: 0
              max: 3
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_numeric_feature(self):
    feat_array = pa.array([[3.1], None, [4.0], [np.NaN], [1.2, 8.5], [2.3],
                           [1.2, 3.2, 3.9]])
    label_array = pa.array([[3.3], None, [4.0], [2.0, 8.0], [1.3, 8.5], [2.3],
                            [1.0, 3.1, 4]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            type: FLOAT
            value_count {
              min: 0
              max: 3
            }
          }
          feature {
            name: "label_key"
            type: FLOAT
            value_count {
              min: 0
              max: 3
            }
          }
          """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "fa"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_categorical_feature(self):
    feat_array = pa.array([
        None, ["A", "C", "C"], ["B", "B"], ["C", "A", "A", "A"],
        ["A", "A", "A", "B", "B"], ["D"], ["C", "C", "C", "C", "C"]
    ])
    label_array = pa.array([None, ["C"], ["B"], ["A"], ["B"], ["D"], ["C"]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            type: BYTES
            value_count {
              min: 1
              max: 5
            }
          }
          feature {
            name: "label_key"
            type: BYTES
            shape {
              dim {
                size: 1
              }
            }
          }
          """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "fa"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0.4808983
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_multivalent_categorical_label(self):
    np.random.seed(0)
    # Generate 100 examples of randomly variable length features with random
    # discrete values of "0", "1", "2"
    feat_array = pa.array(
        [[str(np.random.randint(3))
          for _ in range(np.random.randint(10))]
         for _ in range(100)])
    label_array = pa.array(
        [[str(np.random.randint(3))
          for _ in range(np.random.randint(10))]
         for _ in range(100)])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array, label_array],
                                       ["label_key", "fa", "perfect_feat"])

    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            type: BYTES
            value_count {
              min: 0
              max: 10
            }
          }
          feature {
            name: "perfect_feat"
            type: BYTES
            value_count {
              min: 0
              max: 10
            }
          }
          feature {
            name: "label_key"
            type: BYTES
            value_count {
              min: 0
              max: 10
            }
          }
          """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "fa"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0
          }
        }
        features {
          path {
            step: "perfect_feat"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 4.1630335
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_numerical_univalent_feature_large(self):
    n = 100
    # Set seed so this test is deterministic
    np.random.seed(0)

    # The features have the following labels:
    # Feature | Label
    # -----------------
    # Red     | [0, 1.0)
    # Blue    | [1.0, 2.0)
    # Green   | [2.0, 3.0)

    # Create labels where first n items are [0, 1.0),
    # next n items are [1.0, 2.0), and last n items are [2.0, 3.0).
    label = [np.random.rand() for i in range(n)] + [
        np.random.rand() + 1 for i in range(n)
    ] + [np.random.rand() + 2 for i in range(n)]

    # A categorical feature that maps directly on to the label.
    feat = ["Red"] * n + ["Blue"] * n + ["Green"] * n

    # Shuffle the two arrays together (i.e. the table above still holds, but the
    # order of labels are now mixed.)
    # For example:
    # [0.4, 0.1, 1.2, 2.4]            => [1.2, 0.1, 2.4, 0.4]
    # ["Red", "Red", "Blue", "Green"] => ["Blue", "Red", "Green", "Red"]
    zipped_arrays = list(zip(feat, label))
    np.random.shuffle(zipped_arrays)
    feat_array, label_array = zip(*zipped_arrays)

    batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in label_array]),
        pa.array([[x] for x in feat_array])
    ], ["label_key", "color_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: INT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "color_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "color_feature"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 1.5612983
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_categorical_univalent_feature_large(self):
    labels = ["Red"] * 50 + ["Blue"] * 50

    # Here is the exact mutual information for the almost perfect feature:
    # P(Red, Red) = P(Red|Red) * P(Red) = 49/50 * 1/2 = 49/100 = P(Blue, Blue)
    # P(Red, Blue) = P(Red|Blue) * P(Blue) = 1/50 * 1/2 = 1/100 = P(Blue, Red)
    # MI(X,Y) = 0.47571829 * 2 + -0.04643856 * 2 = 0.85855945
    # Since this generator calculates AMI = MI(X,Y) - Shuffle_MI(X,Y),
    # We should expect the results to be a bit less than 0.85855945
    near_perfect_feature = (["Red"] * 49 + ["Blue"] + ["Red"] + ["Blue"] * 49)

    # The feature is perfectly uncorrelated. The mutual information is:
    # P(Red, Red) = 0 = P(Blue, Blue)
    # P(Red, Blue) = 1 = P(Blue, Red)
    # MI(X,Y) = 0 + 0 + 1*log(1/4) * 2 = -4
    # AMI will thus be floored at 0.
    terrible_feature = (["Red"] * 25 + ["Blue"] * 25) * 2

    batch = pa.RecordBatch.from_arrays([
        pa.array([[x] for x in labels]),
        pa.array([[x] for x in near_perfect_feature]),
        pa.array([[x] for x in terrible_feature])
    ], ["label_key", "near_perfect_feature", "terrible_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "near_perfect_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "terrible_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "terrible_feature"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0
          }
        }
        features {
          path {
            step: "near_perfect_feature"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0.8400134
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_missing_label_key(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([[1]])], ["label", "fa"])
    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            type: FLOAT
            value_count{
              min: 1
              max: 1
            }
          }
          feature {
            name: "label"
            type: FLOAT
              value_count{
              min: 1
              max: 1
            }
          }
          """, schema_pb2.Schema())

    with self.assertRaisesRegex(ValueError,
                                "Feature label_key not found in the schema."):
      mutual_information.MutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED,
          TEST_MAX_ENCODING_LENGTH).compute(batch)

  def test_mi_with_unique_label(self):
    label_array = pa.array([["a"], ["b"], ["c"]], type=pa.list_(pa.binary()))
    multivalent_feat_array = pa.array([["a", "b"], ["b"], ["b"]],
                                      type=pa.list_(pa.binary()))
    univalent_feat_array = pa.array([["a"], ["a"], ["a"]],
                                    type=pa.list_(pa.binary()))
    batch = pa.RecordBatch.from_arrays(
        [label_array, univalent_feat_array, multivalent_feat_array],
        ["label_key", "univalent_feature", "multivalent_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "univalent_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "multivalent_feature"
          type: BYTES
          shape {
            dim {
              size: 2
            }
          }
        }
        feature {
          name: "label_key"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "univalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }
        features {
          path {
            step: "multivalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_unique_feature(self):
    univalent_feat_array = pa.array([["a"], ["b"], ["c"]],
                                    type=pa.list_(pa.binary()))
    multivalent_feat_array = pa.array([["a", "b"], ["b"], ["b"]],
                                      type=pa.list_(pa.binary()))
    label_array = pa.array([["a"], ["b"], ["b"]], type=pa.list_(pa.binary()))
    batch = pa.RecordBatch.from_arrays(
        [label_array, univalent_feat_array, multivalent_feat_array],
        ["label_key", "univalent_feature", "multivalent_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "univalent_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "multivalent_feature"
          type: BYTES
          shape {
            dim {
              size: 2
            }
          }
        }
        feature {
          name: "label_key"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "univalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }
        features {
          path {
            step: "multivalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_unique_categorical_feature_with_regression(self):
    label_array = pa.array([[1.0], [1.5], [2.0], [2.5]])
    multivalent_feat_array = pa.array([["a", "b"], ["c"], ["d"], ["e"]],
                                      type=pa.list_(pa.binary()))
    univalent_feat_array = pa.array([["a"], ["b"], ["c"], ["d"]],
                                    type=pa.list_(pa.binary()))
    batch = pa.RecordBatch.from_arrays(
        [label_array, univalent_feat_array, multivalent_feat_array],
        ["label_key", "univalent_feature", "multivalent_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "univalent_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "multivalent_feature"
          type: BYTES
          shape {
            dim {
              size: 2
            }
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "univalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }
        features {
          path {
            step: "multivalent_feature"
          }
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_missing_multivalent_numeric_feature(self):
    missing_feat_array = pa.array([None, None])
    label_array = pa.array([["a"], ["a"]])
    batch = pa.RecordBatch.from_arrays([label_array, missing_feat_array],
                                       ["label_key", "missing_feature"])
    schema = text_format.Parse(
        """
          feature {
            name: "missing_feature"
            type: FLOAT
            value_count {
              min: 0
              max: 3
            }
          }
          feature {
            name: "label_key"
            type: BYTES
            value_count {
              min: 0
              max: 3
            }
          }
          """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "missing_feature"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_missing_multivalent_categorical_feature(self):
    missing_feat_array = pa.array([None, None])
    label_array = pa.array([["a"], ["a"]])
    batch = pa.RecordBatch.from_arrays([label_array, missing_feat_array],
                                       ["label_key", "missing_feature"])
    schema = text_format.Parse(
        """
          feature {
            name: "missing_feature"
            type: BYTES
            value_count {
              min: 0
              max: 3
            }
          }
          feature {
            name: "label_key"
            type: BYTES
            value_count {
              min: 0
              max: 3
            }
          }
          """, schema_pb2.Schema())

    expected = text_format.Parse(
        """
        features {
          path {
            step: "missing_feature"
          }
          custom_stats {
            name: 'adjusted_mutual_information'
            num: 0.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))

  def test_mi_with_no_schema_or_paths(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([[1]])], ["label_key", "fa"])

    with self.assertRaisesRegex(
        ValueError,
        "Either multivalent feature set or schema must be provided"):
      mutual_information.MutualInformation(
          types.FeaturePath(["label_key"]), None, TEST_SEED,
          TEST_MAX_ENCODING_LENGTH).compute(batch)

  def test_mi_multivalent_too_large_int_value(self):
    label_array = pa.array([[0.1], [0.1], [0.1], [0.1], [0.1]])
    x = 2**53 + 1
    invalid_feat_array = pa.array([[x], [x], [x], [x], []])
    valid_feat_array = pa.array([[1], [1], [1], [1], []])

    batch = pa.RecordBatch.from_arrays(
        [label_array, invalid_feat_array, valid_feat_array],
        ["label_key", "invalid_feat_array", "valid_feat_array"])

    schema = text_format.Parse(
        """
        feature {
          name: "invalid_feat_array"
          type: INT
          value_count {
            min: 0
            max: 2
          }
        }
        feature {
          name: "valid_feat_array"
          type: INT
          value_count {
            min: 0
            max: 2
          }
        }
        feature {
          name: "label_key"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    # The value 2**53 + 1 is too large, and will cause np.histogram to fail.
    # We skip the feature if it cannot be encoded. We still encode the valid
    # features.
    expected = text_format.Parse(
        """
        features {
          custom_stats {
            name: "adjusted_mutual_information"
            num: 0.09617966939259784
          }
          path {
            step: "valid_feat_array"
          }
        }
    """, statistics_pb2.DatasetFeatureStatistics())
    self._assert_ami_output_equal(
        batch,
        expected,
        schema,
        types.FeaturePath(["label_key"]),
        allow_invalid_partitions=True)

  def test_mi_no_feature(self):
    # Tests if there is no feature provided.
    label_array = pa.array([["a"], ["a"]])
    batch = pa.RecordBatch.from_arrays([label_array], ["label_key"])
    schema = text_format.Parse(
        """
          feature {
            name: "label_key"
            type: BYTES
            value_count {
              min: 0
              max: 3
            }
          }
          """, schema_pb2.Schema())

    expected = statistics_pb2.DatasetFeatureStatistics()
    self._assert_ami_output_equal(batch, expected, schema,
                                  types.FeaturePath(["label_key"]))


def _get_test_stats_with_mi(feature_paths):
  """Get stats proto for MI test."""
  result = statistics_pb2.DatasetFeatureStatistics()
  for feature_path in feature_paths:
    feature_proto = text_format.Parse(
        """
            custom_stats {
              name: "max_adjusted_mutual_information"
              num: 0
            }
            custom_stats {
              name: "mean_adjusted_mutual_information"
              num: 0
            }
            custom_stats {
              name: "median_adjusted_mutual_information"
              num: 0
            }
            custom_stats {
              name: "min_adjusted_mutual_information"
              num: 0
            }
            custom_stats {
              name: "num_partitions_adjusted_mutual_information"
              num: 2.0
            }
            custom_stats {
              name: "std_dev_adjusted_mutual_information"
              num: 0
            }
        """, statistics_pb2.FeatureNameStatistics())
    feature_proto.path.CopyFrom(feature_path.to_proto())
    result.features.add().CopyFrom(feature_proto)
  return result


class NonStreamingCustomStatsGeneratorTest(
    test_util.TransformStatsGeneratorTest, parameterized.TestCase):
  """Tests for NonStreamingCustomStatsGenerator."""

  def setUp(self):
    # Integration tests involving Beam and AMI are challenging to write
    # because Beam PCollections are unordered while the results of adjusted MI
    # depend on the order of the data for small datasets. This test case tests
    # MI with one label which will give a value of 0 regardless of
    # the ordering of elements in the PCollection. The purpose of this test is
    # to ensure that the Mutual Information pipeline is able to handle a
    # variety of input types. Unit tests ensuring correctness of the MI value
    # itself are included in mutual_information_test.

    # fa is categorical, fb is numeric, fc is multivalent categorical, fd is
    # multivalent numeric

    self.record_batches = [
        pa.RecordBatch.from_arrays([
            pa.array([["1"]]),
            pa.array([[1.1]]),
            pa.array([["1", "1", "1"]]),
            pa.array([[1.0, 1.2, 0.8]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["0"]]),
            pa.array([[0.3]]),
            pa.array([["0", "1"]]),
            pa.array([[0.1, 0.0]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["1"]]),
            pa.array([[np.NaN]], type=pa.list_(pa.float64())),
            pa.array([["0", "0"]]),
            pa.array([[0.0, 0.2]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([None]),
            pa.array([None]),
            pa.array([["1", "0", "0", "1"]]),
            pa.array([None]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["1"]]),
            pa.array([[1.0]]),
            pa.array([["1", "1"]]),
            pa.array([[1.0]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["0"]]),
            pa.array([[0.3]]),
            pa.array([["0"]]),
            pa.array([[0.0, 0.2]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([None]),
            pa.array([None]),
            pa.array([["1", "0", "0", "1"]]),
            pa.array([None]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["1"]]),
            pa.array([[1.0]]),
            pa.array([["1", "1"]]),
            pa.array([[1.0]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["0"]]),
            pa.array([[0.3]]),
            pa.array([["0"]]),
            pa.array([[0.0, 0.2]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([None]),
            pa.array([None]),
            pa.array([["1", "0", "0", "1"]]),
            pa.array([None]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["1"]]),
            pa.array([[1.0]]),
            pa.array([["1", "1"]]),
            pa.array([[1.0]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
        pa.RecordBatch.from_arrays([
            pa.array([["0"]]),
            pa.array([[0.3]]),
            pa.array([["0"]]),
            pa.array([[0.0, 0.2]]),
            pa.array([["label"]]),
        ], ["fa", "fb", "fc", "fd", "label_key"]),
    ]

    self.schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: BYTES
          value_count {
            min: 0
            max: 1
          }
        }
        feature {
          name: "fb"
          type: FLOAT
          value_count {
            min: 0
            max: 1
          }
        }
        feature {
          name: "fc"
          type: BYTES
          value_count {
            min: 0
            max: 4
          }
        }
        feature {
          name: "fd"
          type: FLOAT
          value_count {
            min: 0
            max: 3
          }
        }
        feature {
          name: "label_key"
          type: BYTES
          value_count {
            min: 1
            max: 1
          }
        }""", schema_pb2.Schema())

  # The number of column partitions should not affect the result, even when
  # that number is much larger than the number of columns.
  @parameterized.parameters([1, 2, 99])
  def test_ranklab_mi(self, column_partitions):
    expected_result = [
        _get_test_stats_with_mi([
            types.FeaturePath(["fa"]),
            types.FeaturePath(["fb"]),
            types.FeaturePath(["fc"]),
            types.FeaturePath(["fd"]),
        ])
    ]

    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        mutual_information.MutualInformation(
            label_feature=types.FeaturePath(["label_key"]),
            schema=self.schema,
            max_encoding_length=TEST_MAX_ENCODING_LENGTH,
            seed=TEST_SEED,
            column_partitions=column_partitions),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        batch_size=1,
        name="NonStreaming Mutual Information")
    self.assertSlicingAwareTransformOutputEqual(
        self.record_batches,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_ranklab_mi_with_paths(self):
    expected_result = [
        _get_test_stats_with_mi([
            types.FeaturePath(["fa"]),
            types.FeaturePath(["fb"]),
            types.FeaturePath(["fc"]),
            types.FeaturePath(["fd"]),
        ])
    ]

    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        mutual_information.MutualInformation(
            label_feature=types.FeaturePath(["label_key"]),
            max_encoding_length=TEST_MAX_ENCODING_LENGTH,
            categorical_features={
                types.FeaturePath(["fa"]),
                types.FeaturePath(["fc"]),
                types.FeaturePath(["label_key"]),
            },
            multivalent_features={
                types.FeaturePath(["fc"]),
                types.FeaturePath(["fd"]),
            },
            seed=TEST_SEED),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        batch_size=1,
        name="NonStreaming Mutual Information")
    self.assertSlicingAwareTransformOutputEqual(
        self.record_batches,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_ranklab_mi_with_slicing(self):
    sliced_record_batches = []
    for slice_key in ["slice1", "slice2"]:
      for record_batch in self.record_batches:
        sliced_record_batches.append((slice_key, record_batch))

    expected_result = [("slice1",
                        _get_test_stats_with_mi([
                            types.FeaturePath(["fa"]),
                            types.FeaturePath(["fb"]),
                            types.FeaturePath(["fc"]),
                            types.FeaturePath(["fd"]),
                        ])),
                       ("slice2",
                        _get_test_stats_with_mi([
                            types.FeaturePath(["fa"]),
                            types.FeaturePath(["fb"]),
                            types.FeaturePath(["fc"]),
                            types.FeaturePath(["fd"]),
                        ]))]
    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        mutual_information.MutualInformation(
            label_feature=types.FeaturePath(["label_key"]),
            schema=self.schema,
            max_encoding_length=TEST_MAX_ENCODING_LENGTH,
            seed=TEST_SEED),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        batch_size=1,
        name="NonStreaming Mutual Information")
    self.assertSlicingAwareTransformOutputEqual(sliced_record_batches,
                                                generator, expected_result)

  def test_row_and_column_partitions_reassemble(self):
    # We'd like to test the row/column partitioning behavior in a non-trivial
    # condition for column partitioning. This test skips the actual MI
    # calculation, and just verifies that RecordBatches passed to it are as we
    # expect.

    # Column names chosen so that
    # yams:     partition 0
    # arugula:  partition 0
    # apple:    partition 1
    #
    # Note that partition indices should be deterministic.
    batch1 = pa.RecordBatch.from_arrays([
        pa.array([1]),
        pa.array([2]),
        pa.array(["a"]),
    ], ["yams", "arugula", "label_key"])
    batch2 = pa.RecordBatch.from_arrays([
        pa.array([3]),
        pa.array(["b"]),
    ], ["yams", "label_key"])
    batch3 = pa.RecordBatch.from_arrays([
        pa.array([4]),
        pa.array(["c"]),
    ], ["apple", "label_key"])

    merged = table_util.MergeRecordBatches([batch1, batch2, batch3]).to_pandas()

    mi = mutual_information.MutualInformation(
        label_feature=types.FeaturePath(["label_key"]),
        schema=self.schema,
        max_encoding_length=TEST_MAX_ENCODING_LENGTH,
        column_partitions=3,
        seed=TEST_SEED)

    def _make_equal_dataframe_items(expected):
      """Compare lists of dataframes without considering order or count."""

      def _assert_fn(dataframes):
        got_expected = [False] * len(expected)
        got_actual = [False] * len(dataframes)
        for i, dfi in enumerate(expected):
          for j, dfj in enumerate(dataframes):
            # Sort by the label to account for non-deterministic PCollection
            # order, and reorder columns for consistency.
            dfi = dfi.sort_values("label_key")
            dfi = dfi[list(sorted(dfi.columns))].reset_index(drop=True)

            dfj = dfj.sort_values("label_key")
            dfj = dfj[list(sorted(dfj.columns))].reset_index(drop=True)
            if dfi.equals(dfj):
              got_expected[i] = True
              got_actual[j] = True
        self.assertTrue(
            min(got_expected),
            msg="some expected outputs missing\ngot: %s\nexpected: %s" %
            (dataframes, expected))
        self.assertTrue(
            min(got_actual),
            msg="some actual outputs not expected\ngot: %s\nexpected: %s" %
            (dataframes, expected))

      return _assert_fn

    with beam.Pipeline() as p:
      result = (
          p | beam.Create([("", batch1), ("", batch2), ("", batch3)])
          | mi.partitioner(1)
          | beam.CombinePerKey(
              partitioned_stats_generator._SampleRecordBatchRows(999))
          | beam.Map(lambda x: x[1].to_pandas()))
      # Note that the batches passed to MI compute are column-wise slices of
      # the merged RecordBatch.
      beam_test_util.assert_that(
          result,
          _make_equal_dataframe_items([
              merged[["yams", "arugula", "label_key"]],
              merged[["apple", "label_key"]],
              merged[["label_key"]],
          ]))


if __name__ == "__main__":
  absltest.main()
