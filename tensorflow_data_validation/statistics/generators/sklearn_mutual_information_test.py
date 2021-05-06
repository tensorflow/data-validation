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
"""Tests for sklearn_mutual_information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import sklearn_mutual_information

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

TEST_SEED = 10


class SkLearnMutualInformationTest(absltest.TestCase):
  """Tests for SkLearnMutualInformationStatsFn."""

  def _assert_mi_output_equal(self, batch, expected, schema, label_feature):
    """Checks that MI computation is correct."""
    actual = sklearn_mutual_information.SkLearnMutualInformation(
        label_feature, schema, TEST_SEED).compute(batch)
    compare.assertProtoEqual(self, actual, expected, normalize_numbers=True)

  def test_mi_regression_with_float_label_and_numeric_features(self):
    label_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.3], [0.9],
        [0.4], [0.1], [0.0], [0.4], [0.6], [0.4], [0.8]])
    # Random floats that do not map onto the label
    terrible_feat_array = pa.array([
        [0.4], [0.1], [0.4], [0.4], [0.8], [0.7], [0.2],
        [0.1], [0.0], [0.4], [0.8], [0.2], [0.5], [0.1]])
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
            name: "sklearn_adjusted_mutual_information"
            num: 1.0096965
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 1.1622766
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.9496162
          }
        }
        features {
          path {
            step: "terrible_feature"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.0211485
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.0211485
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.0161305
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_regression_with_null_array(self):
    label_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.3], [0.9],
        [0.4], [0.1], [0.0], [0.4], [0.6], [0.4], [0.8]])
    # Random floats that do not map onto the label
    terrible_feat_array = pa.array([
        [0.4], [0.1], [0.4], [0.4], [0.8], [0.7], [0.2],
        [0.1], [0.0], [0.4], [0.8], [0.2], [0.5], [0.1]])
    null_array = pa.array([None] * 14, type=pa.null())
    # Note: It is possible to get different results for py2 and py3, depending
    # on the feature name used (e.g., if use 'empty_feature', the results
    # differ). This might be due to the scikit learn function used to compute MI
    # adding a small amount of noise to continuous features before computing MI.
    batch = pa.RecordBatch.from_arrays(
        [label_array, label_array, terrible_feat_array, null_array], [
            "label_key", "perfect_feature", "terrible_feature",
            "values_empty_feature"
        ])

    schema = text_format.Parse(
        """
        feature {
          name: "values_empty_feature"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
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
            name: "sklearn_adjusted_mutual_information"
            num: 1.0742656
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 1.2277528
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 1.0
          }
        }
        features {
          path {
            step: "terrible_feature"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.0392891
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.0392891
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.0299668
          }
        }
        features {
          path {
            step: "values_empty_feature"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.0
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.0
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_regression_with_int_label_and_categorical_feature(self):
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
            name: 'sklearn_adjusted_mutual_information'
            num: 1.0798653
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.0983102
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.2438967
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_classif_with_int_label_and_categorical_feature(self):
    label_array = pa.array([
        [0], [2], [0], [1], [2], [1], [1], [0], [2], [1], [0]])
    # A categorical feature that maps directly on to the label.
    perfect_feat_array = pa.array([
        ["Red"], ["Blue"], ["Red"], ["Green"], ["Blue"], ["Green"], ["Green"],
        ["Red"], ["Blue"], ["Green"], ["Red"]])
    batch = pa.RecordBatch.from_arrays([label_array, perfect_feat_array],
                                       ["label_key", "perfect_feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: INT
          int_domain {
            is_categorical: true
          }
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "perfect_feature"
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
            step: "perfect_feature"
          }
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 0.9297553
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.0900597
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 1.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_classif_with_categorical_all_unique_labels(self):
    label_array = pa.array([[0], [2], [0], [1], [2], [1], [1], [0], [2], [1],
                            [0]])
    # A categorical feature that maps directly on to the label.
    perfect_feat_array = pa.array([["Red"], ["Blue"], ["Red"], ["Green"],
                                   ["Blue"], ["Green"], ["Green"], ["Red"],
                                   ["Blue"], ["Green"], ["Red"]])
    # A categorical feature that has all values unique.
    unique_feat_array = pa.array([["Red1"], ["Red2"], ["Red3"], ["Red4"],
                                  ["Red5"], ["Red6"], ["Red7"], ["Red8"],
                                  ["Red9"], ["Red10"], ["Red11"]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, perfect_feat_array, unique_feat_array],
        ["label_key", "perfect_feature", "unique_feat_array"])

    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: INT
          int_domain {
            is_categorical: true
          }
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "perfect_feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "unique_feat_array"
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
            step: "perfect_feature"
          }
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 0.9297553
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.0900597
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 1.0
          }
        }
        features {
          path {
            step: "unique_feat_array"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.0
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 1.0900597
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_classif_categorical_label_small_sample(self):
    label_array = pa.array([[0]])
    feat_array = pa.array([["Red"]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, feat_array],
        ["label_key", "feature"])
    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: INT
          int_domain {
            is_categorical: true
          }
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "feature"
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
            step: "feature"
          }
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 0
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 0
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_regression_numeric_label_small_sample(self):
    label_array = pa.array([[0], [0]])

    # Make sure the features are not all unique. Otherwise the column will be
    # dropped.
    feat_array = pa.array([["Red"], ["Red"]])
    batch = pa.RecordBatch.from_arrays(
        [label_array, feat_array],
        ["label_key", "feature"])

    schema = text_format.Parse(
        """
        feature {
          name: "label_key"
          type: INT
          int_domain {
            is_categorical: false
          }
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "feature"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        """, schema_pb2.Schema())

    # Since the label is numeric, no mutual information is calculated.
    expected = statistics_pb2.DatasetFeatureStatistics()
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_imputed_categorical_feature(self):
    label_array = pa.array([[0], [2], [0], [1], [2], [1], [1]])
    # A categorical feature with missing values.
    feat_array = pa.array([
        ["Red"], ["Blue"], None, None, ["Blue"], ["Green"], ["Green"]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

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
          int_domain {
            is_categorical: true
          }
        }
        feature {
          name: "fa"
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
            name: 'sklearn_adjusted_mutual_information'
            num: 0.3960841
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 0.8809502
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.4568877
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_imputed_numerical_feature(self):
    label_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3],
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3]])
    feat_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [np.NaN], None,
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

    schema = text_format.Parse(
        """
        feature {
          name: "fa"
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
            step: "fa"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.3849224
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.4063665
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.3268321
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_imputed_categorical_label(self):
    label_array = pa.array([["Red"], ["Blue"], ["Red"], None, None, ["Green"],
                            ["Green"]])
    # A categorical feature with missing values.
    feat_array = pa.array([
        ["Red"], ["Blue"], ["Red"], ["Green"], ["Blue"], ["Green"], ["Green"]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

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
          name: "fa"
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
            name: 'sklearn_adjusted_mutual_information'
            num: 0.1980421
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 0.8809502
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.2960819
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_imputed_numerical_label(self):
    label_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [np.NaN], None,
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3]])
    feat_array = pa.array([
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3],
        [0.1], [0.2], [0.8], [0.7], [0.2], [0.2], [0.3]])
    batch = pa.RecordBatch.from_arrays([label_array, feat_array],
                                       ["label_key", "fa"])

    schema = text_format.Parse(
        """
        feature {
          name: "fa"
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
            step: "fa"
          }
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.2640041
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.3825569
          }
          custom_stats {
            name: "sklearn_normalized_adjusted_mutual_information"
            num: 0.244306
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_invalid_features(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([[1, 2]])],
        ["label_key", "multivalent_feature"])
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
          name: "multivalent_feature"
          type: INT
          value_count: {
            min: 2
            max: 2
          }
        }
        """, schema_pb2.Schema())

    expected = text_format.Parse("""""",
                                 statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_missing_label_key(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([[1]])], ["label", "fa"])

    schema = text_format.Parse(
        """
          feature {
            name: "fa"
            type: FLOAT
              shape {
              dim {
                size: 1
              }
            }
          }
          feature {
            name: "label"
            type: FLOAT
            shape {
              dim {
                size: 1
              }
            }
          }
          """, schema_pb2.Schema())

    with self.assertRaisesRegex(ValueError,
                                "Feature label_key not found in the schema."):
      sklearn_mutual_information.SkLearnMutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED).compute(batch)

  def test_mi_with_multivalent_label(self):
    batch = pa.RecordBatch.from_arrays(
        [pa.array([[1, 2]]), pa.array([[1]])], ["label_key", "fa"])
    schema = text_format.Parse(
        """
          feature {
            name: "fa"
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
            value_count: {
              min: 1
              max: 2
            }
          }
          """, schema_pb2.Schema())

    with self.assertRaisesRegex(ValueError,
                                "Label column contains unsupported data."):
      sklearn_mutual_information.SkLearnMutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED).compute(batch)


if __name__ == "__main__":
  absltest.main()
