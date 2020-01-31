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
    batch = pa.Table.from_arrays([
        label_array, label_array, terrible_feat_array
    ], ["label_key", "perfect_feature", "terrible_feature"])

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
    batch = pa.Table.from_arrays(
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
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_regression_with_int_label_and_categorical_feature(self):
    label_array = pa.array([
        [0], [2], [0], [1], [2], [1], [1], [0], [2], [1], [0]])
    # A categorical feature that maps directly on to the label.
    perfect_feat_array = pa.array([
        ["Red"], ["Blue"], ["Red"], ["Green"], ["Blue"], ["Green"], ["Green"],
        ["Red"], ["Blue"], ["Green"], ["Red"]])
    batch = pa.Table.from_arrays([label_array, perfect_feat_array],
                                 ["label_key", "perfect_feature"])

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
            num: 1.7319986
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.7319986
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
    batch = pa.Table.from_arrays([label_array, perfect_feat_array],
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
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_imputed_categorical_feature(self):
    label_array = pa.array([[0], [2], [0], [1], [2], [1], [1]])
    # A categorical feature with missing values.
    feat_array = pa.array([
        ["Red"], ["Blue"], None, None, ["Blue"], ["Green"], ["Green"]])
    batch = pa.Table.from_arrays([label_array, feat_array], ["label_key", "fa"])

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
            num: 0.4361111
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 0.4361111
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
    batch = pa.Table.from_arrays([label_array, feat_array], ["label_key", "fa"])

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
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema,
                                 types.FeaturePath(["label_key"]))

  def test_mi_with_invalid_features(self):
    batch = pa.Table.from_arrays([pa.array([[1]]), pa.array([[1, 2]])],
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
    with self.assertRaisesRegexp(ValueError, "Found array with 0 sample"):
      sklearn_mutual_information.SkLearnMutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED).compute(batch)

  def test_mi_with_missing_label_key(self):
    batch = pa.Table.from_arrays([pa.array([[1]]), pa.array([[1]])],
                                 ["label", "fa"])

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

    with self.assertRaisesRegexp(ValueError,
                                 "Feature label_key not found in the schema."):
      sklearn_mutual_information.SkLearnMutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED).compute(batch)

  def test_mi_with_multivalent_label(self):
    batch = pa.Table.from_arrays([pa.array([[1, 2]]), pa.array([[1]])],
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
            value_count: {
              min: 1
              max: 2
            }
          }
          """, schema_pb2.Schema())

    with self.assertRaisesRegexp(ValueError,
                                 "Label column contains unsupported data."):
      sklearn_mutual_information.SkLearnMutualInformation(
          types.FeaturePath(["label_key"]), schema, TEST_SEED).compute(batch)


if __name__ == "__main__":
  absltest.main()
