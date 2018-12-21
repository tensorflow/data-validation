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
from tensorflow_data_validation.statistics.generators import sklearn_mutual_information

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

TEST_SEED = 10


class SkLearnMutualInformationTest(absltest.TestCase):
  """Tests for SkLearnMutualInformationStatsFn."""

  def _assert_mi_output_equal(self, batch, expected, schema, label_key):
    """Checks that MI computation is correct."""
    actual = sklearn_mutual_information.SkLearnMutualInformation(
        label_key, schema, TEST_SEED).compute(batch)
    compare.assertProtoEqual(self, actual, expected, normalize_numbers=True)

  def test_mi_regression_with_float_label_and_numeric_features(self):
    batch = {}
    batch["label_key"] = np.array([
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([0.3]),
        np.array([0.9]),
        np.array([0.4]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([0.4]),
        np.array([0.6]),
        np.array([0.4]),
        np.array([0.8])
    ])
    # Maps directly onto the label key
    batch["perfect_feature"] = batch["label_key"]
    # Random floats that do not map onto the label
    batch["terrible_feature"] = np.array([
        np.array([0.4]),
        np.array([0.1]),
        np.array([0.4]),
        np.array([0.4]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([0.4]),
        np.array([0.8]),
        np.array([0.2]),
        np.array([0.5]),
        np.array([0.1])
    ])

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
          name: "perfect_feature"
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
          name: "terrible_feature"
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.0211485
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.0211485
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema, "label_key")

  def test_mi_regression_with_int_label_and_categorical_feature(self):
    batch = {}
    batch["label_key"] = np.array([
        np.array([0]),
        np.array([2]),
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([1]),
        np.array([1]),
        np.array([0]),
        np.array([2]),
        np.array([1]),
        np.array([0])
    ])
    # A categorical feature that maps directly on to the label.
    batch["perfect_feature"] = np.array([
        np.array(["Red"]),
        np.array(["Blue"]),
        np.array(["Red"]),
        np.array(["Green"]),
        np.array(["Blue"]),
        np.array(["Green"]),
        np.array(["Green"]),
        np.array(["Red"]),
        np.array(["Blue"]),
        np.array(["Green"]),
        np.array(["Red"])
    ])

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
          name: 'perfect_feature'
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 1.7319986
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.7319986
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema, "label_key")

  def test_mi_classif_with_int_label_and_categorical_feature(self):
    batch = {}
    batch["label_key"] = np.array([
        np.array([0]),
        np.array([2]),
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([1]),
        np.array([1]),
        np.array([0]),
        np.array([2]),
        np.array([1]),
        np.array([0])
    ])
    # A categorical feature that maps directly on to the label.
    batch["perfect_feature"] = np.array([
        np.array(["Red"]),
        np.array(["Blue"]),
        np.array(["Red"]),
        np.array(["Green"]),
        np.array(["Blue"]),
        np.array(["Green"]),
        np.array(["Green"]),
        np.array(["Red"]),
        np.array(["Blue"]),
        np.array(["Green"]),
        np.array(["Red"])
    ])

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
          name: 'perfect_feature'
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 0.9297553
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 1.0900597
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema, "label_key")

  def test_mi_with_imputed_categorical_feature(self):
    batch = {}
    batch["label_key"] = np.array([
        np.array([0]),
        np.array([2]),
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([1]),
        np.array([1])
    ])
    # A categorical feature with missing values.
    batch["fa"] = np.array([
        np.array(["Red"]),
        np.array(["Blue"]), None, None,
        np.array(["Blue"]),
        np.array(["Green"]),
        np.array(["Green"])
    ])

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
          name: 'fa'
          custom_stats {
            name: 'sklearn_adjusted_mutual_information'
            num: 0.4361111
          }
          custom_stats {
            name: 'sklearn_mutual_information'
            num: 0.4361111
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema, "label_key")

  def test_mi_with_imputed_numerical_feature(self):
    batch = {}
    batch["label_key"] = np.array([
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([0.2]),
        np.array([0.3]),
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([0.2]),
        np.array([0.3])
    ])
    batch["fa"] = np.array([
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([np.NaN]), None,
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.2]),
        np.array([0.2]),
        np.array([0.3])
    ])

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
          name: "fa"
          custom_stats {
            name: "sklearn_adjusted_mutual_information"
            num: 0.3849224
          }
          custom_stats {
            name: "sklearn_mutual_information"
            num: 0.4063665
          }
        }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_mi_output_equal(batch, expected, schema, "label_key")

  def test_mi_with_invalid_features(self):
    batch = {
        "label_key": np.array([np.array([1])]),
        "multivalent_feature": np.array([np.array([1, 2])])
    }

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
          "label_key", schema, TEST_SEED).compute(batch)

  def test_mi_with_missing_label_key(self):
    batch = {
        "fa": np.array([np.array([1.0]), np.array([2.0])]),
        "label": np.array([np.array([1.0]), np.array([2.0])])
    }

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
          "label_key", schema, TEST_SEED).compute(batch)

  def test_mi_with_multivalent_label(self):
    batch = {
        "fa": np.array([np.array([1.0]), np.array([2.0])]),
        "label_key": np.array([np.array([1.0, 2.0]),
                               np.array([2.0])])
    }

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
          "label_key", schema, TEST_SEED).compute(batch)


if __name__ == "__main__":
  absltest.main()
