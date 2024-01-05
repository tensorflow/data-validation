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

"""Tests for display_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized

from google.protobuf import text_format
import pandas as pd
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2
from tensorflow_data_validation.utils import display_util
from tensorflow_data_validation.utils import test_util

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class DisplayUtilTest(parameterized.TestCase):

  def _assert_dict_equal(
      self, expected: Dict[Any, Any], actual: Dict[Any, Any]
  ):
    """Asserts that two dicts are equal.

    The dicts can be arbitrarily nested and contain pandas data frames.

    Args:
      expected: the expected dict.
      actual: the actual dict
    """
    for key, expected_val in expected.items():
      self.assertIn(key, actual, f'Expected key: {key}')
      actual_val = actual[key]
      if isinstance(expected_val, dict):
        self.assertIsInstance(actual_val, dict)
        self._assert_dict_equal(expected_val, actual_val)
      elif isinstance(expected_val, pd.DataFrame):
        self.assertIsInstance(actual_val, pd.DataFrame)
        pd.testing.assert_frame_equal(expected_val, actual_val)
      else:
        self.assertEqual(expected_val, actual_val)

  @parameterized.named_parameters(
      {'testcase_name': 'no_slices', 'slices': False},
      {'testcase_name': 'slices', 'slices': True},
  )
  def test_get_statistics_html(self, slices: bool):
    statistics = statistics = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        name: 'a'
        type: FLOAT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 0
            min_num_values: 1
            max_num_values: 4
            avg_num_values: 2.33333333
            tot_num_values: 7
            num_values_histogram {
              buckets {
                low_value: 1.0
                high_value: 1.0
                sample_count: 1.0
              }
              buckets {
                low_value: 1.0
                high_value: 4.0
                sample_count: 1.0
              }
              buckets {
                low_value: 4.0
                high_value: 4.0
                sample_count: 1.0
              }
              type: QUANTILES
            }
          }
          mean: 2.66666666
          std_dev: 1.49071198
          num_zeros: 0
          min: 1.0
          max: 5.0
          median: 3.0
          histograms {
            num_nan: 1
            buckets {
              low_value: 1.0
              high_value: 2.3333333
              sample_count: 2.9866667
            }
            buckets {
              low_value: 2.3333333
              high_value: 3.6666667
              sample_count: 1.0066667
            }
            buckets {
              low_value: 3.6666667
              high_value: 5.0
              sample_count: 2.0066667
            }
            type: STANDARD
          }
          histograms {
            num_nan: 1
            buckets {
              low_value: 1.0
              high_value: 1.0
              sample_count: 1.5
            }
            buckets {
              low_value: 1.0
              high_value: 3.0
              sample_count: 1.5
            }
            buckets {
              low_value: 3.0
              high_value: 4.0
              sample_count: 1.5
            }
            buckets {
              low_value: 4.0
              high_value: 5.0
              sample_count: 1.5
            }
            type: QUANTILES
          }
        }
      }
      features {
        name: 'c'
        type: INT
        num_stats {
          common_stats {
            num_non_missing: 3
            num_missing: 0
            min_num_values: 500
            max_num_values: 1750
            avg_num_values: 1000.0
            tot_num_values: 3000
            num_values_histogram {
              buckets {
                low_value: 500.0
                high_value: 500.0
                sample_count: 1.0
              }
              buckets {
                low_value: 500.0
                high_value: 1750.0
                sample_count: 1.0
              }
              buckets {
                low_value: 1750.0
                high_value: 1750.0
                sample_count: 1.0
              }
              type: QUANTILES
            }
          }
          mean: 1500.5
          std_dev: 866.025355672
          min: 1.0
          max: 3000.0
          median: 1501.0
          histograms {
            buckets {
              low_value: 1.0
              high_value: 1000.66666667
              sample_count: 999.666666667
            }
            buckets {
              low_value: 1000.66666667
              high_value: 2000.33333333
              sample_count: 999.666666667
            }
            buckets {
              low_value: 2000.33333333
              high_value: 3000.0
              sample_count: 1000.66666667
            }
            type: STANDARD
          }
          histograms {
            buckets {
              low_value: 1.0
              high_value: 751.0
              sample_count: 750.0
            }
            buckets {
              low_value: 751.0
              high_value: 1501.0
              sample_count: 750.0
            }
            buckets {
              low_value: 1501.0
              high_value: 2250.0
              sample_count: 750.0
            }
            buckets {
              low_value: 2250.0
              high_value: 3000.0
              sample_count: 750.0
            }
            type: QUANTILES
          }
        }
      }
      features {
        name: 'b'
        type: STRING
        string_stats {
          common_stats {
            num_non_missing: 3
            min_num_values: 4
            max_num_values: 4
            avg_num_values: 4.0
            tot_num_values: 12
            num_values_histogram {
              buckets {
                low_value: 4.0
                high_value: 4.0
                sample_count: 1.0
              }
              buckets {
                low_value: 4.0
                high_value: 4.0
                sample_count: 1.0
              }
              buckets {
                low_value: 4.0
                high_value: 4.0
                sample_count: 1.0
              }
              type: QUANTILES
            }
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
            buckets {
              low_rank: 2
              high_rank: 2
              label: "d"
              sample_count: 2.0
            }
          }
        }
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )
    if slices:
      statistics.datasets[0].name = constants.DEFAULT_SLICE_KEY
      sliced_dataset = statistics.datasets.add()
      sliced_dataset.MergeFrom(statistics.datasets[0])
      sliced_dataset.name = 'slice1'
    # pylint: disable=line-too-long,anomalous-backslash-in-string
    expected_output = """<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>"""
    # pylint: enable=line-too-long

    display_html = display_util.get_statistics_html(statistics, statistics)

    self.assertEqual(display_html, expected_output)

  def test_get_statistics_html_with_empty_dataset(self):
    expected_output = '<p>Empty dataset.</p>'
    statistics = text_format.Parse(
        'datasets { num_examples: 0 }',
        statistics_pb2.DatasetFeatureStatisticsList(),
    )
    display_html = display_util.get_statistics_html(statistics)
    self.assertEqual(display_html, expected_output)

  def test_visualize_statistics_invalid_allowlist_denylist(self):
    statistics = text_format.Parse(
        """
    datasets {
      name: 'test'
      features {
        path { step: 'a' }
        type: FLOAT
      }
      features {
        path { step: 'c' }
        type: INT
      }
      features {
        path { step: 'b' }
        type: STRING
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )
    with self.assertRaisesRegex(AssertionError, '.*specify one of.*'):
      display_util.visualize_statistics(
          statistics,
          allowlist_features=[types.FeaturePath(['a'])],
          denylist_features=[types.FeaturePath(['c'])],
      )

  def test_get_combined_statistics_allowlist_features(self):
    statistics = text_format.Parse(
        """
    datasets {
      name: 'test'
      features {
        path { step: 'a' }
        type: FLOAT
      }
      features {
        path { step: 'c' }
        type: INT
      }
      features {
        path { step: 'b' }
        type: STRING
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    expected_output = text_format.Parse(
        """
    datasets {
      name: 'test'
      features {
        path { step: 'a' }
        type: FLOAT
      }
      features {
        path { step: 'b' }
        type: STRING
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    actual_output = display_util._get_combined_statistics(
        statistics,
        allowlist_features=[types.FeaturePath(['a']), types.FeaturePath(['b'])],
    )
    self.assertLen(actual_output.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self, actual_output.datasets[0], expected_output.datasets[0]
    )

  def test_get_combined_statistics_denylist_features(self):
    statistics = text_format.Parse(
        """
    datasets {
      name: 'test'
      features {
        path { step: 'a' }
        type: FLOAT
      }
      features {
        path { step: 'c' }
        type: INT
      }
      features {
        path { step: 'b' }
        type: STRING
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    expected_output = text_format.Parse(
        """
    datasets {
      name: 'test'
      features {
        path { step: 'a' }
        type: FLOAT
      }
      features {
        path { step: 'b' }
        type: STRING
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    actual_output = display_util._get_combined_statistics(
        statistics, denylist_features=[types.FeaturePath(['c'])]
    )
    self.assertLen(actual_output.datasets, 1)
    test_util.assert_dataset_feature_stats_proto_equal(
        self, actual_output.datasets[0], expected_output.datasets[0]
    )

  def test_get_schema_dataframe(self):
    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        feature {
          name: "fb"
          type: BYTES
        }
        feature {
          name: "fc"
          type: FLOAT
        }
        string_domain {
          name: "timezone"
          value: "America/Los_Angeles"
        }
        """,
        schema_pb2.Schema(),
    )
    actual_features, actual_domains = display_util.get_schema_dataframe(schema)
    # The resulting features DataFrame has a row for each feature and columns
    # for type, presence, valency, and domain.
    self.assertEqual(actual_features.shape, (3, 4))
    # The resulting domain DataFrame has a row for each domain and a column for
    # domain values.
    self.assertEqual(actual_domains.shape, (1, 1))

  def test_get_anomalies_dataframe(self):
    anomalies = text_format.Parse(
        """
    anomaly_info {
     key: "feature_1"
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
      key: "feature_2"
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
    }
    """,
        anomalies_pb2.Anomalies(),
    )
    actual_output = display_util.get_anomalies_dataframe(anomalies)
    # The resulting DataFrame has a row for each feature and a column for each
    # of the short description and long description.
    self.assertEqual(actual_output.shape, (2, 2))

  def test_get_anomalies_dataframe_with_no_toplevel_description(self):
    anomalies = text_format.Parse(
        """
    anomaly_info {
     key: "feature_1"
     value {
        severity: ERROR
        reason {
          type: ENUM_TYPE_BYTES_NOT_STRING
          short_description: "Bytes not string"
          description: "Expected bytes but got string."
        }
      }
    }
    anomaly_info {
      key: "feature_2"
      value {
        severity: ERROR
        reason {
          type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
          short_description: "Unexpected string values"
          description: "Examples contain values missing from the "
            "schema."
        }
      }
    }
    """,
        anomalies_pb2.Anomalies(),
    )
    actual_output = display_util.get_anomalies_dataframe(anomalies)
    # The resulting DataFrame has a row for each feature and a column for each
    # of the short description and long description.
    self.assertEqual(actual_output.shape, (2, 2))

    # Confirm Anomaly short/long description is not empty
    self.assertNotEmpty(actual_output['Anomaly short description'][0])
    self.assertNotEmpty(actual_output['Anomaly long description'][0])

  def test_get_drift_skew_dataframe(self):
    anomalies = text_format.Parse(
        """
    drift_skew_info {
     path: {step: "feature_1"}
     drift_measurements {
       type: JENSEN_SHANNON_DIVERGENCE
       value: 0.4
       threshold: 0.1
     }
    }
    drift_skew_info {
     path: {step: "feature_2"}
     drift_measurements {
      type: L_INFTY
      value: 0.5
      threshold: 0.1
    }
    }
    """,
        anomalies_pb2.Anomalies(),
    )
    actual_output = display_util.get_drift_skew_dataframe(anomalies)
    expected = pd.DataFrame(
        [
            ['feature_1', 'JENSEN_SHANNON_DIVERGENCE', 0.4, 0.1],
            ['feature_2', 'L_INFTY', 0.5, 0.1],
        ],
        columns=['path', 'type', 'value', 'threshold'],
    ).set_index('path')
    self.assertTrue(actual_output.equals(expected))

  def test_get_anomalies_dataframe_no_anomalies(self):
    anomalies = anomalies_pb2.Anomalies()
    actual_output = display_util.get_anomalies_dataframe(anomalies)
    self.assertEqual(actual_output.shape, (0, 2))

  def test_get_natural_language_statistics_dataframes(self):
    statistics = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        name: 'feature_name'
        type: BYTES
        custom_stats {
          name: "nl_statistics"
        }
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    nl_stats = text_format.Parse(
        """
        feature_coverage: 1.0
        avg_token_length: 3.6760780287474333
        token_length_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 194.8
          }
        }
        token_statistics {
          int_token: 88
        }
        token_statistics {
          string_token: "[UNK]"
        }
        token_statistics {
          string_token: "[PAD]"
          frequency: 48852.0
          fraction_of_sequences: 1.0
          per_sequence_min_frequency: 220.0
          per_sequence_avg_frequency: 244.26
          per_sequence_max_frequency: 251.0
          positions {
            buckets {
              high_value: 0.1
              sample_count: 2866.0
            }
          }
        }
        sequence_length_histogram {
          buckets {
            low_value: 5.0
            high_value: 7.0
            sample_count: 20.0
          }
        }
        min_sequence_length: 5
        max_sequence_length: 36
        """,
        statistics_pb2.NaturalLanguageStatistics(),
    )

    statistics.datasets[0].features[0].custom_stats[0].any.Pack(nl_stats)
    actual = display_util.get_natural_language_statistics_dataframes(statistics)

    expected = {
        'lhs_statistics': {
            'feature_name': {
                'token_length_histogram': pd.DataFrame.from_dict({
                    'high_values': [1.0],
                    'low_values': [1.0],
                    'sample_counts': [194.8],
                }),
                'token_statistics': pd.DataFrame.from_dict({
                    'token_name': [88, '[UNK]', '[PAD]'],
                    'frequency': [0.0, 0.0, 48852.0],
                    'fraction_of_sequences': [0.0, 0.0, 1.0],
                    'per_sequence_min_frequency': [0.0, 0.0, 220.0],
                    'per_sequence_max_frequency': [0.0, 0.0, 251.0],
                    'per_sequence_avg_frequency': [0.0, 0.0, 244.26],
                    'positions': [
                        pd.DataFrame.from_dict({
                            'high_values': [],
                            'low_values': [],
                            'sample_counts': [],
                        }),
                        pd.DataFrame.from_dict({
                            'high_values': [],
                            'low_values': [],
                            'sample_counts': [],
                        }),
                        pd.DataFrame.from_dict({
                            'high_values': [0.1],
                            'low_values': [0.0],
                            'sample_counts': [2866.0],
                        }),
                    ],
                }),
            }
        }
    }

    self._assert_dict_equal(expected, actual)

  def test_get_natural_language_statistics_dataframes_feature_path(self):
    statistics = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        path {
          step: "my"
          step: "feature"
        }
        type: BYTES
        custom_stats {
          name: "nl_statistics"
        }
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    nl_stats = text_format.Parse(
        """
        feature_coverage: 1.0
        avg_token_length: 3.6760780287474333
        token_length_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 194.8
          }
        }
        token_statistics {
          int_token: 88
        }
        token_statistics {
          string_token: "[UNK]"
        }
        token_statistics {
          string_token: "[PAD]"
          frequency: 48852.0
          fraction_of_sequences: 1.0
          per_sequence_min_frequency: 220.0
          per_sequence_avg_frequency: 244.26
          per_sequence_max_frequency: 251.0
          positions {
            buckets {
              high_value: 0.1
              sample_count: 2866.0
            }
          }
        }
        sequence_length_histogram {
          buckets {
            low_value: 5.0
            high_value: 7.0
            sample_count: 20.0
          }
        }
        min_sequence_length: 5
        max_sequence_length: 36
        """,
        statistics_pb2.NaturalLanguageStatistics(),
    )

    statistics.datasets[0].features[0].custom_stats[0].any.Pack(nl_stats)
    actual = display_util.get_natural_language_statistics_dataframes(statistics)

    expected = {
        'lhs_statistics': {
            'my.feature': {
                'token_length_histogram': pd.DataFrame.from_dict({
                    'high_values': [1.0],
                    'low_values': [1.0],
                    'sample_counts': [194.8],
                }),
                'token_statistics': pd.DataFrame.from_dict({
                    'token_name': [88, '[UNK]', '[PAD]'],
                    'frequency': [0.0, 0.0, 48852.0],
                    'fraction_of_sequences': [0.0, 0.0, 1.0],
                    'per_sequence_min_frequency': [0.0, 0.0, 220.0],
                    'per_sequence_max_frequency': [0.0, 0.0, 251.0],
                    'per_sequence_avg_frequency': [0.0, 0.0, 244.26],
                    'positions': [
                        pd.DataFrame.from_dict({
                            'high_values': [],
                            'low_values': [],
                            'sample_counts': [],
                        }),
                        pd.DataFrame.from_dict({
                            'high_values': [],
                            'low_values': [],
                            'sample_counts': [],
                        }),
                        pd.DataFrame.from_dict({
                            'high_values': [0.1],
                            'low_values': [0.0],
                            'sample_counts': [2866.0],
                        }),
                    ],
                }),
            }
        }
    }

    self._assert_dict_equal(expected, actual)

  def test_get_natural_language_statistics_many_features_dataframes(self):
    statistics = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        name: 'feature_name'
        type: BYTES
        custom_stats {
          name: "nl_statistics"
        }
      }
      features {
        name: 'feature_name_2'
        type: BYTES
        custom_stats {
          name: "nl_statistics"
        }
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )

    nl_stats = text_format.Parse(
        """
        feature_coverage: 1.0
        avg_token_length: 3.6760780287474333
        token_length_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 194.8
          }
        }
        token_statistics {
          int_token: 88
        }
        token_statistics {
          string_token: "[UNK]"
        }
        token_statistics {
          string_token: "[PAD]"
          frequency: 48852.0
          fraction_of_sequences: 1.0
          per_sequence_min_frequency: 220.0
          per_sequence_avg_frequency: 244.26
          per_sequence_max_frequency: 251.0
          positions {
            buckets {
              high_value: 0.1
              sample_count: 2866.0
            }
          }
        }
        sequence_length_histogram {
          buckets {
            low_value: 5.0
            high_value: 7.0
            sample_count: 20.0
          }
        }
        min_sequence_length: 5
        max_sequence_length: 36
        """,
        statistics_pb2.NaturalLanguageStatistics(),
    )

    statistics.datasets[0].features[0].custom_stats[0].any.Pack(nl_stats)
    statistics.datasets[0].features[1].custom_stats[0].any.Pack(nl_stats)
    actual = display_util.get_natural_language_statistics_dataframes(
        statistics, statistics
    )

    token_length_histogram = pd.DataFrame.from_dict(
        {'high_values': [1.0], 'low_values': [1.0], 'sample_counts': [194.8]}
    )
    token_statistics = pd.DataFrame.from_dict({
        'token_name': [88, '[UNK]', '[PAD]'],
        'frequency': [0.0, 0.0, 48852.0],
        'fraction_of_sequences': [0.0, 0.0, 1.0],
        'per_sequence_min_frequency': [0.0, 0.0, 220.0],
        'per_sequence_max_frequency': [0.0, 0.0, 251.0],
        'per_sequence_avg_frequency': [0.0, 0.0, 244.26],
        'positions': [
            pd.DataFrame.from_dict(
                {'high_values': [], 'low_values': [], 'sample_counts': []}
            ),
            pd.DataFrame.from_dict(
                {'high_values': [], 'low_values': [], 'sample_counts': []}
            ),
            pd.DataFrame.from_dict({
                'high_values': [0.1],
                'low_values': [0.0],
                'sample_counts': [2866.0],
            }),
        ],
    })
    expected = {
        'lhs_statistics': {
            'feature_name': {
                'token_length_histogram': token_length_histogram,
                'token_statistics': token_statistics,
            },
            'feature_name_2': {
                'token_length_histogram': token_length_histogram,
                'token_statistics': token_statistics,
            },
        },
        'rhs_statistics': {
            'feature_name': {
                'token_length_histogram': token_length_histogram,
                'token_statistics': token_statistics,
            },
            'feature_name_2': {
                'token_length_histogram': token_length_histogram,
                'token_statistics': token_statistics,
            },
        },
    }

    self._assert_dict_equal(expected, actual)

  def test_get_nonexistent_natural_language_statistics_dataframes(self):
    statistics = text_format.Parse(
        """
    datasets {
      num_examples: 3
      features {
        name: 'a'
        type: BYTES
      }
    }
    """,
        statistics_pb2.DatasetFeatureStatisticsList(),
    )
    actual = display_util.get_natural_language_statistics_dataframes(statistics)
    self.assertIsNone(actual)


class FeatureSkewTest(absltest.TestCase):

  def test_formats_skew_results(self):
    skew_results = [
        text_format.Parse(
            """
        feature_name: 'foo'
        base_count: 101
        test_count: 102
        match_count: 103
        base_only: 104
        test_only: 105
        mismatch_count: 106
        diff_count: 107
        """,
            feature_skew_results_pb2.FeatureSkew(),
        ),
        text_format.Parse(
            """
        feature_name: 'bar'
        base_count: 201
        test_count: 202
        match_count: 203
        base_only: 204
        test_only: 205
        mismatch_count: 206
        diff_count: 207
        """,
            feature_skew_results_pb2.FeatureSkew(),
        ),
        text_format.Parse(
            """
        feature_name: 'baz'
        """,
            feature_skew_results_pb2.FeatureSkew(),
        ),
    ]
    df = display_util.get_skew_result_dataframe(skew_results)
    expected = pd.DataFrame(
        [
            ['bar', 201, 202, 203, 204, 205, 206, 207],
            ['baz', 0, 0, 0, 0, 0, 0, 0],
            ['foo', 101, 102, 103, 104, 105, 106, 107],
        ],
        columns=[
            'feature_name',
            'base_count',
            'test_count',
            'match_count',
            'base_only',
            'test_only',
            'mismatch_count',
            'diff_count',
        ],
    )
    self.assertTrue(df.equals(expected))

  def test_formats_empty_skew_results(self):
    skew_results = []
    df = display_util.get_skew_result_dataframe(skew_results)
    expected = pd.DataFrame(
        [],
        columns=[
            'feature_name',
            'base_count',
            'test_count',
            'match_count',
            'base_only',
            'test_only',
            'mismatch_count',
            'diff_count',
        ],
    )
    self.assertTrue(df.equals(expected))

  def test_formats_confusion_counts(self):
    confusion = [
        text_format.Parse(
            """
        feature_name: "foo"
        base {
          bytes_value: "val1"
        }
        test {
          bytes_value: "val1"
        }
        count: 99
        """,
            feature_skew_results_pb2.ConfusionCount(),
        ),
        text_format.Parse(
            """
        feature_name: "foo"
        base {
          bytes_value: "val1"
        }
        test {
          bytes_value: "val2"
        }
        count: 1
        """,
            feature_skew_results_pb2.ConfusionCount(),
        ),
        text_format.Parse(
            """
        feature_name: "foo"
        base {
          bytes_value: "val2"
        }
        test {
          bytes_value: "val3"
        }
        count: 1
        """,
            feature_skew_results_pb2.ConfusionCount(),
        ),
        text_format.Parse(
            """
        feature_name: "foo"
        base {
          bytes_value: "val3"
        }
        test {
          bytes_value: "val3"
        }
        count: 100
        """,
            feature_skew_results_pb2.ConfusionCount(),
        ),
        text_format.Parse(
            """
        feature_name: "bar"
        base {
          bytes_value: "val1"
        }
        test {
          bytes_value: "val2"
        }
        count: 1
        """,
            feature_skew_results_pb2.ConfusionCount(),
        ),
    ]
    dfs = display_util.get_confusion_count_dataframes(confusion)
    self.assertSameElements(dfs.keys(), ['foo', 'bar'])
    self.assertTrue(
        dfs['foo'].equals(
            pd.DataFrame(
                [[b'val1', b'val2', 1, 100, 1], [b'val2', b'val3', 1, 1, 101]],
                columns=[
                    'Base value',
                    'Test value',
                    'Pair count',
                    'Base count',
                    'Test count',
                ],
            )
        )
    )
    self.assertTrue(
        dfs['bar'].equals(
            pd.DataFrame(
                [[b'val1', b'val2', 1, 1, 1]],
                columns=[
                    'Base value',
                    'Test value',
                    'Pair count',
                    'Base count',
                    'Test count',
                ],
            )
        )
    )


if __name__ == '__main__':
  absltest.main()
