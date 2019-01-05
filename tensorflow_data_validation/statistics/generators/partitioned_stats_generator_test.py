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
"""Tests for partitioned_stats_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import partitioned_stats_generator
from tensorflow_data_validation.statistics.generators import sklearn_mutual_information
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

TEST_SEED = 10


class FlattenExamplesTest(absltest.TestCase):
  """Tests for _flatten_examples."""

  def _assert_output_equal(self, batches, expected):
    actual = partitioned_stats_generator._flatten_examples(batches)

    actual_features = set(actual.keys())
    expected_features = set(expected.keys())

    self.assertSetEqual(actual_features, expected_features)

    for feature in actual_features:
      actual_values = actual[feature]
      expected_values = expected[feature]
      np.testing.assert_array_equal(actual_values, expected_values)

  def test_custom_stats_combiner_two_features(self):
    # Input with two batches: first batch has two examples with two features
    # and second batch has a single example with two features.

    batches = [{
        'a': np.array([np.array([1.5]), np.array([1.5])]),
        'b': np.array([np.array([2.0]), np.array([3.0])])
    }, {
        'a': np.array([np.array([3.0])]),
        'b': np.array([np.array([4.0])])
    }]
    expected_result = {
        'a': np.array([np.array([1.5]),
                       np.array([1.5]),
                       np.array([3.0])]),
        'b': np.array([np.array([2.0]),
                       np.array([3.0]),
                       np.array([4.0])])
    }
    self._assert_output_equal(batches, expected_result)

  def test_custom_stats_combiner_stats_generator_missing_feature(self):
    # Input with two batches: first batch has three examples (one missing)
    # with one feature and second batch has two examples (one missing) with
    # one feature.
    batches = [{
        'a': np.array(
            [np.array([1.0]), None, np.array([3.0])], dtype=np.object)
    }, {
        'a': np.array([np.array([2.0]), None], dtype=np.object)
    }]
    expected_result = {
        'a':
            np.array(
                [np.array([1.0]), None,
                 np.array([3]),
                 np.array([2]), None])
    }
    self._assert_output_equal(batches, expected_result)

  def test_custom_stats_combiner_with_empty_batch(self):
    # Input with two batches: first batch has one example with one feature and
    # second batch has no examples.
    batches = [{'a': np.array([np.array([1.0])])}, {}]
    expected_result = {'a': np.array([np.array([1.0])])}
    self._assert_output_equal(batches, expected_result)

  def test_custom_stats_combiner_with_missing_feature_name(self):
    # Input with two batches: first batch has one example with two features and
    # second batch has one example with one feature.
    batches = [{
        'a': np.array([np.array([1.5])]),
        'b': np.array([np.array([3.0])])
    }, {
        'a': np.array([np.array([3.0])])
    }]
    expected_result = {
        'a': np.array([np.array([1.5]), np.array([3.0])]),
        'b': np.array([np.array([3.0]), None])
    }
    self._assert_output_equal(batches, expected_result)


class AssignToPartitionTest(absltest.TestCase):
  """Tests for _asssign_to_partition."""

  def test_partitioner(self):
    """Tests that examples are randomly partitioned.

    Tests an input batch with one univalent feature taking on values in {0,1,2}.
    The input batch has 4500 examples. The partitioner is configured to have 3
    partitions. So, we expect there to be around 4500/3/3 = 500 of each.
    {0,1,2} in each partition.
    """

    np.random.seed(TEST_SEED)
    examples = [{'a': x} for x in np.random.randint(0, 3, (4500, 1))]
    num_partitions = 3

    # The i,jth value of result represents the number of examples with value j
    # assigned to partition i.
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    partitioned_examples = [
        partitioned_stats_generator._assign_to_partition(example,
                                                         num_partitions)
        for example in examples]
    for partition_key, example in partitioned_examples:
      result[partition_key][example['a'][0]] += 1

    for partition in result:
      for count in partition:
        self.assertBetween(count, 400, 600)


class PartitionedStatisticsAnalyzer(absltest.TestCase):
  """Tests PartitionedStatisticsAnalyzer."""

  def _assert_combiner_output_equal(self, statistics, combiner, expected):
    accumulators = [
        combiner.add_input(combiner.create_accumulator(), statistic)
        for statistic in statistics
    ]
    actual = combiner.extract_output(combiner.merge_accumulators(accumulators))
    compare.assertProtoEqual(self, actual, expected, normalize_numbers=True)

  def test_statistic_analyzer_with_invalid_featureeature(self):
    statistics = [
        text_format.Parse(
            """
      features {
        name: 'valid_feature'
        custom_stats {
              name: 'MI'
              num: 0.5
            }
        custom_stats {
              name: 'Cov'
              num: 0.3
            }
      }
      features {
        name: 'invalid_feature'
        custom_stats {
          name: 'MI'
          num: 0.5
        }
        custom_stats {
          name: 'Cov'
          num: 0.3
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        name: 'valid_feature'
        custom_stats {
              name: 'MI'
              num: 1.5
            }
        custom_stats {
              name: 'Cov'
              num: 0.7
            }
      }""", statistics_pb2.DatasetFeatureStatistics())
    ]
    expected = text_format.Parse(
        """
      features {
        name: 'valid_feature'
        custom_stats {
              name: 'max_Cov'
              num: 0.7
            }
        custom_stats {
              name: 'max_MI'
              num: 1.5
            }
        custom_stats {
              name: 'mean_Cov'
              num: 0.5
            }
        custom_stats {
              name: 'mean_MI'
              num: 1
            }
        custom_stats {
              name: 'median_Cov'
              num: 0.5
            }
        custom_stats {
              name: 'median_MI'
              num: 1
            }
        custom_stats {
              name: 'min_Cov'
              num: 0.3
            }
        custom_stats {
              name: 'min_MI'
              num: 0.5
            }
        custom_stats {
              name: 'num_partitions_Cov'
              num: 2
            }
        custom_stats {
              name: 'num_partitions_MI'
              num: 2
            }
        custom_stats {
              name: 'std_dev_Cov'
              num: 0.2
            }
        custom_stats {
              name: 'std_dev_MI'
              num: 0.5
            }
      }""", statistics_pb2.DatasetFeatureStatistics())
    self._assert_combiner_output_equal(
        statistics,
        partitioned_stats_generator.PartitionedStatisticsAnalyzer(
            min_partitions_stat_presence=2), expected)


class NonStreamingCustomStatsGeneratorTest(
    test_util.TransformStatsGeneratorTest):
  """Tests for NonStreamingCustomStatsGenerator."""

  def test_sklearn_mi(self):
    # Integration tests involving Beam and AMI are challenging to write
    # because Beam PCollections are unordered while the results of adjusted MI
    # depend on the order of the data for small datasets. This test case tests
    # MI with one label which will give a value of 0 regardless of
    # the ordering of elements in the PCollection. The purpose of this test is
    # to ensure that the Mutual Information pipeline is able to handle a
    # variety of input types. Unit tests ensuring correctness of the MI value
    # itself are included in sklearn_mutual_information_test.

    # fa is categorical, fb is numeric, fc is multivalent and fd has null values
    examples = [
        {'fa': np.array(['Red']),
         'fb': np.array([1.0]),
         'fc': np.array([1, 3, 1]),
         'fd': np.array([0.4]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Green']),
         'fb': np.array([2.2]),
         'fc': np.array([2, 6]),
         'fd': np.array([0.4]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Blue']),
         'fb': np.array([3.3]),
         'fc': np.array([4, 6]),
         'fd': np.array([0.3]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Green']),
         'fb': np.array([1.3]),
         'fc': None,
         'fd': np.array([0.2]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Red']),
         'fb': np.array([1.2]),
         'fc': np.array([1]),
         'fd': np.array([0.3]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Blue']),
         'fb': np.array([0.5]),
         'fc': np.array([3, 2]),
         'fd': np.array([0.4]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Blue']),
         'fb': np.array([1.3]),
         'fc': np.array([1, 4]),
         'fd': np.array([1.7]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Green']),
         'fb': np.array([2.3]),
         'fc': np.array([0]),
         'fd': np.array([np.NaN]),
         'label_key': np.array(['Label'])},
        {'fa': np.array(['Green']),
         'fb': np.array([0.3]),
         'fc': np.array([3]),
         'fd': np.array([4.4]),
         'label_key': np.array(['Label'])}
    ]

    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: BYTES
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "fb"
          type: FLOAT
          shape {
            dim {
              size: 1
            }
          }
        }
        feature {
          name: "fc"
          type: INT
          value_count: {
            min: 0
            max: 2
          }
        }
        feature {
          name: "fd"
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
        }""", schema_pb2.Schema())

    expected_result = [
        text_format.Parse(
            """
              features {
                name: "fa"
                custom_stats {
                  name: "max_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "max_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_adjusted_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "std_dev_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "std_dev_sklearn_mutual_information"
                  num: 0.0
                }
              }
              features {
                name: "fb"
                custom_stats {
                  name: "max_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "max_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_adjusted_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "std_dev_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "std_dev_sklearn_mutual_information"
                  num: 0.0
                }
              }
              features {
                name: "fd"
                custom_stats {
                  name: "max_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "max_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "mean_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "median_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "min_sklearn_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_adjusted_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "num_partitions_sklearn_mutual_information"
                  num: 2.0
                }
                custom_stats {
                  name: "std_dev_sklearn_adjusted_mutual_information"
                  num: 0.0
                }
                custom_stats {
                  name: "std_dev_sklearn_mutual_information"
                  num: 0.0
                }
              }""", statistics_pb2.DatasetFeatureStatistics())
    ]
    generator = partitioned_stats_generator.NonStreamingCustomStatsGenerator(
        sklearn_mutual_information.SkLearnMutualInformation(
            label_feature='label_key', schema=schema, seed=TEST_SEED),
        num_partitions=2,
        min_partitions_stat_presence=2,
        seed=TEST_SEED,
        max_examples_per_partition=1000,
        name='NonStreaming Mutual Information')
    self.assertTransformOutputEqual(examples, generator, expected_result)

if __name__ == '__main__':
  absltest.main()
