# Copyright 2019 Google LLC
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

"""Tests for TopKUniques statistics generator."""

from absl.testing import absltest
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import top_k_uniques_stats_generator
from tensorflow_data_validation.utils import test_util
from tensorflow_data_validation.utils.example_weight_map import ExampleWeightMap

from google.protobuf import text_format

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class TopkUniquesStatsGeneratorTest(test_util.TransformStatsGeneratorTest):
  """Tests for TopkUniquesStatsGenerator."""

  def test_topk_uniques_with_single_string_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'

    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([
                ['a', 'b', 'c', 'e'],
                ['a', 'c', 'd', 'a'],
                ['a', 'b', 'c', 'd'],
            ])
        ], ['fa'])
    ]

    # Note that if two feature values have the same frequency, the one with the
    # lexicographically larger feature value will be higher in the order.
    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
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
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_weights(self):
    # non-weighted ordering
    # fa: 3 'a', 2 'e', 2 'd', 2 'c', 1 'b'
    # fb: 1 'v', 1 'w', 1 'x', 1 'y', 1 'z'
    # weighted ordering
    # fa: 20 'e', 20 'd', 15 'a', 10 'c', 5 'b'
    # fb: 6 'z', 4 'x', 4 'y', 4 'w', 2 'v'
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([
                ['a', 'b', 'c', 'e'],
                ['a', 'c', 'd', 'a'],
                ['d', 'e'],
            ]),
            pa.array([[5.0], [5.0], [15.0]]),
            pa.array([['v'], ['w', 'x', 'y'], ['z']]),
            pa.array([[2], [4], [6]]),
        ], ['fa', 'w', 'fb', 'w_b'])
    ]

    expected_result = [
        text_format.Parse(
            """
            features {
              path {
                step: 'fa'
              }
              string_stats {
                top_values {
                  value: 'a'
                  frequency: 3.0
                }
                top_values {
                  value: 'e'
                  frequency: 2.0
                }
                top_values {
                  value: 'd'
                  frequency: 2.0
                }
                top_values {
                  value: 'c'
                  frequency: 2.0
                }
                rank_histogram {
                  buckets {
                    low_rank: 0
                    high_rank: 0
                    label: "a"
                    sample_count: 3.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "e"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "d"
                    sample_count: 2.0
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              string_stats {
                top_values {
                  value: "z"
                  frequency: 1.0
                }
                top_values {
                  value: "y"
                  frequency: 1.0
                }
                top_values {
                  value: "x"
                  frequency: 1.0
                }
                top_values {
                  value: "w"
                  frequency: 1.0
                }
                rank_histogram {
                  buckets {
                    label: "z"
                    sample_count: 1.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "y"
                    sample_count: 1.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "x"
                    sample_count: 1.0
                  }
                }
              }
              path {
                step: "fb"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              path {
                step: 'fa'
              }
              string_stats {
                weighted_string_stats {
                  top_values {
                    value: 'e'
                    frequency: 20.0
                  }
                  top_values {
                    value: 'd'
                    frequency: 20.0
                  }
                  top_values {
                    value: 'a'
                    frequency: 15.0
                  }
                  top_values {
                    value: 'c'
                    frequency: 10.0
                  }
                  rank_histogram {
                    buckets {
                      low_rank: 0
                      high_rank: 0
                      label: "e"
                      sample_count: 20.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "d"
                      sample_count: 20.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "a"
                      sample_count: 15.0
                    }
                  }
                }
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              string_stats {
                weighted_string_stats {
                  top_values {
                    value: "z"
                    frequency: 6.0
                  }
                  top_values {
                    value: "y"
                    frequency: 4.0
                  }
                  top_values {
                    value: "x"
                    frequency: 4.0
                  }
                  top_values {
                    value: "w"
                    frequency: 4.0
                  }
                  rank_histogram {
                    buckets {
                      label: "z"
                      sample_count: 6.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "y"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "x"
                      sample_count: 4.0
                    }
                  }
                }
              }
              path {
                step: "fb"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              path {
                step: 'fa'
              }
              string_stats {
                unique: 5
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              string_stats {
                unique: 5
              }
              path {
                step: "fb"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        example_weight_map=ExampleWeightMap(
            weight_feature='w',
            per_feature_override={types.FeaturePath(['fb']): 'w_b'}),
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_single_unicode_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([
                [u'a', u'b', u'c', u'e'],
                [u'a', u'c', u'd', u'a'],
                [u'a', u'b', u'c', u'd'],
            ])
        ], ['fa'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
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
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_multiple_features(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 2 'b', 3 'c'
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd'],
                      ['a', 'a', 'b', 'c', 'd'], None]),
            pa.array([['a', 'c', 'c'], ['b'], None, None, ['b', 'c']])
        ], ['fa', 'fb'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
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
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'b'
            frequency: 2
          }
          top_values {
            value: 'a'
            frequency: 1
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "c"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "b"
              sample_count: 2.0
            }
            buckets {
              low_rank: 2
              high_rank: 2
              label: "a"
              sample_count: 1.0
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fb'
        }
        string_stats {
          unique: 3
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_empty_input(self):
    examples = []
    expected_result = []
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(examples, generator,
                                                expected_result)

  def test_topk_uniques_with_empty_record_batch(self):
    examples = [pa.RecordBatch.from_arrays([], [])]
    expected_result = []
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_missing_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 1 'b', 2 'c'
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None]),
            pa.array([
                ['a', 'c', 'c'],
                ['b'],
            ])
        ], ['fa', 'fb']),
        pa.RecordBatch.from_arrays(
            [pa.array([['a', 'c', 'd'], ['a', 'a', 'b', 'c', 'd'], None])],
            ['fa']),
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
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
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          top_values {
            value: 'c'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 1
          }
          top_values {
            value: 'a'
            frequency: 1
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "c"
              sample_count: 2.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "b"
              sample_count: 1.0
            }
            buckets {
              low_rank: 2
              high_rank: 2
              label: "a"
              sample_count: 1.0
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fb'
        }
        string_stats {
          unique: 3
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_numeric_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'

    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd'],
                      ['a', 'a', 'b', 'c', 'd']]),
            pa.array([[1.0, 2.0, 3.0], [4.0, 5.0], None, None]),
        ], ['fa', 'fb'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
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
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=2, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_bytes_feature(self):
    # fa: 4 'a', 2 'b', 3 'c', 2 'd', 1 'e'
    # fb: 1 'a', 2 'b', 3 'c'
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'b', 'c', 'e'], None, ['a', 'c', 'd'],
                      ['a', 'a', 'b', 'c', 'd'], None]),
            pa.array([['a', 'c', 'c'], ['b'], None, None, ['b', 'c']])
        ], ['fa', 'fb'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 4
          }
          top_values {
            value: 'c'
            frequency: 3
          }
          top_values {
            value: 'd'
            frequency: 2
          }
          top_values {
            value: 'b'
            frequency: 2
          }
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
      }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    schema = text_format.Parse(
        """
        feature {
          name: "fb"
          type: BYTES
          image_domain { }
        }
        """, schema_pb2.Schema())
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        schema=schema, num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_categorical_feature(self):
    examples = [
        pa.RecordBatch.from_arrays(
            [pa.array([[12, 23, 34, 12], [45, 23], [12, 12, 34, 45]])], ['fa']),
        pa.RecordBatch.from_arrays([pa.array([None, None], type=pa.null())],
                                   ['fa'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        type: INT
        string_stats {
          top_values {
            value: '12'
            frequency: 4
          }
          top_values {
            value: '45'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "12"
              sample_count: 4.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "45"
              sample_count: 2.0
            }
            buckets {
              low_rank: 2
              high_rank: 2
              label: "34"
              sample_count: 2.0
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        type: INT
        string_stats {
          unique: 4
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    schema = text_format.Parse(
        """
        feature {
          name: "fa"
          type: INT
          int_domain {
            is_categorical: true
          }
        }
        """, schema_pb2.Schema())
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        schema=schema, num_top_values=2, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_frequency_threshold(self):
    examples = [
        pa.RecordBatch.from_arrays([
            pa.array([['a', 'b', 'y', 'b'], ['a', 'x', 'a', 'z']]),
            pa.array([[5.0], [15.0]])
        ], ['fa', 'w'])
    ]

    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 3
          }
          top_values {
            value: 'b'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "b"
              sample_count: 2.0
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          weighted_string_stats {
            top_values {
              value: 'a'
              frequency: 35.0
            }
            top_values {
              value: 'z'
              frequency: 15.0
            }
            top_values {
              value: 'x'
              frequency: 15.0
            }
            rank_histogram {
              buckets {
                low_rank: 0
                high_rank: 0
                label: "a"
                sample_count: 35.0
              }
              buckets {
                low_rank: 1
                high_rank: 1
                label: "z"
                sample_count: 15.0
              }
              buckets {
                low_rank: 2
                high_rank: 2
                label: "x"
                sample_count: 15.0
              }
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        example_weight_map=ExampleWeightMap(weight_feature='w'),
        num_top_values=5,
        frequency_threshold=2,
        weighted_frequency_threshold=15,
        num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_invalid_utf8_value(self):
    examples = [
        pa.RecordBatch.from_arrays(
            [pa.array([[b'a', b'\x80abc', b'a', b'\x80abc', b'a']])], ['fa'])
    ]
    expected_result = [
        text_format.Parse(
            """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 3
          }
          top_values {
            value: '__BYTES_VALUE__'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "__BYTES_VALUE__"
              sample_count: 2.0
            }
          }
        }
    }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
    features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 2
        }
      }""", statistics_pb2.DatasetFeatureStatistics()),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=4, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        examples,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_topk_uniques_with_slicing(self):
    examples = [
        ('slice1',
         pa.RecordBatch.from_arrays(
             [pa.array([['a', 'b', 'c', 'e']]),
              pa.array([['1', '1', '0']])], ['fa', 'fb'])),
        ('slice2',
         pa.RecordBatch.from_arrays(
             [pa.array([['b', 'a', 'e', 'c']]),
              pa.array([['0', '0', '1']])], ['fa', 'fb'])),
        ('slice1',
         pa.RecordBatch.from_arrays([pa.array([['a', 'c', 'd', 'a']])],
                                    ['fa'])),
        ('slice2',
         pa.RecordBatch.from_arrays([pa.array([['b', 'e', 'd', 'b']])], ['fa']))
    ]

    # Note that if two feature values have the same frequency, the one with the
    # lexicographically larger feature value will be higher in the order.
    expected_result = [
        ('slice1',
         text_format.Parse(
             """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'a'
            frequency: 3
          }
          top_values {
            value: 'c'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "a"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "c"
              sample_count: 2.0
            }
          }
        }
      }
    """, statistics_pb2.DatasetFeatureStatistics())),
        ('slice1',
         text_format.Parse(
             """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          top_values {
            value: '1'
            frequency: 2
          }
          top_values {
            value: '0'
            frequency: 1
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "1"
              sample_count: 2.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "0"
              sample_count: 1.0
            }
          }
        }
      }
    """, statistics_pb2.DatasetFeatureStatistics())),
        ('slice1',
         text_format.Parse(
             """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
    }""", statistics_pb2.DatasetFeatureStatistics())),
        ('slice1',
         text_format.Parse(
             """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          unique: 2
        }
    }""", statistics_pb2.DatasetFeatureStatistics())),
        ('slice2',
         text_format.Parse(
             """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          top_values {
            value: 'b'
            frequency: 3
          }
          top_values {
            value: 'e'
            frequency: 2
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "b"
              sample_count: 3.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "e"
              sample_count: 2.0
            }
          }
        }
      }
    """, statistics_pb2.DatasetFeatureStatistics())),
        ('slice2',
         text_format.Parse(
             """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          top_values {
            value: '0'
            frequency: 2
          }
          top_values {
            value: '1'
            frequency: 1
          }
          rank_histogram {
            buckets {
              low_rank: 0
              high_rank: 0
              label: "0"
              sample_count: 2.0
            }
            buckets {
              low_rank: 1
              high_rank: 1
              label: "1"
              sample_count: 1.0
            }
          }
        }
      }
    """, statistics_pb2.DatasetFeatureStatistics())),
        ('slice2',
         text_format.Parse(
             """
      features {
        path {
          step: 'fa'
        }
        string_stats {
          unique: 5
        }
    }""", statistics_pb2.DatasetFeatureStatistics())),
        ('slice2',
         text_format.Parse(
             """
      features {
        path {
          step: 'fb'
        }
        string_stats {
          unique: 2
        }
    }""", statistics_pb2.DatasetFeatureStatistics())),
    ]

    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        num_top_values=2, num_rank_histogram_buckets=2)
    self.assertSlicingAwareTransformOutputEqual(examples, generator,
                                                expected_result)

  def test_topk_uniques_with_struct_leaves(self):
    inputs = [
        pa.RecordBatch.from_arrays([
            pa.array([[1.0], [2.0]]),
            pa.array([[{
                'f1': ['a', 'b'],
                'f2': [1, 2]
            }, {
                'f1': ['b'],
            }], [{
                'f1': ['c', 'd'],
                'f2': [2, 3]
            }, {
                'f2': [3]
            }]]),
        ], ['w', 'c']),
        pa.RecordBatch.from_arrays([
            pa.array([[3.0]]),
            pa.array([[{
                'f1': ['d'],
                'f2': [4]
            }]]),
        ], ['w', 'c']),
    ]
    expected_result = [
        text_format.Parse(
            """
            features{
              string_stats {
                top_values {
                  value: "d"
                  frequency: 2.0
                }
                top_values {
                  value: "b"
                  frequency: 2.0
                }
                top_values {
                  value: "c"
                  frequency: 1.0
                }
                rank_histogram {
                  buckets {
                    label: "d"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "b"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "c"
                    sample_count: 1.0
                  }
                }
              }
              path {
                step: "c"
                step: "f1"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse(
            """
            features {
              string_stats {
                top_values {
                  value: "3"
                  frequency: 2.0
                }
                top_values {
                  value: "2"
                  frequency: 2.0
                }
                top_values {
                  value: "4"
                  frequency: 1.0
                }
                rank_histogram {
                  buckets {
                    label: "3"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 1
                    high_rank: 1
                    label: "2"
                    sample_count: 2.0
                  }
                  buckets {
                    low_rank: 2
                    high_rank: 2
                    label: "4"
                    sample_count: 1.0
                  }
                }
              }
              path {
                step: "c"
                step: "f2"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse("""
            features {
              string_stats {
                unique: 4
              }
              path {
                step: "c"
                step: "f1"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse("""
            features {
              type: INT
              string_stats {
                unique: 4
              }
              path {
                step: "c"
                step: "f2"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse("""
            features {
              string_stats {
                weighted_string_stats {
                  top_values {
                    value: "d"
                    frequency: 5.0
                  }
                  top_values {
                    value: "c"
                    frequency: 2.0
                  }
                  top_values {
                    value: "b"
                    frequency: 2.0
                  }
                  rank_histogram {
                    buckets {
                      label: "d"
                      sample_count: 5.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "c"
                      sample_count: 2.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "b"
                      sample_count: 2.0
                    }
                  }
                }
              }
              path {
                step: "c"
                step: "f1"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),
        text_format.Parse("""
            features {
              string_stats {
                weighted_string_stats {
                  top_values {
                    value: "3"
                    frequency: 4.0
                  }
                  top_values {
                    value: "4"
                    frequency: 3.0
                  }
                  top_values {
                    value: "2"
                    frequency: 3.0
                  }
                  rank_histogram {
                    buckets {
                      label: "3"
                      sample_count: 4.0
                    }
                    buckets {
                      low_rank: 1
                      high_rank: 1
                      label: "4"
                      sample_count: 3.0
                    }
                    buckets {
                      low_rank: 2
                      high_rank: 2
                      label: "2"
                      sample_count: 3.0
                    }
                  }
                }
              }
              path {
                step: "c"
                step: "f2"
              }
            }""", statistics_pb2.DatasetFeatureStatistics()),

    ]
    schema = text_format.Parse(
        """
        feature {
          name: "c"
          type: STRUCT
          struct_domain {
            feature {
              name: "f2"
              type: INT
              int_domain {
                is_categorical: true
              }
            }
          }
        }
        """, schema_pb2.Schema())
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        schema=schema,
        example_weight_map=ExampleWeightMap(weight_feature='w'),
        num_top_values=3, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        inputs,
        generator,
        expected_result,
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

  def test_schema_claims_categorical_but_actually_float(self):
    schema = text_format.Parse("""
    feature {
      name: "a"
      type: INT
      int_domain { is_categorical: true }
    }""", schema_pb2.Schema())
    inputs = [pa.RecordBatch.from_arrays([
        pa.array([], type=pa.list_(pa.float32()))], ['a'])]
    generator = top_k_uniques_stats_generator.TopKUniquesStatsGenerator(
        schema=schema,
        num_top_values=3, num_rank_histogram_buckets=3)
    self.assertSlicingAwareTransformOutputEqual(
        inputs,
        generator,
        expected_results=[],
        add_default_slice_key_to_input=True,
        add_default_slice_key_to_output=True)

if __name__ == '__main__':
  absltest.main()
