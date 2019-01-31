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
"""Tests for image statistics generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl.testing import absltest
import numpy as np
from tensorflow_data_validation.statistics.generators import image_stats_generator
from tensorflow_data_validation.utils import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class FakeImageDecoder(image_stats_generator.ImageDecoderInterface):
  """Fake ImageDecoderInterface implementation for testing."""

  @staticmethod
  def encode_image_metadata(image_format, image_height, image_width):
    image_metadata = {
        'format': image_format,
        'height': image_height,
        'width': image_width
    }
    return json.dumps(image_metadata)

  def get_format(self, content):
    image_metadata = json.loads(content)
    return image_metadata['format']

  def get_size(self, content):
    image_metadata = json.loads(content)
    return (image_metadata['height'], image_metadata['width'])


# TODO(b/119735769): use parameterized test case here.
class ImageStatsGeneratorTest(test_util.CombinerStatsGeneratorTest):

  # Input batch only having one feature, and all feature values are images.
  def test_image_stats_generator_single_feature_all_images(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                        FakeImageDecoder.encode_image_metadata('JPEG', 4, 2)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                        FakeImageDecoder.encode_image_metadata('JPEG', 1, 1),
                        FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
                    ])
                ])
        },
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('GIF', 2, 1)
                    ])
                ])
        }]
    expected_result = {
        'a':
            text_format.Parse(
                """
            name: 'a'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 7.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 5.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'GIF'
                  sample_count: 1
                }
                buckets {
                  label: 'JPEG'
                  sample_count: 2
                }
                buckets {
                  label: 'PNG'
                  sample_count: 1
                }
                buckets {
                  label: 'TIFF'
                  sample_count: 2
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  # Input batch only having one feature, and one of the value is not an image
  # type (image_type being an empty string). The ratio of image value is
  # above 0.8, thus the feature is still an image feature.
  def test_image_stats_generator_single_feature_one_non_image(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                        FakeImageDecoder.encode_image_metadata('JPEG', 4, 2)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                        FakeImageDecoder.encode_image_metadata('', -1, -1),
                        FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
                    ])
                ])
        },
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('GIF', 2, 1)
                    ])
                ])
        }]
    expected_result = {
        'a':
            text_format.Parse(
                """
            name: 'a'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 7.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 5.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'UNKNOWN'
                  sample_count: 1
                }
                buckets {
                  label: 'GIF'
                  sample_count: 1
                }
                buckets {
                  label: 'JPEG'
                  sample_count: 1
                }
                buckets {
                  label: 'PNG'
                  sample_count: 1
                }
                buckets {
                  label: 'TIFF'
                  sample_count: 2
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder, is_image_ratio_threshold=0.8)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  # Input batch only having one feature, and many of the values are not an image
  # type (image_type being an empty string). The ratio of image value is
  # below 0.8, thus the feature is still an image feature.
  def test_image_stats_generator_single_feature_many_non_images(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                        FakeImageDecoder.encode_image_metadata('', -1, -1)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                        FakeImageDecoder.encode_image_metadata('', -1, -1),
                        FakeImageDecoder.encode_image_metadata('', -1, -1)
                    ])
                ])
        },
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('GIF', 2, 1)
                    ])
                ])
        }]
    expected_result = {}
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder, is_image_ratio_threshold=0.8)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_multiple_features(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                        FakeImageDecoder.encode_image_metadata('JPEG', 4, 2)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                        FakeImageDecoder.encode_image_metadata('JPEG', 1, 1),
                        FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
                    ])
                ]),
            'b':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('HEIF', 2, 4)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1)
                    ])
                ])
        },
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('GIF', 2, 1)
                    ])
                ]),
            'b':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('JPEG', 10, 1)
                    ])
                ])
        }]
    expected_result = {
        'a':
            text_format.Parse(
                """
            name: 'a'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 7.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 5.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'GIF'
                  sample_count: 1
                }
                buckets {
                  label: 'JPEG'
                  sample_count: 2
                }
                buckets {
                  label: 'PNG'
                  sample_count: 1
                }
                buckets {
                  label: 'TIFF'
                  sample_count: 2
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b':
            text_format.Parse(
                """
            name: 'b'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 4.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 10.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'HEIF'
                  sample_count: 1
                }
                buckets {
                  label: 'JPEG'
                  sample_count: 1
                }
                buckets {
                  label: 'TIFF'
                  sample_count: 1
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_with_missing_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example. The first batch is missing feature 'b'
    batches = [
        {
            'a':
                np.array([
                    np.array([
                        FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                        FakeImageDecoder.encode_image_metadata('JPEG', 4, 2)
                    ]),
                    np.array([
                        FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                        FakeImageDecoder.encode_image_metadata('JPEG', 1, 1),
                        FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
                    ])
                ])
        },
        {
            'a':
                np.array([
                    np.array(
                        [FakeImageDecoder.encode_image_metadata('GIF', 2, 1)])
                ]),
            'b':
                np.array([
                    np.array(
                        [FakeImageDecoder.encode_image_metadata('JPEG', 10, 1)])
                ])
        }
    ]
    expected_result = {
        'a':
            text_format.Parse(
                """
            name: 'a'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 7.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 5.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'GIF'
                  sample_count: 1
                }
                buckets {
                  label: 'JPEG'
                  sample_count: 2
                }
                buckets {
                  label: 'PNG'
                  sample_count: 1
                }
                buckets {
                  label: 'TIFF'
                  sample_count: 2
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics()),
        'b':
            text_format.Parse(
                """
            name: 'b'
            custom_stats {
              name: 'is_image'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_width'
              num: 1.0
            }
            custom_stats {
              name: 'max_image_height'
              num: 10.0
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'JPEG'
                  sample_count: 1
                }
              }
            }
            """, statistics_pb2.FeatureNameStatistics())
    }
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_numerical_feature(self):
    # input with two batches: first batch has two examples and second batch
    # has a single example.
    batches = [{
        'a': np.array([np.array([1, 0]), np.array([0, 1, 0])])
    }, {
        'a': np.array([np.array([1])])
    }]
    expected_result = {}
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_empty_batch(self):
    batches = [{'a': np.array([])}]
    expected_result = {}
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_empty_dict(self):
    batches = [{}]
    expected_result = {}
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_empty_list(self):
    batches = []
    expected_result = {}
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder)
    self.assertCombinerOutputEqual(batches, generator, expected_result)


if __name__ == '__main__':
  absltest.main()
