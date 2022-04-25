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
import os
import pickle
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pyarrow as pa
import tensorflow as tf
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

  def get_formats(self, value_list):
    return np.array([json.loads(value)['format'] for value in value_list],
                    dtype=object)

  def get_sizes(self, value_list):
    loaded_metadata = [json.loads(value) for value in value_list]
    return np.array([[meta['height'], meta['width']]
                     for meta in loaded_metadata])


class ImageStatsGeneratorTest(test_util.CombinerFeatureStatsGeneratorTest,
                              parameterized.TestCase):

  @parameterized.named_parameters(
      ('EmptyList', []),  # Line-break comment for readability.
      ('EmptyBatch', [pa.array([])]),
      ('NumericalShouldInvalidateImageStats', [
          pa.array([[
              FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
              FakeImageDecoder.encode_image_metadata('JPEG', 1, 1),
              FakeImageDecoder.encode_image_metadata('TIFF', 3, 7),
          ]]),
          pa.array([[1]]),
      ]))
  def test_cases_with_no_image_stats(self, batches):
    """Test cases that should not generate image statistics."""
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        values_threshold=1,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator,
                                   statistics_pb2.FeatureNameStatistics())

  def test_image_stats_generator_with_missing_feature(self):
    """Test with missing values for a batch."""
    batches = [
        pa.array([]),
        pa.array([[
            FakeImageDecoder.encode_image_metadata('JPEG', 10, 1),
        ]]),
    ]
    expected_result = text_format.Parse(
        """
            custom_stats {
              name: 'domain_info'
              str: 'image_domain {}'
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
            custom_stats {
              name: 'image_max_width'
              num: 1.0
            }
            custom_stats {
              name: 'image_max_height'
              num: 10.0
            }""", statistics_pb2.FeatureNameStatistics())
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        values_threshold=1,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_values_threshold_check(self):
    """Check values_threshold with a feature that is all images."""
    batches = [
        pa.array([
            [
                FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                FakeImageDecoder.encode_image_metadata('JPEG', 4, 2),
            ],
            [
                FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                FakeImageDecoder.encode_image_metadata('JPEG', -1, -1),
                FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
            ],
        ]),
        pa.array([[
            FakeImageDecoder.encode_image_metadata('GIF', 2, 1),
        ]]),
    ]

    # With values_threshold = 7 statistics should not be generated.
    image_decoder = FakeImageDecoder()
    expected_result = statistics_pb2.FeatureNameStatistics()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        values_threshold=7,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

    # With values_threshold = 6 statistics should be generated.
    expected_result = text_format.Parse(
        """
            custom_stats {
              name: 'domain_info'
              str: 'image_domain {}'
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
            custom_stats {
              name: 'image_max_width'
              num: 7.0
            }
            custom_stats {
              name: 'image_max_height'
              num: 5.0
            }
            """, statistics_pb2.FeatureNameStatistics())
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        values_threshold=6,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_check_is_image_ratio(self):
    """Check is_image_ratio with a feature that has partially images."""
    # The image ratio is: 0.83
    batches = [
        pa.array([
            [
                FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                FakeImageDecoder.encode_image_metadata('JPEG', 4, 2),
            ],
            [
                FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                FakeImageDecoder.encode_image_metadata('', -1, -1),
                FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
            ],
        ]),
        pa.array([[
            FakeImageDecoder.encode_image_metadata('GIF', 2, 1),
        ]]),
    ]
    # For image_ratio_threshold=0.85 we for not expect stats.
    expected_result = statistics_pb2.FeatureNameStatistics()
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        is_image_ratio_threshold=0.85,
        values_threshold=1,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

    # For image_ratio_threshold=0.8 we expect stats.
    expected_result = text_format.Parse(
        """
            custom_stats {
              name: 'domain_info'
              str: 'image_domain {}'
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
            custom_stats {
              name: 'image_max_width'
              num: 7.0
            }
            custom_stats {
              name: 'image_max_height'
              num: 5.0
            }
            """, statistics_pb2.FeatureNameStatistics())
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        is_image_ratio_threshold=0.8,
        values_threshold=1,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_disable_size_stats(self):
    """Test the enable_size_stats_option."""
    # Identical input to test_image_stats_generator_check_is_image_ratio
    batches = [
        pa.array([
            [
                FakeImageDecoder.encode_image_metadata('PNG', 2, 4),
                FakeImageDecoder.encode_image_metadata('JPEG', 4, 2),
            ],
            [
                FakeImageDecoder.encode_image_metadata('TIFF', 5, 1),
                FakeImageDecoder.encode_image_metadata('', -1, -1),
                FakeImageDecoder.encode_image_metadata('TIFF', 3, 7)
            ],
        ]),
        pa.array([[
            FakeImageDecoder.encode_image_metadata('GIF', 2, 1),
        ]]),
    ]
    # Stats should be identical but without stats for image size.
    expected_result = text_format.Parse(
        """
            custom_stats {
              name: 'domain_info'
              str: 'image_domain {}'
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
    image_decoder = FakeImageDecoder()
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        is_image_ratio_threshold=0.8,
        values_threshold=1,
        enable_size_stats=False)
    self.assertCombinerOutputEqual(batches, generator, expected_result)


def _read_file(filepath):
  """Helper method for reading a file in binary mode."""
  f = tf.io.gfile.GFile(filepath, mode='rb')
  return f.read()


class ImageStatsGeneratorRealImageTest(
    test_util.CombinerFeatureStatsGeneratorTest):

  def test_image_stats_generator_real_image(self):
    test_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    batches = [
        pa.array([
            [
                _read_file(os.path.join(test_data_dir, 'image1.gif')),
                _read_file(os.path.join(test_data_dir, 'image2.png')),
                _read_file(os.path.join(test_data_dir, 'image5.jpg')),
                _read_file(os.path.join(test_data_dir, 'image6.jpg')),
                _read_file(os.path.join(test_data_dir, 'not_a_image.abc'))
            ],
            [
                _read_file(os.path.join(test_data_dir, 'image3.bmp')),
                b'not_a_image'
            ],
        ]),
        pa.array([[
            _read_file(os.path.join(test_data_dir, 'image4.png')),
        ]]),
    ]
    expected_result = text_format.Parse(
        """
            custom_stats {
              name: 'domain_info'
              str: 'image_domain {}'
            }
            custom_stats {
              name: 'image_format_histogram'
              rank_histogram {
                buckets {
                  label: 'UNKNOWN'
                  sample_count: 2
                }
                buckets {
                  label: 'bmp'
                  sample_count: 1
                }
                buckets {
                  label: 'gif'
                  sample_count: 1
                }
                buckets {
                  label: 'jpeg'
                  sample_count: 2
                }
                buckets {
                  label: 'png'
                  sample_count: 2
                }
              }
            }
            custom_stats {
              name: 'image_max_width'
              num: 300.0
            }
            custom_stats {
              name: 'image_max_height'
              num: 300.0
            }
            """, statistics_pb2.FeatureNameStatistics())
    generator = image_stats_generator.ImageStatsGenerator(
        is_image_ratio_threshold=0.6,
        values_threshold=1,
        enable_size_stats=True)
    self.assertCombinerOutputEqual(batches, generator, expected_result)

  def test_image_stats_generator_pickle_success(self):
    """Ensure that decoder and generator implementations are pickle-able."""
    image_decoder = image_stats_generator.TfImageDecoder()
    pickle.dumps(image_decoder)
    generator = image_stats_generator.ImageStatsGenerator(
        image_decoder=image_decoder,
        is_image_ratio_threshold=0.6,
        values_threshold=1)
    pickle.dumps(generator)


if __name__ == '__main__':
  absltest.main()
