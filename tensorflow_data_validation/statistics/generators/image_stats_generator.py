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
"""Module that computes statistics for features of image format.

Specifically, the following statistics are computed:
- Maximum image heigh and width
- Histogram of example count by image format
- If the rate of recognized formats is high enough and enough examples
  have been considered, features get marked with domain_info: image_domain
  used for schema inference.

The current implementation is using imghdr for identifying image formats
(efficient, based on metadata) and tf.image.decode_image for image height,
width (possibly expensive, performs decoding).
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import collections
import imghdr
import numpy as np
import six

import tensorflow as tf

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import List, Iterable, Text, Tuple
from tensorflow_metadata.proto.v0 import statistics_pb2

_DOMAIN_INFO = 'domain_info'
_IMAGE_DOMAIN = 'image_domain {}'
_IMAGE_MAX_WIDTH_STATISTICS = 'image_max_width'
_IMAGE_MAX_HEIGHT_STATISTICS = 'image_max_height'
_IMAGE_FORMAT_HISTOGRAM = 'image_format_histogram'

# ImageStatsGenerator default initialization values.
_IS_IMAGE_RATIO = 0.8
_EXAMPLES_THRESHOLD = 100


class ImageDecoderInterface(six.with_metaclass(abc.ABCMeta)):
  """Interface for extracting image formats and sizes."""

  @abc.abstractmethod
  def get_formats(self, value_list):
    """Returns the image format name for each value if it represents an image.

    Args:
      value_list: a list of values in bytes to check the image format.

    Returns:
      A list of string image formats (e.g: 'jpeg', 'bmp', ...) or empty string
      if the value is not a supported image, in the same order as the input
      value_list.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_sizes(self, value_list):
    """Returns the image size for each value if it represents an image.

    Args:
      value_list: a list of values in bytes to check the image size.

    Returns:
      A list of (image_height, image_width) tuple (if the value represents an
      image) in the same order as the input value list.
    """
    raise NotImplementedError


class TfImageDecoder(ImageDecoderInterface):
  """ImageDecoderInterface implementation based on tensorflow library.

  This image decoder only supports image formats supported by:
  tf.image.decode_image, ['bmp', 'gif', 'jpeg', 'png'].

  Image sizes are computed using tf.image.decode_image, which requires tf.
  Initializating and pickling tf objects can be non-trivial, so:
  - Initialization is done lazily when get_sizes computation is needed.
  - __reduce__() is overridden so that tf state is ignored. It is lazily
    initialized as needed, after deserialization.
  """

  def __init__(self):
    self._lazy_get_sizes_callable = None

  def __reduce__(self):
    return TfImageDecoder, tuple()

  def _initialize_lazy_get_sizes_callable(self):
    # Initialize the tensorflow graph for decoding images.
    graph = tf.Graph()
    self._session = tf.Session(graph=graph)

    def get_image_shape(value):
      image_shape = tf.shape(tf.image.decode_image(value))
      # decode_image returns a 3-D array ([height, width, num_channels]) for
      # BMP/JPEG/PNG images, but 4-D array ([num_frames, height, width, 3])
      # for GIF images.
      return tf.cond(
          tf.equal(tf.size(image_shape), 4),
          lambda: image_shape[1:3],
          lambda: image_shape[0:2],
      )

    with self._session.graph.as_default(), self._session.as_default():
      self._batch_image_input = tf.placeholder(dtype=tf.string, shape=[None])
      self._image_shapes = tf.map_fn(
          get_image_shape,
          elems=self._batch_image_input,
          dtype=tf.int32,
          infer_shape=False)
      graph.finalize()
    self._lazy_get_sizes_callable = self._session.make_callable(
        fetches=self._image_shapes, feed_list=[self._batch_image_input])

  def get_formats(self, value_list):
    """Returns the image format name for each value if it represents an image.

    Args:
      value_list: a list of value in bytes to check the image format.

    Returns:
      A list of image format name (e.g. 'JPG'/'GIF'/etc, or empty string if the
      value is not an image) in the same order as the input value list.
    """
    # Extract all formats.
    formats = [
        # Ensure input is bytes, in py3 imghdr.what requires bytes input.
        # TODO(b/126249134): If TFDV provided a guarantee that image features
        # are never represented as py3 strings, strings should be skipped
        # instead of manually casting to bytes here.
        imghdr.what(None,
                    v.encode('utf-8') if isinstance(v, six.text_type) else v)
        for v in value_list
    ]

    # Replace non supported formats by ''.
    # Note: Using str(f) to help pytype infer types.
    return [
        str(f) if f in ('bmp', 'gif', 'jpeg', 'png') else '' for f in formats
    ]

  def get_sizes(self, value_list):
    """Returns the image size for each value if it represents an image.

    Args:
      value_list: a list of value in bytes to check the image size.

    Returns:
      A list of (image_height, image_width) tuple (if the value represents an
      image) in the same order as the input value list.

    Raises:
      ValueError: If any of the input value does not represents an image.
    """
    if not self._lazy_get_sizes_callable:
      self._initialize_lazy_get_sizes_callable()
    image_shapes = self._lazy_get_sizes_callable(value_list)
    # Transform shape from list to tuple.
    return [tuple(shape) for shape in image_shapes]


class _PartialImageStats(object):
  """Partial feature stats for images.

  Attributes:
    total_num_values: The total number of values processed for this feature.
    max_width: The maximum image width among all the values for this feature.
    max_height: The maximum image height among all the values for this feature.
    counter_by_format: A dict from image format string to the number of images
      in this format. The format / key '' is used for non supported.
    invalidate: True only if this feature should never be considered, e.g: some
      value_lists have inconsistent formats.
  """

  def __init__(self):
    self.total_num_values = 0
    self.max_width = 0
    self.max_height = 0
    self.counter_by_format = collections.Counter()
    self.invalidate = False

  def __iadd__(self, other):
    """Merge two partial image stats."""
    self.total_num_values += other.total_num_values
    self.max_width = max(self.max_width, other.max_width)
    self.max_height = max(self.max_height, other.max_height)
    self.counter_by_format += other.counter_by_format
    self.invalidate |= other.invalidate
    return self


class ImageStatsGenerator(stats_generator.CombinerFeatureStatsGenerator):
  """Computes the statistics for features of image format."""

  def __init__(self,
               image_decoder = None,
               name = 'ImageStatsGenerator',
               is_image_ratio_threshold = _IS_IMAGE_RATIO,
               examples_threshold = _EXAMPLES_THRESHOLD,
               enable_size_stats = False):
    """Initializes an image statistics generator.

    Args:
      image_decoder: ImageDecoderInterface instance for fetching image metadata.
      name: The unique name associated with this statistics generator.
      is_image_ratio_threshold: In order for a feature to be considered "image"
        type and respective stats to be generated, at least this ratio of values
        should be supported images.
      examples_threshold: In order for a feature to be considered "image" type
        and respective stats to be generated, at least so many values should be
        considered.
      enable_size_stats: If True statistics about image sizes are generated.
        This currently requires decoding through TF that could have performance
        implications.
    """
    super(ImageStatsGenerator, self).__init__(name)
    if image_decoder is None:
      image_decoder = TfImageDecoder()
    self._image_decoder = image_decoder
    self._is_image_ratio_threshold = is_image_ratio_threshold
    self._examples_threshold = examples_threshold
    self._enable_size_stats = enable_size_stats

  def create_accumulator(self):
    """Return a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    return _PartialImageStats()

  def add_input(self, accumulator,
                input_batch):
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      input_batch: A list representing a batch of feature value_lists (one per
        example) which should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    if accumulator.invalidate:
      return accumulator

    if self._enable_size_stats:
      # A list of images to decode in order to get image size stats.
      # This is done separately as a performance optimization to avoid costs
      # of using tf. The values that have valid formats is being kept in
      # images_to_decode and a single call to .get_sizes is performed for the
      # full input_batch.
      images_to_decode = []

    for value_list in input_batch:
      if value_list is None or value_list.size == 0:
        continue

      # Check if the numpy array is of bytes type, if not invalidate the stats.
      # in examples/features to run image stas gen on.
      if stats_util.get_feature_type(
          value_list.dtype) != statistics_pb2.FeatureNameStatistics.STRING:
        accumulator.invalidate = True
        return accumulator

      accumulator.total_num_values += len(value_list)
      image_formats = self._image_decoder.get_formats(value_list)
      for (value, image_format) in zip(value_list, image_formats):
        accumulator.counter_by_format[image_format] += 1
        if self._enable_size_stats and image_format:
          images_to_decode.append(value)

    # Update the partial image stats.
    if self._enable_size_stats and images_to_decode:
      image_sizes = self._image_decoder.get_sizes(images_to_decode)
      # Update the max image height/width with all image values.
      accumulator.max_height = max(accumulator.max_height,
                                   max(x[0] for x in image_sizes))
      accumulator.max_width = max(accumulator.max_width,
                                  max(x[1] for x in image_sizes))
    return accumulator

  def merge_accumulators(
      self, accumulators):
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    result = _PartialImageStats()
    for accumulator in accumulators:
      result += accumulator
    return result

  def extract_output(self, accumulator
                    ):
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.FeatureNameStatistics()
    # Only generate an image statistics proto if the ratio of image feature
    # values is above a threshold.
    if (accumulator.invalidate or
        accumulator.total_num_values < self._examples_threshold or
        (1 - (float(accumulator.counter_by_format['']) /
              accumulator.total_num_values)) < self._is_image_ratio_threshold):
      return result

    result.custom_stats.add(name=_DOMAIN_INFO, str=_IMAGE_DOMAIN)
    # Image format histogram.
    custom_stats = result.custom_stats.add(name=_IMAGE_FORMAT_HISTOGRAM)

    # Add the buckets with sorted image format.
    for image_format in sorted(accumulator.counter_by_format):
      custom_stats.rank_histogram.buckets.add(
          label=image_format if image_format else 'UNKNOWN',
          sample_count=accumulator.counter_by_format[image_format])

    if self._enable_size_stats:
      result.custom_stats.add(
          name=_IMAGE_MAX_WIDTH_STATISTICS, num=accumulator.max_width)
      result.custom_stats.add(
          name=_IMAGE_MAX_HEIGHT_STATISTICS, num=accumulator.max_height)
    return result
