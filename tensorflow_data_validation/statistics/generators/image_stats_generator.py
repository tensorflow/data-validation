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
- Histogram of value count by image format
- If the rate of recognized formats is high enough and enough values
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
from typing import Iterable, List, Optional, Text
import numpy as np
import pandas as pd
import pyarrow as pa
import six
import tensorflow as tf

from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import array_util
from tensorflow_metadata.proto.v0 import statistics_pb2

_DOMAIN_INFO = 'domain_info'
_IMAGE_DOMAIN = 'image_domain {}'
_IMAGE_MAX_WIDTH_STATISTICS = 'image_max_width'
_IMAGE_MAX_HEIGHT_STATISTICS = 'image_max_height'
_IMAGE_FORMAT_HISTOGRAM = 'image_format_histogram'

# ImageStatsGenerator default initialization values.
_IS_IMAGE_RATIO = 0.8
_VALUES_THRESHOLD = 100

# Magic bytes (hex) signature for each image format.
# Source: https://en.wikipedia.org/wiki/List_of_file_signatures.
_IMAGE_FORMAT_SIGNATURES = {
    'bmp': b'\x42\x4d',
    'gif': b'\x47\x49\x46\x38',
    # The 4th byte of JPEG is '\xe0' or '\xe1', so check just the first three.
    'jpeg': b'\xff\xd8\xff',
    'png': b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
}


class ImageDecoderInterface(six.with_metaclass(abc.ABCMeta)):
  """Interface for extracting image formats and sizes."""

  @abc.abstractmethod
  def get_formats(self, values: np.ndarray) -> np.ndarray:
    """Returns the image format name for each value if it represents an image.

    Args:
      values: a list of values in bytes to check the image format.

    Returns:
      A list of string image formats (e.g: 'jpeg', 'bmp', ...) or None
      if the value is not a supported image, in the same order as the input
      value_list.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_sizes(self, values: np.ndarray) -> np.ndarray:
    """Returns the image size for each value if it represents an image.

    Args:
      values: a list of values in bytes to check the image size.

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

  def __init__(self):  # pylint: disable=super-init-not-called
    self._lazy_get_sizes_callable = None

  def __reduce__(self):
    return TfImageDecoder, tuple()

  def _initialize_lazy_get_sizes_callable(self):
    # Initialize the tensorflow graph for decoding images.
    graph = tf.Graph()
    self._session = tf.compat.v1.Session(graph=graph)

    def get_image_shape(value):
      image_shape = tf.shape(input=tf.image.decode_image(value))
      # decode_image returns a 3-D array ([height, width, num_channels]) for
      # BMP/JPEG/PNG images, but 4-D array ([num_frames, height, width, 3])
      # for GIF images.
      return tf.cond(
          pred=tf.equal(tf.size(input=image_shape), 4),
          true_fn=lambda: image_shape[1:3],
          false_fn=lambda: image_shape[0:2],
      )

    with self._session.graph.as_default(), self._session.as_default():
      self._batch_image_input = tf.compat.v1.placeholder(
          dtype=tf.string, shape=[None])
      self._image_shapes = tf.map_fn(
          get_image_shape,
          elems=self._batch_image_input,
          dtype=tf.int32,
          infer_shape=False)
      graph.finalize()
    self._lazy_get_sizes_callable = self._session.make_callable(
        fetches=self._image_shapes, feed_list=[self._batch_image_input])

  def get_formats(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, values: List[object]) -> np.ndarray:
    """Returns the image format name for each value if it represents an image.

    Args:
      values: a list of value in bytes to check the image format.

    Returns:
      A list of image format name (e.g. 'JPG'/'GIF'/etc, or None if the
      value is not an image) in the same order as the input value list.
    """
    def get_image_format(image_bytes):
      for image_format, signature in _IMAGE_FORMAT_SIGNATURES.items():
        if bytes(image_bytes[:len(signature)]) == signature:
          return image_format
      return None
    return np.vectorize(get_image_format, otypes=[object])(values)

  def get_sizes(self, values: np.ndarray) -> np.ndarray:
    """Returns the image size for each value if it represents an image.

    Args:
      values: a list of value in bytes to check the image size.

    Returns:
      A numpy array containing (image_height, image_width) tuples (if the value
        represents an image) in the same order as the input value list.

    Raises:
      ValueError: If any of the input value does not represents an image.
    """
    if not self._lazy_get_sizes_callable:
      self._initialize_lazy_get_sizes_callable()
    assert self._lazy_get_sizes_callable is not None
    return self._lazy_get_sizes_callable(values)


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

  def __iadd__(self, other: '_PartialImageStats') -> '_PartialImageStats':
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
               image_decoder: Optional[ImageDecoderInterface] = None,
               name: Text = 'ImageStatsGenerator',
               is_image_ratio_threshold: float = _IS_IMAGE_RATIO,
               values_threshold: int = _VALUES_THRESHOLD,
               enable_size_stats: bool = False):
    """Initializes an image statistics generator.

    Args:
      image_decoder: ImageDecoderInterface instance for fetching image metadata.
      name: The unique name associated with this statistics generator.
      is_image_ratio_threshold: In order for a feature to be considered "image"
        type and respective stats to be generated, at least this ratio of values
        should be supported images.
      values_threshold: In order for a feature to be considered "image" type
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
    self._values_threshold = values_threshold
    self._enable_size_stats = enable_size_stats

  def create_accumulator(self) -> _PartialImageStats:
    """Return a fresh, empty accumulator.

    Returns:
      An empty accumulator.
    """
    return _PartialImageStats()

  def add_input(self, accumulator: _PartialImageStats,
                feature_path: types.FeaturePath,
                feature_array: pa.Array) -> _PartialImageStats:
    """Return result of folding a batch of inputs into accumulator.

    Args:
      accumulator: The current accumulator.
      feature_path: The path of the feature.
      feature_array: An arrow array representing a batch of feature values
        which should be added to the accumulator.

    Returns:
      The accumulator after updating the statistics for the batch of inputs.
    """
    if accumulator.invalidate:
      return accumulator
    feature_type = stats_util.get_feature_type_from_arrow_type(
        feature_path, feature_array.type)
    # Ignore null array.
    if feature_type is None:
      return accumulator
    # If we see a different type, invalidate.
    if feature_type != statistics_pb2.FeatureNameStatistics.STRING:
      accumulator.invalidate = True
      return accumulator

    # Consider using memoryview to avoid copying after upgrading to
    # arrow 0.12. Note that this would involve modifying the subsequent logic
    # to iterate over the values in a loop.
    values = np.asarray(array_util.flatten_nested(feature_array)[0])
    accumulator.total_num_values += values.size
    image_formats = self._image_decoder.get_formats(values)
    valid_mask = ~pd.isnull(image_formats)
    valid_formats = image_formats[valid_mask]
    format_counts = np.unique(valid_formats, return_counts=True)
    for (image_format, count) in zip(*format_counts):
      accumulator.counter_by_format[image_format] += count
    unknown_count = image_formats.size - valid_formats.size
    if unknown_count > 0:
      accumulator.counter_by_format[''] += unknown_count

    if self._enable_size_stats:
      # Get image height and width.
      image_sizes = self._image_decoder.get_sizes(values[valid_mask])
      if image_sizes.any():
        max_sizes = np.max(image_sizes, axis=0)
        # Update the max image height/width with all image values.
        accumulator.max_height = max(accumulator.max_height, max_sizes[0])
        accumulator.max_width = max(accumulator.max_width, max_sizes[1])

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_PartialImageStats]) -> _PartialImageStats:
    """Merges several accumulators to a single accumulator value.

    Args:
      accumulators: The accumulators to merge.

    Returns:
      The merged accumulator.
    """
    it = iter(accumulators)
    result = next(it)
    for accumulator in it:
      result += accumulator
    return result

  def extract_output(self, accumulator: _PartialImageStats
                    ) -> statistics_pb2.FeatureNameStatistics:
    """Return result of converting accumulator into the output value.

    Args:
      accumulator: The final accumulator value.

    Returns:
      A proto representing the result of this stats generator.
    """
    result = statistics_pb2.FeatureNameStatistics()
    # Only generate an image statistics proto if the ratio of image feature
    # values is at or above a threshold.
    if (accumulator.invalidate or
        accumulator.total_num_values < self._values_threshold or
        (1 - (float(accumulator.counter_by_format['']) /
              accumulator.total_num_values)) < self._is_image_ratio_threshold):
      return result

    result.custom_stats.add(name=_DOMAIN_INFO, str=_IMAGE_DOMAIN)
    # Image format histogram.
    custom_stats = result.custom_stats.add(name=_IMAGE_FORMAT_HISTOGRAM)

    # Add the buckets with sorted image format.
    for image_format in sorted(accumulator.counter_by_format):
      custom_stats.rank_histogram.buckets.add(
          # LINT.IfChange
          label=image_format if image_format else 'UNKNOWN',
          # image_domain_util.cc relies on unsupported image formats being
          # being assigned to the bucket labeled 'UNKNOWN'. If this labeling
          # changes, change the corresponding code accordingly.
          # LINT.ThenChange(../../anomalies/image_domain_util.cc)
          sample_count=accumulator.counter_by_format[image_format])
    if self._enable_size_stats:
      result.custom_stats.add(
          name=_IMAGE_MAX_WIDTH_STATISTICS, num=accumulator.max_width)
      result.custom_stats.add(
          name=_IMAGE_MAX_HEIGHT_STATISTICS, num=accumulator.max_height)
    return result
