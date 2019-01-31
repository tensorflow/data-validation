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
"""Module that computes statistics for features of image type.

Specifically, we compute the following image statistics:
  - Max image height
  - Max image width
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import collections
import logging
import numpy as np
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics.generators import stats_generator
from tensorflow_data_validation.utils import stats_util
from tensorflow_data_validation.types_compat import Dict, List, Text, Tuple

from tensorflow_metadata.proto.v0 import statistics_pb2

_IS_IMAGE_STATISTICS = 'is_image'
_MAX_IMAGE_WIDTH_STATISTICS = 'max_image_width'
_MAX_IMAGE_HEIGHT_STATISTICS = 'max_image_height'
_IMAGE_FORMAT_HISTOGRAM = 'image_format_histogram'


class ImageDecoderInterface(six.with_metaclass(abc.ABCMeta)):
  """Util class interface for getting image content metadata."""

  @abc.abstractmethod
  def get_format(self, content):
    """Returns the image type name if the content represents an image.

    Args:
      content: content in bytes to check the image format.

    Returns:
      Image type name (e.g. 'JPG'/'GIF'/etc) if the content represents an
      image. Otherwise returns an empty string.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_size(self, content):
    """Returns the image size.

    Args:
      content: content in bytes to get the image size.

    Returns:
      A (image_height, image_width) tuple if the content represents an
      image.
    """
    raise NotImplementedError


class _PartialImageStats(object):
  """Partial statistics for computing the image statistics for a single feature.
  """

  def __init__(self):
    # A dict from image format string to the number of images in this format.
    self.counter_by_format = collections.Counter()
    # The total number of values processed for this feature.
    self.total_num_values = 0
    # The maximum image width among all the values for this feature.
    self.max_image_width = 0
    # The maximum image height among all the values for this feature.
    self.max_image_height = 0


def _update_partial_image_stats(
    image_stats,
    value,
    image_decoder):
  """Updates the partial image statistics using the input value."""
  for v in value:
    image_stats.total_num_values += 1
    image_type = image_decoder.get_format(v)
    image_stats.counter_by_format[image_type] += 1

    # If this is actually an image, update the max image height/width.
    if image_type:
      image_height, image_width = image_decoder.get_size(v)
      image_stats.max_image_height = max(image_stats.max_image_height,
                                         image_height)
      image_stats.max_image_width = max(image_stats.max_image_width,
                                        image_width)


def _merge_image_stats(
    left, right):
  """Merges two partial image statistics and return the merged statistics."""
  result = _PartialImageStats()
  result.total_num_values = left.total_num_values + right.total_num_values
  result.max_image_width = max(left.max_image_width, right.max_image_width)
  result.max_image_height = max(left.max_image_height, right.max_image_height)
  result.counter_by_format = left.counter_by_format + right.counter_by_format
  return result


def _make_feature_stats_proto(
    image_stats,
    feature_name):
  """Converts the partial image statistics into FeatureNameStatistics proto."""
  result = statistics_pb2.FeatureNameStatistics()
  result.name = feature_name

  # TODO(mingzhong): Need to add set any other fields?

  result.custom_stats.add(name=_IS_IMAGE_STATISTICS, num=1)

  # Max image width stats.
  result.custom_stats.add(
      name=_MAX_IMAGE_WIDTH_STATISTICS, num=image_stats.max_image_width)

  # Max image height stats.
  result.custom_stats.add(
      name=_MAX_IMAGE_HEIGHT_STATISTICS, num=image_stats.max_image_height)

  # Image type histogram.
  custom_stats = result.custom_stats.add()
  custom_stats.name = _IMAGE_FORMAT_HISTOGRAM

  # Add the buckets with sorted image type.
  for image_type in sorted(image_stats.counter_by_format):
    custom_stats.rank_histogram.buckets.add(
        label=image_type if image_type else 'UNKNOWN',
        sample_count=image_stats.counter_by_format[image_type])
  return result


class ImageStatsGenerator(stats_generator.CombinerStatsGenerator):
  """Computes the statistics for features of image type.

  """

  def __init__(self,
               image_decoder,
               name = 'ImageStatsGenerator',
               is_image_ratio_threshold = 0.8):
    """Initializes an image statistics generator.

    Args:
      image_decoder: ImageDecoderInterface instance for fetching image metadata.
      name: An optional unique name associated with the statistics generator.
      is_image_ratio_threshold: An optional threshold for checking if a feature
        should be treated as image feature. If the percentage of image features
        values for a feature is below this threshold, we won't generate the
        image statistics for this feature.
    """
    super(ImageStatsGenerator, self).__init__(name)
    self._image_decoder = image_decoder
    self._is_image_ratio_threshold = is_image_ratio_threshold

  # Creates an accumulator, which maps feature name to the partial image stats
  # associated with the feature.
  def create_accumulator(self):
    return {}

  # Incorporates the input (a Python dict whose keys are feature names and
  # values are numpy arrays representing a batch of examples) into the
  # accumulator.
  def add_input(self, accumulator,
                input_batch
               ):
    # Iterate through each feature and update the partial image stats.
    for feature_name, values in six.iteritems(input_batch):
      # TODO(mingzhong): Maybe skip features specified as categorical/numerical
      # features in schema.
      for value in values:
        # Check if we have a numpy arary with at least one value.
        if not isinstance(value, np.ndarray) or value.size == 0:
          continue

        # Check if the numpy array is of bytes type -- we skip checking image
        # statistics for numeric data.
        #
        # Note that it is intentional to not include count this in
        # total_num_values, as tfdv assumes that all values for the same feature
        # have the same numpy array dtype.
        # TODO(b/120138003): Run image stats gen for only a sample of examples.
        # TODO(b/118066476): Make image stas gen accept a function for filtering
        # in examples/features to run image stas gen on.
        if stats_util.get_feature_type(
            value.dtype) != statistics_pb2.FeatureNameStatistics.STRING:
          continue

        # If we encounter this feature for the first time, create a new partial
        # numeric stats.
        if feature_name not in accumulator:
          accumulator[feature_name] = _PartialImageStats()

        # Update the partial image stats.
        _update_partial_image_stats(accumulator[feature_name], value,
                                    self._image_decoder)
    return accumulator

  # Merges together a list of partial image statistics.
  def merge_accumulators(
      self, accumulators
  ):
    result = {}
    for accumulator in accumulators:
      for feature_name, image_stats in accumulator.items():
        if feature_name not in result:
          result[feature_name] = image_stats
        else:
          result[feature_name] = _merge_image_stats(result[feature_name],
                                                    image_stats)
    return result

  # Return final stats as a DatasetFeatureStatistics proto.
  def extract_output(self,
                     accumulator
                    ):
    # Create a new DatasetFeatureStatistics proto.
    result = statistics_pb2.DatasetFeatureStatistics()

    for feature_name, image_stats in six.iteritems(accumulator):
      # Only generate an image statistics proto if the ratio of image feature
      # values is above a threshold.
      if image_stats.total_num_values <= 0 or (
          1 - (image_stats.counter_by_format[''] /
               image_stats.total_num_values)) < self._is_image_ratio_threshold:
        logging.warning(
            'Ratio of image values for feature %s is below threshold',
            feature_name)
        continue

      feature_stats_proto = _make_feature_stats_proto(image_stats, feature_name)
      new_feature_stats_proto = result.features.add()
      new_feature_stats_proto.CopyFrom(feature_stats_proto)
    return result

