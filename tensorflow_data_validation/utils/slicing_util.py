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
"""Utility function for generating slicing functions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import itertools
import six

from tensorflow_data_validation import types

from tensorflow_data_validation.types_compat import Generator, Iterable, Mapping, Optional, Text, Tuple, Union

_ValueType = Iterable[Union[Text, int]]


def get_feature_value_slicer(
    features
):
  """Returns a function that generates a list of slice keys for a given example.

  The returned function returns slice keys based on the combination of all
  features specified in `features`. To slice on features separately (e.g., slice
  on age feature and separately slice on interests feature), you must use
  separate slice functions.

  Examples:
  # Slice on each value of the specified features.
  slice_fn = get_feature_value_slicer(
      features={'age': None, 'interests': None})
  example = {'age': np.array([10]), 'interests': np.array(['dogs', 'cats'])}
  slice_fn(example) = ['age_10_interests_cats', 'age_10_interests_dogs']

  # Slice on a specified feature value.
  slice_fn = get_feature_value_slicer(features={'interests': ['dogs']})
  example = {'age': np.array([10]), 'interests': np.array(['dogs', 'cats'])}
  slice_fn(example) = ['interests_dogs']

  # Slice on each value of one feature and a specified value of another.
  slice_fn = get_feature_value_slicer(
      features={'fruits': None, 'numbers': [1]})
  example_with_integer_numbers = {
      'fruits': np.array(['apples', 'pears']),
      'numbers': np.array([1, 2])}
  slice_fn(example_with_integer_numbers) = [
      'fruits_apples_numbers_1', 'fruits_pears_numbers_1']
  example_with_string_numbers = {
      'fruits': np.array(['apples', 'pears']),
      'numbers': np.array(['1', '2'])}
  # Type is taken into account when matching values. This returns an empty list
  # because 'numbers' does not contain an integer value of 1 in
  # example_with_string_numbers.
  slice_fn(example_with_string_numbers) = []

  Args:
    features: A mapping of features to an optional iterable of values that the
      returned function will slice on. If values is None for a feature, then the
      slice keys will reflect each distinct value found for that feature in the
      example. If values are specified for a feature, then the slice keys will
      reflect only those values for the feature, if found in the example. Values
      must be an iterable of strings or integers.

  Returns:
    A function that takes as input a single example and returns a list of
    zero or more slice keys.

  Raises:
    TypeError: If feature values are not specified in an iterable.
    NotImplementedError: If a value of a type other than string or integer is
      specified in the values iterable in `features`.
  """
  for values in features.values():
    if values is not None:
      if not isinstance(values, collections.Iterable):
        raise TypeError('Feature values must be specified in an iterable.')
      for value in values:
        if not isinstance(value, six.string_types) and not isinstance(
            value, int):
          raise NotImplementedError(
              'Only string and int values are supported as the slice value.')

  def feature_value_slicer(example):
    """A function that generates a list of slice keys for a given example."""
    per_feature_value_matches = []

    for feature_name, values in features.items():
      if feature_name not in example:
        # If the example does not have a specified feature, do not return any
        # slice keys.
        return []
      values_in_example = set(example[feature_name])
      if values is not None:
        for value in values:
          if value not in values_in_example:
            # If the example does not have a specified feature value, the
            # resulting function should not return any slice keys.
            return []
          else:
            # This match is appended within a list so that we can use
            # itertools.product to get the combinations of matches.
            per_feature_value_matches.append(
                ['_'.join([feature_name, str(value)])])
      else:
        # If a feature was specified without values, include each distinct
        # value found for that feature in the slice keys.
        per_feature_value_matches.append([
            '_'.join([feature_name, str(value)])
            for value in values_in_example
        ])

    slice_keys = []
    if per_feature_value_matches:
      # Take the Cartesian product of the lists of matching values per feature
      # to get all feature-value combinations. The result is sorted so that the
      # slice key generated for a given combination is deterministic.
      for feature_value_combination in itertools.product(
          *per_feature_value_matches):
        slice_keys.append('_'.join(sorted(list(feature_value_combination))))
    return slice_keys

  return feature_value_slicer


def generate_slices(
    example, slice_functions,
    **kwargs):
  """Yields (slice_key, example) tuples based on provided slice functions.

  Args:
    example: An input example.
    slice_functions: An iterable of functions each of which takes as input an
      example (and zero or more kwargs) and returns a list of slice keys.
    **kwargs: Keyword arguments to pass to each of the slice_functions.

  Yields:
    A (slice_key, example) tuple for each slice_key in the combined list
    of slice keys that the slice_functions return.
  """
  for slice_function in slice_functions:
    try:
      for slice_key in slice_function(example, **kwargs):
        yield (slice_key, example)
    except Exception as e:
      raise ValueError('One of the slice_functions %s raised an exception: %s.'
                       % (slice_function.__name__, repr(e)))
