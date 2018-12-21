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
"""Tests for the slicing utilities."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_data_validation.utils import slicing_util

GENERATE_SLICER_WITH_ALL_UNIVALENT_FEATURES_TESTS = [
    {
        'testcase_name': 'feature_value_does_not_match',
        'features': {'age': [99]},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': []
    },
    {
        'testcase_name': 'no_such_feature_in_example',
        'features': {'no_such_feature': None},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': []
    },
    {
        'testcase_name': 'feature_without_value',
        'features': {'age': None},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': ['age_5']
    },
    {
        'testcase_name': 'feature_with_value',
        'features': {'age': [5]},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': ['age_5']
    },
    {
        'testcase_name': 'value_type_mismatch',
        'features': {'age': ['5']},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': []
    },
    {
        'testcase_name': 'features_with_and_without_values',
        'features': {'gender': None, 'age': [5]},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': ['age_5_gender_f']
    },
    {
        'testcase_name': 'multiple_features_with_and_without_values',
        'features': {'interests': None,
                     'gender': None,
                     'age': [5],
                     'fruits': ['apples']},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars']),
                    'fruits': np.array(['apples'])},
        'expected_slice_keys': ['age_5_fruits_apples_gender_f_interests_cars']
    },
]
GENERATE_SLICER_WITH_MULTIVALENT_FEATURES_TESTS = [
    {
        'testcase_name': 'feature_without_values',
        'features': {'fruits': None},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars', 'dogs']),
                    'fruits': np.array(['apples', 'apples', 'pears'])},
        'expected_slice_keys': ['fruits_apples', 'fruits_pears']
    },
    {
        'testcase_name':
            'multiple_features_without_values',
        'features': {'fruits': None, 'interests': None},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars', 'dogs']),
                    'fruits': np.array(['apples', 'apples', 'pears'])},
        'expected_slice_keys': [
            'fruits_apples_interests_cars',
            'fruits_apples_interests_dogs',
            'fruits_pears_interests_cars',
            'fruits_pears_interests_dogs'
        ]
    },
    {
        'testcase_name': 'feature_with_values',
        'features': {'interests': ['cars']},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars', 'dogs']),
                    'fruits': np.array(['apples', 'apples', 'pears'])},
        'expected_slice_keys': ['interests_cars']
    },
    {
        'testcase_name': 'multiple_features_with_values',
        'features': {'interests': ['cars'], 'gender': ['f']},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars', 'dogs']),
                    'fruits': np.array(['apples', 'apples', 'pears'])},
        'expected_slice_keys': [
            'gender_f_interests_cars'
        ]
    },
    {
        'testcase_name':
            'multiple_features_with_and_without_values',
        'features': {'fruits': None,
                     'interests': None,
                     'age': [5],
                     'gender': ['f']},
        'example': {'gender': np.array(['f']),
                    'age': np.array([5]),
                    'interests': np.array(['cars', 'dogs']),
                    'fruits': np.array(['apples', 'apples', 'pears'])},
        'expected_slice_keys': [
            'age_5_fruits_apples_gender_f_interests_cars',
            'age_5_fruits_pears_gender_f_interests_cars',
            'age_5_fruits_apples_gender_f_interests_dogs',
            'age_5_fruits_pears_gender_f_interests_dogs'
        ]
    },
]


class SlicingUtilTest(parameterized.TestCase):

  def test_get_feature_value_slicer_with_float_feature_value(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Only string and int.*as the slice value.'):
      slicing_util.get_feature_value_slicer({'feature_name': [1.1]})

  def test_get_feature_value_slicer_with_values_not_in_iterable(self):
    with self.assertRaisesRegexp(TypeError, 'Feature values must be specified '
                                 'in an iterable.'):
      slicing_util.get_feature_value_slicer({'feature_name': 1})

  @parameterized.named_parameters(
      *GENERATE_SLICER_WITH_ALL_UNIVALENT_FEATURES_TESTS)
  def test_get_feature_value_slicer_for_univalent_features(
      self, features, example, expected_slice_keys):
    slicer_function = slicing_util.get_feature_value_slicer(features)
    actual_slice_keys = slicer_function(example)
    self.assertCountEqual(expected_slice_keys, actual_slice_keys)

  @parameterized.named_parameters(
      *GENERATE_SLICER_WITH_MULTIVALENT_FEATURES_TESTS)
  def test_get_feature_value_slicer_for_multivalent_features(
      self, features, example, expected_slice_keys):
    slicer_function = slicing_util.get_feature_value_slicer(features)
    actual_slice_keys = slicer_function(example)
    self.assertCountEqual(expected_slice_keys, actual_slice_keys)

  def test_generate_slices_without_kwargs(self):

    def slice_function_1(example):
      if example.get('test_column_1'):
        return ['test_slice_key_1']

    def slice_function_2(example):
      if example.get('test_column_2'):
        return ['test_slice_key_2']

    input_example = {
        'test_column_1': np.array([1]),
        'test_column_2': np.array(['a'])
    }

    expected_result = [('test_slice_key_1', input_example),
                       ('test_slice_key_2', input_example)]
    # Each slice function returns a list of one slice key.
    actual_result = list(
        slicing_util.generate_slices(input_example,
                                     [slice_function_1, slice_function_2]))
    self.assertCountEqual(expected_result, actual_result)

  def test_generate_slices_with_kwargs(self):

    def slice_function_1(example, **kwargs):
      slice_keys = []
      test_num = kwargs['test_num']
      if example.get('test_column_1'):
        slice_keys.append('test_slice_key_a_' + str(test_num))
      if example.get('test_column_2'):
        slice_keys.append('test_slice_key_b_' + str(test_num))
      return slice_keys

    def slice_function_2(example, **kwargs):  # pylint: disable=unused-argument
      return []

    input_example = {
        'test_column_1': np.array([1]),
        'test_column_2': np.array([2])
    }

    expected_result = [('test_slice_key_a_1', input_example),
                       ('test_slice_key_b_1', input_example)]
    test_kwargs = {'test_num': 1}
    # slice_function_1 returns a list of multiple slice keys, and
    # slice_function_2 returns an empty list.
    actual_result = list(
        slicing_util.generate_slices(
            input_example, [slice_function_1, slice_function_2], **test_kwargs))
    self.assertCountEqual(expected_result, actual_result)

  def test_generate_slices_bad_slice_function(self):

    def bad_slice_function(example):  # pylint: disable=unused-argument
      return 1 / 0

    input_example = {'test_column': np.array([1])}

    with self.assertRaisesRegexp(
        ValueError, 'One of the slice_functions '
        'bad_slice_function raised an exception: '
        'ZeroDivisionError.*'):
      list(slicing_util.generate_slices(input_example, [bad_slice_function]))


if __name__ == '__main__':
  absltest.main()
