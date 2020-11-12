# Copyright 2020 Google LLC
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
"""Tests for tensorflow_data_validation.utils.example_weight_map."""


from absl.testing import absltest
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import example_weight_map


class ExampleWeightMapTest(absltest.TestCase):

  def test_no_weight_feature(self):
    m = example_weight_map.ExampleWeightMap()
    self.assertIsNone(m.get(types.FeaturePath(['feature'])))
    self.assertEmpty(m.all_weight_features())

  def test_only_global_weight_feature(self):
    m = example_weight_map.ExampleWeightMap(weight_feature='w')
    self.assertEqual(m.get(types.FeaturePath(['feature'])), 'w')
    self.assertEqual(m.all_weight_features(), frozenset(['w']))

  def test_per_feature_override(self):
    m = example_weight_map.ExampleWeightMap(
        weight_feature='w',
        per_feature_override={
            types.FeaturePath(['foo']): 'w1',
            types.FeaturePath(['bar']): 'w2'
        })
    self.assertEqual('w1', m.get(types.FeaturePath(['foo'])))
    self.assertEqual('w2', m.get(types.FeaturePath(['bar'])))
    self.assertEqual('w', m.get(types.FeaturePath(['feature'])))
    self.assertEqual(m.all_weight_features(), frozenset(['w', 'w1', 'w2']))

  def test_only_per_feature_override(self):
    m = example_weight_map.ExampleWeightMap(per_feature_override={
        types.FeaturePath(['foo']): 'w1',
    })
    self.assertEqual('w1', m.get(types.FeaturePath(['foo'])))
    self.assertIsNone(m.get(types.FeaturePath(['feature'])))
    self.assertEqual(m.all_weight_features(), frozenset(['w1']))


if __name__ == '__main__':
  absltest.main()
