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
"""ExampleWeightMap."""

from typing import FrozenSet, Mapping, Optional

from tensorflow_data_validation import types


# Implementation notes:
# For now this map is essentially as a defaultdict, but in the future we may
# want to implement more semantics for nested structures (for example, if
# an override for path ["x", "y"] if specified, then any children of that path
# should share the same override).
class ExampleWeightMap(object):
  """Maps a feature path to its weight feature.

  This map can be created with a "global" weight feature and path-specific
  overrides. For any given FeaturePath, its weight column is the override, if
  specified, or the "global" one.
  """

  def __init__(
      self,
      weight_feature: Optional[types.FeatureName] = None,
      per_feature_override: Optional[Mapping[types.FeaturePath,
                                             types.FeatureName]] = None):
    self._weight_feature = weight_feature
    self._per_feature_override = per_feature_override
    all_weight_features = []
    if self._per_feature_override is not None:
      all_weight_features.extend(self._per_feature_override.values())
    if self._weight_feature is not None:
      all_weight_features.append(self._weight_feature)
    self._all_weight_features = frozenset(all_weight_features)

  def get(self, feature_path: types.FeaturePath) -> Optional[types.FeatureName]:
    if self._per_feature_override is None:
      return self._weight_feature
    override = self._per_feature_override.get(feature_path)
    return self._weight_feature if override is None else override

  def all_weight_features(self) -> FrozenSet[types.FeatureName]:
    return self._all_weight_features
