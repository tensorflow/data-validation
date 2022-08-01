# Copyright 2022 Google LLC
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
"""Path type definition."""

import json
from typing import Iterable, Tuple
from tensorflow_metadata.proto.v0 import path_pb2


# Type of the feature name we support in the input batch.
FeatureName = str

# Type of FeaturePath.steps(). Pickling types.FeaturePath is slow, so we use
# tuples directly where pickling happens frequently. Ellipsis due to
# b/152929669.
FeaturePathTuple = Tuple[FeatureName, ...]


class FeaturePath(object):
  """Represents the path to a feature in an input example.

  An input example might contain nested structure. FeaturePath is to identify
  a node in such a structure.
  """

  __slot__ = ["_steps"]

  def __init__(self, steps: Iterable[FeatureName]):
    self._steps = tuple(steps)

  def to_proto(self) -> path_pb2.Path:
    return path_pb2.Path(step=self._steps)

  def to_json(self) -> str:
    return json.dumps(self._steps)

  @staticmethod
  def from_proto(path_proto: path_pb2.Path):
    return FeaturePath(path_proto.step)

  @staticmethod
  def from_json(path_json: str):
    steps = json.loads(path_json)
    if not isinstance(steps, list):
      raise TypeError("Invalid FeaturePath json: %s" % path_json)
    for s in steps:
      if not isinstance(s, str):
        raise TypeError("Invalid FeaturePath json: %s" % path_json)
    return FeaturePath(steps)

  @staticmethod
  def from_string(path_string: str):
    steps = path_string.split(".")
    return FeaturePath(steps)

  def steps(self) -> FeaturePathTuple:
    return self._steps

  def parent(self) -> "FeaturePath":
    if not self._steps:
      raise ValueError("Root does not have parent.")
    return FeaturePath(self._steps[:-1])

  def child(self, child_step: FeatureName) -> "FeaturePath":
    return FeaturePath(self._steps + (child_step,))

  def __str__(self) -> str:
    return ".".join(self._steps)

  def __repr__(self) -> str:
    return self._steps.__repr__()

  def __eq__(self, other) -> bool:
    return self._steps == other._steps  # pylint: disable=protected-access

  def __lt__(self, other) -> bool:
    # lexicographic order.
    return self._steps < other._steps  # pylint: disable=protected-access

  def __hash__(self) -> int:
    return hash(self._steps)

  def __len__(self) -> int:
    return len(self._steps)

  def __bool__(self) -> bool:
    return bool(self._steps)
