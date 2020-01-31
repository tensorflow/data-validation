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
"""Types."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np
import pyarrow as pa
import six
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

from tensorflow_metadata.proto.v0 import path_pb2

# Type for representing a CSV record and a field value.
CSVRecord = Union[bytes, Text]
CSVCell = Union[bytes, Text]

# Type of the feature name we support in the input batch.
FeatureName = Union[bytes, Text]

# Type of the feature cross.
FeatureCross = Tuple[FeatureName, FeatureName]

# Feature type enum value.
FeatureNameStatisticsType = int

# Type of the input example.
Example = Dict[FeatureName, Optional[np.ndarray]]

# Type of batched values.
ValueBatch = List[Optional[np.ndarray]]

# Type of the input batch.
ExampleBatch = Dict[FeatureName, ValueBatch]

# Type of slice keys.
SliceKey = Optional[Union[bytes, Text]]

# Type of list of slice keys.
SliceKeysList = List[SliceKey]

# Type of the tuple containing an input example along with the slice key.
SlicedExample = Tuple[SliceKey, Example]

# Type of the tuple containing an input arrow table along with the slice key.
SlicedTable = Tuple[SliceKey, pa.Table]

# Type of the function that is used to generate a list of slice keys for a given
# example. This function should take the form: slice_fn(example, **kwargs) ->
# SliceKeysList.
# TODO(b/110842625): Replace with SliceFunction = Callable[..., SliceKeysList]
# once we no longer replace Callable in types_compat.
SliceFunction = Callable

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
# TODO(b/111217539): Remove this once Beam supports arbitrary Python types
# to be used in the type annotations.
BeamFeatureName = beam.typehints.Union[bytes, Text]
BeamExample = beam.typehints.Dict[BeamFeatureName, beam.typehints
                                  .Optional[np.ndarray]]
BeamSliceKey = beam.typehints.Optional[beam.typehints.Union[bytes, Text]]
BeamSlicedTable = beam.typehints.Tuple[BeamSliceKey, pa.Table]
BeamCSVRecord = beam.typehints.Union[bytes, Text]
BeamCSVCell = beam.typehints.Union[bytes, Text]

# pa.Column has been removed since pyarrow 0.15. (Table.data returns a
# ChunkedArray instead, but a ChunkedArray has all the methods that
# pa.Column used to have for compatibility reasons.)
# TODO(b/142894895): remove the version switch once tfx-bsl starts requiring
# pyarrow>=0.15.
if pa.__version__ >= "0.15":
  ArrowColumn = pa.ChunkedArray
else:
  ArrowColumn = pa.Column


@six.python_2_unicode_compatible
class FeaturePath(object):
  """Represents the path to a feature in an input example.

  An input example might contain nested structure. FeaturePath is to identify
  a node in such a structure.
  """

  __slot__ = ["_steps"]

  def __init__(self, steps: Iterable[FeatureName]):
    self._steps = tuple(
        s if isinstance(s, six.text_type) else s.decode("utf-8") for s in steps)

  def to_proto(self) -> path_pb2.Path:
    return path_pb2.Path(step=self._steps)

  @staticmethod
  def from_proto(path_proto: path_pb2.Path):
    return FeaturePath(path_proto.step)

  def steps(self) -> Tuple[FeatureName]:
    return self._steps

  def parent(self) -> "FeaturePath":
    if not self._steps:
      raise ValueError("Root does not have parent.")
    return FeaturePath(self._steps[:-1])

  def child(self, child_step: FeatureName) -> "FeaturePath":
    if isinstance(child_step, six.text_type):
      return FeaturePath(self._steps + (child_step,))
    return FeaturePath(self._steps + (child_step.decode("utf-8"),))

  def __str__(self) -> Text:
    return u".".join(self._steps)

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
