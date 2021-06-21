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
import json
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa

from tensorflow_metadata.proto.v0 import path_pb2

# Type of the feature name we support in the input batch.
FeatureName = Text

# Type of the feature cross.
FeatureCross = Tuple[FeatureName, FeatureName]

# Feature type enum value.
FeatureNameStatisticsType = int

# Vocab name.
VocabName = Text

# Vocab path.
VocabPath = Text

# Type of the input example.
Example = Dict[FeatureName, Optional[np.ndarray]]

# Type of batched values.
ValueBatch = List[Optional[np.ndarray]]

# Type of the input batch.
ExampleBatch = Dict[FeatureName, ValueBatch]

# Type of slice keys.
SliceKey = Optional[Text]

# Type of list of slice keys.
SliceKeysList = List[SliceKey]

# Type of the tuple containing an input example along with the slice key.
SlicedExample = Tuple[SliceKey, Example]

# Type of the tuple containing an input arrow record batch along with the slice
# key.
SlicedRecordBatch = Tuple[SliceKey, pa.RecordBatch]

SliceFunction = Callable[[pa.RecordBatch], Iterable[SlicedRecordBatch]]

# Type of FeaturePath.steps(). Pickling types.FeaturePath is slow, so we use
# tuples directly where pickling happens frequently. Ellipsis due to
# b/152929669.
FeaturePathTuple = Tuple[FeatureName, ...]

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
# TODO(b/111217539): Remove this once Beam supports arbitrary Python types
# to be used in the type annotations.
BeamExample = beam.typehints.Dict[FeatureName, beam.typehints
                                  .Optional[np.ndarray]]
BeamSliceKey = beam.typehints.Optional[Text]
BeamSlicedRecordBatch = beam.typehints.Tuple[BeamSliceKey, pa.RecordBatch]


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

  def to_json(self) -> Text:
    return json.dumps(self._steps)

  @staticmethod
  def from_proto(path_proto: path_pb2.Path):
    return FeaturePath(path_proto.step)

  @staticmethod
  def from_json(path_json: Text):
    steps = json.loads(path_json)
    if not isinstance(steps, list):
      raise TypeError("Invalid FeaturePath json: %s" % path_json)
    for s in steps:
      if not isinstance(s, str):
        raise TypeError("Invalid FeaturePath json: %s" % path_json)
    return FeaturePath(steps)

  def steps(self) -> FeaturePathTuple:
    return self._steps

  def parent(self) -> "FeaturePath":
    if not self._steps:
      raise ValueError("Root does not have parent.")
    return FeaturePath(self._steps[:-1])

  def child(self, child_step: FeatureName) -> "FeaturePath":
    return FeaturePath(self._steps + (child_step,))

  def __str__(self) -> Text:
    return ".".join(self._steps)

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


# Do not use multiple threads to encode record batches, as parallelism
# should be managed by beam.
_ARROW_CODER_IPC_OPTIONS = pa.ipc.IpcWriteOptions(use_threads=False)


# TODO(b/190756453): Make this into the upstream
# (preference: Arrow, Beam, tfx_bsl).
class _ArrowRecordBatchCoder(beam.coders.Coder):
  """Custom coder for Arrow record batches."""

  def encode(self, value: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(
        sink, value.schema, options=_ARROW_CODER_IPC_OPTIONS)
    writer.write_batch(value)
    writer.close()
    return sink.getvalue().to_pybytes()

  def decode(self, encoded: bytes) -> pa.RecordBatch:
    reader = pa.ipc.open_stream(encoded)
    result = reader.read_next_batch()
    try:
      reader.read_next_batch()
    except StopIteration:
      pass
    else:
      raise ValueError("Expected only one RecordBatch in the stream.")
    return result

  def to_type_hint(self):
    return pa.RecordBatch


beam.coders.typecoders.registry.register_coder(pa.RecordBatch,
                                               _ArrowRecordBatchCoder)
