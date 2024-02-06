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
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tensorflow_data_validation.utils import path

# TODO(b/239944944): Eliminate these aliases, and move tests.
FeatureName = path.FeatureName

FeaturePath = path.FeaturePath

FeaturePathTuple = path.FeaturePathTuple

# Type of the feature cross.
FeatureCross = Tuple[FeatureName, FeatureName]

# Feature type enum value.
FeatureNameStatisticsType = int

# Vocab name.
VocabName = Text

# Vocab path.
VocabPath = Text

# Type of slice keys.
SliceKey = Optional[Text]

# Type of list of slice keys.
SliceKeysList = List[SliceKey]

# Type of the tuple containing an input arrow record batch along with the slice
# key.
SlicedRecordBatch = Tuple[SliceKey, pa.RecordBatch]

SliceFunction = Callable[[pa.RecordBatch], Iterable[SlicedRecordBatch]]

# TODO(b/221152546): Deprecate this.
LegacyExample = Dict[FeatureName, Optional[np.ndarray]]

# Do not use multiple threads to encode record batches, as parallelism
# should be managed by beam.
_ARROW_CODER_IPC_OPTIONS = pa.ipc.IpcWriteOptions(use_threads=False)


class PerFeatureStatsConfig:
  """Supports enabling / disabling stats per-feature. Experimental.
  
  NOTE: disabling histograms *also* disables median calculation for numeric
  features.
  """

  INCLUDE = "include"
  EXCLUDE = "exclude"
  histogram_paths: list[FeaturePath]
  histogram_mode: str

  def __init__(
      self,
      histogram_paths: list[FeaturePath],
      histogram_mode: str,
  ):
    self._histogram_paths = set(histogram_paths)
    self._histogram_mode = histogram_mode

  @classmethod
  def default(cls):
    return cls([], PerFeatureStatsConfig.EXCLUDE)

  def should_compute_histograms(self, p: FeaturePath) -> bool:
    if self._histogram_mode == self.INCLUDE:
      return p in self._histogram_paths
    elif self._histogram_mode == self.EXCLUDE:
      return p not in self._histogram_paths
    raise ValueError(
        f"Unknown quantiles histogram mode: {self._histogram_mode}"
    )


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
