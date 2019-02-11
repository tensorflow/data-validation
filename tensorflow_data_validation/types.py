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

from tensorflow_data_validation.types_compat import Callable, Dict, List, Optional, Text, Tuple, Union

# Type of the feature name we support in the input batch.
FeatureName = Union[bytes, Text]

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
