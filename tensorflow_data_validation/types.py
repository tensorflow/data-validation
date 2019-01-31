"""Types."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np

from tensorflow_data_validation.types_compat import Callable, Dict, List, Optional, Text, Union

# Type of the feature name we support in the input batch.
FeatureName = Union[bytes, Text]

# Feature type enum value.
FeatureNameStatisticsType = int

# Type of the input example.
Example = Dict[FeatureName, Optional[np.ndarray]]

# Type of the input batch.
ExampleBatch = Dict[FeatureName, np.ndarray]

# Type of slice keys.
SliceKey = Union[bytes, Text]

# Type of list of slice keys.
SliceKeysList = List[SliceKey]

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
BeamSliceKey = beam.typehints.Union[bytes, Text]
