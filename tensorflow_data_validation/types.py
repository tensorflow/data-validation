"""Types."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np

from tensorflow_data_validation.types_compat import Dict, Text, Union

FeatureName = Union[bytes, Text]

# Type of the input batch.
ExampleBatch = Dict[FeatureName, np.ndarray]

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
BeamFeatureName = beam.typehints.Union[bytes, Text]
# pylint: enable=invalid-name
