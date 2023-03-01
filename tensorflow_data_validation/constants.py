# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in TensorFlow Data Validation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx_bsl.telemetry import util


# Name of the default slice containing all examples.
# LINT.IfChange
DEFAULT_SLICE_KEY = 'All Examples'
# LINT.ThenChange(../anomalies/custom_validation.cc)

# Name of the invalid slice containing all examples in the RecordBatch.
INVALID_SLICE_KEY = 'Invalid Slice'

# Namespace for all TFDV metrics.
METRICS_NAMESPACE = util.MakeTfxNamespace(['DataValidation'])

# Default input batch size.
# This needs to be large enough to allow for efficient TF invocations during
# batch flushing, but shouldn't be too large as it also acts as cap on the
# maximum memory usage of the computation.
DEFAULT_DESIRED_INPUT_BATCH_SIZE = 1000

# Placeholder for non-utf8 sequences in top-k results.
NON_UTF8_PLACEHOLDER = '__BYTES_VALUE__'
# Placeholder for large sequences in top-k results.
LARGE_BYTES_PLACEHOLDER = '__LARGE_BYTES__'
