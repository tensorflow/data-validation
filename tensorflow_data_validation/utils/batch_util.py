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

"""Utilities for batching input examples."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import List


def merge_single_batch(batch):
  """Merges batched input examples to proper batch format."""
  batch_size = len(batch)
  result = {}
  for idx, example in enumerate(batch):
    for feature in example.keys():
      if feature not in result.keys():
        # New feature. Initialize the list with None.
        result[feature] = [None] * batch_size
      result[feature][idx] = example[feature]
  return result
