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

import apache_beam as beam
import numpy as np
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import List, Optional


def merge_single_batch(batch):
  """Merges batched input examples to proper batch format."""
  batch_size = len(batch)
  result = {}
  for idx, example in enumerate(batch):
    for feature in example.keys():
      if feature not in result.keys():
        # New feature. Initialize the list with None.
        result[feature] = np.empty(batch_size, dtype=np.object)
      result[feature][idx] = example[feature]
  return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Example)
@beam.typehints.with_output_types(types.ExampleBatch)
def BatchExamples(  # pylint: disable=invalid-name
    examples,
    desired_batch_size = None):
  """Batches input examples to proper batch format.

  Each input example is a dict of feature name to np.ndarray of feature values.
  The output batched example format is also a dict of feature name to a
  np.ndarray. However, this np.ndarray contains either np.ndarray of feature
  values (if one example have this feature), or np.NaN (if one example is
  missing this feature).

  For example, if two input examples are
  {
    'a': [1, 2, 3],
    'b': ['a', 'b', 'c']
  },
  {
    'a': [4, 5, 6],
  }

  Then the output batched examples will be
  {
    'a': [[1, 2, 3], [4, 5, 6]],
    'b': [['a', 'b', 'c'], np.NaN]
  }

  Args:
    examples: PCollection of examples. Each example should be a dict of
      feature name to a numpy array of values (OK to be empty).
    desired_batch_size: Optional batch size for batching examples when
      computing data statistics.

  Returns:
    PCollection of batched examples.
  """
  batch_args = {}
  if desired_batch_size:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)
  return (examples
          | 'BatchExamples' >> beam.BatchElements(**batch_args)
          | 'MergeBatch' >> beam.Map(merge_single_batch))
