# Copyright 2019 Google LLC
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
"""Utility function for generating slicing functions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import functools

import numpy as np
import pandas as pd
import pyarrow as pa
import six
from tensorflow_data_validation import constants
from tensorflow_data_validation import types
from tensorflow_data_validation.arrow import arrow_util
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow import table_util
from typing import Any, Dict, Iterable, Optional, Text, Union

_ValueType = Iterable[Union[Text, int]]

_PARENT_INDEX_COLUMN = '__TFDV_INTERNAL_PARENT_INDEX__'
_SLICE_KEY_COLUMN = '__TFDV_INTERNAL_SLICE_KEY__'


def default_slicer(
    record_batch: pa.RecordBatch) -> Iterable[types.SlicedRecordBatch]:
  """Default slicing function that adds the default slice key to the input."""
  yield (constants.DEFAULT_SLICE_KEY, record_batch)


def get_feature_value_slicer(
    features: Dict[types.FeatureName, Optional[_ValueType]]
) -> types.SliceFunction:
  """Returns a function that generates sliced record batches for a given one.

  The returned function returns sliced record batches based on the combination
  of all features specified in `features`. To slice on features separately (
  e.g., slice on age feature and separately slice on interests feature), you
  must use separate slice functions.

  Examples:
  # Slice on each value of the specified features.
  slice_fn = get_feature_value_slicer(
      features={'age': None, 'interests': None})

  # Slice on a specified feature value.
  slice_fn = get_feature_value_slicer(features={'interests': ['dogs']})

  # Slice on each value of one feature and a specified value of another.
  slice_fn = get_feature_value_slicer(
      features={'fruits': None, 'numbers': [1]})

  Args:
    features: A mapping of features to an optional iterable of values that the
      returned function will slice on. If values is None for a feature, then the
      slice keys will reflect each distinct value found for that feature in the
      input record batch. If values are specified for a feature, then the slice
      keys will reflect only those values for the feature, if found in the input
      record batch. Values must be an iterable of strings or integers.

  Returns:
    A function that takes as input a single record batch and returns a list of
    sliced record batches (slice_key, record_batch).

  Raises:
    TypeError: If feature values are not specified in an iterable.
    NotImplementedError: If a value of a type other than string or integer is
      specified in the values iterable in `features`.
  """
  for values in features.values():
    if values is not None:
      if not isinstance(values, collections.Iterable):
        raise TypeError('Feature values must be specified in an iterable.')
      for value in values:
        if (not isinstance(value, (six.string_types, six.binary_type)) and
            not isinstance(value, int)):
          raise NotImplementedError(
              'Only string and int values are supported as the slice value.')
  # Extract the unique slice values per feature.
  for feature_name in features:
    if features[feature_name] is not None:
      features[feature_name] = set(features[feature_name])

  def feature_value_slicer(record_batch: pa.RecordBatch) -> Iterable[
      types.SlicedRecordBatch]:
    """A function that generates sliced record batches.

    The naive approach of doing this would be to iterate each row, identify
    slice keys for the row and keep track of index ranges for each slice key.
    And then generate an arrow record batch for each slice key based on the
    index ranges. This would be expensive as we are identifying the slice keys
    for each row individually and we would have to loop over the feature values
    including crossing them when we have to slice on multiple features. The
    current approach generates the slice keys for a batch by performing joins
    over indices of individual features. And then groups the joined record batch
    by slice key to get the row indices corresponding to a slice.

    Args:
      record_batch: Arrow RecordBatch.

    Yields:
      Sliced record batch (slice_key, record_batch) where record_batch contains
      the rows corresponding to a slice.
    """
    per_feature_parent_indices = []
    for feature_name, values in six.iteritems(features):
      feature_array = record_batch.column(
          record_batch.schema.get_field_index(feature_name))
      flattened, value_parent_indices = arrow_util.flatten_nested(
          feature_array, True)
      non_missing_values = np.asarray(flattened)
      # Create dataframe with feature value and parent index.
      df = pd.DataFrame({feature_name: non_missing_values,
                         _PARENT_INDEX_COLUMN: value_parent_indices})
      df.drop_duplicates(inplace=True)
      # Filter based on slice values
      if values is not None:
        df = df.loc[df[feature_name].isin(values)]
      per_feature_parent_indices.append(df)

    # Join dataframes based on parent indices.
    # Note that we want the parent indices per slice key to be sorted in the
    # merged dataframe. The individual dataframes have the parent indices in
    # sorted order. We use "inner" join type to preserve the order of the left
    # keys (also note that same parent index rows would be consecutive). Hence
    # we expect the merged dataframe to have sorted parent indices per
    # slice key.
    merged_df = functools.reduce(
        lambda base, update: pd.merge(base, update, how='inner',  # pylint: disable=g-long-lambda
                                      on=_PARENT_INDEX_COLUMN),
        per_feature_parent_indices)

    # Construct a new column in the merged dataframe with the slice keys.
    merged_df[_SLICE_KEY_COLUMN] = ''
    index = 0
    for col_name in sorted(merged_df.columns):
      if col_name in [_PARENT_INDEX_COLUMN, _SLICE_KEY_COLUMN]:
        continue
      slice_key_col = (_to_slice_key(col_name) + '_' +
                       merged_df[col_name].apply(_to_slice_key))
      if index == 0:
        merged_df[_SLICE_KEY_COLUMN] = slice_key_col
        index += 1
      else:
        merged_df[_SLICE_KEY_COLUMN] += ('_' + slice_key_col)

    # Since the parent indices are sorted per slice key, the groupby would
    # preserve the sorted order within each group.
    per_slice_parent_indices = merged_df.groupby(
        _SLICE_KEY_COLUMN, sort=False)[_PARENT_INDEX_COLUMN]
    for slice_key, parent_indices in per_slice_parent_indices:
      yield (slice_key,
             table_util.RecordBatchTake(record_batch,
                                        pa.array(parent_indices.to_numpy())))

  return feature_value_slicer


def _to_slice_key(feature_value: Any):
  """Decode slice key as UTF-8."""
  # For bytes features we try decoding it as utf-8 (and throw an error if
  # fails). This is because in stats proto the slice name (dataset name) is a
  # string field which can only accept valid unicode.
  if isinstance(feature_value, six.binary_type):
    decoded_value = stats_util.maybe_get_utf8(feature_value)
    if decoded_value is None:
      raise ValueError('Feature names and slicing feature values must be valid'
                       ' UTF-8. Found value {}.'.format(feature_value))
    return decoded_value
  return str(feature_value)


def generate_slices(
    record_batch: pa.RecordBatch,
    slice_functions: Iterable[types.SliceFunction], **kwargs
    ) -> Iterable[types.SlicedRecordBatch]:
  """Generates sliced record batches based on provided slice functions.

  Args:
    record_batch: Arrow RecordBatch.
    slice_functions: An iterable of functions each of which takes as input an
      example (and zero or more kwargs) and returns a list of slice keys.
    **kwargs: Keyword arguments to pass to each of the slice_functions.

  Yields:
    Sliced record batch (slice_key, record batch).
  """
  for slice_fn in slice_functions:
    try:
      for sliced_record_batch in slice_fn(record_batch, **kwargs):
        yield sliced_record_batch
    except Exception as e:
      raise ValueError('One of the slice_functions %s raised an exception: %s.'
                       % (slice_fn.__name__, repr(e)))
