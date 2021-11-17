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
"""Utilities for manipulating an Anomalies proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, FrozenSet, List, Text, Tuple
import pyarrow as pa
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import io_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2

# LINT.IfChange
MULTIPLE_ERRORS_SHORT_DESCRIPTION = 'Multiple errors'


def _make_updated_descriptions(
    reasons: List[anomalies_pb2.AnomalyInfo.Reason]) -> Tuple[Text, Text]:
  """Returns descriptions based on the specified reasons."""
  # If we only have one reason, use its descriptions. Alternatively, if the only
  # reasons for the anomaly are of type SCHEMA_NEW_COLUMN, then just use one of
  # them for the description.
  if len(reasons) == 1 or all(
      reason.type == anomalies_pb2.AnomalyInfo.SCHEMA_NEW_COLUMN
      for reason in reasons):
    return (reasons[0].description, reasons[0].short_description)
  else:
    return (' '.join([reason.description for reason in reasons]),
            MULTIPLE_ERRORS_SHORT_DESCRIPTION)
# LINT.ThenChange(../anomalies/schema_anomalies.cc)


def remove_anomaly_types(
    anomalies: anomalies_pb2.Anomalies,
    types_to_remove: FrozenSet['anomalies_pb2.AnomalyInfo.Type']) -> None:
  """Removes the specified types of anomaly reasons from an Anomalies proto.

  If all reasons for a given feature's anomalies are removed, the entire feature
  will be removed from the Anomalies proto.

  Args:
    anomalies: The Anomalies proto from which to remove anomaly reasons of
      the specified types.
    types_to_remove: A set of the types of reasons to remove.
  """
  features_to_remove = []
  for feature_name, anomaly_info in anomalies.anomaly_info.items():
    retained_reasons = [
        reason for reason in anomaly_info.reason
        if reason.type not in types_to_remove
    ]

    # Clear the diff regions entirely since we do not have a way of readily
    # separating the comparisons that are attributable to the retained reasons
    # from those that are attributable to the removed reasons.
    anomaly_info.ClearField('diff_regions')
    anomaly_info.ClearField('reason')
    if retained_reasons:
      # If there are anomaly reasons that are retained, update the anomaly info
      # for the feature to include only those retained reasons.
      anomaly_info.reason.extend(retained_reasons)
      # Replace the description and short_description based on the reasons that
      # are retained.
      (updated_description,
       updated_short_description) = _make_updated_descriptions(retained_reasons)
      anomaly_info.description = updated_description
      anomaly_info.short_description = updated_short_description

    else:
      # If there are no anomaly types that are retained for a given feature,
      # remove that feature from the anomaly_info map entirely.
      features_to_remove.append(feature_name)

  for feature_name in features_to_remove:
    del anomalies.anomaly_info[feature_name]


def get_anomalies_slicer(
    anomalies: anomalies_pb2.Anomalies) -> types.SliceFunction:
  """Returns a SliceFunction that For each anomaly, yields (anomaly, example).

  Args:
    anomalies: An Anomalies proto from which to generate the list of slice keys.
  """
  def slice_fn(example: pa.RecordBatch) -> Iterable[types.SlicedRecordBatch]:
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
      for anomaly_reason in anomaly_info.reason:
        yield (feature_name + '_' +
               anomalies_pb2.AnomalyInfo.Type.Name(anomaly_reason.type),
               example)

  return slice_fn


def write_anomalies_text(anomalies: anomalies_pb2.Anomalies,
                         output_path: Text) -> None:
  """Writes the Anomalies proto to a file in text format.

  Args:
    anomalies: An Anomalies protocol buffer.
    output_path: File path to which to write the Anomalies proto.

  Raises:
    TypeError: If the input Anomalies proto is not of the expected type.
  """
  if not isinstance(anomalies, anomalies_pb2.Anomalies):
    raise TypeError(
        'anomalies is of type %s; should be an Anomalies proto.' %
        type(anomalies).__name__)

  anomalies_text = text_format.MessageToString(anomalies)
  io_util.write_string_to_file(output_path, anomalies_text)


def load_anomalies_text(input_path: Text) -> anomalies_pb2.Anomalies:
  """Loads the Anomalies proto stored in text format in the input path.

  Args:
    input_path: File path from which to load the Anomalies proto.

  Returns:
    An Anomalies protocol buffer.
  """
  anomalies = anomalies_pb2.Anomalies()
  anomalies_text = io_util.read_file_to_string(input_path)
  text_format.Parse(anomalies_text, anomalies)
  return anomalies


def load_anomalies_binary(input_path: Text) -> anomalies_pb2.Anomalies:
  """Loads the Anomalies proto stored in binary format in the input path.

  Args:
    input_path: File path from which to load the Anomalies proto.

  Returns:
    An Anomalies protocol buffer.
  """
  anomalies_proto = anomalies_pb2.Anomalies()

  anomalies_proto.ParseFromString(io_util.read_file_to_string(
      input_path, binary_mode=True))

  return anomalies_proto
