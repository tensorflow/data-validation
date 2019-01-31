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

from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import List, Set, Text, Tuple
from tensorflow_metadata.proto.v0 import anomalies_pb2

# LINT.IfChange
MULTIPLE_ERRORS_SHORT_DESCRIPTION = 'Multiple errors'


def _make_updated_descriptions(
    reasons):
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
    anomaly_proto,
    types_to_remove):
  """Removes the specified types of anomaly reasons from an Anomalies proto.

  If all reasons for a given feature's anomalies are removed, the entire feature
  will be removed from the Anomalies proto.

  Args:
    anomaly_proto: The Anomalies proto from which to remove anomaly reasons of
      the specified types.
    types_to_remove: A set of the types of reasons to remove.
  """
  features_to_remove = []
  for feature_name, anomaly_info in anomaly_proto.anomaly_info.items():
    retained_reasons = [
        reason for reason in anomaly_info.reason
        if reason.type not in types_to_remove
    ]

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
    del anomaly_proto.anomaly_info[feature_name]


def anomalies_slicer(
    unused_example,
    anomaly_proto):
  """Returns slice keys for an example based on the given Anomalies proto.

  This slicer will generate a slice key for each anomaly reason in the proto.

  Args:
    unused_example: The example for which to generate slice keys.
    anomaly_proto: An Anomalies proto from which to generate the list of slice
      keys.

  Returns:
    A list of slice keys.
  """
  slice_keys = []
  for feature_name, anomaly_info in anomaly_proto.anomaly_info.items():
    for anomaly_reason in anomaly_info.reason:
      slice_keys.append(
          feature_name + '_' +
          anomalies_pb2.AnomalyInfo.Type.Name(anomaly_reason.type))
  return slice_keys
