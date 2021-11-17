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
# ==============================================================================
"""Validation options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Mapping, Text
from tensorflow_data_validation.anomalies.proto import validation_config_pb2
from tensorflow_data_validation.types import FeaturePath

# TODO(https://issues.apache.org/jira/browse/SPARK-22674): Switch to
# `collections.namedtuple` or `typing.NamedTuple` once the Spark issue is
# resolved.
from tfx_bsl.types import tfx_namedtuple  # pylint: disable=g-bad-import-order


class ReasonFeatureNeeded(
    tfx_namedtuple.namedtuple('ReasonFeatureNeeded', ['comment'])):
  """A named tuple to indicate why a feature is needed for struct2tensor."""

  def __new__(cls, comment: Text):
    return super(ReasonFeatureNeeded, cls).__new__(cls, comment=comment)


class ValidationOptions(object):
  """Options for example validation."""

  def __init__(
      self,
      features_needed: Optional[Mapping[FeaturePath,
                                        List[ReasonFeatureNeeded]]] = None,
      new_features_are_warnings: Optional[bool] = False,
      severity_overrides: Optional[List[
          validation_config_pb2.SeverityOverride]] = None):
    self._features_needed = features_needed
    self._new_features_are_warnings = new_features_are_warnings
    self._severity_overrides = severity_overrides or []

  @property
  def features_needed(
      self) -> Optional[Mapping[FeaturePath, List[ReasonFeatureNeeded]]]:
    return self._features_needed

  @property
  def new_features_are_warnings(self) -> bool:
    return self._new_features_are_warnings

  @property
  def severity_overrides(self) -> List[validation_config_pb2.SeverityOverride]:
    return self._severity_overrides
