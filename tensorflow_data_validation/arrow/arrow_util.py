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
# limitations under the License
"""Util functions regarding to Arrow objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_data_validation.pywrap import pywrap_tensorflow_data_validation as pywrap

# The following are function aliases thus valid function names.
# pylint: disable=invalid-name
FlattenListArray = pywrap.TFDV_Arrow_FlattenListArray
ListLengthsFromListArray = pywrap.TFDV_Arrow_ListLengthsFromListArray
GetFlattenedArrayParentIndices = pywrap.TFDV_Arrow_GetFlattenedArrayParentIndices
GetArrayNullBitmapAsByteArray = pywrap.TFDV_Arrow_GetArrayNullBitmapAsByteArray
GetBinaryArrayTotalByteSize = pywrap.TFDV_Arrow_GetBinaryArrayTotalByteSize
ValueCounts = pywrap.TFDV_Arrow_ValueCounts
MakeListArrayFromParentIndicesAndValues = (
    pywrap.TFDV_Arrow_MakeListArrayFromParentIndicesAndValues)
