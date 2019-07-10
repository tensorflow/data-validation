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

"""Import module for Pyarrow and Tensorflow."""

# We use this module to import Pyarrow or Tensorflow in TFDV. This is needed
# to ensure we import tensorflow and pyarrow in correct order. Importing
# pyarrow first and then tensorflow can result in segmentation fault.
# TODO(b/137200365): Avoid doing this special handling once this is no longer
# necessary.
# Issue: https://issues.apache.org/jira/browse/ARROW-5894

# pylint: disable=unused-import
import tensorflow
import pyarrow  # pylint: disable=g-bad-import-order
# pylint: enable=unused-import
