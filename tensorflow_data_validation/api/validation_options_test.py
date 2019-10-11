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
"""Tests for validation_options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensorflow_data_validation.api import validation_options
from tensorflow_data_validation.types import FeaturePath


class ValidationOptionsTest(absltest.TestCase):

  def test_access_attributes(self):
    features_needed = {
        FeaturePath(['a', 'b']): [
            validation_options.ReasonFeatureNeeded(comment='reason1'),
            validation_options.ReasonFeatureNeeded(comment='reason2')
        ]
    }
    new_features_are_warnings = True
    severity_overrides = []
    options = validation_options.ValidationOptions(features_needed,
                                                   new_features_are_warnings,
                                                   severity_overrides)

    # Test getters
    self.assertEqual(features_needed, options.features_needed)
    self.assertEqual(new_features_are_warnings,
                     options.new_features_are_warnings)
    self.assertEqual(severity_overrides, options.severity_overrides)


if __name__ == '__main__':
  absltest.main()
