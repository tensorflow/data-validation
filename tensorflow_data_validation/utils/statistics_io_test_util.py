# Copyright 2022 Google LLC
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
"""Provides test runner for io tests."""


def test_runner_impl():
  """Beam runner for testing record_sink_impl.

  Usage:

  with beam.Pipeline(runner=test_runner_impl()):
     ...

  Defined only for tests.

  Returns:
    A beam runner.
  """
  return None  # default runner.
