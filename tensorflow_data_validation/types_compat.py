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
"""Types for backwards compatibility with versions that don't support typing."""


from apache_beam.typehints import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union  # pylint: disable=unused-import,g-multiple-import

# pylint: disable=invalid-name
Callable = None
Text = Any
TypeVar = None

# pylint: enable=invalid-name
