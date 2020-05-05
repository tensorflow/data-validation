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

# Note:
# 1. Before open source, "from typing import" will be replaced by
#    "from tensorflow_data_validation.types_compat import" via copybara.
# 2. During development:
# (a) Any types import from typing directly should be defined within this file.
# (b) For types supported by typing only, e.g. Callable, all related codes
#     should be removed by copybara and/or pystrip, except for the import line
#     which will be dealt by note 1.

from apache_beam.typehints import Any, Dict, Generator, Iterable, Iterator, List, Optional, Set, Tuple, TypeVariable, Union  # pylint: disable=unused-import,g-multiple-import

# pylint: disable=invalid-name
Callable = None
FrozenSet = None
Hashable = None
Mapping = None
Pattern = None
Text = Any
TypeVar = TypeVariable

# pylint: enable=invalid-name
