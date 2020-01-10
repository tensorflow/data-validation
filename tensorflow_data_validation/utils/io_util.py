# Copyright 2020 Google LLC
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
"""IO utilities."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf
from typing import Text, Union


def write_string_to_file(filename: Text, file_content: Text) -> None:
  """Writes a string to a given file.

  Args:
    filename: path to a file.
    file_content: contents that need to be written to the file.
  """
  with tf.io.gfile.GFile(filename, mode="w") as f:
    f.write(file_content)


def read_file_to_string(filename: Text,
                        binary_mode: bool = False) -> Union[Text, bytes]:
  """Reads the entire contents of a file to a string.

  Args:
    filename: path to a file
    binary_mode: whether to open the file in binary mode or not. This changes
      the type of the object returned.

  Returns:
    contents of the file as a string or bytes.
  """
  if binary_mode:
    f = tf.io.gfile.GFile(filename, mode="rb")
  else:
    f = tf.io.gfile.GFile(filename, mode="r")
  return f.read()
