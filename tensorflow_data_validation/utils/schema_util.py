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
"""Utilities for manipulating the schema."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import logging
import six
from tensorflow_data_validation import types
from tensorflow_data_validation.types_compat import List, Union
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2


def get_feature(schema,
                feature_name):
  """Get a feature from the schema.

  Args:
    schema: A Schema protocol buffer.
    feature_name: The name of the feature to obtain from the schema.

  Returns:
    A Feature protocol buffer.

  Raises:
    TypeError: If the input schema is not of the expected type.
    ValueError: If the input feature is not found in the schema.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  for feature in schema.feature:
    if feature.name == feature_name:
      return feature

  raise ValueError('Feature %s not found in the schema.' % feature_name)




def get_domain(schema,
               feature_name):
  """Get the domain associated with the input feature from the schema.

  Args:
    schema: A Schema protocol buffer.
    feature_name: The name of the feature whose domain needs to be found.

  Returns:
    The domain protocol buffer (one of IntDomain, FloatDomain, StringDomain or
        BoolDomain) associated with the input feature.

  Raises:
    TypeError: If the input schema is not of the expected type.
    ValueError: If the input feature is not found in the schema or there is
        no domain associated with the feature.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  feature = get_feature(schema, feature_name)
  domain_info = feature.WhichOneof('domain_info')

  if domain_info is None:
    raise ValueError(
        'Feature %s has no domain associated with it.' % feature_name)

  if domain_info == 'int_domain':
    return feature.int_domain
  elif domain_info == 'float_domain':
    return feature.float_domain
  elif domain_info == 'string_domain':
    return feature.string_domain
  elif domain_info == 'domain':
    for domain in schema.string_domain:
      if domain.name == feature.domain:
        return domain
  elif domain_info == 'bool_domain':
    return feature.bool_domain

  raise ValueError('Feature %s has an unsupported domain %s.'
                   % (feature_name, domain_info))


def set_domain(schema,
               feature_name,
               domain):
  """Sets the domain for the input feature in the schema.

  If the input feature already has a domain, it is overwritten with the newly
  provided input domain. This method cannot be used to add a new global domain.

  Args:
    schema: A Schema protocol buffer.
    feature_name: The name of the feature whose domain needs to be set.
    domain: A domain protocol buffer (one of IntDomain, FloatDomain,
        StringDomain or BoolDomain) or the name of a global string domain
        present in the input schema.

  Example:

  ```python
    >>> from tensorflow_metadata.proto.v0 import schema_pb2
    >>> import tensorflow_data_validation as tfdv
    >>> schema = schema_pb2.Schema()
    >>> schema.feature.add(name='feature')
    # Setting a int domain.
    >>> int_domain = schema_pb2.IntDomain(min=3, max=5)
    >>> tfdv.set_domain(schema, "feature", int_domain)
    # Setting a string domain.
    >>> str_domain = schema_pb2.StringDomain(value=['one', 'two', 'three'])
    >>> tfdv.set_domain(schema, "feature", str_domain)
  ```

  Raises:
    TypeError: If the input schema or the domain is not of the expected type.
    ValueError: If an invalid global string domain is provided as input.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  if not isinstance(domain, (schema_pb2.IntDomain, schema_pb2.FloatDomain,
                             schema_pb2.StringDomain, schema_pb2.BoolDomain,
                             six.string_types)):
    raise TypeError('domain is of type %s, should be one of IntDomain, '
                    'FloatDomain, StringDomain, BoolDomain proto or a string '
                    'denoting the name of a global domain in the schema.' %
                    type(domain).__name__)

  feature = get_feature(schema, feature_name)

  if feature.WhichOneof('domain_info') is not None:
    logging.warning('Replacing existing domain of feature "%s".', feature_name)

  if isinstance(domain, schema_pb2.IntDomain):
    feature.int_domain.CopyFrom(domain)
  elif isinstance(domain, schema_pb2.FloatDomain):
    feature.float_domain.CopyFrom(domain)
  elif isinstance(domain, schema_pb2.StringDomain):
    feature.string_domain.CopyFrom(domain)
  elif isinstance(domain, schema_pb2.BoolDomain):
    feature.bool_domain.CopyFrom(domain)
  else:
    # If we have a domain name provided as input, check if we have a valid
    # global string domain with the specified name.
    found_domain = False
    for global_domain in schema.string_domain:
      if global_domain.name == domain:
        found_domain = True
        break
    if not found_domain:
      raise ValueError('Invalid global string domain "{}".'.format(domain))
    feature.domain = domain


def write_schema_text(schema, output_path):
  """Writes input schema to a file in text format.

  Args:
    schema: A Schema protocol buffer.
    output_path: File path to write the input schema.

  Raises:
    TypeError: If the input schema is not of the expected type.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  schema_text = text_format.MessageToString(schema)
  file_io.write_string_to_file(output_path, schema_text)


def load_schema_text(input_path):
  """Loads the schema stored in text format in the input path.

  Args:
    input_path: File path to load the schema from.

  Returns:
    A Schema protocol buffer.
  """
  schema = schema_pb2.Schema()
  schema_text = file_io.read_file_to_string(input_path)
  text_format.Parse(schema_text, schema)
  return schema


def is_categorical_feature(feature):
  """Checks if the input feature is categorical."""
  if feature.type == schema_pb2.BYTES:
    return True
  elif feature.type == schema_pb2.INT:
    return ((feature.HasField('int_domain') and
             feature.int_domain.is_categorical) or
            feature.HasField('bool_domain'))
  else:
    return False


def get_categorical_numeric_features(
    schema):
  """Get the list of numeric features that should be treated as categorical.

  Args:
    schema: The schema for the data.

  Returns:
    A list of int features that should be considered categorical.
  """
  categorical_features = []
  for feature in schema.feature:
    if feature.type == schema_pb2.INT and is_categorical_feature(feature):
      categorical_features.append(feature.name)
  return categorical_features
