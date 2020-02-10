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
# ==============================================================================

"""Utils for example notebooks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import base64
from IPython.display import display
from IPython.display import HTML
import pandas as pd
from tensorflow_data_validation import types
from typing import Optional, Text
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


def _add_quotes(input_str: types.FeatureName) -> types.FeatureName:
  return "'" + input_str.replace("'", "\\'") + "'"


def display_schema(schema: schema_pb2.Schema) -> None:
  """Displays the input schema.

  Args:
    schema: A Schema protocol buffer.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError('schema is of type %s, should be a Schema proto.' %
                    type(schema).__name__)

  # Extract all the string domains at the schema level.
  domain_rows = []
  for domain in schema.string_domain:
    domain_rows.append(
        [_add_quotes(domain.name),
         ', '.join(_add_quotes(v) for v in domain.value)])

  feature_rows = []
  # Iterate over the features in the schema and extract the properties of each
  # feature.
  for feature in schema.feature:
    # Extract the presence information of the feature.
    if feature.HasField('presence'):
      if feature.presence.min_fraction == 1.0:
        feature_presence = 'required'
      else:
        feature_presence = 'optional'
    else:
      feature_presence = ''

    # Extract the valency information of the feature.
    valency = ''
    if feature.HasField('value_count'):
      if (feature.value_count.min == feature.value_count.max and
          feature.value_count.min == 1):
        valency = 'single'
      else:
        min_value_count = ('[%d' % feature.value_count.min
                           if feature.value_count.HasField('min') else '[0')
        max_value_count = ('%d]' % feature.value_count.max
                           if feature.value_count.HasField('max') else 'inf)')
        valency = min_value_count + ',' + max_value_count

    # Extract the feature type.
    feature_type = schema_pb2.FeatureType.Name(feature.type)
    # If the feature has a string domain, treat it as a string feature.
    if feature_type == 'BYTES' and (feature.HasField('domain') or
                                    feature.HasField('string_domain')):
      feature_type = 'STRING'

    # Extract the domain (if any) of the feature.
    domain = '-'
    if feature.HasField('domain'):
      domain = _add_quotes(feature.domain)
    elif feature.HasField('int_domain'):
      left_value = ('[%d' % feature.int_domain.min
                    if feature.int_domain.HasField('min') else '(-inf')
      right_value = ('%d]' % feature.int_domain.max
                     if feature.int_domain.HasField('max') else 'inf)')
      domain = left_value + ',' + right_value
    elif feature.HasField('float_domain'):
      left_value = ('[%f' % feature.float_domain.min
                    if feature.float_domain.HasField('min') else '(-inf')
      right_value = ('%f]' % feature.float_domain.max
                     if feature.float_domain.HasField('max') else 'inf)')
      domain = left_value + ',' + right_value
    elif feature.HasField('string_domain'):
      domain = _add_quotes(feature.string_domain.name if
                           feature.string_domain.name else
                           feature.name + '_domain')
      domain_rows.append([domain,
                          ', '.join(_add_quotes(v) for v in
                                    feature.string_domain.value)])

    feature_rows.append(
        [_add_quotes(feature.name), feature_type, feature_presence, valency,
         domain])

  # Construct a DataFrame consisting of the properties of the features
  # and display it.
  features = pd.DataFrame(
      feature_rows,
      columns=['Feature name', 'Type', 'Presence', 'Valency',
               'Domain']).set_index('Feature name')
  display(features)

  # Construct a DataFrame consisting of the domain values and display it.
  if domain_rows:
    domains = pd.DataFrame(
        domain_rows, columns=['Domain',
                              'Values']).set_index('Domain')
    # Do not truncate columns.
    pd.set_option('max_colwidth', -1)
    display(domains)


def display_anomalies(anomalies: anomalies_pb2.Anomalies) -> None:
  """Displays the input anomalies.

  Args:
    anomalies: An Anomalies protocol buffer.
  """
  if not isinstance(anomalies, anomalies_pb2.Anomalies):
    raise TypeError('anomalies is of type %s, should be an Anomalies proto.' %
                    type(anomalies).__name__)

  anomaly_rows = []
  for feature_name, anomaly_info in anomalies.anomaly_info.items():
    anomaly_rows.append([
        _add_quotes(feature_name), anomaly_info.short_description,
        anomaly_info.description
    ])

  if not anomaly_rows:
    display(HTML('<h4 style="color:green;">No anomalies found.</h4>'))
  else:
    # Construct a DataFrame consisting of the anomalies and display it.
    anomalies_df = pd.DataFrame(
        anomaly_rows,
        columns=['Feature name', 'Anomaly short description',
                 'Anomaly long description']).set_index('Feature name')
    # Do not truncate columns.
    pd.set_option('max_colwidth', -1)
    display(anomalies_df)


def get_statistics_html(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics'
) -> Text:
  """Build the HTML for visualizing the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.

  Returns:
    HTML to be embedded for visualization.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  if not isinstance(lhs_statistics,
                    statistics_pb2.DatasetFeatureStatisticsList):
    raise TypeError(
        'lhs_statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.' % type(lhs_statistics).__name__)

  if len(lhs_statistics.datasets) != 1:
    raise ValueError('lhs_statistics proto contains multiple datasets. Only '
                     'one dataset is currently supported.')

  if lhs_statistics.datasets[0].name:
    lhs_name = lhs_statistics.datasets[0].name

  # Add lhs stats.
  combined_statistics = statistics_pb2.DatasetFeatureStatisticsList()
  lhs_stats_copy = combined_statistics.datasets.add()
  lhs_stats_copy.MergeFrom(lhs_statistics.datasets[0])

  if rhs_statistics is not None:
    if not isinstance(rhs_statistics,
                      statistics_pb2.DatasetFeatureStatisticsList):
      raise TypeError('rhs_statistics is of type %s, should be a '
                      'DatasetFeatureStatisticsList proto.'
                      % type(rhs_statistics).__name__)
    if len(rhs_statistics.datasets) != 1:
      raise ValueError('rhs_statistics proto contains multiple datasets. Only '
                       'one dataset is currently supported.')

    if rhs_statistics.datasets[0].name:
      rhs_name = rhs_statistics.datasets[0].name

    # If we have same name, revert to default names.
    if lhs_name == rhs_name:
      lhs_name, rhs_name = 'lhs_statistics', 'rhs_statistics'

    # Add rhs stats.
    rhs_stats_copy = combined_statistics.datasets.add()
    rhs_stats_copy.MergeFrom(rhs_statistics.datasets[0])
    rhs_stats_copy.name = rhs_name

  # Update lhs name.
  lhs_stats_copy.name = lhs_name

  protostr = base64.b64encode(
      combined_statistics.SerializeToString()).decode('utf-8')

  # pylint: disable=line-too-long
  # Note that in the html template we currently assign a temporary id to the
  # facets element and then remove it once we have appended the serialized proto
  # string to the element. We do this to avoid any collision of ids when
  # displaying multiple facets output in the notebook.
  html_template = """<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="protostr"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>"""
  # pylint: enable=line-too-long
  html = html_template.replace('protostr', protostr)

  return html


def visualize_statistics(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics') -> None:
  """Visualize the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  html = get_statistics_html(lhs_statistics, rhs_statistics, lhs_name, rhs_name)
  display(HTML(html))
