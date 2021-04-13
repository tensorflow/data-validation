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

"""Utils for displaying TFDV outputs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import base64
import sys
from typing import List, Optional, Text, Tuple

import pandas as pd
from tensorflow_data_validation import types
from tensorflow_data_validation.utils import stats_util
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


try:
  # pylint: disable=g-import-not-at-top
  from IPython.display import display
  from IPython.display import HTML
except ImportError as e:

  def display(unused_input):
    print('IPython is not installed. Unable to display.')

  def HTML(s):  # pylint: disable=invalid-name
    return s

  sys.stderr.write('Unable to import IPython: {}. \n'
                   'TFDV visualization APIs will not function. To use '
                   'visualization features, make sure IPython is installed, or '
                   'install TFDV using '
                   '"pip install tensorflow-data-validation[visualization]"\n'
                   .format(e))


def _add_quotes(input_str: types.FeatureName) -> types.FeatureName:
  return "'" + input_str.replace("'", "\\'") + "'"


def get_schema_dataframe(
    schema: schema_pb2.Schema) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Returns a tuple of DataFrames containing the input schema information.

  Args:
    schema: A Schema protocol buffer.
  Returns:
    A tuple of DataFrames containing the features and domains of the schema.
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
    def combine_min_max_strings(min_string, max_string):
      if min_string is not None and max_string is not None:
        domain_string = min_string + '; ' + max_string
      elif min_string is not None:
        domain_string = min_string
      elif max_string is not None:
        domain_string = max_string
      else:
        domain_string = '-'
      return domain_string

    domain = '-'
    if feature.HasField('domain'):
      domain = _add_quotes(feature.domain)
    elif feature.HasField('int_domain'):
      min_string = ('min: %d' % feature.int_domain.min
                    if feature.int_domain.HasField('min') else None)
      max_string = ('max: %d' % feature.int_domain.max
                    if feature.int_domain.HasField('max') else None)
      domain = combine_min_max_strings(min_string, max_string)
    elif feature.HasField('float_domain'):
      if feature.float_domain.HasField('min'):
        min_string = 'min: %f' % feature.float_domain.min
      elif feature.float_domain.disallow_inf:
        min_string = None
      else:
        min_string = 'min: -inf'
      if feature.float_domain.HasField('max'):
        max_string = 'max: %f' % feature.float_domain.max
      elif feature.float_domain.disallow_inf:
        max_string = None
      else:
        max_string = 'max: inf'
      domain = combine_min_max_strings(min_string, max_string)
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

  features = pd.DataFrame(
      feature_rows,
      columns=['Feature name', 'Type', 'Presence', 'Valency',
               'Domain']).set_index('Feature name')

  domains = pd.DataFrame(
      domain_rows, columns=['Domain', 'Values']).set_index('Domain')

  return features, domains


def display_schema(schema: schema_pb2.Schema) -> None:
  """Displays the input schema (for use in a Jupyter notebook).

  Args:
    schema: A Schema protocol buffer.
  """
  features_df, domains_df = get_schema_dataframe(schema)
  display(features_df)
  # Do not truncate columns.
  if not domains_df.empty:
    pd.set_option('max_colwidth', -1)
    display(domains_df)


def get_anomalies_dataframe(anomalies: anomalies_pb2.Anomalies) -> pd.DataFrame:
  """Returns a DataFrame containing the input anomalies.

  Args:
    anomalies: An Anomalies protocol buffer.
  Returns:
    A DataFrame containing the input anomalies, or an empty DataFrame if there
    are no anomalies.
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
  if anomalies.HasField('dataset_anomaly_info'):
    anomaly_rows.append([
        '[dataset anomaly]', anomalies.dataset_anomaly_info.short_description,
        anomalies.dataset_anomaly_info.description
    ])

  # Construct a DataFrame consisting of the anomalies and display it.
  anomalies_df = pd.DataFrame(
      anomaly_rows,
      columns=[
          'Feature name', 'Anomaly short description',
          'Anomaly long description'
      ]).set_index('Feature name')
  # Do not truncate columns.
  pd.set_option('max_colwidth', -1)
  return anomalies_df


def display_anomalies(anomalies: anomalies_pb2.Anomalies) -> None:
  """Displays the input anomalies (for use in a Jupyter notebook).

  Args:
    anomalies: An Anomalies protocol buffer.
  """
  anomalies_df = get_anomalies_dataframe(anomalies)
  if anomalies_df.empty:
    display(HTML('<h4 style="color:green;">No anomalies found.</h4>'))
  else:
    display(anomalies_df)


def _project_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Project statistics proto based on allowlist and denylist features."""
  if allowlist_features is None and denylist_features is None:
    return statistics
  result = statistics_pb2.DatasetFeatureStatisticsList()
  for dataset_stats in statistics.datasets:
    result_dataset_stats = result.datasets.add()
    result_dataset_stats.MergeFrom(dataset_stats)
    del result_dataset_stats.features[:]
    if allowlist_features is not None:
      allowlist_features = set(allowlist_features)
      for feature in dataset_stats.features:
        if types.FeaturePath.from_proto(feature.path) in allowlist_features:
          result_dataset_stats.features.add().MergeFrom(feature)
    else:
      denylist_features = set(denylist_features)
      for feature in dataset_stats.features:
        if types.FeaturePath.from_proto(feature.path) in denylist_features:
          continue
        result_dataset_stats.features.add().MergeFrom(feature)
  return result


def _get_combined_statistics(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Get combined datatset statistics list proto."""
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
  lhs_statistics = _project_statistics(
      lhs_statistics, allowlist_features, denylist_features)
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
    rhs_statistics = _project_statistics(
        rhs_statistics, allowlist_features, denylist_features)
    rhs_stats_copy = combined_statistics.datasets.add()
    rhs_stats_copy.MergeFrom(rhs_statistics.datasets[0])
    rhs_stats_copy.name = rhs_name

  # Update lhs name.
  lhs_stats_copy.name = lhs_name
  return combined_statistics


def get_statistics_html(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None
) -> Text:
  """Build the HTML for visualizing the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.

  Returns:
    HTML to be embedded for visualization.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  combined_statistics = _get_combined_statistics(
      lhs_statistics, rhs_statistics, lhs_name, rhs_name, allowlist_features,
      denylist_features)
  protostr = base64.b64encode(
      combined_statistics.SerializeToString()).decode('utf-8')

  # pylint: disable=line-too-long,anomalous-backslash-in-string
  # Note that in the html template we currently assign a temporary id to the
  # facets element and then remove it once we have appended the serialized proto
  # string to the element. We do this to avoid any collision of ids when
  # displaying multiple facets output in the notebook.
  #
  # Note that a string literal including '</script>' in a <script> tag needs to
  # escape it as <\/script> to avoid early closing the wrapping <script> tag.
  html_template = """<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="protostr"></facets-overview>';
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
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None) -> None:
  """Visualize the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  assert (not allowlist_features or not denylist_features), (
      'Only specify one of allowlist_features and denylist_features.')
  html = get_statistics_html(lhs_statistics, rhs_statistics, lhs_name, rhs_name,
                             allowlist_features, denylist_features)
  display(HTML(html))


def compare_slices(statistics: statistics_pb2.DatasetFeatureStatisticsList,
                   lhs_slice_key: Text, rhs_slice_key: Text):
  """Compare statistics of two slices using Facets.

  Args:
    statistics: A DatasetFeatureStatisticsList protocol buffer.
    lhs_slice_key: Slice key of the first slice.
    rhs_slice_key: Slice key of the second slice.

  Raises:
    ValueError: If the input statistics proto does not have the specified slice
      statistics.
  """
  lhs_stats = stats_util.get_slice_stats(statistics, lhs_slice_key)
  rhs_stats = stats_util.get_slice_stats(statistics, rhs_slice_key)
  visualize_statistics(lhs_stats, rhs_stats,
                       lhs_name=lhs_slice_key, rhs_name=rhs_slice_key)
