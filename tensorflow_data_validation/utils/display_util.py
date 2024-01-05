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
import collections
import sys
from typing import Dict, Iterable, List, Optional, Text, Tuple, Union

import pandas as pd
from tensorflow_data_validation import types
from tensorflow_data_validation.skew.protos import feature_skew_results_pb2
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

  sys.stderr.write(
      'Unable to import IPython: {}. \n'
      'TFDV visualization APIs will not function. To use '
      'visualization features, make sure IPython is installed, or '
      'install TFDV using '
      '"pip install tensorflow-data-validation[visualization]"\n'.format(e)
  )

_NL_CUSTOM_STATS_NAME = 'nl_statistics'
_TOKEN_NAME_KEY = 'token_name'
_FREQUENCY_KEY = 'frequency'
_FRACTION_OF_SEQ_KEY = 'fraction_of_sequences'
_PER_SEQ_MIN_FREQ_KEY = 'per_sequence_min_frequency'
_PER_SEQ_MAX_FREQ_KEY = 'per_sequence_max_frequency'
_PER_SEQ_AVG_FREQ_KEY = 'per_sequence_avg_frequency'
_POSITIONS_KEY = 'positions'


def _add_quotes(input_str: types.FeatureName) -> types.FeatureName:
  return "'" + input_str.replace("'", "\\'") + "'"


def get_schema_dataframe(
    schema: schema_pb2.Schema,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Returns a tuple of DataFrames containing the input schema information.

  Args:
    schema: A Schema protocol buffer.

  Returns:
    A tuple of DataFrames containing the features and domains of the schema.
  """
  if not isinstance(schema, schema_pb2.Schema):
    raise TypeError(
        'schema is of type %s, should be a Schema proto.'
        % type(schema).__name__
    )

  # Extract all the string domains at the schema level.
  domain_rows = []
  for domain in schema.string_domain:
    domain_rows.append([
        _add_quotes(domain.name),
        ', '.join(_add_quotes(v) for v in domain.value),
    ])

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
      if (
          feature.value_count.min == feature.value_count.max
          and feature.value_count.min == 1
      ):
        valency = 'single'
      else:
        min_value_count = (
            '[%d' % feature.value_count.min
            if feature.value_count.HasField('min')
            else '[0'
        )
        max_value_count = (
            '%d]' % feature.value_count.max
            if feature.value_count.HasField('max')
            else 'inf)'
        )
        valency = min_value_count + ',' + max_value_count

    # Extract the feature type.
    feature_type = schema_pb2.FeatureType.Name(feature.type)
    # If the feature has a string domain, treat it as a string feature.
    if feature_type == 'BYTES' and (
        feature.HasField('domain') or feature.HasField('string_domain')
    ):
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
      min_string = (
          'min: %d' % feature.int_domain.min
          if feature.int_domain.HasField('min')
          else None
      )
      max_string = (
          'max: %d' % feature.int_domain.max
          if feature.int_domain.HasField('max')
          else None
      )
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
      domain = _add_quotes(
          feature.string_domain.name
          if feature.string_domain.name
          else feature.name + '_domain'
      )
      domain_rows.append([
          domain,
          ', '.join(_add_quotes(v) for v in feature.string_domain.value),
      ])

    feature_rows.append([
        _add_quotes(feature.name),
        feature_type,
        feature_presence,
        valency,
        domain,
    ])

  features = pd.DataFrame(
      feature_rows,
      columns=['Feature name', 'Type', 'Presence', 'Valency', 'Domain'],
  ).set_index('Feature name')

  domains = pd.DataFrame(domain_rows, columns=['Domain', 'Values']).set_index(
      'Domain'
  )

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
    pd.set_option('display.max_colwidth', None)
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
    raise TypeError(
        'anomalies is of type %s, should be an Anomalies proto.'
        % type(anomalies).__name__
    )

  anomaly_rows = []
  for feature_name, anomaly_info in anomalies.anomaly_info.items():
    if not anomaly_info.short_description:
      anomaly_info_short_description = ('; ').join(
          [r.short_description for r in anomaly_info.reason]
      )
    else:
      anomaly_info_short_description = anomaly_info.short_description
    if not anomaly_info.description:
      anomaly_info_description = ('; ').join(
          [r.description for r in anomaly_info.reason]
      )
    else:
      anomaly_info_description = anomaly_info.description
    anomaly_rows.append([
        _add_quotes(feature_name),
        anomaly_info_short_description,
        anomaly_info_description,
    ])
  if anomalies.HasField('dataset_anomaly_info'):
    if not anomalies.dataset_anomaly_info.short_description:
      dataset_anomaly_info_short_description = ('; ').join(
          [r.short_description for r in anomalies.dataset_anomaly_info.reason]
      )
    else:
      dataset_anomaly_info_short_description = (
          anomalies.dataset_anomaly_info.short_description
      )
    if not anomalies.dataset_anomaly_info.description:
      dataset_anomaly_info_description = ('; ').join(
          [r.description for r in anomalies.dataset_anomaly_info.reason]
      )
    else:
      dataset_anomaly_info_description = (
          anomalies.dataset_anomaly_info.description
      )
    anomaly_rows.append([
        '[dataset anomaly]',
        dataset_anomaly_info_short_description,
        dataset_anomaly_info_description,
    ])

  # Construct a DataFrame consisting of the anomalies.
  anomalies_df = pd.DataFrame(
      anomaly_rows,
      columns=[
          'Feature name',
          'Anomaly short description',
          'Anomaly long description',
      ],
  ).set_index('Feature name')
  # Do not truncate columns.
  pd.set_option('display.max_colwidth', None)
  return anomalies_df


def get_drift_skew_dataframe(anomalies):
  """Get drift_skew_info as a Pandas dataframe."""
  result = []
  for info in anomalies.drift_skew_info:
    for measurement in info.drift_measurements:
      result.append((
          str(types.FeaturePath.from_proto(info.path)),
          anomalies_pb2.DriftSkewInfo.Measurement.Type.Name(measurement.type),
          measurement.value,
          measurement.threshold,
      ))
  return pd.DataFrame(
      result, columns=['path', 'type', 'value', 'threshold']
  ).set_index('path')


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
    denylist_features: Optional[List[types.FeaturePath]] = None,
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


def _get_default_slice_stats(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  if len(statistics.datasets) == 1:
    return statistics
  view = stats_util.DatasetListView(statistics)
  return statistics_pb2.DatasetFeatureStatisticsList(
      datasets=[view.get_default_slice_or_die().proto()]
  )


def _get_combined_statistics(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList
    ] = None,
    lhs_name: Optional[str] = None,
    rhs_name: Optional[str] = None,
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None,
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Get combined datatset statistics list proto."""
  if not isinstance(
      lhs_statistics, statistics_pb2.DatasetFeatureStatisticsList
  ):
    raise TypeError(
        'lhs_statistics is of type %s, should be '
        'a DatasetFeatureStatisticsList proto.'
        % type(lhs_statistics).__name__
    )

  lhs_statistics = _get_default_slice_stats(lhs_statistics)
  if lhs_name is None:
    if lhs_statistics.datasets[0].name:
      lhs_name = lhs_statistics.datasets[0].name
    else:
      lhs_name = 'lhs_statistics'

  # Add lhs stats.
  lhs_statistics = _project_statistics(
      lhs_statistics, allowlist_features, denylist_features
  )
  combined_statistics = statistics_pb2.DatasetFeatureStatisticsList()
  lhs_stats_copy = combined_statistics.datasets.add()
  lhs_stats_copy.MergeFrom(lhs_statistics.datasets[0])

  if rhs_statistics is not None:
    if not isinstance(
        rhs_statistics, statistics_pb2.DatasetFeatureStatisticsList
    ):
      raise TypeError(
          'rhs_statistics is of type %s, should be a '
          'DatasetFeatureStatisticsList proto.'
          % type(rhs_statistics).__name__
      )
    rhs_statistics = _get_default_slice_stats(rhs_statistics)
    if rhs_name is None:
      if rhs_statistics.datasets[0].name:
        rhs_name = rhs_statistics.datasets[0].name
      else:
        rhs_name = 'rhs_statistics'

    # If we have same name, revert to default names.
    if lhs_name == rhs_name:
      lhs_name, rhs_name = 'lhs_statistics', 'rhs_statistics'

    # Add rhs stats.
    rhs_statistics = _project_statistics(
        rhs_statistics, allowlist_features, denylist_features
    )
    rhs_stats_copy = combined_statistics.datasets.add()
    rhs_stats_copy.MergeFrom(rhs_statistics.datasets[0])
    rhs_stats_copy.name = rhs_name

  # Update lhs name.
  lhs_stats_copy.name = lhs_name
  return combined_statistics


def get_statistics_html(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList
    ] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None,
) -> Text:
  """Build the HTML for visualizing the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name to use for the lhs_statistics dataset if a name is not
      already provided within the protocol buffer.
    rhs_name: Name to use for the rhs_statistics dataset if a name is not
      already provided within the protocol buffer.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.

  Returns:
    HTML to be embedded for visualization.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  combined_statistics = _get_combined_statistics(
      lhs_statistics,
      rhs_statistics,
      lhs_name,
      rhs_name,
      allowlist_features,
      denylist_features,
  )
  if (
      len(combined_statistics.datasets) == 1
      and combined_statistics.datasets[0].num_examples == 0
  ):
    return '<p>Empty dataset.</p>'

  protostr = base64.b64encode(combined_statistics.SerializeToString()).decode(
      'utf-8'
  )

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
        statistics_pb2.DatasetFeatureStatisticsList
    ] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None,
) -> None:
  """Visualize the input statistics using Facets.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name to use for the lhs_statistics dataset if a name is not
      already provided within the protocol buffer.
    rhs_name: Name to use for the rhs_statistics dataset if a name is not
      already provided within the protocol buffer.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.

  Raises:
    TypeError: If the input argument is not of the expected type.
    ValueError: If the input statistics protos does not have only one dataset.
  """
  assert (
      not allowlist_features or not denylist_features
  ), 'Only specify one of allowlist_features and denylist_features.'
  html = get_statistics_html(
      lhs_statistics,
      rhs_statistics,
      lhs_name,
      rhs_name,
      allowlist_features,
      denylist_features,
  )
  display(HTML(html))


def compare_slices(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    lhs_slice_key: Text,
    rhs_slice_key: Text,
):
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
  visualize_statistics(
      lhs_stats, rhs_stats, lhs_name=lhs_slice_key, rhs_name=rhs_slice_key
  )


def get_natural_language_statistics_dataframes(
    lhs_statistics: statistics_pb2.DatasetFeatureStatisticsList,
    rhs_statistics: Optional[
        statistics_pb2.DatasetFeatureStatisticsList
    ] = None,
    lhs_name: Text = 'lhs_statistics',
    rhs_name: Text = 'rhs_statistics',
    allowlist_features: Optional[List[types.FeaturePath]] = None,
    denylist_features: Optional[List[types.FeaturePath]] = None,
) -> Optional[
    Dict[
        str, Dict[Union[int, str], Union[Dict[str, pd.DataFrame], pd.DataFrame]]
    ]
]:
  """Gets the `NaturalLanguageStatistics` as a dict of pandas.DataFrame.

  Each pd.DataFrame can be fed into a plot with little to no manipulation.

  For example, to plot the `token_length_histogram` in plot.ly:
  ```
  import pandas a pd
  import plotly
  import tensorflow_data_validation as tfdv
  from tensorflow_data_validation.utils import display_util as tfdv_display_util

  data = pd.DataFrame.from_dict({"col": [1, 2, 3]})
  statistics = tfdv.generate_statistics_from_dataframe(data)

  df = tfdv_display_util.get_natural_language_statistics_dataframes(statistics)
  hist, bin_edges = np.histogram(df[ds_name][feature_name][
                      'token_length_histogram']['high_values'])
  fig = plotly.graph_objs.Figure(data=[
      plotly.graph_objs.Bar(x=bin_edges, y=hist, name='Histogram'),
  ])
  ```

  The resulting dict contains `token_length_histogram` and each token name as
  its keys. For each token, the data frame represents a list of stats as well
  as the token's positions histogram.

  Args:
    lhs_statistics: A DatasetFeatureStatisticsList protocol buffer.
    rhs_statistics: An optional DatasetFeatureStatisticsList protocol buffer to
      compare with lhs_statistics.
    lhs_name: Name of the lhs_statistics dataset.
    rhs_name: Name of the rhs_statistics dataset.
    allowlist_features: Set of features to be visualized.
    denylist_features: Set of features to ignore for visualization.

  Returns:
    A dict of pandas data frames. Returns None if natural language statistics
    does not exist in the statistics proto.
  """
  combined_statistics = _get_combined_statistics(
      lhs_statistics,
      rhs_statistics,
      lhs_name,
      rhs_name,
      allowlist_features,
      denylist_features,
  )
  nlp_stats = _get_natural_language_statistics(combined_statistics)
  if not nlp_stats:
    return None

  result = {}
  for ds_name, features_dict in nlp_stats.items():
    result[ds_name] = {}
    for feature_name, nlp_stat in features_dict.items():
      result[ds_name][feature_name] = {
          'token_length_histogram': _get_histogram_dataframe(
              nlp_stat.token_length_histogram
          ),
          'token_statistics': _get_token_statistics(
              list(nlp_stat.token_statistics)
          ),
      }
  return result


def _get_natural_language_statistics(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
) -> Dict[str, Dict[str, statistics_pb2.NaturalLanguageStatistics]]:
  """Gets the Natural Language stat out of the custom statistic."""
  result = {}
  for dataset in statistics.datasets:
    if not dataset.name:
      continue
    features_dict = {}
    for feature in dataset.features:
      for custom_stats in feature.custom_stats:
        if custom_stats.name == _NL_CUSTOM_STATS_NAME:
          nlp_stat = statistics_pb2.NaturalLanguageStatistics()
          custom_stats.any.Unpack(nlp_stat)
          if feature.name:
            feature_name = feature.name
          else:
            feature_name = str(types.FeaturePath.from_proto(feature.path))
          features_dict[feature_name] = nlp_stat
    if features_dict:
      result[dataset.name] = features_dict
  return result


def _get_token_statistics(
    token_statistic: List[
        statistics_pb2.NaturalLanguageStatistics.TokenStatistics
    ],
) -> pd.DataFrame:
  """Returns a dict of each token's stats."""
  nlp_stats_dict = {
      _TOKEN_NAME_KEY: [],
      _FREQUENCY_KEY: [],
      _FRACTION_OF_SEQ_KEY: [],
      _PER_SEQ_MIN_FREQ_KEY: [],
      _PER_SEQ_MAX_FREQ_KEY: [],
      _PER_SEQ_AVG_FREQ_KEY: [],
      _POSITIONS_KEY: [],
  }
  for token in token_statistic:
    if token.WhichOneof('token') == 'string_token':
      token_name = token.string_token
    else:
      token_name = token.int_token
    nlp_stats_dict[_TOKEN_NAME_KEY].append(token_name)
    nlp_stats_dict[_FREQUENCY_KEY].append(token.frequency)
    nlp_stats_dict[_FRACTION_OF_SEQ_KEY].append(token.fraction_of_sequences)
    nlp_stats_dict[_PER_SEQ_MIN_FREQ_KEY].append(
        token.per_sequence_min_frequency
    )
    nlp_stats_dict[_PER_SEQ_MAX_FREQ_KEY].append(
        token.per_sequence_max_frequency
    )
    nlp_stats_dict[_PER_SEQ_AVG_FREQ_KEY].append(
        token.per_sequence_avg_frequency
    )
    nlp_stats_dict[_POSITIONS_KEY].append(
        _get_histogram_dataframe(token.positions)
    )
  return pd.DataFrame.from_dict(nlp_stats_dict)


def _get_histogram_dataframe(
    histogram: statistics_pb2.Histogram,
) -> pd.DataFrame:
  """Gets the `Histogram` as a pandas.DataFrame."""
  return pd.DataFrame.from_dict({
      'high_values': [b.high_value for b in histogram.buckets],
      'low_values': [b.low_value for b in histogram.buckets],
      'sample_counts': [b.sample_count for b in histogram.buckets],
  })


def get_skew_result_dataframe(
    skew_results: Iterable[feature_skew_results_pb2.FeatureSkew],
) -> pd.DataFrame:
  """Formats FeatureSkew results as a pandas dataframe."""
  result = []
  for feature_skew in skew_results:
    result.append((
        feature_skew.feature_name,
        feature_skew.base_count,
        feature_skew.test_count,
        feature_skew.match_count,
        feature_skew.base_only,
        feature_skew.test_only,
        feature_skew.mismatch_count,
        feature_skew.diff_count,
    ))
  # Preserve deterministic order from the proto.
  columns = [
      'feature_name',
      'base_count',
      'test_count',
      'match_count',
      'base_only',
      'test_only',
      'mismatch_count',
      'diff_count',
  ]
  return (
      pd.DataFrame(result, columns=columns)
      .sort_values('feature_name')
      .reset_index(drop=True)
  )


def get_match_stats_dataframe(
    match_stats: feature_skew_results_pb2.MatchStats,
) -> pd.DataFrame:
  """Formats MatchStats as a pandas dataframe."""
  return pd.DataFrame.from_dict({
      'base_with_id_count': [match_stats.base_with_id_count],
      'test_with_id_count': [match_stats.test_with_id_count],
      'identifiers_count': [match_stats.identifiers_count],
      'ids_missing_in_base_count': [match_stats.ids_missing_in_base_count],
      'ids_missing_in_test_count': [match_stats.ids_missing_in_test_count],
      'matching_pairs_count': [match_stats.matching_pairs_count],
      'base_missing_id_count': [match_stats.base_missing_id_count],
      'test_missing_id_count': [match_stats.test_missing_id_count],
      'duplicate_id_count': [match_stats.duplicate_id_count],
  })


def get_confusion_count_dataframes(
    confusion: Iterable[feature_skew_results_pb2.ConfusionCount],
) -> Dict[str, pd.DataFrame]:
  """Returns a pandas dataframe representation of a sequence of ConfusionCount.

  Args:
    confusion: An interable over ConfusionCount protos.
  Returns: A map from feature name to a pandas dataframe containing match counts
    along with base and test counts for all unequal value pairs in the input.
  """
  confusion = list(confusion)
  confusion_per_feature = collections.defaultdict(list)
  for c in confusion:
    confusion_per_feature[c.feature_name].append(c)

  def _build_df(confusion):
    base_count_per_value = collections.defaultdict(lambda: 0)
    test_count_per_value = collections.defaultdict(lambda: 0)
    value_counts = []
    for c in confusion:
      base_count_per_value[c.base.bytes_value] += c.count
      test_count_per_value[c.test.bytes_value] += c.count
      value_counts.append((c.base.bytes_value, c.test.bytes_value, c.count))
    df = pd.DataFrame(
        value_counts, columns=('Base value', 'Test value', 'Pair count')
    )
    df['Base count'] = df['Base value'].apply(lambda x: base_count_per_value[x])
    df['Test count'] = df['Test value'].apply(lambda x: test_count_per_value[x])
    df['Fraction of base'] = df['Pair count'] / df['Base count']
    df = (
        df[df['Base value'] != df['Test value']]
        .sort_values(['Base value', 'Fraction of base'])
        .reset_index(drop=True)
    )
    return df[
        ['Base value', 'Test value', 'Pair count', 'Base count', 'Test count']
    ]

  return {k: _build_df(v) for k, v in confusion_per_feature.items()}
