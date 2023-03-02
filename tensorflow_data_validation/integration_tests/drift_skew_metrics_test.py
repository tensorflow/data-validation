# Copyright 2023 Google LLC
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
"""End to end tests of the validation API which are easier to do in Python."""

from absl import flags
from absl.testing import absltest
import numpy as np
import pandas as pd
import tensorflow_data_validation as tfdv

from tensorflow_metadata.proto.v0 import schema_pb2

FLAGS = flags.FLAGS


def get_js(
    array1: np.ndarray,
    array2: np.ndarray,
    hist_type: schema_pb2.HistogramSelection.Type,
    quantiles_buckets: int = 10,
) -> float:
  opts = tfdv.StatsOptions()
  opts.num_quantiles_histogram_buckets = quantiles_buckets
  stats1 = tfdv.generate_statistics_from_dataframe(
      pd.DataFrame({'foo': array1}), stats_options=opts
  )
  stats2 = tfdv.generate_statistics_from_dataframe(
      pd.DataFrame({'foo': array2}), stats_options=opts
  )
  schema = tfdv.infer_schema(stats1)
  f = tfdv.get_feature(schema, 'foo')
  f.drift_comparator.jensen_shannon_divergence.threshold = 0
  f.drift_comparator.jensen_shannon_divergence.source.type = hist_type
  anomalies = tfdv.validate_statistics(
      stats1, schema, previous_statistics=stats2
  )
  return anomalies.drift_skew_info[0].drift_measurements[0].value


class DriftSkewMetricsTest(absltest.TestCase):

  def test_standard_quantiles_similar_outcomes_with_normal_dist(self):
    gen = np.random.default_rng(44)
    for shift in np.linspace(0, 2, 10):
      array1 = gen.standard_normal(1000)
      array2 = shift + gen.standard_normal(1000)
      js_standard = get_js(
          array1, array2, schema_pb2.HistogramSelection.STANDARD
      )
      js_quantiles = get_js(
          array1, array2, schema_pb2.HistogramSelection.QUANTILES
      )
      self.assertAlmostEqual(js_standard, js_quantiles, delta=0.1)

  def test_outlier_sensitivity(self):
    gen = np.random.default_rng(44)
    array1 = gen.standard_normal(10000)
    array2 = np.concatenate([array1, np.array([1e8])])
    js_quantiles = get_js(
        array1, array2, schema_pb2.HistogramSelection.QUANTILES
    )
    js_quantiles_100 = get_js(
        array1, array2, schema_pb2.HistogramSelection.QUANTILES, 100
    )
    js_standard = get_js(array1, array2, schema_pb2.HistogramSelection.STANDARD)
    js_standard_100 = get_js(
        array1, array2, schema_pb2.HistogramSelection.STANDARD, 100
    )
    # The idealized JS is very close to zero, but in practice we expect a value
    # around 0.1 because there are only ten bins, and the last bin is affected
    # by the outlier.
    self.assertLess(js_quantiles, 0.15)
    # QUANTILES JS with more bins is better here.
    self.assertLess(js_quantiles_100, 0.02)
    # STANDARD JS is very affected by outliers.
    self.assertGreater(js_standard, 0.99)
    # Adding more bins doesn't help.
    self.assertGreater(js_standard_100, 0.99)


if __name__ == '__main__':
  absltest.main()
