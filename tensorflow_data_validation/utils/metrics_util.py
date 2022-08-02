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
"""Metrics utilities."""

from typing import Mapping

import apache_beam as beam
from tensorflow_data_validation import constants


class IncrementJobCounters(beam.PTransform):
  """Increments beam counters from values available at graph construction."""

  def __init__(self, values: Mapping[str, int]):
    self._values = values

  def expand(self, pcoll: beam.PCollection):

    def _incr(unused_value):
      for name, value in self._values.items():
        beam.metrics.Metrics.counter(constants.METRICS_NAMESPACE,
                                     name).inc(value)
      return None

    _ = (
        pcoll.pipeline
        | 'CreateSingleton' >> beam.Create([1])
        | 'IncrementCounters' >> beam.Map(_incr))
