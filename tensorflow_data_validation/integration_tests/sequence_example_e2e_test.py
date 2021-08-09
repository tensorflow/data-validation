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
"""Integration tests to cover TFDV consuming tf.SequenceExamples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import test_util
from tfx_bsl.tfxio import tf_sequence_example_record

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

FLAGS = flags.FLAGS
_EXAMPLE_A = text_format.Parse(
    """
feature_lists: {
  feature_list: {
    key: 'sequence_int64_feature'
    value: {
      # sequence length = 3
      feature: {
        # missing
      }
      feature: {
        int64_list: {
          value: [1, 2]
        }
      }
      feature: {
        int64_list: {
          value: []
        }
      }
    }
  }
  feature_list: {
    key: 'sequence_float_feature'
    value: {
      feature: {
        float_list: {
          value: [0.0, 0.0]
        }
      }
    }
  }
}
context: {
  feature: {
    key: 'context_bytes_feature'
    value: {
      bytes_list: {
        value: ['0']
      }
    }
  }
  feature: {
    key: 'context_int64_feature'
    value: {
      int64_list: {
        value: []
      }
    }
  }
  feature: {
    key: 'label'
    value: {
      float_list: {
        value: [1]
      }
    }
  }
  feature: {
    key: 'example_weight'
    value: {
      float_list: {
        value: [5]
      }
    }
  }
}""", tf.train.SequenceExample()).SerializeToString()

_EXAMPLE_B = text_format.Parse(
    """
feature_lists: {
  feature_list: {
    key: 'sequence_int64_feature'
    # sequence length = 1
    value: {
      feature: {
        int64_list: {
          value: [2, 3, 4]
        }
      }
    }
  }
  # 'sequence_float_feature' is missing.
}
context: {
  feature: {
    key: 'context_bytes_feature'
    value: {
      bytes_list: {
        value: ['1']
      }
    }
  }
  feature: {
    key: 'label'
    value: {
      float_list: {
        value: [2]
      }
    }
  }
  feature: {
    key: 'example_weight'
    value: {
      float_list: {
        value: [10]
      }
    }
  }
}
""", tf.train.SequenceExample()).SerializeToString()

_LABEL = 'label'
_EXAMPLE_WEIGHT = 'example_weight'

_BASIC_GOLDEN_STATS = """
datasets {
  num_examples: 20
  features {
    type: STRING
    string_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        tot_num_values: 20
      }
      unique: 2
      top_values {
        value: "1"
        frequency: 10.0
      }
      top_values {
        value: "0"
        frequency: 10.0
      }
      avg_length: 1.0
      rank_histogram {
        buckets {
          label: "1"
          sample_count: 10.0
        }
        buckets {
          low_rank: 1
          high_rank: 1
          label: "0"
          sample_count: 10.0
        }
      }
    }
    path {
      step: "context_bytes_feature"
    }
  }
  features {
    num_stats {
      common_stats {
        num_non_missing: 10
        num_missing: 10
        num_values_histogram {
          buckets {
            sample_count: 3.3333333333333335
          }
          buckets {
            sample_count: 3.3333333333333335
          }
          buckets {
            sample_count: 3.3333333333333335
          }
          type: QUANTILES
        }
      }
    }
    path {
      step: "context_int64_feature"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        tot_num_values: 20
      }
      mean: 7.5
      std_dev: 2.5
      min: 5.0
      median: 10.0
      max: 10.0
      histograms {
        buckets {
          low_value: 5.0
          high_value: 6.666666666666667
          sample_count: 9.955555555555556
        }
        buckets {
          low_value: 6.666666666666667
          high_value: 8.333333333333334
          sample_count: 0.022222222222222227
        }
        buckets {
          low_value: 8.333333333333334
          high_value: 10.0
          sample_count: 10.022222222222222
        }
      }
      histograms {
        buckets {
          low_value: 5.0
          high_value: 5.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 5.0
          high_value: 10.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 10.0
          high_value: 10.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
    }
    path {
      step: "example_weight"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        tot_num_values: 20
      }
      mean: 1.5
      std_dev: 0.5
      min: 1.0
      median: 2.0
      max: 2.0
      histograms {
        buckets {
          low_value: 1.0
          high_value: 1.3333333333333333
          sample_count: 9.955555555555556
        }
        buckets {
          low_value: 1.3333333333333333
          high_value: 1.6666666666666665
          sample_count: 0.022222222222222216
        }
        buckets {
          low_value: 1.6666666666666665
          high_value: 2.0
          sample_count: 10.022222222222222
        }
      }
      histograms {
        buckets {
          low_value: 1.0
          high_value: 1.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
    }
    path {
      step: "label"
    }
  }
  features {
    type: STRUCT
    struct_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        tot_num_values: 20
      }
    }
    path {
      step: "##SEQUENCE##"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 10
        num_missing: 10
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          type: QUANTILES
        }
        tot_num_values: 10
        presence_and_valency_stats {
          num_non_missing: 10
          num_missing: 10
          min_num_values: 1
          max_num_values: 1
          tot_num_values: 10
        }
        presence_and_valency_stats {
          num_non_missing: 10
          min_num_values: 2
          max_num_values: 2
          tot_num_values: 20
        }
      }
      num_zeros: 20
      histograms {
        buckets {
          sample_count: 20.0
        }
      }
      histograms {
        buckets {
          sample_count: 6.666666666666667
        }
        buckets {
          sample_count: 6.666666666666667
        }
        buckets {
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
    }
    custom_stats {
      name: "level_2_value_list_length"
      histogram {
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        type: QUANTILES
      }
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_float_feature"
    }
  }
  features {
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 3
        avg_num_values: 2.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 3.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 3.0
            high_value: 3.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        tot_num_values: 40
        presence_and_valency_stats {
          num_non_missing: 20
          min_num_values: 1
          max_num_values: 3
          tot_num_values: 40
        }
        presence_and_valency_stats {
          num_non_missing: 30
          num_missing: 10
          max_num_values: 3
          tot_num_values: 50
        }
      }
      mean: 2.4
      std_dev: 1.019803902718557
      min: 1.0
      median: 2.0
      max: 4.0
      histograms {
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 9.999999999999998
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 20.0
        }
        buckets {
          low_value: 3.0
          high_value: 4.0
          sample_count: 20.0
        }
      }
      histograms {
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 16.666666666666664
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 16.666666666666664
        }
        buckets {
          low_value: 3.0
          high_value: 4.0
          sample_count: 16.666666666666664
        }
        type: QUANTILES
      }
    }
    custom_stats {
      name: "level_2_value_list_length"
      histogram {
        buckets {
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 3.0
          high_value: 3.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_int64_feature"
    }
  }
}
"""

_WEIGHT_AND_LABEL_GOLDEN_STATS = """
datasets {
  num_examples: 20
  features {
    type: STRING
    string_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 150.0
          avg_num_values: 1.0
          tot_num_values: 150.0
        }
        tot_num_values: 20
      }
      unique: 2
      top_values {
        value: "1"
        frequency: 10.0
      }
      top_values {
        value: "0"
        frequency: 10.0
      }
      avg_length: 1.0
      rank_histogram {
        buckets {
          label: "1"
          sample_count: 10.0
        }
        buckets {
          low_rank: 1
          high_rank: 1
          label: "0"
          sample_count: 10.0
        }
      }
      weighted_string_stats {
        top_values {
          value: "1"
          frequency: 100.0
        }
        top_values {
          value: "0"
          frequency: 50.0
        }
        rank_histogram {
          buckets {
            label: "1"
            sample_count: 100.0
          }
          buckets {
            low_rank: 1
            high_rank: 1
            label: "0"
            sample_count: 50.0
          }
        }
      }
    }
    path {
      step: "context_bytes_feature"
    }
  }
  features {
    num_stats {
      common_stats {
        num_non_missing: 10
        num_missing: 10
        num_values_histogram {
          buckets {
            sample_count: 3.3333333333333335
          }
          buckets {
            sample_count: 3.3333333333333335
          }
          buckets {
            sample_count: 3.3333333333333335
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 50.0
          num_missing: 100.0
        }
      }
    }
    path {
      step: "context_int64_feature"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 150.0
          avg_num_values: 1.0
          tot_num_values: 150.0
        }
        tot_num_values: 20
      }
      mean: 7.5
      std_dev: 2.5
      min: 5.0
      median: 10.0
      max: 10.0
      histograms {
        buckets {
          low_value: 5.0
          high_value: 6.666666666666667
          sample_count: 9.955555555555556
        }
        buckets {
          low_value: 6.666666666666667
          high_value: 8.333333333333334
          sample_count: 0.022222222222222227
        }
        buckets {
          low_value: 8.333333333333334
          high_value: 10.0
          sample_count: 10.022222222222222
        }
      }
      histograms {
        buckets {
          low_value: 5.0
          high_value: 5.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 5.0
          high_value: 10.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 10.0
          high_value: 10.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
      weighted_numeric_stats {
        mean: 8.333333333333334
        std_dev: 2.357022603955156
        median: 10.0
        histograms {
          buckets {
            low_value: 5.0
            high_value: 6.666666666666667
            sample_count: 49.666666666666664
          }
          buckets {
            low_value: 6.666666666666667
            high_value: 8.333333333333334
            sample_count: 0.16666666666666669
          }
          buckets {
            low_value: 8.333333333333334
            high_value: 10.0
            sample_count: 100.16666666666667
          }
        }
        histograms {
          buckets {
            low_value: 5.0
            high_value: 10.0
            sample_count: 50.0
          }
          buckets {
            low_value: 10.0
            high_value: 10.0
            sample_count: 50.0
          }
          buckets {
            low_value: 10.0
            high_value: 10.0
            sample_count: 50.0
          }
          type: QUANTILES
        }
      }
    }
    path {
      step: "example_weight"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 150.0
          avg_num_values: 1.0
          tot_num_values: 150.0
        }
        tot_num_values: 20
      }
      mean: 1.5
      std_dev: 0.5
      min: 1.0
      median: 2.0
      max: 2.0
      histograms {
        buckets {
          low_value: 1.0
          high_value: 1.3333333333333333
          sample_count: 9.955555555555556
        }
        buckets {
          low_value: 1.3333333333333333
          high_value: 1.6666666666666665
          sample_count: 0.022222222222222216
        }
        buckets {
          low_value: 1.6666666666666665
          high_value: 2.0
          sample_count: 10.022222222222222
        }
      }
      histograms {
        buckets {
          low_value: 1.0
          high_value: 1.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
      weighted_numeric_stats {
        mean: 1.6666666666666667
        std_dev: 0.4714045207910313
        median: 2.0
        histograms {
          buckets {
            low_value: 1.0
            high_value: 1.3333333333333333
            sample_count: 49.666666666666664
          }
          buckets {
            low_value: 1.3333333333333333
            high_value: 1.6666666666666665
            sample_count: 0.16666666666666663
          }
          buckets {
            low_value: 1.6666666666666665
            high_value: 2.0
            sample_count: 100.16666666666667
          }
        }
        histograms {
          buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 50.0
          }
          buckets {
            low_value: 2.0
            high_value: 2.0
            sample_count: 50.0
          }
          buckets {
            low_value: 2.0
            high_value: 2.0
            sample_count: 50.0
          }
          type: QUANTILES
        }
      }
    }
    path {
      step: "label"
    }
  }
  features {
    type: STRUCT
    struct_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 150.0
          avg_num_values: 1.0
          tot_num_values: 150.0
        }
        tot_num_values: 20
      }
    }
    path {
      step: "##SEQUENCE##"
    }
  }
  features {
    type: FLOAT
    num_stats {
      common_stats {
        num_non_missing: 10
        num_missing: 10
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 3.3333333333333335
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 50.0
          num_missing: 100.0
          avg_num_values: 1.0
          tot_num_values: 50.0
        }
        tot_num_values: 10
        presence_and_valency_stats {
          num_non_missing: 10
          num_missing: 10
          min_num_values: 1
          max_num_values: 1
          tot_num_values: 10
        }
        presence_and_valency_stats {
          num_non_missing: 10
          min_num_values: 2
          max_num_values: 2
          tot_num_values: 20
        }
        weighted_presence_and_valency_stats {
          num_non_missing: 50.0
          num_missing: 100.0
          avg_num_values: 1.0
          tot_num_values: 50.0
        }
        weighted_presence_and_valency_stats {
          num_non_missing: 50.0
          avg_num_values: 2.0
          tot_num_values: 100.0
        }
      }
      num_zeros: 20
      histograms {
        buckets {
          sample_count: 20.0
        }
      }
      histograms {
        buckets {
          sample_count: 6.666666666666667
        }
        buckets {
          sample_count: 6.666666666666667
        }
        buckets {
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
      weighted_numeric_stats {
        histograms {
          buckets {
            sample_count: 100.0
          }
        }
        histograms {
          buckets {
            sample_count: 33.33333333333333
          }
          buckets {
            sample_count: 33.33333333333333
          }
          buckets {
            sample_count: 33.33333333333333
          }
          type: QUANTILES
        }
      }
    }
    custom_stats {
      name: "level_2_value_list_length"
      histogram {
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        buckets {
          low_value: 2.0
          high_value: 2.0
          sample_count: 3.3333333333333335
        }
        type: QUANTILES
      }
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_float_feature"
    }
  }
  features {
    num_stats {
      common_stats {
        num_non_missing: 20
        min_num_values: 1
        max_num_values: 3
        avg_num_values: 2.0
        num_values_histogram {
          buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 1.0
            high_value: 3.0
            sample_count: 6.666666666666667
          }
          buckets {
            low_value: 3.0
            high_value: 3.0
            sample_count: 6.666666666666667
          }
          type: QUANTILES
        }
        weighted_common_stats {
          num_non_missing: 150.0
          avg_num_values: 1.6666666666666667
          tot_num_values: 250.0
        }
        tot_num_values: 40
        presence_and_valency_stats {
          num_non_missing: 20
          min_num_values: 1
          max_num_values: 3
          tot_num_values: 40
        }
        presence_and_valency_stats {
          num_non_missing: 30
          num_missing: 10
          max_num_values: 3
          tot_num_values: 50
        }
        weighted_presence_and_valency_stats {
          num_non_missing: 150.0
          avg_num_values: 1.6666666666666667
          tot_num_values: 250.0
        }
        weighted_presence_and_valency_stats {
          num_non_missing: 200.0
          num_missing: 50.0
          avg_num_values: 2.0
          tot_num_values: 400.0
        }
      }
      mean: 2.4
      std_dev: 1.019803902718557
      min: 1.0
      median: 2.0
      max: 4.0
      histograms {
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 9.999999999999998
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 20.0
        }
        buckets {
          low_value: 3.0
          high_value: 4.0
          sample_count: 20.0
        }
      }
      histograms {
        buckets {
          low_value: 1.0
          high_value: 2.0
          sample_count: 16.666666666666664
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 16.666666666666664
        }
        buckets {
          low_value: 3.0
          high_value: 4.0
          sample_count: 16.666666666666664
        }
        type: QUANTILES
      }
      weighted_numeric_stats {
        mean: 2.625
        std_dev: 0.9921567416492215
        median: 3.0
        histograms {
          buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 50.666666666666664
          }
          buckets {
            low_value: 2.0
            high_value: 3.0
            sample_count: 149.33333333333334
          }
          buckets {
            low_value: 3.0
            high_value: 4.0
            sample_count: 200.0
          }
        }
        histograms {
          buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 133.33333333333331
          }
          buckets {
            low_value: 2.0
            high_value: 3.0
            sample_count: 133.33333333333331
          }
          buckets {
            low_value: 3.0
            high_value: 4.0
            sample_count: 133.33333333333331
          }
          type: QUANTILES
        }
      }
    }
    custom_stats {
      name: "level_2_value_list_length"
      histogram {
        buckets {
          high_value: 2.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 2.0
          high_value: 3.0
          sample_count: 6.666666666666667
        }
        buckets {
          low_value: 3.0
          high_value: 3.0
          sample_count: 6.666666666666667
        }
        type: QUANTILES
      }
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_int64_feature"
    }
  }
  weighted_num_examples: 150.0
}
"""


_BASIC_GOLDEN_INFERRED_SCHEMA = """
feature {
  name: "context_bytes_feature"
  type: BYTES
  bool_domain {
    true_value: "1"
    false_value: "0"
  }
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "context_int64_feature"
  type: INT
  presence {
    min_count: 1
  }
}
feature {
  name: "example_weight"
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "label"
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "##SEQUENCE##"
  type: STRUCT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  value_count {
    min: 1
    max: 1
  }
  struct_domain {
    feature {
      name: "sequence_float_feature"
      value_counts {
        value_count {
          min: 1
          max: 1
        }
        value_count {
          min: 2
          max: 2
        }
      }
      type: FLOAT
      presence {
        min_count: 1
      }
    }
    feature {
      name: "sequence_int64_feature"
      value_counts {
        value_count {
          min: 1
        }
        value_count {
        }
      }
      type: INT
      presence {
        min_fraction: 1.0
        min_count: 1
      }
    }
  }
}
"""

_BASIC_SCHEMA_FOR_VALIDATION = """
feature {
  name: "context_bytes_feature"
  value_counts {
    value_count {
      min: 1
      max: 1
    }
    value_count {
      min: 1
      max: 1
    }
  }
  type: BYTES
  bool_domain {
    true_value: "1"
    false_value: "0"
  }
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "context_int64_feature"
  type: INT
  presence {
    min_count: 1
  }
}
feature {
  name: "example_weight"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "label"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "##SEQUENCE##"
  value_count {
    min: 1
    max: 1
  }
  type: STRUCT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  struct_domain {
    feature {
      name: "sequence_float_feature"
      type: FLOAT
      presence {
        min_count: 1
      }
      value_count {
          min: 1
          max: 1
      }
    }
    feature {
      name: "sequence_int64_feature"
      type: INT
      presence {
        min_fraction: 1.0
        min_count: 1
      }
      value_counts {
        value_count {
          min: 1
        }
        value_count {
          min: 2
          max: 2
        }
      }
    }
  }
}
"""

_BASIC_GOLDEN_ANOMALIES = """
anomaly_info {
  key: "context_bytes_feature"
  value {
    description: "The values have a different nest level than expected. Value "
      "counts will not be checked."
    severity: ERROR
    short_description: "Mismatched value nest level"
    reason {
      type: VALUE_NESTEDNESS_MISMATCH
      short_description: "Mismatched value nest level"
      description: "The values have a different nest level than expected. "
        "Value counts will not be checked."
    }
    path {
      step: "context_bytes_feature"
    }
  }
}
anomaly_info {
  key: "##SEQUENCE##.sequence_float_feature"
  value {
    description: "This feature has a value_count, but the nestedness level of "
      "the feature > 1. For features with nestedness levels greater than 1, "
      "value_counts, not value_count, should be specified."
    severity: ERROR
    short_description: "Mismatched value nest level"
    reason {
      type: VALUE_NESTEDNESS_MISMATCH
      short_description: "Mismatched value nest level"
      description: "This feature has a value_count, but the nestedness level "
        "of the feature > 1. For features with nestedness levels greater than "
        "1, value_counts, not value_count, should be specified."
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_float_feature"
    }
  }
}
anomaly_info {
  key: "##SEQUENCE##.sequence_int64_feature"
  value {
    description: "Some examples have fewer values than expected at nestedness "
      "level 1. Some examples have more values than expected at nestedness "
      "level 1."
    severity: ERROR
    short_description: "Multiple errors"
    reason {
      type: FEATURE_TYPE_LOW_NUMBER_VALUES
      short_description: "Missing values"
      description: "Some examples have fewer values than expected at "
        "nestedness level 1."
    }
    reason {
      type: FEATURE_TYPE_HIGH_NUMBER_VALUES
      short_description: "Superfluous values"
      description: "Some examples have more values than expected at "
        "nestedness level 1."
    }
    path {
      step: "##SEQUENCE##"
      step: "sequence_int64_feature"
    }
  }
}
anomaly_name_format: SERIALIZED_PATH
"""

_BASIC_SCHEMA_FROM_UPDATE = """
feature {
  name: "context_bytes_feature"
  value_count {
    min: 1
    max: 1
  }
  type: BYTES
  bool_domain {
    true_value: "1"
    false_value: "0"
  }
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "context_int64_feature"
  type: INT
  presence {
    min_count: 1
  }
}
feature {
  name: "example_weight"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "label"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "##SEQUENCE##"
  value_count {
    min: 1
    max: 1
  }
  type: STRUCT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  struct_domain {
    feature {
      name: "sequence_float_feature"
      type: FLOAT
      presence {
        min_count: 1
      }
      value_counts {
        value_count {
          min: 1
          max: 1
        }
        value_count {
          min: 2
          max: 2
        }
      }
    }
    feature {
      name: "sequence_int64_feature"
      type: INT
      presence {
        min_fraction: 1.0
        min_count: 1
      }
      value_counts {
        value_count {
          min: 1
        }
        value_count {
          max: 3
        }
      }
    }
  }
}
"""

# Do not inline the goldens in _TEST_CASES. This way indentation is easier to
# manage. The rule is to have no first level indent for goldens.
_TEST_CASES = [
    dict(
        testcase_name='basic',
        stats_options=tfdv.StatsOptions(
            num_rank_histogram_buckets=3,
            num_values_histogram_buckets=3,
            num_histogram_buckets=3,
            num_quantiles_histogram_buckets=3,
            enable_semantic_domain_stats=True),
        expected_stats_pbtxt=_BASIC_GOLDEN_STATS,
        expected_inferred_schema_pbtxt=_BASIC_GOLDEN_INFERRED_SCHEMA,
        schema_for_validation_pbtxt=_BASIC_SCHEMA_FOR_VALIDATION,
        expected_anomalies_pbtxt=_BASIC_GOLDEN_ANOMALIES,
        expected_updated_schema_pbtxt=_BASIC_SCHEMA_FROM_UPDATE,
    ),
    dict(
        testcase_name='weight_and_label',
        stats_options=tfdv.StatsOptions(
            label_feature=_LABEL,
            weight_feature=_EXAMPLE_WEIGHT,
            num_rank_histogram_buckets=3,
            num_values_histogram_buckets=3,
            num_histogram_buckets=3,
            num_quantiles_histogram_buckets=3,
            enable_semantic_domain_stats=True),
        expected_stats_pbtxt=_WEIGHT_AND_LABEL_GOLDEN_STATS,
        expected_inferred_schema_pbtxt=_BASIC_GOLDEN_INFERRED_SCHEMA,
        schema_for_validation_pbtxt=_BASIC_SCHEMA_FOR_VALIDATION,
        expected_anomalies_pbtxt=_BASIC_GOLDEN_ANOMALIES,
        expected_updated_schema_pbtxt=_BASIC_SCHEMA_FROM_UPDATE,
    )
]


class SequenceExampleStatsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(SequenceExampleStatsTest, cls).setUpClass()
    cls._input_file = os.path.join(FLAGS.test_tmpdir,
                                   'sequence_example_stats_test', 'input')
    cls._output_dir = os.path.join(FLAGS.test_tmpdir,
                                   'sequence_example_stats_test', 'output')
    tf.io.gfile.makedirs(os.path.dirname(cls._input_file))
    examples = []
    for _ in range(10):
      examples.append(_EXAMPLE_A)
      examples.append(_EXAMPLE_B)
    with tf.io.TFRecordWriter(cls._input_file) as w:
      for e in examples:
        w.write(e)

  def _assert_schema_equal(self, lhs, rhs):
    def _assert_features_equal(lhs, rhs):
      lhs_feature_map = {f.name: f for f in lhs.feature}
      rhs_feature_map = {f.name: f for f in rhs.feature}
      self.assertEmpty(set(lhs_feature_map) - set(rhs_feature_map))
      self.assertEmpty(set(rhs_feature_map) - set(lhs_feature_map))
      for feature_name, lhs_feature in lhs_feature_map.items():
        rhs_feature = rhs_feature_map[feature_name]
        if lhs_feature.type != schema_pb2.STRUCT:
          self.assertEqual(
              lhs_feature, rhs_feature,
              'feature: {}\n{}\nvs\n{}'.format(feature_name, lhs_feature,
                                               rhs_feature))
        else:
          lhs_feature_copy = copy.copy(lhs_feature)
          rhs_feature_copy = copy.copy(rhs_feature)
          lhs_feature_copy.ClearField('struct_domain')
          rhs_feature_copy.ClearField('struct_domain')
          self.assertEqual(
              lhs_feature_copy, rhs_feature_copy,
              '{} \nvs\n {}'.format(lhs_feature_copy, rhs_feature_copy))
          _assert_features_equal(lhs_feature.struct_domain,
                                 rhs_feature.struct_domain)

    lhs_schema_copy = schema_pb2.Schema()
    lhs_schema_copy.CopyFrom(lhs)
    rhs_schema_copy = schema_pb2.Schema()
    rhs_schema_copy.CopyFrom(rhs)
    lhs_schema_copy.ClearField('feature')
    rhs_schema_copy.ClearField('feature')
    self.assertEqual(lhs_schema_copy, rhs_schema_copy)
    _assert_features_equal(lhs, rhs)

  @parameterized.named_parameters(*_TEST_CASES)
  def test_e2e(self, stats_options, expected_stats_pbtxt,
               expected_inferred_schema_pbtxt, schema_for_validation_pbtxt,
               expected_anomalies_pbtxt, expected_updated_schema_pbtxt):
    tfxio = tf_sequence_example_record.TFSequenceExampleRecord(
        self._input_file, ['tfdv', 'test'])
    stats_file = os.path.join(self._output_dir, 'stats')
    with beam.Pipeline() as p:
      _ = (
          p
          | 'TFXIORead' >> tfxio.BeamSource()
          | 'GenerateStats' >> tfdv.GenerateStatistics(stats_options)
          | 'WriteStats' >> tfdv.WriteStatisticsToTFRecord(stats_file))

    actual_stats = tfdv.load_statistics(stats_file)
    test_util.make_dataset_feature_stats_list_proto_equal_fn(
        self,
        text_format.Parse(expected_stats_pbtxt,
                          statistics_pb2.DatasetFeatureStatisticsList()))(
                              [actual_stats])
    actual_inferred_schema = tfdv.infer_schema(
        actual_stats, infer_feature_shape=True)

    if hasattr(actual_inferred_schema, 'generate_legacy_feature_spec'):
      actual_inferred_schema.ClearField('generate_legacy_feature_spec')
    self._assert_schema_equal(
        actual_inferred_schema,
        text_format.Parse(expected_inferred_schema_pbtxt, schema_pb2.Schema()))

    schema_for_validation = text_format.Parse(schema_for_validation_pbtxt,
                                              schema_pb2.Schema())
    actual_anomalies = tfdv.validate_statistics(actual_stats,
                                                schema_for_validation)
    actual_anomalies.ClearField('baseline')
    self.assertEqual(
        actual_anomalies,
        text_format.Parse(expected_anomalies_pbtxt, anomalies_pb2.Anomalies()))

    actual_updated_schema = tfdv.update_schema(
        schema_for_validation, actual_stats)
    self._assert_schema_equal(
        actual_updated_schema,
        text_format.Parse(expected_updated_schema_pbtxt, schema_pb2.Schema()))


if __name__ == '__main__':
  absltest.main()
