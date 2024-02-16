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
"""Tests for StatsOptions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_data_validation import types
from tensorflow_data_validation.statistics import stats_options
from tensorflow_data_validation.statistics.generators import lift_stats_generator
from tensorflow_data_validation.utils import slicing_util
from tfx_bsl.public.proto import slicing_spec_pb2

from google.protobuf import text_format
from tensorflow.python.util.protobuf import compare  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


INVALID_STATS_OPTIONS = [
    {
        'testcase_name': 'invalid_generators',
        'stats_options_kwargs': {
            'generators': {}
        },
        'exception_type': TypeError,
        'error_message': 'generators is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_generator',
        'stats_options_kwargs': {
            'generators': [{}]
        },
        'exception_type': TypeError,
        'error_message': 'Statistics generator must extend one of '
                         'CombinerStatsGenerator, TransformStatsGenerator, '
                         'or CombinerFeatureStatsGenerator '
                         'found object of type dict.'
    },
    {
        'testcase_name': 'invalid_feature_allowlist',
        'stats_options_kwargs': {
            'feature_allowlist': {}
        },
        'exception_type': TypeError,
        'error_message': 'feature_allowlist is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_schema',
        'stats_options_kwargs': {
            'schema': {}
        },
        'exception_type': TypeError,
        'error_message': 'schema is of type dict, should be a Schema proto.'
    },
    {
        'testcase_name': 'invalid_vocab_paths',
        'stats_options_kwargs': {
            'vocab_paths': []
        },
        'exception_type': TypeError,
        'error_message': 'vocab_paths is of type list, should be a dict.'
    },
    {
        'testcase_name': 'invalid_slice_functions_list',
        'stats_options_kwargs': {
            'slice_functions': {}
        },
        'exception_type': TypeError,
        'error_message': 'slice_functions is of type dict, should be a list.'
    },
    {
        'testcase_name': 'invalid_slice_function_type',
        'stats_options_kwargs': {
            'slice_functions': [1]
        },
        'exception_type': TypeError,
        'error_message': 'slice_functions must contain functions only.'
    },
    {
        'testcase_name': 'sample_rate_zero',
        'stats_options_kwargs': {
            'sample_rate': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate 0'
    },
    {
        'testcase_name': 'sample_rate_negative',
        'stats_options_kwargs': {
            'sample_rate': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate -1'
    },
    {
        'testcase_name': 'sample_rate_above_one',
        'stats_options_kwargs': {
            'sample_rate': 2
        },
        'exception_type': ValueError,
        'error_message': 'Invalid sample_rate 2'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_one',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': 1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets 1'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_zero',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets 0'
    },
    {
        'testcase_name': 'num_values_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_values_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_values_histogram_buckets -1'
    },
    {
        'testcase_name': 'num_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_histogram_buckets -1'
    },
    {
        'testcase_name': 'num_quantiles_histogram_buckets_negative',
        'stats_options_kwargs': {
            'num_quantiles_histogram_buckets': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid num_quantiles_histogram_buckets -1'
    },
    {
        'testcase_name': 'desired_batch_size_zero',
        'stats_options_kwargs': {
            'desired_batch_size': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid desired_batch_size 0'
    },
    {
        'testcase_name': 'desired_batch_size_negative',
        'stats_options_kwargs': {
            'desired_batch_size': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid desired_batch_size -1'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_zero',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': 0
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate 0'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_negative',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': -1
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate -1'
    },
    {
        'testcase_name': 'semantic_domain_stats_sample_rate_above_one',
        'stats_options_kwargs': {
            'semantic_domain_stats_sample_rate': 2
        },
        'exception_type': ValueError,
        'error_message': 'Invalid semantic_domain_stats_sample_rate 2'
    },
    {
        'testcase_name':
            'categorical_float_without_sketch_generators',
        'stats_options_kwargs': {
            'use_sketch_based_topk_uniques':
                False,
            'schema':
                schema_pb2.Schema(
                    feature=[
                        schema_pb2.Feature(
                            name='f',
                            type=schema_pb2.FLOAT,
                            float_domain=schema_pb2.FloatDomain(
                                is_categorical=True))
                    ],),
        },
        'exception_type':
            ValueError,
        'error_message': ('Categorical float features set in schema require '
                          'use_sketch_based_topk_uniques'),
    },
    {
        'testcase_name': 'both_slice_fns_and_slice_sqls_specified',
        'stats_options_kwargs': {
            'experimental_slice_functions': [lambda x: (None, x)],
            'experimental_slice_sqls': ['']
        },
        'exception_type': ValueError,
        'error_message': 'Only one of experimental_slice_functions or'
    },
    {
        'testcase_name':
            'both_slicing_config_and_slice_fns_specified',
        'stats_options_kwargs': {
            'experimental_slice_functions': [lambda x: (None, x)],
            'slicing_config':
                text_format.Parse(
                    """
              slicing_specs {
                feature_keys: ["country", "city"]
              }
              """, slicing_spec_pb2.SlicingConfig()),
        },
        'exception_type':
            ValueError,
        'error_message':
            'Specify only one of slicing_config or experimental_slice_functions.'
    },
    {
        'testcase_name':
            'both_slicing_config_and_slice_sqls_specified',
        'stats_options_kwargs': {
            'experimental_slice_sqls': [''],
            'slicing_config':
                text_format.Parse(
                    """
              slicing_specs {
                feature_keys: ["country", "city"]
              }
              """, slicing_spec_pb2.SlicingConfig()),
        },
        'exception_type':
            ValueError,
        'error_message':
            'Specify only one of slicing_config or experimental_slice_sqls.'
    },
    {
        'testcase_name': 'both_functions_and_sqls_in_slicing_config',
        'stats_options_kwargs': {
            'slicing_config':
                text_format.Parse(
                    """
            slicing_specs {
              feature_keys: ["country", "city"]
            }
            slicing_specs {
              slice_keys_sql: "SELECT STRUCT(education) FROM example.education"
            }
            """, slicing_spec_pb2.SlicingConfig()),
        },
        'exception_type': ValueError,
        'error_message':
            'Only one of slicing features or slicing sql queries can be '
            'specified in the slicing config.'
    },
]


class StatsOptionsTest(parameterized.TestCase):

  @parameterized.named_parameters(*INVALID_STATS_OPTIONS)
  def test_stats_options(self, stats_options_kwargs, exception_type,
                         error_message):
    with self.assertRaisesRegex(exception_type, error_message):
      stats_options.StatsOptions(**stats_options_kwargs)

  def test_stats_options_invalid_slicing_sql_query(self):
    schema = schema_pb2.Schema(
        feature=[schema_pb2.Feature(name='feat1', type=schema_pb2.BYTES),
                 schema_pb2.Feature(name='feat3', type=schema_pb2.INT)],)
    experimental_slice_sqls = [
        """
        SELECT
          STRUCT(feat1, feat2)
        FROM
          example.feat1, example.feat2
        """
    ]
    with self.assertLogs(level='ERROR') as log_output:
      stats_options.StatsOptions(
          experimental_slice_sqls=experimental_slice_sqls, schema=schema)
      self.assertLen(log_output.records, 1)
      self.assertRegex(log_output.records[0].message,
                       'One of the slice SQL query .*')

  def test_valid_stats_options_json_round_trip(self):
    feature_allowlist = ['a']
    schema = schema_pb2.Schema(feature=[schema_pb2.Feature(name='f')])
    vocab_paths = {'a': '/path/to/a'}
    label_feature = 'label'
    weight_feature = 'weight'
    sample_rate = 0.01
    num_top_values = 21
    frequency_threshold = 2
    weighted_frequency_threshold = 2.0
    num_rank_histogram_buckets = 1001
    num_values_histogram_buckets = 11
    num_histogram_buckets = 11
    num_quantiles_histogram_buckets = 11
    epsilon = 0.02
    infer_type_from_schema = True
    desired_batch_size = 100
    enable_semantic_domain_stats = True
    semantic_domain_stats_sample_rate = 0.1
    per_feature_weight_override = {types.FeaturePath(['a']): 'w'}
    add_default_generators = True
    use_sketch_based_topk_uniques = True
    experimental_result_partitions = 3
    slicing_config = text_format.Parse(
        """
        slicing_specs {
          feature_keys: ["country", "city"]
        }
        """, slicing_spec_pb2.SlicingConfig())

    options = stats_options.StatsOptions(
        feature_allowlist=feature_allowlist,
        schema=schema,
        vocab_paths=vocab_paths,
        label_feature=label_feature,
        weight_feature=weight_feature,
        sample_rate=sample_rate,
        num_top_values=num_top_values,
        frequency_threshold=frequency_threshold,
        weighted_frequency_threshold=weighted_frequency_threshold,
        num_rank_histogram_buckets=num_rank_histogram_buckets,
        num_values_histogram_buckets=num_values_histogram_buckets,
        num_histogram_buckets=num_histogram_buckets,
        num_quantiles_histogram_buckets=num_quantiles_histogram_buckets,
        epsilon=epsilon,
        infer_type_from_schema=infer_type_from_schema,
        desired_batch_size=desired_batch_size,
        enable_semantic_domain_stats=enable_semantic_domain_stats,
        semantic_domain_stats_sample_rate=semantic_domain_stats_sample_rate,
        per_feature_weight_override=per_feature_weight_override,
        add_default_generators=add_default_generators,
        experimental_use_sketch_based_topk_uniques=use_sketch_based_topk_uniques,
        experimental_result_partitions=experimental_result_partitions,
        slicing_config=slicing_config,
    )

    options_json = options.to_json()
    options = stats_options.StatsOptions.from_json(options_json)

    self.assertEqual(feature_allowlist, options.feature_allowlist)
    compare.assertProtoEqual(self, schema, options.schema)
    self.assertEqual(vocab_paths, options.vocab_paths)
    self.assertEqual(label_feature, options.label_feature)
    self.assertEqual(weight_feature, options.weight_feature)
    self.assertEqual(sample_rate, options.sample_rate)
    self.assertEqual(num_top_values, options.num_top_values)
    self.assertEqual(frequency_threshold, options.frequency_threshold)
    self.assertEqual(weighted_frequency_threshold,
                     options.weighted_frequency_threshold)
    self.assertEqual(num_rank_histogram_buckets,
                     options.num_rank_histogram_buckets)
    self.assertEqual(num_values_histogram_buckets,
                     options.num_values_histogram_buckets)
    self.assertEqual(num_histogram_buckets, options.num_histogram_buckets)
    self.assertEqual(num_quantiles_histogram_buckets,
                     options.num_quantiles_histogram_buckets)
    self.assertEqual(epsilon, options.epsilon)
    self.assertEqual(infer_type_from_schema, options.infer_type_from_schema)
    self.assertEqual(desired_batch_size, options.desired_batch_size)
    self.assertEqual(enable_semantic_domain_stats,
                     options.enable_semantic_domain_stats)
    self.assertEqual(semantic_domain_stats_sample_rate,
                     options.semantic_domain_stats_sample_rate)
    self.assertEqual(per_feature_weight_override,
                     options._per_feature_weight_override)
    self.assertEqual(add_default_generators, options.add_default_generators)
    self.assertEqual(use_sketch_based_topk_uniques,
                     options.use_sketch_based_topk_uniques)
    self.assertEqual(experimental_result_partitions,
                     options.experimental_result_partitions)
    self.assertEqual(slicing_config, options.slicing_config)

  def test_stats_options_with_generators_to_json(self):
    generators = [
        lift_stats_generator.LiftStatsGenerator(
            schema=None,
            y_path=types.FeaturePath(['label']),
            x_paths=[types.FeaturePath(['feature'])])
    ]
    options = stats_options.StatsOptions(
        generators=generators)
    with self.assertRaisesRegex(ValueError, 'StatsOptions cannot be converted'):
      options.to_json()

  def test_stats_options_with_slice_fns_to_json(self):
    slice_functions = [slicing_util.get_feature_value_slicer({'b': None})]
    options = stats_options.StatsOptions(
        experimental_slice_functions=slice_functions)
    with self.assertRaisesRegex(ValueError, 'StatsOptions cannot be converted'):
      options.to_json()

  @parameterized.named_parameters(
      {'testcase_name': 'no_type_name'},
      {
          'testcase_name': 'type_name_correct',
          'type_name': 'StatsOptions'
      },
      {
          'testcase_name': 'type_name_incorrect',
          'type_name': 'BorkBorkBork',
          'want_exception': True
      },
  )
  def test_stats_options_from_json(self,
                                   type_name: Optional[str] = None,
                                   want_exception: bool = False):
    if type_name:
      type_name_line = f',\n"TYPE_NAME": "{type_name}"\n'
    else:
      type_name_line = ''
    options_json = """{
      "_generators": null,
      "_feature_allowlist": null,
      "_schema": null,
      "_vocab_paths": null,
      "weight_feature": null,
      "label_feature": null,
      "_slice_functions": null,
      "_sample_rate": null,
      "num_top_values": 20,
      "frequency_threshold": 1,
      "weighted_frequency_threshold": 1.0,
      "num_rank_histogram_buckets": 1000,
      "_num_values_histogram_buckets": 10,
      "_num_histogram_buckets": 10,
      "_num_quantiles_histogram_buckets": 10,
      "epsilon": 0.01,
      "infer_type_from_schema": false,
      "_desired_batch_size": null,
      "enable_semantic_domain_stats": false,
      "_semantic_domain_stats_sample_rate": null,
      "_per_feature_weight_override": null,
      "_add_default_generators": true,
      "_use_sketch_based_topk_uniques": false,
      "_slice_sqls": null,
      "_experimental_result_partitions": 1,
      "_experimental_num_feature_partitions": 1,
      "_slicing_config": null,
      "_experimental_filter_read_paths": false,
      "_per_feature_stats_config": null
    """
    options_json += type_name_line + '}'
    if want_exception:
      with self.assertRaises(ValueError):
        _ = stats_options.StatsOptions.from_json(options_json)
    else:
      actual_options = stats_options.StatsOptions.from_json(options_json)
      expected_options_dict = stats_options.StatsOptions().__dict__
      self.assertEqual(expected_options_dict, actual_options.__dict__)

  def test_example_weight_map(self):
    options = stats_options.StatsOptions()
    self.assertIsNone(options.example_weight_map.get(types.FeaturePath(['f'])))
    self.assertEqual(frozenset([]),
                     options.example_weight_map.all_weight_features())

    options = stats_options.StatsOptions(weight_feature='w')
    self.assertEqual('w',
                     options.example_weight_map.get(types.FeaturePath(['f'])))
    self.assertEqual(
        frozenset(['w']),
        options.example_weight_map.all_weight_features())

    options = stats_options.StatsOptions(
        per_feature_weight_override={types.FeaturePath(['x']): 'w'})
    self.assertIsNone(options.example_weight_map.get(types.FeaturePath(['f'])))
    self.assertEqual('w',
                     options.example_weight_map.get(types.FeaturePath(['x'])))
    self.assertEqual(frozenset(['w']),
                     options.example_weight_map.all_weight_features())

  def test_sketch_based_uniques_set_both_fields(self):
    with self.assertRaises(ValueError):
      stats_options.StatsOptions(
          experimental_use_sketch_based_topk_uniques=True,
          use_sketch_based_topk_uniques=True)

  def test_sketch_based_uniques_construct_old(self):
    opts = stats_options.StatsOptions(
        experimental_use_sketch_based_topk_uniques=True)
    self.assertTrue(opts.use_sketch_based_topk_uniques)

  def test_sketch_based_uniques_construct_new(self):
    opts = stats_options.StatsOptions(use_sketch_based_topk_uniques=True)
    self.assertTrue(opts.use_sketch_based_topk_uniques)

  def test_sketch_based_uniques_set_old(self):
    opts = stats_options.StatsOptions()
    opts.experimental_use_sketch_based_topk_uniques = True
    self.assertTrue(opts.use_sketch_based_topk_uniques)

  def test_sketch_based_uniques_set_new(self):
    opts = stats_options.StatsOptions()
    opts.use_sketch_based_topk_uniques = True
    self.assertTrue(opts.use_sketch_based_topk_uniques)


if __name__ == '__main__':
  absltest.main()
