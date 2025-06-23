# Copyright 2019 Google LLC
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
"""Tests for the slicing utilities."""

import pyarrow as pa
from absl.testing import absltest
from google.protobuf import text_format
from tfx_bsl.public.proto import slicing_spec_pb2

from tensorflow_data_validation.utils import slicing_util


class SlicingUtilTest(absltest.TestCase):
    # This should be simply self.assertCountEqual(), but
    # RecordBatch.__eq__ is not implemented.
    # TODO(zhuo): clean-up after ARROW-8277 is available.
    def _check_results(self, got, expected):
        got_dict = {g[0]: g[1] for g in got}
        expected_dict = {e[0]: e[1] for e in expected}

        self.assertCountEqual(got_dict.keys(), expected_dict.keys())
        for k, got_record_batch in got_dict.items():
            expected_record_batch = expected_dict[k]
            self.assertTrue(got_record_batch.equals(expected_record_batch))

    def test_get_feature_value_slicer(self):
        features = {"a": None, "b": None}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1], [3], [2, 1, 1], [3]]),
                pa.array([["dog"], ["cat"], ["wolf"], ["dog", "wolf"], ["wolf"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "a_1_b_dog",
                pa.RecordBatch.from_arrays(
                    [pa.array([[1], [2, 1, 1]]), pa.array([["dog"], ["dog", "wolf"]])],
                    ["a", "b"],
                ),
            ),
            (
                "a_1_b_cat",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
            (
                "a_2_b_cat",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
            (
                "a_2_b_dog",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1, 1]]), pa.array([["dog", "wolf"]])], ["a", "b"]
                ),
            ),
            (
                "a_1_b_wolf",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1, 1]]), pa.array([["dog", "wolf"]])], ["a", "b"]
                ),
            ),
            (
                "a_2_b_wolf",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1, 1]]), pa.array([["dog", "wolf"]])], ["a", "b"]
                ),
            ),
            (
                "a_3_b_wolf",
                pa.RecordBatch.from_arrays(
                    [pa.array([[3], [3]]), pa.array([["wolf"], ["wolf"]])], ["a", "b"]
                ),
            ),
        ]
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_one_feature_not_in_batch(self):
        features = {"not_an_actual_feature": None, "a": None}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "a_1",
                pa.RecordBatch.from_arrays(
                    [pa.array([[1], [2, 1]]), pa.array([["dog"], ["cat"]])], ["a", "b"]
                ),
            ),
            (
                "a_2",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
        ]
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_single_feature(self):
        features = {"a": [2]}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "a_2",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
        ]
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_no_slice(self):
        features = {"a": [3]}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = []
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_feature_not_in_record_batch(self):
        features = {"c": [0]}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = []
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_feature_not_in_record_batch_all_values(self):
        features = {"c": None}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = []
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_bytes_feature_valid_utf8(self):
        features = {"b": None}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([[b"dog"], [b"cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "b_dog",
                pa.RecordBatch.from_arrays(
                    [pa.array([[1]]), pa.array([[b"dog"]])], ["a", "b"]
                ),
            ),
            (
                "b_cat",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([[b"cat"]])], ["a", "b"]
                ),
            ),
        ]
        self._check_results(
            slicing_util.get_feature_value_slicer(features)(input_record_batch),
            expected_result,
        )

    def test_get_feature_value_slicer_non_utf8_slice_key(self):
        features = {"a": None}
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[b"\xf0"], ["cat"]]),
            ],
            ["a"],
        )
        with self.assertRaisesRegex(ValueError, "must be valid UTF-8"):
            _ = list(
                slicing_util.get_feature_value_slicer(features)(input_record_batch)
            )

    def test_convert_slicing_config_to_fns(self):
        slicing_config = text_format.Parse(
            """
        slicing_specs {}
        slicing_specs {
          feature_keys: ["country"]
        }
        slicing_specs {
          feature_keys: ["state"]
          feature_values: [{key: "age", value: "20"}]
        }
        """,
            slicing_spec_pb2.SlicingConfig(),
        )

        slicing_fns = slicing_util.convert_slicing_config_to_slice_functions(
            slicing_config
        )
        self.assertLen(slicing_fns, 2)

        slicing_config = text_format.Parse(
            """
        slicing_specs {
          feature_values: [{key: "a", value: "2"}]
        }
        """,
            slicing_spec_pb2.SlicingConfig(),
        )
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([["1"], ["2", "1"]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "a_2",
                pa.RecordBatch.from_arrays(
                    [pa.array([["2", "1"]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
        ]
        slicing_fns = slicing_util.convert_slicing_config_to_slice_functions(
            slicing_config
        )
        self._check_results(slicing_fns[0](input_record_batch), expected_result)

    def test_convert_slicing_config_to_fns_on_int_field(self):
        slicing_config = text_format.Parse(
            """
        slicing_specs {
          feature_values: [{key: "a", value: "2"}]
        }
        """,
            slicing_spec_pb2.SlicingConfig(),
        )
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )
        expected_result = [
            (
                "a_2",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
        ]
        slicing_fns = slicing_util.convert_slicing_config_to_slice_functions(
            slicing_config
        )
        self._check_results(slicing_fns[0](input_record_batch), expected_result)

    def test_convert_slicing_config_to_fns_on_int_invalid(self):
        slicing_config = text_format.Parse(
            """
        slicing_specs {
          feature_values: [{key: "a", value: "2.5"}]
        }
        """,
            slicing_spec_pb2.SlicingConfig(),
        )
        input_record_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1], [2, 1]]),
                pa.array([["dog"], ["cat"]]),
            ],
            ["a", "b"],
        )

        expected_result = [
            (
                "a_2",
                pa.RecordBatch.from_arrays(
                    [pa.array([[2, 1]]), pa.array([["cat"]])], ["a", "b"]
                ),
            ),
        ]
        slicing_fns = slicing_util.convert_slicing_config_to_slice_functions(
            slicing_config
        )

        with self.assertRaisesRegex(
            ValueError, "The feature to slice on has integer values but*"
        ):
            self._check_results(slicing_fns[0](input_record_batch), expected_result)


if __name__ == "__main__":
    absltest.main()
