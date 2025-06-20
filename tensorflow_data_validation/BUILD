load("@bazel_skylib//lib:selects.bzl", "selects")

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_arm64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
)

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = select({
        "//conditions:default": [
            "//tensorflow_data_validation/anomalies/proto:validation_config_proto_py_pb2",
            "//tensorflow_data_validation/anomalies/proto:validation_metadata_proto_py_pb2",
            "//tensorflow_data_validation/pywrap:tensorflow_data_validation_extension.so",
            "//tensorflow_data_validation/skew/protos:feature_skew_results_proto_py_pb2",
        ],
    }),
)
