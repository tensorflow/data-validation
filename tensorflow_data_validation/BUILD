licenses(["notice"])  # Apache 2.0

load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

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
        ":macos_arm64"
    ],
)

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = select({
        ":windows": [
            "//tensorflow_data_validation/anomalies/proto:custom_validation_config_pb2.py",
            "//tensorflow_data_validation/anomalies/proto:validation_config_pb2.py",
            "//tensorflow_data_validation/anomalies/proto:validation_metadata_pb2.py",
            "//tensorflow_data_validation/pywrap:tensorflow_data_validation_extension.pyd",
            "//tensorflow_data_validation/skew/protos:feature_skew_results_pb2.py",
        ],
        "//conditions:default": [
            "//tensorflow_data_validation/anomalies/proto:custom_validation_config_pb2.py",
            "//tensorflow_data_validation/anomalies/proto:validation_config_pb2.py",
            "//tensorflow_data_validation/anomalies/proto:validation_metadata_pb2.py",
            "//tensorflow_data_validation/pywrap:tensorflow_data_validation_extension.so",
            "//tensorflow_data_validation/skew/protos:feature_skew_results_pb2.py",
        ],
    }),
)
