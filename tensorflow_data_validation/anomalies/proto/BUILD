package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_data_validation:data_validation.bzl", "tfdv_proto_library")

tfdv_proto_library(
    name = "feature_statistics_to_proto_proto",
    srcs = ["feature_statistics_to_proto.proto"],
    cc_api_version = 2,
)

tfdv_proto_library(
    name = "validation_config_proto",
    srcs = ["validation_config.proto"],
    cc_api_version = 2,
)
