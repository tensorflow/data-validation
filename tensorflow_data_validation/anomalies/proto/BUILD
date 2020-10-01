package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow_data_validation:data_validation.bzl", "tfdv_proto_library", "tfdv_proto_library_py")

tfdv_proto_library(
    name = "feature_statistics_to_proto_proto",
    srcs = ["feature_statistics_to_proto.proto"],
    cc_api_version = 2,
    deps = [
        ":validation_config_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc",
    ],
)

tfdv_proto_library(
    name = "validation_config_proto",
    srcs = ["validation_config.proto"],
    cc_api_version = 2,
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc"],
)

tfdv_proto_library_py(
    name = "validation_config_proto_py_pb2",
    srcs = ["validation_config.proto"],
    proto_library = "validation_config_proto",
    deps = [":validation_config_proto"],
)

tfdv_proto_library(
    name = "validation_metadata_proto",
    srcs = ["validation_metadata.proto"],
    cc_api_version = 2,
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc"],
)

tfdv_proto_library_py(
    name = "validation_metadata_proto_py_pb2",
    srcs = ["validation_metadata.proto"],
    proto_library = "validation_metadata_proto",
    deps = [":validation_metadata_proto"],
)
