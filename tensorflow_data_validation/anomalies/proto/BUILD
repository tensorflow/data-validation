load("//tensorflow_data_validation:data_validation.bzl", "tfdv_proto_library", "tfdv_proto_library_py")

package(
    default_visibility = ["//tensorflow_data_validation:__subpackages__"],
)

licenses(["notice"])

tfdv_proto_library(
    name = "feature_statistics_to_proto_proto",
    srcs = ["feature_statistics_to_proto.proto"],
    deps = [
        ":validation_config_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto",
    ],
)

tfdv_proto_library(
    name = "validation_config_proto",
    srcs = ["validation_config.proto"],
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto"],
)

tfdv_proto_library_py(
    name = "validation_config_proto_py_pb2",
    deps = [":validation_config_proto"],
)

tfdv_proto_library(
    name = "validation_metadata_proto",
    srcs = ["validation_metadata.proto"],
    deps = ["@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto"],
)

tfdv_proto_library_py(
    name = "validation_metadata_proto_py_pb2",
    deps = [":validation_metadata_proto"],
)
