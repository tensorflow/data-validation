load("//tensorflow_data_validation:data_validation.bzl", "tfdv_proto_library", "tfdv_proto_library_py")

package(default_visibility = [
    "//tensorflow_data_validation:__subpackages__",
])

licenses(["notice"])

tfdv_proto_library(
    name = "feature_skew_results_proto",
    srcs = ["feature_skew_results.proto"],
    cc_api_version = 2,
    deps = ["@org_tensorflow//tensorflow/core:protos_all_py"],
)

tfdv_proto_library_py(
    name = "feature_skew_results_proto_py_pb2",
    srcs = ["feature_skew_results.proto"],
    proto_library = "feature_skew_results_proto",
    deps = [":feature_skew_results_proto"],
)

tfdv_proto_library(
    name = "skew_config_proto",
    srcs = ["skew_config.proto"],
    cc_api_version = 2,
)

tfdv_proto_library_py(
    name = "skew_config_proto_py_pb2",
    srcs = ["skew_config.proto"],
    proto_library = "skew_config_proto",
    deps = [":skew_config_proto"],
)
