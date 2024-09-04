load("//tensorflow_data_validation:data_validation.bzl", "tfdv_proto_library", "tfdv_proto_library_py")

package(
    default_visibility = [
        "//tensorflow_data_validation:__subpackages__",
    ],
)

licenses(["notice"])

tfdv_proto_library(
    name = "feature_skew_results_proto",
    srcs = ["feature_skew_results.proto"],
)

tfdv_proto_library_py(
    name = "feature_skew_results_proto_py_pb2",
    srcs = ["feature_skew_results.proto"],
    proto_library = "feature_skew_results_proto",
    deps = [":feature_skew_results_proto"],
)
