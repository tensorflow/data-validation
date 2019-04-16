# Fast example coder.

package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "fast_example_decoder",
    srcs = ["fast_example_decoder.cc"],
    hdrs = ["fast_example_decoder.h"],
    deps = [
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
