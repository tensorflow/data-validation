licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow_data_validation/pywrap:__subpackages__"])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    deps = [],
)

cc_library(
    name = "arrow_util",
    srcs = ["arrow_util.cc"],
    hdrs = ["arrow_util.h"],
    deps = [
        ":macros",
        "@arrow",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "decoded_examples_to_arrow",
    srcs = ["decoded_examples_to_arrow.cc"],
    hdrs = ["decoded_examples_to_arrow.h"],
    deps = [
        ":macros",
        "@arrow",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
    ],
)
