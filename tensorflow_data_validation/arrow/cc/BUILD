licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow_data_validation/pywrap:__subpackages__"])

cc_library(
    name = "arrow_util",
    srcs = ["arrow_util.cc"],
    hdrs = ["arrow_util.h"],
    deps = [
        "@arrow",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",
    ],
)
