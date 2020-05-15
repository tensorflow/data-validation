# Description:
#   C++ libraries in TFDV.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(default_visibility = [
    "//tensorflow_data_validation:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

pybind_extension(
    name = "tensorflow_data_validation_extension",
    srcs = [
        "tensorflow_data_validation_extension.cc",
    ],
    module_name = "tensorflow_data_validation_extension",
    deps = [
        ":validation_submodule",
        "@pybind11",
    ],
)

py_library(
    name = "tensorflow_data_validation_extension_lib",
    srcs = ["__init__.py"],
    deps = [
        ":tensorflow_data_validation_extension",
    ],
)

cc_library(
    name = "validation_submodule",
    srcs = ["validation_submodule.cc"],
    hdrs = ["validation_submodule.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow_data_validation/anomalies:feature_statistics_validator",
        "@org_tensorflow//tensorflow/core:lib",
        "@pybind11",
    ],
)
