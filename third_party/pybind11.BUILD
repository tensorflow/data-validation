# pybind11 - Seamless operability between C++11 and Python.
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

OPTIONS = [
    "-fexceptions",
    # Useless warnings
    "-Xclang-only=-Wno-undefined-inline",
    "-Xclang-only=-Wno-pragma-once-outside-header",
    "-Xgcc-only=-Wno-error",  # no way to just disable the pragma-once warning in gcc
]

INCLUDES = [
    "include/pybind11/*.h",
    "include/pybind11/detail/*.h",
]

EXCLUDES = [
    # Deprecated file that just emits a warning
    "include/pybind11/common.h",
]

cc_library(
    name = "pybind11",
    hdrs = glob(
        INCLUDES,
        exclude = EXCLUDES,
    ),
    copts = OPTIONS,
    includes = ["include"],
    deps = ["@local_config_python//:python_headers"],
)
