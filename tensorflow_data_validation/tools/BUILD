# Opensource tools, not part of the pip package.

load("@rules_python//python:defs.bzl", "py_binary")

licenses(["notice"])

package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    deps = [
        "//tensorflow_data_validation",
    ],
)
