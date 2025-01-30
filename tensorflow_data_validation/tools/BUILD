# Opensource tools, not part of the pip package.

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

licenses(["notice"])

package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    deps = [
        "# Implicit absl dependency:app",
        "# Implicit apache_beam dependency.",
        "# Implicit tensorflow_docs dependency./api_generator",
        "//tensorflow_data_validation",
    ],
)
