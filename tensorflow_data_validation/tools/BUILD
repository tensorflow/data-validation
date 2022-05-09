# Opensource tools, not part of the pip package.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_data_validation",
        "//third_party/py/absl:app",
        "//third_party/py/apache_beam",
        "//third_party/py/tensorflow_docs/api_generator",
    ],
)
