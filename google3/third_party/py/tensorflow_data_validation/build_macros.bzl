"""BUILD macros."""

load("//third_party/bazel_rules/rules_python/python:py_extension.bzl", "py_extension")

def tfdv_pybind_extension(
        name,
        srcs,
        module_name,
        deps = [],
        visibility = None):
    py_extension(
        name = name,
        module_name = module_name,
        srcs = srcs,
        srcs_version = "PY3ONLY",
        copts = [
            "-fno-strict-aliasing",
            "-fexceptions",
        ],
        features = ["-use_header_modules"],
        deps = deps,
        visibility = visibility,
    )
