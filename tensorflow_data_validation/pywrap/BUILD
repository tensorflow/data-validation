# Package TFDV SWIG/C++ code.

package(default_visibility = [
    "//tensorflow_data_validation:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

# Note that the name of the target should follow specific naming
# pattern specified in tensorflow/tf_exported_symbols.lds in order
# for the init function in the generated .so file to be exported.
tf_py_wrap_cc(
    name = "pywrap_tensorflow_data_validation",
    srcs = ["tensorflow_data_validation.i"],
    swig_includes = [
        "validation_api.i",
    ],
    deps = [
        "//tensorflow_data_validation/anomalies:feature_statistics_validator",
        "@local_config_python//:python_headers",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
