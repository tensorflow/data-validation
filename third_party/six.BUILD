# This file is copied from https://github.com/abseil/abseil-py/blob/master/third_party/six.BUILD.
# It is needed to get TFDV to build with a dependency on Zetasql.
# Description:
#   Six provides simple utilities for wrapping over differences between Python 2
#   and Python 3.

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "six",
    srcs = ["six.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
