# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "arrow_format",
    srcs = [
        "cpp/src/arrow/ipc/feather.fbs",
        "format/File.fbs",
        "format/Message.fbs",
        "format/Schema.fbs",
        "format/Tensor.fbs",
    ],
    flatc_args = [
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "cpp/src/arrow/ipc/",
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/compute/**/*.cc",
            "cpp/src/arrow/io/**/*.cc",
            "cpp/src/arrow/ipc/**/*.cc",
            "cpp/src/arrow/python/**/*.cc",
            "cpp/src/arrow/util/**/*.cc",
        ],
        exclude = [
            # Excluding files which we don't depend on, but needs
            # additional dependencies like boost, snappy etc.
            "cpp/src/arrow/util/compression*",
            "cpp/src/arrow/**/*test*.cc",
            "cpp/src/arrow/**/*benchmark*.cc",
            "cpp/src/arrow/**/*hdfs*.cc",
            "cpp/src/arrow/ipc/json*.cc",
            "cpp/src/arrow/ipc/stream-to-file.cc",
            "cpp/src/arrow/ipc/file-to-stream.cc",
        ],
    ),
    hdrs = glob(["cpp/src/arrow/**/*.h"]),
    includes = [
        "cpp/src",
    ],
    deps = [
        ":arrow_format",
        "@boost//:algorithm",
        "@boost//:filesystem",
        # These libs are defined by TensorFlow.
        "@double_conversion//:double-conversion",
        "@local_config_python//:python_headers",
    ],
)
