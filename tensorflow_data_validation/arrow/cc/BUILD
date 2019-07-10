licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//tensorflow_data_validation/google/coders:__subpackages__",
    "//tensorflow_data_validation/google/scratch/arrow:__subpackages__",
    "//tensorflow_data_validation/pywrap:__subpackages__",
])

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@arrow",
        "@local_config_python//:numpy_headers",
    ],
)

cc_library(
    name = "arrow_util",
    srcs = ["arrow_util.cc"],
    hdrs = ["arrow_util.h"],
    deps = [
        ":common",
        ":init_numpy",
        "@arrow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "decoded_examples_to_arrow",
    srcs = ["decoded_examples_to_arrow.cc"],
    hdrs = ["decoded_examples_to_arrow.h"],
    deps = [
        ":common",
        ":init_numpy",
        ":pyarrow_numpy_stub",
        "@arrow",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "init_numpy",
    srcs = ["init_numpy.cc"],
    hdrs = ["init_numpy.h"],
    deps = [
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "pyarrow_numpy_stub",
    srcs = ["pyarrow_numpy_stub.cc"],
    hdrs = ["pyarrow_numpy_stub.h"],
    deps = [
        "@arrow",
        "@local_config_python//:numpy_headers",
    ],
)
