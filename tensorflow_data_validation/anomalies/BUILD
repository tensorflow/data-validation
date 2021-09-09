# Description:
# Code for anomaly detection: example-based validation, and checks of data statistics.
# Code for automatically generating and modifying schemas.

package(default_visibility = ["//tensorflow_data_validation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "features_needed",
    srcs = ["features_needed.cc"],
    hdrs = ["features_needed.h"],
    deps = [
        ":path",
        "//tensorflow_data_validation/anomalies/proto:validation_metadata_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "features_needed_test",
    srcs = ["features_needed_test.cc"],
    deps = [
        ":features_needed",
        ":path",
        ":test_util",
        "//tensorflow_data_validation/anomalies/proto:validation_metadata_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_schema_protos",
    testonly = 1,
    srcs = ["test_schema_protos.cc"],
    hdrs = ["test_schema_protos.h"],
    deps = [
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
    ],
)

cc_test(
    name = "statistics_view_test",
    srcs = ["statistics_view_test.cc"],
    deps = [
        ":statistics_view",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "schema_anomalies_test",
    srcs = ["schema_anomalies_test.cc"],
    deps = [
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "statistics_view",
    srcs = ["statistics_view.cc"],
    hdrs = ["statistics_view.h"],
    deps = [
        ":map_util",
        ":path",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "statistics_view_test_util",
    testonly = 1,
    srcs = ["statistics_view_test_util.cc"],
    hdrs = ["statistics_view_test_util.h"],
    deps = [
        ":statistics_view",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "internal_types",
    hdrs = ["internal_types.h"],
    deps = [
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "metrics",
    srcs = [
        "metrics.cc",
    ],
    hdrs = ["metrics.h"],
    deps = [
        ":map_util",
        ":statistics_view",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "float_domain_test",
    srcs = ["float_domain_test.cc"],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "metrics_test",
    srcs = ["metrics_test.cc"],
    deps = [
        ":metrics",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "diff_util",
    srcs = ["diff_util.cc"],
    hdrs = ["diff_util.h"],
    deps = [
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "schema",
    srcs = [
        "bool_domain_util.cc",
        "custom_domain_util.cc",
        "dataset_constraints_util.cc",
        "feature_util.cc",
        "float_domain_util.cc",
        "image_domain_util.cc",
        "int_domain_util.cc",
        "natural_language_domain_util.cc",
        "schema.cc",
        "schema_anomalies.cc",
        "schema_util.cc",
        "string_domain_util.cc",
    ],
    hdrs = [
        "bool_domain_util.h",
        "custom_domain_util.h",
        "dataset_constraints_util.h",
        "feature_util.h",
        "float_domain_util.h",
        "image_domain_util.h",
        "int_domain_util.h",
        "natural_language_domain_util.h",
        "schema.h",
        "schema_anomalies.h",
        "schema_util.h",
        "string_domain_util.h",
    ],
    deps = [
        ":diff_util",
        ":features_needed",
        ":internal_types",
        ":map_util",
        ":metrics",
        ":path",
        ":statistics_view",
        "//tensorflow_data_validation/anomalies/proto:feature_statistics_to_proto_proto",
        "//tensorflow_data_validation/anomalies/proto:validation_config_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "dataset_constraints_util_test",
    srcs = ["dataset_constraints_util_test.cc"],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "feature_util_test",
    srcs = [
        "feature_util_test.cc",
    ],
    deps = [
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "bool_domain_test",
    srcs = ["bool_domain_test.cc"],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "image_domain_test",
    srcs = ["image_domain_test.cc"],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "string_domain_test",
    srcs = ["string_domain_test.cc"],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "//tensorflow_data_validation/anomalies/proto:feature_statistics_to_proto_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "int_domain_test",
    srcs = [
        "int_domain_test.cc",
    ],
    deps = [
        ":internal_types",
        ":schema",
        ":statistics_view_test_util",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "schema_test",
    srcs = [
        "schema_test.cc",
        "schema_util_test.cc",
    ],
    deps = [
        ":schema",
        ":statistics_view_test_util",
        ":test_schema_protos",
        ":test_util",
        "//tensorflow_data_validation/anomalies/proto:validation_config_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "feature_statistics_validator",
    srcs = ["feature_statistics_validator.cc"],
    hdrs = ["feature_statistics_validator.h"],
    visibility = ["//tensorflow_data_validation:__subpackages__"],
    deps = [
        ":features_needed",
        ":path",
        ":schema",
        ":statistics_view",
        "//tensorflow_data_validation/anomalies/proto:feature_statistics_to_proto_proto",
        "//tensorflow_data_validation/anomalies/proto:validation_config_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "feature_statistics_validator_test",
    srcs = ["feature_statistics_validator_test.cc"],
    deps = [
        ":feature_statistics_validator",
        ":test_util",
        "//tensorflow_data_validation/anomalies/proto:validation_config_proto",
        "//tensorflow_data_validation/anomalies/proto:validation_metadata_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":map_util",
        ":path",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "path_test",
    srcs = ["path_test.cc"],
    deps = [
        ":path",
        ":test_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:metadata_v0_proto_cc_pb2",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "map_util",
    srcs = ["map_util.cc"],
    hdrs = ["map_util.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "map_util_test",
    srcs = ["map_util_test.cc"],
    deps = [
        ":map_util",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "custom_domain_util_test",
    srcs = ["custom_domain_util_test.cc"],
    deps = [
        ":schema",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)
