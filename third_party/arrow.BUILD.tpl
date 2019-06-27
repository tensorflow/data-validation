# BUILD template for Apache Arrow.
# This template is instantiated by arrow_configure.bzl and the BUILD file
# generated will serve as the BUILD of the arrow repository.

package(default_visibility = ["//visibility:public"])

load("@org_tensorflow//tensorflow:tensorflow.bzl", "clean_dep")

cc_library(
    name = "arrow",
    hdrs = [":arrow_header_include"],
    includes = ["include"],
    deps = [
        ":libarrow",
        ":libarrow_python",
        "@local_config_python//:python_headers",
    ],
)

cc_import(
    name = "libarrow",
    shared_library = select({
        clean_dep("@org_tensorflow//tensorflow:macos"): "libarrow.dylib",
        clean_dep("@org_tensorflow//tensorflow:windows"): None,
        "//conditions:default": "libarrow.so",
    }),
    interface_library = select({
        clean_dep("@org_tensorflow//tensorflow:windows"): "arrow.lib",
        "//conditions:default": None
    }),
    system_provided = select({
        clean_dep("@org_tensorflow//tensorflow:windows"): 1,
        "//conditions:default": 0
    }),
)

cc_import(
    name = "libarrow_python",
    shared_library = select({
        clean_dep("@org_tensorflow//tensorflow:macos"): "libarrow_python.dylib",
        clean_dep("@org_tensorflow//tensorflow:windows"): None,
        "//conditions:default": "libarrow_python.so",
    }),
    interface_library = select({
        clean_dep("@org_tensorflow//tensorflow:windows"): "arrow_python.lib",
        "//conditions:default": None,
    }),
    system_provided = select({
        clean_dep("@org_tensorflow//tensorflow:windows"): 1,
        "//conditions:default": 0
    }),
)

%{ARROW_HEADER_GENRULE}
%{LIBARROW_GENRULE}
%{LIBARROW_PYTHON_GENRULE}
