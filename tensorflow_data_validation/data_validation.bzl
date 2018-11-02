load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")
load("@protobuf_archive//:protobuf.bzl", "py_proto_library")

def tfdv_proto_library(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None,
        cc_api_version = 2):
    """Opensource cc_proto_library."""
    _ignore = [has_services, cc_api_version]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True
    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        cc_libs = ["@protobuf_archive//:protobuf"],
        protoc = "@protobuf_archive//:protoc",
        default_runtime = "@protobuf_archive//:protobuf",
        use_grpc_plugin = use_grpc_plugin,
        testonly = testonly,
        visibility = visibility,
    )

def tfdv_proto_library_py(name, proto_library, srcs = [], deps = [], visibility = None, testonly = 0):
    """Opensource py_proto_library."""
    _ignore = [proto_library]
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        deps = ["@protobuf_archive//:protobuf_python"] + deps,
        default_runtime = "@protobuf_archive//:protobuf_python",
        protoc = "@protobuf_archive//:protoc",
        visibility = visibility,
        testonly = testonly,
    )
