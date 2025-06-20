"""Open-source versions of TFDV proto build rules."""

load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

def tfdv_proto_library(name, **kwargs):
    """Google proto_library and cc_proto_library.

    Args:
        name: Name of the cc proto library.
        **kwargs: Keyword arguments to pass to the proto libraries."""
    well_known_protos = [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:wrappers_proto",
    ]
    kwargs["deps"] = kwargs.get("deps", []) + well_known_protos
    native.proto_library(name = name, **kwargs)  # buildifier: disable=native-proto
    cc_proto_kwargs = {
        "deps": [":" + name],
    }
    if "visibility" in kwargs:
        cc_proto_kwargs["visibility"] = kwargs["visibility"]
    if "testonly" in kwargs:
        cc_proto_kwargs["testonly"] = kwargs["testonly"]
    if "compatible_with" in kwargs:
        cc_proto_kwargs["compatible_with"] = kwargs["compatible_with"]
    cc_proto_library(name = name + "_cc_pb2", **cc_proto_kwargs)

def tfdv_proto_library_py(
        name,
        deps,
        visibility = None,
        testonly = 0):
    """Opensource py_proto_library."""
    py_proto_library(
        name = name,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
    )
