workspace(name = "tensorflow_data_validation")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Install version 0.9.0 of rules_foreign_cc, as default version causes an
# invalid escape sequence error to be raised, which can't be avoided with
# the --incompatible_restrict_string_escapes=false flag (flag was removed in
# Bazel 5.0).
RULES_FOREIGN_CC_VERSION = "0.9.0"
http_archive(
    name = "rules_foreign_cc",
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-%s" % RULES_FOREIGN_CC_VERSION,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/%s.tar.gz" % RULES_FOREIGN_CC_VERSION,
    patch_tool = "patch",
    patches = ["//third_party:rules_foreign_cc.patch",],
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

# To update package to a new revision, for example ABSL
# 1. Update the 'COM_GOOGLE_ABSL_COMMIT' var below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/abseil/abseil-cpp/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

# TODO(caveness): Clean up dependencies that were needed by TF but are no longer
# required.
# Needed by tf_py_wrap_cc rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
# LINT.IfChange(io_bazel_rules_clousure)
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2020-02-14
    ],
)
# LINT.ThenChange(:tf_commit)

# External proto rules.
http_archive(
    name = "rules_proto",
    sha256 = "66bfdf8782796239d3875d37e7de19b1d94301e8972b3cbd2446b332429b4df1",
    strip_prefix = "rules_proto-4.0.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
    ],
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

# TODO(b/239095455): Change to using a tfx-bsl workspace macro to load these
# dependencies.
# Needed by zetasql.
PROTOBUF_COMMIT = "fde7cf7358ec7cd69e8db9be4f1fa6a5c431386a" # 3.13.0
http_archive(
    name = "com_google_protobuf",
    sha256 = "e589e39ef46fb2b3b476b3ca355bd324e5984cbdfac19f0e1625f0042e99c276",
    strip_prefix = "protobuf-%s" % PROTOBUF_COMMIT,
    urls = [
        "https://storage.googleapis.com/grpc-bazel-mirror/github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
        "https://github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
    ],
)

# Needed by abseil-py by zetasql.
http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "//third_party:six.BUILD"
)

COM_GOOGLE_ABSL_COMMIT = "92fdbfb301f8b301b28ab5c99e7361e775c2fb8a" # 2022-08-25 Abseil Logging library first release
http_archive(
  name = "com_google_absl",
  url = "https://github.com/abseil/abseil-cpp/archive/%s.tar.gz" % COM_GOOGLE_ABSL_COMMIT,
  sha256 = "71d38c5f44997a5ccbc338f904c8682b40c25cad60b9cbaf27087a917228d5fa",
  strip_prefix = "abseil-cpp-%s" % COM_GOOGLE_ABSL_COMMIT
)

ZETASQL_COMMIT = "ac37cf5c0d80b5605176fc0f29e87b12f00be693"  # 08/10/2022
http_archive(
    name = "com_google_zetasql",
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,
    sha256 = "651a768cd51627f58aa6de7039aba9ddab22f4b0450521169800555269447840",
)

load("@com_google_zetasql//bazel:zetasql_deps_step_1.bzl", "zetasql_deps_step_1")

zetasql_deps_step_1()

load("@com_google_zetasql//bazel:zetasql_deps_step_2.bzl", "zetasql_deps_step_2")

zetasql_deps_step_2(
    analyzer_deps = True,
    evaluator_deps = True,
    tools_deps = False,
    java_deps = False,
    testing_deps = False,
)

# This is part of what zetasql_deps_step_3() does.
load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,
)

_PLATFORMS_VERSION = "0.0.6"
http_archive(
    name = "platforms",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
        "https://github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
    ],
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
)

# Please add all new TensorFlow Data Validation dependencies in workspace.bzl.
load("//tensorflow_data_validation:workspace.bzl", "tf_data_validation_workspace")

tf_data_validation_workspace()

load("@com_github_tfx_bsl//third_party:python_configure.bzl", "local_python_configure")
local_python_configure(name = "local_config_python")

PYBIND11_COMMIT = "8a099e44b3d5f85b20f05828d919d2332a8de841"  # 2.11.1
http_archive(
  name = "pybind11",
  build_file = "@com_github_tfx_bsl//third_party:pybind11.BUILD",
  strip_prefix = "pybind11-%s" % PYBIND11_COMMIT,
  urls = ["https://github.com/pybind/pybind11/archive/%s.zip" % PYBIND11_COMMIT],
  sha256 = "8f4b7f28d214e36301435c055076c36186388dc9617117802cba8a059347cb00",
)

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("5.3.0")
