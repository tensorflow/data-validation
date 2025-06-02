workspace(name = "tensorflow_data_validation")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "google_bazel_common",
    sha256 = "82a49fb27c01ad184db948747733159022f9464fc2e62da996fa700594d9ea42",
    strip_prefix = "bazel-common-2a6b6406e12208e02b2060df0631fb30919080f3",
    urls = ["https://github.com/google/bazel-common/archive/2a6b6406e12208e02b2060df0631fb30919080f3.zip"],
)

################################################################################
# Generic Bazel Support                                                        #
################################################################################

http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/releases/download/6.0.2/rules_proto-6.0.2.tar.gz",
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:setup.bzl", "rules_proto_setup")

rules_proto_setup()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

# Install version 0.9.0 of rules_foreign_cc, as default version causes an
# invalid escape sequence error to be raised, which can't be avoided with
# the --incompatible_restrict_string_escapes=false flag (flag was removed in
# Bazel 5.0).
RULES_FOREIGN_CC_VERSION = "0.9.0"

http_archive(
    name = "rules_foreign_cc",
    patch_tool = "patch",
    patches = ["//third_party:rules_foreign_cc.patch"],
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-%s" % RULES_FOREIGN_CC_VERSION,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/%s.tar.gz" % RULES_FOREIGN_CC_VERSION,
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

_PROTOBUF_COMMIT = "4.25.6"  # 4.25.6

http_archive(
    name = "com_google_protobuf",
    sha256 = "ff6e9c3db65f985461d200c96c771328b6186ee0b10bc7cb2bbc87cf02ebd864",
    strip_prefix = "protobuf-%s" % _PROTOBUF_COMMIT,
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v4.25.6.zip",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Use the last commit on the relevant release branch to update.
# LINT.IfChange(arrow_archive_version)
ARROW_COMMIT = "347a88ff9d20e2a4061eec0b455b8ea1aa8335dc"  # 6.0.1
# LINT.ThenChange(third_party/arrow.BUILD:arrow_gen_version)

# `shasum -a 256` can be used to get `sha256` from the downloaded archive on
# Linux.
http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    patches = ["//third_party:arrow.patch"],
    sha256 = "55fc466d0043c4cce0756bc18e1e62b3233be74c9afe8dc0d18420b9a5fd9714",
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
)

COM_GOOGLE_ABSL_COMMIT = "4447c7562e3bc702ade25105912dce503f0c4010"  # lts_2023_08_0

http_archive(
    name = "com_google_absl",
    sha256 = "df8b3e0da03567badd9440377810c39a38ab3346fa89df077bb52e68e4d61e74",
    strip_prefix = "abseil-cpp-%s" % COM_GOOGLE_ABSL_COMMIT,
    url = "https://github.com/abseil/abseil-cpp/archive/%s.tar.gz" % COM_GOOGLE_ABSL_COMMIT,
)

# Will be loaded by workspace.bzl from head
# TFMD_COMMIT = "404805761e614561cceedc429e67c357c62be26d"  # 1.17.1

# http_archive(
#     name = "com_tensorflow_metadata",
#     sha256 = "1b72e0e5085812cd9b19e004a381b544542f9545a081f0f738c5ed6b8bb886a2",
#     strip_prefix = "metadata-%s" % TFMD_COMMIT,
#     urls = ["https://github.com/tensorflow/metadata/archive/%s.zip" % TFMD_COMMIT],
# )

# TODO(b/177694034): Follow the new format for tensorflow import after TF 2.5.
#here
TENSORFLOW_COMMIT = "3c92ac03cab816044f7b18a86eb86aa01a294d95"  # 2.17.1

http_archive(
    name = "org_tensorflow_no_deps",
    patches = [
        "//third_party:tensorflow_expose_example_proto.patch",
    ],
    sha256 = "317dd95c4830a408b14f3e802698eb68d70d81c7c7cfcd3d28b0ba023fe84a68",
    strip_prefix = "tensorflow-%s" % TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % TENSORFLOW_COMMIT,
    ],
)

PYBIND11_COMMIT = "8a099e44b3d5f85b20f05828d919d2332a8de841"  # 2.11.1

http_archive(
    name = "pybind11",
    build_file = "//third_party:pybind11.BUILD",
    sha256 = "8f4b7f28d214e36301435c055076c36186388dc9617117802cba8a059347cb00",
    strip_prefix = "pybind11-%s" % PYBIND11_COMMIT,
    urls = ["https://github.com/pybind/pybind11/archive/%s.zip" % PYBIND11_COMMIT],
)

load("//third_party:python_configure.bzl", "local_python_configure")

local_python_configure(name = "local_config_python")

http_archive(
    name = "com_google_farmhash",
    build_file = "//third_party:farmhash.BUILD",
    sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",  # SHARED_FARMHASH_SHA
    strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
    urls = [
        "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
    ],
)

################################################################################
# Google APIs protos                                                           #
################################################################################
http_archive(
    name = "com_google_googleapis",
    patch_args = ["-p1"],
    patches = ["//third_party:googleapis.patch"],
    sha256 = "28e7fe3a640dd1f47622a4c263c40d5509c008cc20f97bd366076d5546cccb64",
    strip_prefix = "googleapis-4ce00b00904a7ce1df8c157e54fcbf96fda0dc49",
    url = "https://github.com/googleapis/googleapis/archive/4ce00b00904a7ce1df8c157e54fcbf96fda0dc49.tar.gz",
)

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,
    go = True,
)

###############################################################################
# Gazelle Support                                                             #
###############################################################################

_rules_go_version = "v0.48.1"

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "b2038e2de2cace18f032249cb4bb0048abf583a36369fa98f687af1b3f880b26",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/{0}/rules_go-{0}.zip".format(_rules_go_version),
        "https://github.com/bazelbuild/rules_go/releases/download/{0}/rules_go-{0}.zip.format(_rules_go_version)",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.21.11")

_bazel_gazelle_version = "0.36.0"

http_archive(
    name = "bazel_gazelle",
    sha256 = "75df288c4b31c81eb50f51e2e14f4763cb7548daae126817247064637fd9ea62",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v{0}/bazel-gazelle-v{0}.tar.gz".format(_bazel_gazelle_version),
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v{0}/bazel-gazelle-v{0}.tar.gz".format(_bazel_gazelle_version),
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")  #, "go_repository")

gazelle_dependencies()

_PLATFORMS_VERSION = "0.0.6"

http_archive(
    name = "platforms",
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
        "https://github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
    ],
)

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check("6.5.0")

# Please add all new TensorFlow Data Validation dependencies in workspace.bzl.
load("//tensorflow_data_validation:workspace.bzl", "tf_data_validation_workspace")

tf_data_validation_workspace()
