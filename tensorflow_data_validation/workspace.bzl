"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""

    git_repository(
        name = "com_github_tensorflow_metadata",
        commit = "da17f4557749ae122284c5be3e43a2f211d3e6e2",
        remote = "https://github.com/tensorflow/metadata.git",
    )

    git_repository(
        name = "com_github_tfx_bsl",
        commit = "79a4c97cec4ecbad3e95838e990f9b3a0c6439c0",
        remote = "https://github.com/tensorflow/tfx-bsl.git",
    )
