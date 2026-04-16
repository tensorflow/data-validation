"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""

    git_repository(
        name = "com_github_tensorflow_metadata",
        branch = "align-tf-2.21",
        remote = "https://github.com/vkarampudi/metadata.git",
    )

    git_repository(
        name = "com_github_tfx_bsl",
        branch = "testing",
        remote = "https://github.com/vkarampudi/tfx-bsl.git",
    )
