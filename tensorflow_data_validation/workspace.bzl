"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""

    git_repository(
        name = "com_github_tensorflow_metadata",
        branch = "master",
        remote = "https://github.com/tensorflow/metadata.git",
    )

    git_repository(
        name = "com_github_tfx_bsl",
        branch = "master",
        remote = "https://github.com/tensorflow/tfx-bsl",
    )
