"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""

    # LINT.IfChange
    # The next line (a comment) is important because it is used to
    # locate the git_repository repo rule. Therefore if it's changed, also
    # change copy.bara.sky.
    #
    # Fetch tf.Metadata repo from GitHub.
    git_repository(
        name = "com_github_tensorflow_metadata",
        branch = "master",
        remote = "https://github.com/tensorflow/metadata.git",
    )
    # LINT.ThenChange(//tensorflow_data_validation/placeholder/files)

    git_repository(
        name = "com_github_tfx_bsl",
        branch = "master",
        remote = "https://github.com/tensorflow/tfx-bsl",
    )
