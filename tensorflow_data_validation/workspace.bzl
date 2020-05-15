"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    # LINT.IfChange
    # The next line (a comment) is important because it is used to
    # locate the git_repository repo rule. Therefore if it's changed, also
    # change copy.bara.sky.
    #
    # Fetch tf.Metadata repo from GitHub.
    git_repository(
        name = "com_github_tensorflow_metadata",
        commit = "8b0aa117f5850b1a61e1a976dcf5a17a8fcb34cd",
        remote = "https://github.com/tensorflow/metadata.git",
    )
    # LINT.ThenChange(//third_party/py/tensorflow_data_validation/google/copy.bara.sky)
