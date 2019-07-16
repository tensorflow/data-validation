"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
load("//third_party:arrow_configure.bzl", "arrow_configure")

def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    # LINT.IfChange
    # Fetch tf.Metadata repo from GitHub.
    git_repository(
        name = "com_github_tensorflow_metadata",
        commit = "992edd17e0f020458084c031c42f85d520e6f6af",
        remote = "https://github.com/tensorflow/metadata.git",
    )
    # LINT.ThenChange(//third_party/py/tensorflow_data_validation/google/copy.bara.sky)

    arrow_configure(name = "arrow")
