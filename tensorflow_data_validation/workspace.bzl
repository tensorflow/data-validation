"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

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
        commit = "8f9ea5ea1942aff98b96adfbaec9f0772c92d9f9",
        remote = "https://github.com/tensorflow/metadata.git",
    )
    # LINT.ThenChange(//third_party/py/tensorflow_data_validation/google/copy.bara.sky)

    boost_deps()

    # Fetch arrow from GitHub.
    ARROW_COMMIT = "b65beb625fb14a6b627be667a32c136a79cb5c6f"  # v0.11.1
    new_git_repository(
        name = "arrow",
        build_file = "//third_party:arrow.BUILD",
        commit = ARROW_COMMIT,
        remote = "https://github.com/apache/arrow.git",
    )
