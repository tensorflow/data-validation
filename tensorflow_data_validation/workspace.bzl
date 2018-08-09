"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_data_validation_workspace():
  '''All TensorFlow Data Validation external dependencies.'''
  tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
  )

  # Fetch tf.Metadata repo from GitHub.
  native.git_repository(
      name = "com_github_tensorflow_metadata",
      # v0.6.0
      commit = "b402775d13b705970090493c81c89655821d8249",
      remote = "https://github.com/tensorflow/metadata.git",
  )
