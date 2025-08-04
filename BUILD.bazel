load("@bazel_gazelle//:def.bzl", "gazelle")

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

exports_files(["LICENSE"])

gazelle(
    name = "gazelle-update-repos",
    args = [
        "-from_file=go.mod",
        "-to_macro=deps.bzl%go_dependencies",
    ],
    command = "update-repos",
)
