load("@bazel_gazelle//:def.bzl", "gazelle")

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

exports_files(["LICENSE"])

sh_binary(
    name = "patch_wrapped_clang",
    srcs = ["patch_wrapped_clang.sh"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "patch_local_config_apple_cc",
    tools = ["//:patch_wrapped_clang"],
    outs = ["patch_local_config_apple_cc.stamp"],
    cmd = "$(execpath //:patch_wrapped_clang) && touch $@",
    tags = [
        "local",
        "no-cache",
        "no-remote",
        "no-sandbox",
    ],
    target_compatible_with = ["@platforms//os:osx"],
)

gazelle(
    name = "gazelle-update-repos",
    args = [
        "-from_file=go.mod",
        "-to_macro=deps.bzl%go_dependencies",
    ],
    command = "update-repos",
)
