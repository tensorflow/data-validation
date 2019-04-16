licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = select({
        ":windows": [
            "//tensorflow_data_validation/pywrap:_pywrap_tensorflow_data_validation.pyd",
            "//tensorflow_data_validation/pywrap:pywrap_tensorflow_data_validation.py",
        ],
        "//conditions:default": [
            "//tensorflow_data_validation/pywrap:_pywrap_tensorflow_data_validation.so",
            "//tensorflow_data_validation/pywrap:pywrap_tensorflow_data_validation.py",
        ],
    }),
)
