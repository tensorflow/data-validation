licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//tensorflow_data_validation/pywrap:_pywrap_tensorflow_data_validation.so",
        "//tensorflow_data_validation/pywrap:pywrap_tensorflow_data_validation.py",
    ],
)
