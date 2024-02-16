"""BUILD macros used in OSS builds."""

def tfdv_pybind_extension(
        name,
        srcs,
        module_name,  # buildifier: disable=unused-variable
        deps = [],
        visibility = None):
    """Builds a generic Python extension module.

    Args:
      name: Name of the target.
      srcs: C++ source files.
      module_name: Ignored.
      deps: Dependencies.
      visibility: Visibility.
    """
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ]

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.cc_binary(
        name = so_file,
        srcs = srcs,
        copts = [
            "-fno-strict-aliasing",
            "-fexceptions",
        ] + select({
            "//conditions:default": [
                "-fvisibility=hidden",
            ],
        }),
        linkopts = select({
            "//tensorflow_data_validation:macos": [
                # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                # not being exported.  There should be a better way to deal with this.
                # "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
                "-Wl,-w",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script",
                "$(location %s)" % version_script_file,
            ],
        }),
        deps = deps + [
            exported_symbols_file,
            version_script_file,
        ],
        features = ["-use_header_modules"],
        linkshared = 1,
        visibility = visibility,
    )
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
    )
    native.py_library(
        name = name,
        data = select({
            "//conditions:default": [so_file],
        }),
        srcs_version = "PY3",
        visibility = visibility,
    )
