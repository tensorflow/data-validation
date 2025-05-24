"""Repository rule for configuring a local Python repo.

`local_python_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: location of python binary.
  * `PYTHON_LIB_PATH`: Location of python libraries.

Add the following to your WORKSPACE FILE:

```python
local_python_configure(name = "local_config_python")
```

Args:
  name: A unique name for this workspace rule.
"""

BAZEL_SH = "BAZEL_SH"
PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
PYTHON_LIB_PATH = "PYTHON_LIB_PATH"

def _auto_config_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sConfiguration Error:%s %s\n" % (red, no_color, msg))

def _which(repository_ctx, program_name):
    """Returns the full path to a program on the execution platform.

    Args:
      repository_ctx: the repository_ctx
      program_name: name of the program on the PATH

    Returns:
      The full path to a program on the execution platform.
    """
    if _is_windows(repository_ctx):
        if not program_name.endswith(".exe"):
            program_name = program_name + ".exe"
        result = _execute(repository_ctx, ["C:\\Windows\\System32\\where.exe", program_name])
    else:
        result = _execute(repository_ctx, ["which", program_name])
    return result.stdout.rstrip()

def _get_environ(repository_ctx, name, default_value = None):
    """Returns the value of an environment variable on the execution platform.

    Args:
      repository_ctx: the repository_ctx
      name: the name of environment variable
      default_value: the value to return if not set

    Returns:
      The value of the environment variable 'name' on the execution platform
      or 'default_value' if it's not set.
    """
    if _is_windows(repository_ctx):
        result = _execute(
            repository_ctx,
            ["C:\\Windows\\System32\\cmd.exe", "/c", "echo", "%" + name + "%"],
            empty_stdout_fine = True,
        )
    else:
        cmd = "echo -n \"$%s\"" % name
        result = _execute(
            repository_ctx,
            [_get_bash_bin(repository_ctx), "-c", cmd],
            empty_stdout_fine = True,
        )
    if len(result.stdout) == 0:
        return default_value
    return result.stdout

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Args:
      repository_ctx: the repository_ctx object
      cmdline: list of strings, the command to execute
      error_msg: string, a summary of the error if the command fails
      error_details: string, details about the error or steps to fix it
      empty_stdout_fine: bool, if True, an empty stdout result is fine,
        otherwise it's an error
    Returns:
      The result of repository_ctx.execute(cmdline)
    """
    result = _raw_exec(repository_ctx, cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
                "Command: " + " ".join(cmdline),
            ]),
        )
    return result

def _get_bash_bin(repository_ctx):
    """Gets the bash bin path.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      The bash bin path.
    """
    bash_bin = _get_host_environ(repository_ctx, BAZEL_SH)
    if bash_bin != None:
        return bash_bin
    bash_bin_path = _which(repository_ctx, "bash")
    if bash_bin_path == None:
        _auto_config_fail("Cannot find bash in PATH, please make sure " +
                          "bash is installed and add its directory in PATH, or --define " +
                          "%s='/path/to/bash'.\nPATH=%s" % (
                              BAZEL_SH,
                              _get_environ("PATH", ""),
                          ))
    return bash_bin_path

def _get_host_environ(repository_ctx, name):
    """Returns the value of an environment variable on the host platform.

    The host platform is the machine that Bazel runs on.

    Args:
      repository_ctx: the repository_ctx
      name: the name of environment variable

    Returns:
      The value of the environment variable 'name' on the host platform.
    """
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ.get(name).strip()

    if hasattr(repository_ctx.attr, "environ") and name in repository_ctx.attr.environ:
        return repository_ctx.attr.environ.get(name).strip()

    return None

def _get_python_bin(repository_ctx):
    """Gets the python bin path.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      The python bin path.
    """
    python_bin = _get_host_environ(repository_ctx, PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin

    # First check for an explicit "python3"
    python_bin = _which(repository_ctx, "python3")
    if python_bin != None:
        return python_bin

    # Some systems just call pythone3 "python"
    python_bin = _which(repository_ctx, "python")
    if python_bin != None:
        return python_bin

    _auto_config_fail("Cannot find python in PATH, please make sure " +
                      "python is installed and add its directory in PATH, or --define " +
                      "%s='/something/else'.\nPATH=%s" % (
                          PYTHON_BIN_PATH,
                          _get_environ("PATH", ""),
                      ))
    return python_bin  # unreachable

def _is_windows(repository_ctx):
    """Returns true if the execution platform is Windows.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      If the execution platform is Windows.
    """
    os_name = ""
    if hasattr(repository_ctx.attr, "exec_properties") and "OSFamily" in repository_ctx.attr.exec_properties:
        os_name = repository_ctx.attr.exec_properties["OSFamily"]
    else:
        os_name = repository_ctx.os.name

    return os_name.lower().find("windows") != -1

def _raw_exec(repository_ctx, cmdline):
    """Executes a command via repository_ctx.execute() and returns the result.

    This method is useful for debugging purposes. For example, to print all
    commands executed as well as their return code.

    Args:
      repository_ctx: the repository_ctx
      cmdline: the list of args

    Returns:
      The 'exec_result' of repository_ctx.execute().
    """
    return repository_ctx.execute(cmdline)

def _read_dir(repository_ctx, src_dir):
    """Returns a sorted list with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks.

    Args:
      repository_ctx: the repository_ctx
      src_dir: the directory to traverse

    Returns:
      A sorted list with all files in a directory.
    """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["C:\\Windows\\System32\\cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return sorted(result.splitlines())

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _symlink_genrule_for_dir(
        repository_ctx,
        src_dir,
        dest_dir,
        genrule_name,
        src_files = [],
        dest_files = []):
    """Returns a genrule to symlink(or copy if on Windows) a set of files.

    If src_dir is passed, files will be read from the given directory; otherwise
    we assume files are in src_files and dest_files
    """
    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        files = "\n".join(_read_dir(repository_ctx, src_dir))

        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()
    command = []
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]

            # Copy the headers to create a sandboxable setup.
            cmd = "cp -f"
            command.append(cmd + ' "%s" "%s"' % (src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')
    genrule = _genrule(
        src_dir,
        genrule_name,
        " && ".join(command),
        "\n".join(outs),
    )
    return genrule

def _get_python_lib(repository_ctx, python_bin):
    """Gets the python lib path."""
    python_lib = _get_host_environ(repository_ctx, PYTHON_LIB_PATH)
    if python_lib != None:
        return python_lib

    # The interesting program to execute.
    print_lib = [
        "from __future__ import print_function",
        "import site",
        "import os",
        "python_paths = []",
        "if os.getenv('PYTHONPATH') is not None:",
        "  python_paths = os.getenv('PYTHONPATH').split(':')",
        "try:",
        "  library_paths = site.getsitepackages()",
        "except AttributeError:",
        "  from distutils.sysconfig import get_python_lib",
        "  library_paths = [get_python_lib()]",
        "all_paths = set(python_paths + library_paths)",
        "paths = []",
        "for path in all_paths:",
        "  if os.path.isdir(path):",
        "    paths.append(path)",
        "if len(paths) >=1:",
        "  print(paths[0])",
    ]

    # The below script writes the above program to a file
    # and executes it. This is to work around the limitation
    # of not being able to upload files as part of execute.
    cmd = "from os import linesep;"
    cmd += "f = open('script.py', 'w');"
    for line in print_lib:
        cmd += "f.write(\"%s\" + linesep);" % line
    cmd += "f.close();"
    cmd += "from os import system;"
    cmd += "system(\"%s script.py\");" % python_bin

    result = _execute(repository_ctx, [python_bin, "-c", cmd])
    return result.stdout.strip()

def _check_python_lib(repository_ctx, python_lib):
    """Checks the python lib path."""
    cmd = 'test -d "%s" -a -x "%s"' % (python_lib, python_lib)
    result = _raw_exec(repository_ctx, [_get_bash_bin(repository_ctx), "-c", cmd])
    if result.return_code == 1:
        _auto_config_fail("Invalid python library path: %s" % python_lib)

def _check_python_bin(repository_ctx, python_bin):
    """Checks the python bin path."""
    cmd = '[[ -x "%s" ]] && [[ ! -d "%s" ]]' % (python_bin, python_bin)
    result = _raw_exec(repository_ctx, [_get_bash_bin(repository_ctx), "-c", cmd])
    if result.return_code == 1:
        _auto_config_fail("--define %s='%s' is not executable. Is it the python binary?" % (
            PYTHON_BIN_PATH,
            python_bin,
        ))

def _get_python_include(repository_ctx, python_bin):
    """Gets the python include path."""
    result = _execute(
        repository_ctx,
        [
            python_bin,
            "-Wignore",
            "-c",
            "import importlib; " +
            "import importlib.util; " +
            "print(importlib.import_module('distutils.sysconfig').get_python_inc() " +
            "if importlib.util.find_spec('distutils.sysconfig') " +
            "else importlib.import_module('sysconfig').get_path('include'))",
        ],
        error_msg = "Problem getting python include path.",
        error_details = ("Is the Python binary path set up right? " +
                         "(See ./configure or " + PYTHON_BIN_PATH + ".) " +
                         "Is distutils installed?"),
    )
    return result.stdout.splitlines()[0]

def _get_python_import_lib_name(repository_ctx, python_bin):
    """Get Python import library name (pythonXY.lib) on Windows."""
    result = _execute(
        repository_ctx,
        [
            python_bin,
            "-c",
            "import sys;" +
            'print("python" + str(sys.version_info[0]) + ' +
            '      str(sys.version_info[1]) + ".lib")',
        ],
        error_msg = "Problem getting python import library.",
        error_details = ("Is the Python binary path set up right? " +
                         "(See ./configure or " + PYTHON_BIN_PATH + ".) "),
    )
    return result.stdout.splitlines()[0]

def _get_numpy_include(repository_ctx, python_bin):
    """Gets the numpy include path."""
    return _execute(
        repository_ctx,
        [
            python_bin,
            "-c",
            "from __future__ import print_function;" +
            "import numpy;" +
            " print(numpy.get_include());",
        ],
        error_msg = "Problem getting numpy include path.",
        error_details = "Is numpy installed?",
    ).stdout.splitlines()[0]

def _create_local_python_repository(repository_ctx):
    """Creates the repository containing files set up to build with Python."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    build_tpl = repository_ctx.path(Label("//third_party:local_python.BUILD.tpl"))

    python_bin = _get_python_bin(repository_ctx)
    _check_python_bin(repository_ctx, python_bin)
    python_lib = _get_python_lib(repository_ctx, python_bin)
    _check_python_lib(repository_ctx, python_lib)
    python_include = _get_python_include(repository_ctx, python_bin)
    numpy_include = _get_numpy_include(repository_ctx, python_bin) + "/numpy"
    python_include_rule = _symlink_genrule_for_dir(
        repository_ctx,
        python_include,
        "python_include",
        "python_include",
    )
    python_import_lib_genrule = ""

    # To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
    # See https://docs.python.org/3/extending/windows.html
    if _is_windows(repository_ctx):
        python_bin = python_bin.replace("\\", "/")
        python_include = _norm_path(python_include)
        python_import_lib_name = _get_python_import_lib_name(repository_ctx, python_bin)
        python_import_lib_src = python_include.rsplit("/", 1)[0] + "/libs/" + python_import_lib_name
        python_import_lib_genrule = _symlink_genrule_for_dir(
            repository_ctx,
            None,
            "",
            "python_import_lib",
            [python_import_lib_src],
            [python_import_lib_name],
        )
    numpy_include_rule = _symlink_genrule_for_dir(
        repository_ctx,
        numpy_include,
        "numpy_include/numpy",
        "numpy_include",
    )

    platform_constraint = ""
    if repository_ctx.attr.platform_constraint:
        platform_constraint = "\"%s\"" % repository_ctx.attr.platform_constraint
    repository_ctx.template("BUILD", build_tpl, {
        "%{PYTHON_BIN_PATH}": python_bin,
        "%{PYTHON_INCLUDE_GENRULE}": python_include_rule,
        "%{PYTHON_IMPORT_LIB_GENRULE}": python_import_lib_genrule,
        "%{NUMPY_INCLUDE_GENRULE}": numpy_include_rule,
        "%{PLATFORM_CONSTRAINT}": platform_constraint,
    })

_ENVIRONS = [
    BAZEL_SH,
    PYTHON_BIN_PATH,
    PYTHON_LIB_PATH,
]

local_python_configure = repository_rule(
    implementation = _create_local_python_repository,
    environ = _ENVIRONS,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
    },
)
