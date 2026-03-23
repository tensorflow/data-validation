#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

find_wrapper_dir() {
  local -a candidates=()
  local cwd="$(pwd)"

  candidates+=("${cwd}/external/local_config_cc")
  candidates+=("${cwd}/../external/local_config_cc")

  candidates+=("${ROOT_DIR}/bazel-data-validation/external/local_config_cc")

  if command -v bazel >/dev/null 2>&1; then
    local output_base
    if output_base="$(cd "${ROOT_DIR}" && bazel info output_base 2>/dev/null)"; then
      candidates+=("${output_base}/external/local_config_cc")
    fi
  fi

  for dir in "${candidates[@]}"; do
    if [[ -d "${dir}" ]]; then
      echo "${dir}"
      return 0
    fi
  done

  return 1
}

write_wrapper() {
  local path="$1"
  cat >"${path}" <<EOF
#!/bin/bash
set -euo pipefail
WRAPPER_DEVDIR="\${DEVELOPER_DIR:-\$(xcode-select -p)}"
SDKROOT_PATH="\${SDKROOT:-\$(xcrun --sdk macosx --show-sdk-path)}"
tool="clang"
if [[ "\$(basename "\$0")" == "wrapped_clang_pp" ]]; then
  tool="clang++"
fi
args=()
for arg in "\$@"; do
  if [[ "\$arg" == "DEBUG_PREFIX_MAP_PWD=." ]]; then
    args+=("-fdebug-prefix-map=\$(pwd)=.")
    continue
  fi
  arg="\${arg//__BAZEL_XCODE_DEVELOPER_DIR__/\${WRAPPER_DEVDIR}}"
  arg="\${arg//__BAZEL_XCODE_SDKROOT__/\${SDKROOT_PATH}}"
  args+=("\$arg")
done
exec /usr/bin/xcrun "\${tool}" "\${args[@]}"
EOF
  chmod +x "${path}"
}

write_libtool_check_unique() {
  local path="$1"
  cat >"${path}" <<'EOF'
#!/bin/bash
set -euo pipefail

TMP_INPUTS="$(mktemp "${TMPDIR:-/tmp}/libtool_unique.XXXXXX")"
trap 'rm -f "$TMP_INPUTS"' EXIT

EXPECT_FILELIST=0

add_object() {
  local obj="$1"
  [[ -n "$obj" ]] || return 0
  basename "$obj" >>"$TMP_INPUTS"
}

parse_token() {
  local token="$1"

  if [[ "$EXPECT_FILELIST" == "1" ]]; then
    EXPECT_FILELIST=0
    if [[ -f "$token" ]]; then
      while IFS= read -r obj; do
        add_object "$obj"
      done <"$token"
    fi
    return 0
  fi

  case "$token" in
    -filelist)
      EXPECT_FILELIST=1
      ;;
    @*)
      local params_file="${token:1}"
      if [[ -f "$params_file" ]]; then
        while IFS= read -r opt; do
          parse_token "$opt"
        done <"$params_file"
      fi
      ;;
    *.o)
      add_object "$token"
      ;;
  esac
}

for arg in "$@"; do
  parse_token "$arg"
done

if sort "$TMP_INPUTS" | uniq -d | grep -q .; then
  exit 1
fi

exit 0
EOF
  chmod +x "${path}"
}

main() {
  local wrapper_dir
  if ! wrapper_dir="$(find_wrapper_dir)"; then
    echo "Could not find local_config_cc wrapper directory." >&2
    echo "Run a Bazel build once, then rerun this script." >&2
    exit 1
  fi

  local clang_wrapper="${wrapper_dir}/wrapped_clang"
  local clangpp_wrapper="${wrapper_dir}/wrapped_clang_pp"
  local libtool_check_unique="${wrapper_dir}/libtool_check_unique"

  if [[ ! -e "${clang_wrapper}" || ! -e "${clangpp_wrapper}" || ! -e "${libtool_check_unique}" ]]; then
    echo "Missing wrapped_clang binaries under ${wrapper_dir}" >&2
    exit 1
  fi

  cp -f "${clang_wrapper}" "${clang_wrapper}.bak"
  cp -f "${clangpp_wrapper}" "${clangpp_wrapper}.bak"
  cp -f "${libtool_check_unique}" "${libtool_check_unique}.bak"

  write_wrapper "${clang_wrapper}"
  write_wrapper "${clangpp_wrapper}"
  write_libtool_check_unique "${libtool_check_unique}"

  echo "Patched wrappers in: ${wrapper_dir}"
  echo "Backups saved as wrapped_clang.bak, wrapped_clang_pp.bak, libtool_check_unique.bak"
}

main "$@"
