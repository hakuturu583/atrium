#!/usr/bin/env bash
#
# One-time bootstrap so a plain `uv sync` builds the OpenShell CLI on this host.
#
# OpenShell (the sandbox runtime atrium.sandbox drives) is a Rust CLI that
# publishes to PyPI *only* as manylinux_2_39 wheels (glibc >= 2.39, i.e. Ubuntu
# 24.04+). On an older glibc the wheel won't install, so uv builds it from source
# with maturin, which compiles against the local glibc. That source build needs
# two native prerequisites that a bare `uv sync` cannot provide on its own:
#   * a Rust toolchain
#   * the Z3 SMT solver's C library + header (the z3-sys crate links it)
#
# uv has no pre-sync hook, so instead of wrapping every `uv sync` we make the
# build self-configuring: this script provisions Z3 without root and writes the
# z3-sys/bindgen build env (+ an rpath to libz3) into ~/.cargo/config.toml. uv
# drives `cargo` under the hood and cargo always reads that file, so after
# running this ONCE a plain `uv sync` builds openshell automatically — no wrapper,
# no exported env. Re-run it only if Z3 moves or the toolchain is reinstalled.
#
# On glibc >= 2.39 none of this is needed: drop the [tool.uv.sources] openshell
# pin in pyproject.toml and `uv sync` installs the PyPI wheel directly.
#
# Usage:  scripts/build-openshell.sh [extra uv sync args]
#
set -euo pipefail

# Prebuilt Z3 and the pinned gcc/target wiring below are x86_64-only. Fail
# loudly here rather than 404-ing on a missing asset or writing a broken config.
if [ "$(uname -m)" != "x86_64" ]; then
  echo "error: this bootstrap supports x86_64 only (host is $(uname -m))." >&2
  echo "       On other arches use OpenShell's PyPI wheel (glibc >= 2.39) or" >&2
  echo "       provision Z3 + build openshell by hand." >&2
  exit 1
fi

Z3_VERSION="${Z3_VERSION:-4.13.4}"
# Pick the prebuilt Z3 whose glibc tag is <= the host glibc (4.13.4 ships a 2.35
# build; override Z3_GLIBC/Z3_VERSION if your host differs).
Z3_GLIBC="${Z3_GLIBC:-2.35}"
Z3_PREFIX="${Z3_PREFIX:-$HOME/.local/z3-${Z3_VERSION}}"
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
CARGO_CONFIG="$CARGO_HOME/config.toml"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) Rust toolchain (rustup drops an env file that puts cargo on PATH).
if ! command -v cargo >/dev/null 2>&1 && [ -f "$CARGO_HOME/env" ]; then
  # shellcheck disable=SC1091
  . "$CARGO_HOME/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: Rust toolchain not found. Install it (no root) with:" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal" >&2
  exit 1
fi

# 2) Z3 prebuilt (idempotent: skip if already provisioned).
if [ ! -f "$Z3_PREFIX/include/z3.h" ] || [ ! -f "$Z3_PREFIX/bin/libz3.so" ]; then
  echo "provisioning Z3 ${Z3_VERSION} (glibc ${Z3_GLIBC}) -> ${Z3_PREFIX}"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  url="https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/z3-${Z3_VERSION}-x64-glibc-${Z3_GLIBC}.zip"
  curl -fsSL -o "$tmp/z3.zip" "$url"
  unzip -q "$tmp/z3.zip" -d "$tmp/x"
  inner=("$tmp"/x/z3-*/); inner="${inner[0]}"
  mkdir -p "$Z3_PREFIX"
  cp -r "${inner}bin" "${inner}include" "$Z3_PREFIX/"
fi

# 3) Persist the build env into ~/.cargo/config.toml so every cargo build (i.e.
#    every `uv sync` that rebuilds openshell) picks it up automatically:
#      * Z3_SYS_Z3_HEADER / Z3_LIBRARY_PATH_OVERRIDE — point z3-sys at our Z3.
#      * CPATH — give clang/bindgen the extra include dirs (z3.h's siblings + the
#        compiler's freestanding headers for stdbool.h). We use CPATH, NOT
#        BINDGEN_EXTRA_CLANG_ARGS: OpenShell's own .cargo/config.toml sets that
#        key and a project-local config outranks this global one for the same
#        key (and a force table form errors on a string↔table merge). CPATH is a
#        key OpenShell does not set, so ours applies cleanly.
#      * rustflags — bake an rpath so the binary finds libz3 without LD_LIBRARY_PATH.
#    Managed between sentinels and rewritten in place, leaving any other config
#    untouched (a plain create when the file does not yet exist).
# Ask gcc for its freestanding-header dir (holds stdbool.h) and rustc for the
# host target triple, so the block below isn't pinned to one gcc version or triple.
gcc_inc="$(gcc -print-file-name=include)"
if [ ! -f "$gcc_inc/stdbool.h" ]; then
  echo "error: could not locate the compiler's freestanding headers (stdbool.h);" >&2
  echo "       install a C compiler (gcc) first." >&2
  exit 1
fi
host_triple="$(rustc -vV | awk '/^host:/{print $2}')"
mkdir -p "$CARGO_HOME"
BEGIN="# >>> atrium openshell build env >>>"
END="# <<< atrium openshell build env <<<"
block="$BEGIN
[env]
Z3_SYS_Z3_HEADER = \"$Z3_PREFIX/include/z3.h\"
Z3_LIBRARY_PATH_OVERRIDE = \"$Z3_PREFIX/bin\"
CPATH = \"$Z3_PREFIX/include:$gcc_inc\"

[target.$host_triple]
rustflags = [\"-C\", \"link-args=-Wl,-rpath,$Z3_PREFIX/bin\"]
$END"

BLOCK="$block" BEGIN="$BEGIN" END="$END" CFG="$CARGO_CONFIG" python3 - <<'PY'
import os, re, sys
cfg, begin, end, block = os.environ["CFG"], os.environ["BEGIN"], os.environ["END"], os.environ["BLOCK"]
existing = ""
if os.path.exists(cfg):
    with open(cfg, encoding="utf-8") as f:
        existing = f.read()
pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
if pat.search(existing):
    out = pat.sub(lambda _: block, existing)                      # update in place
elif existing.strip():
    # File exists with unrelated content. Refuse to risk a duplicate [env]/[target]
    # table; ask the user to merge our block by hand.
    if "[env]" in existing or "[target." in existing:
        sys.stderr.write(
            f"error: {cfg} already defines [env]/[target.*]; merge this block manually:\n\n{block}\n")
        sys.exit(2)
    out = existing.rstrip() + "\n\n" + block + "\n"
else:
    out = block + "\n"
if out == existing:
    print(f"openshell build env already current in {cfg}")
else:
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote openshell build env to {cfg}")
PY

# 4) Build openshell from source into the project venv (default dependency).
echo "building openshell into the project venv (uv sync)..."
cd "$repo_root"
uv sync "$@"

echo "done: $(uv run openshell --version)"
echo "from now on a plain \`uv sync\` rebuilds openshell automatically."
