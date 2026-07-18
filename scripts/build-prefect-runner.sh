#!/usr/bin/env bash
#
# Build the PrefectRunnerAgent image chain (the least-privilege executor that runs
# a generated Prefect flow under WAN isolation) and smoke-test it offline.
#
# The runner image derives from the code-workspace chain, so this builds all three
# in order:
#   codeworkspace_base            (git + gh toolchain base)
#     -> python_code_workspace_agent   (+ Python interpreter, uv)
#         -> prefect_runner_agent      (+ prefect + atrium + the atrium_dispatch primitive)
#
# In production the (fixed, non-evolving) BuilderAgent builds these with rootless
# Kaniko and pushes to local-registry/<slug>:<version>; this script is the
# developer-local equivalent using a plain `docker build` so the chain is
# reproducible without standing up the whole build pipeline.
#
# The runner runs WAN-isolated (NetworkMode.INTERNAL): everything a generated flow
# may import is baked in, so the final smoke test runs a sample flow with
# `--network none` and asserts it completes offline (exit 0).
#
# Usage:  scripts/build-prefect-runner.sh [VERSION]      (VERSION default: 0.1.0)
#
set -euo pipefail

VERSION="${1:-0.1.0}"
REGISTRY="${ATRIUM_REGISTRY:-local-registry}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_TAG="$REGISTRY/codeworkspace_base:$VERSION"
PY_TAG="$REGISTRY/python_code_workspace_agent:$VERSION"
RUNNER_TAG="$REGISTRY/prefect_runner_agent:$VERSION"

CW_DIR="src/atrium/agents/code_workspace_agent/sandbox"
RUNNER_DIR="src/atrium/agents/prefect_runner_agent/sandbox"

echo ">> building $BASE_TAG"
docker build -t "$BASE_TAG" -f "$CW_DIR/Dockerfile" .

echo ">> building $PY_TAG"
docker build -t "$PY_TAG" --build-arg "BASE_IMAGE=$BASE_TAG" -f "$CW_DIR/python/Dockerfile" .

echo ">> building $RUNNER_TAG"
docker build -t "$RUNNER_TAG" --build-arg "BASE_IMAGE=$PY_TAG" -f "$RUNNER_DIR/Dockerfile" .

echo ">> smoke test: run a sample flow OFFLINE (--network none) as the non-root user"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/flow.py" <<'PY'
from prefect import flow
from atrium_dispatch import atrium_dispatch, resolve_endpoint  # noqa: F401 (import proves it ships)

@flow
def main():
    print("SMOKE_OK")
    return "ok"

if __name__ == "__main__":
    main()
PY
# mktemp -d is 0700 owned by the host user; make it readable by the container's
# non-root `coder` (uid 10001) so the mounted flow.py can actually be opened.
chmod -R a+rX "$WORK"

if docker run --rm --network none -v "$WORK:/workspace" -w /workspace "$RUNNER_TAG" \
        python flow.py | grep -q "SMOKE_OK"; then
    echo ">> OK: $RUNNER_TAG built and runs a flow offline"
else
    echo ">> FAIL: offline smoke test did not print SMOKE_OK" >&2
    exit 1
fi
