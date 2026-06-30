# Python code-workspace image — a *derived* workspace for PythonCodeWorkspaceAgent.
#
# It inherits the Base Docker Image (codeworkspace_base) so `git` + `gh` (the
# push toolchain) are guaranteed by construction, and adds the Python toolchain
# the agent needs to actually build & test Python code: an interpreter, a C
# compiler (build-essential — many wheels compile native extensions) and a fast
# package manager (`uv`).
#
# Note the inheritance mirrors the Python class hierarchy:
#   Docker:  codeworkspace_base   ->  python_code_workspace_agent
#   Python:  CodeWorkSpaceAgent   ->  PythonCodeWorkspaceAgent
#
# Built by the BuilderAgent with rootless Kaniko and pushed to
# local-registry/python_code_workspace_agent:<version>.
#
#   docker build -t local-registry/python_code_workspace_agent:0.1.0 \
#       --build-arg BASE_IMAGE=local-registry/codeworkspace_base:0.1.0 \
#       -f src/atrium/agents/code_workspace_agent/sandbox/python/Dockerfile .

# Inherit the Base Docker Image (the gh/git guarantee comes with it).
ARG BASE_IMAGE=local-registry/codeworkspace_base:0.1.0

# Pull the `uv` binary from its official image via a multi-stage COPY (same
# pattern as the base's gh-fetch: a pinned, provenance-fixed package manager
# independent of the distro feed).
ARG UV_VERSION=0.5.11
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${BASE_IMAGE}

# Toolchain install needs root; we drop back to `coder` at the end.
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    # Let uv manage venvs/caches under the coder home (writable, see policy.yaml).
    UV_CACHE_DIR=/home/coder/.cache/uv

# Interpreter + compiler/build prerequisites for native extension wheels.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev build-essential; \
    rm -rf /var/lib/apt/lists/*

# Package manager: `uv` (also exposes `uvx`); copied from the pinned stage above.
COPY --from=uv /uv /uvx /usr/local/bin/

# Build-time assertion that the Python toolchain (interpreter + compiler + package
# manager) really materialised — so a published image is never half-provisioned —
# and prepare the writable uv cache for the non-root user, in one layer.
RUN python3 --version && cc --version && uv --version \
    && mkdir -p /home/coder/.cache/uv \
    && chown -R coder:coder /home/coder/.cache

# Drop back to the inherited non-root user (WORKDIR /workspace and the CMD are
# inherited from the base image unchanged).
USER coder
