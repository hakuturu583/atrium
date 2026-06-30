# Base Docker Image for Atrium *code workspace* sandboxes.
#
# This is the image every coding-agent workspace inherits from (directly, via
# `FROM local-registry/codeworkspace_base:<version>`). Its single, hard guarantee
# is that the **minimum toolchain needed to get code onto GitHub is always
# present** — `git` and the GitHub CLI (`gh`) — no matter what a derived,
# language-specific image (PythonCodeWorkspaceAgent, ...) layers on top.
#
# That guarantee is made with a Docker **multi-stage build**:
#
#   * a dedicated, pinned `gh-fetch` stage downloads the `gh` release (a static
#     Go binary) in isolation, so the tool's provenance/version is fixed here and
#     does not depend on the final base distro's package feed; and
#   * the final stage `COPY --from=gh-fetch` brings just that binary in, then
#     asserts at build time (`gh --version && git --version`) that the toolchain
#     really materialised — the image build *fails* otherwise, so a broken base
#     can never be published.
#
# Built by the (fixed, non-evolving) BuilderAgent with rootless Kaniko and pushed
# to local-registry/codeworkspace_base:<version>.
#
#   docker build -t local-registry/codeworkspace_base:0.1.0 \
#       -f src/atrium/agents/code_workspace_agent/sandbox/Dockerfile .

# --------------------------------------------------------------------------- #
# Stage 1 — fetch a pinned `gh` release (isolated, deterministic).            #
# --------------------------------------------------------------------------- #
# Pin the CLI version here so the push toolchain is reproducible; override at
# build time with --build-arg GH_VERSION=... when bumping.
ARG GH_VERSION=2.62.0
FROM debian:stable-slim AS gh-fetch
ARG GH_VERSION
# BuildKit/Kaniko populate TARGETARCH from the build platform; default to amd64
# so a plain `docker build` (no buildx) still resolves a concrete asset.
ARG TARGETARCH=amd64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fsSL -o /tmp/gh.tar.gz \
        "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${TARGETARCH}.tar.gz"; \
    mkdir -p /opt/gh; \
    tar -xzf /tmp/gh.tar.gz -C /opt/gh --strip-components=1; \
    /opt/gh/bin/gh --version  # fail the stage immediately if the asset is wrong

# --------------------------------------------------------------------------- #
# Final stage — the Base Docker Image all code workspaces inherit from.       #
# --------------------------------------------------------------------------- #
FROM debian:stable-slim

ENV DEBIAN_FRONTEND=noninteractive \
    # `gh`/`git` read config from $HOME; pin it for the non-root coder user.
    HOME=/home/coder \
    LANG=C.UTF-8 \
    # /workspace is the project tree the coding agent writes & tests in.
    ATRIUM_WORKSPACE=/workspace

# The always-present VCS/push toolchain (git + openssh-client cover SSH remotes,
# ca-certificates + curl cover HTTPS remotes and `gh`, bash backs OpenShell's
# `bash -lc` exec, less is `gh`'s default pager), plus the non-root `coder` user
# that the coding agent runs as — one layer, since neither depends on `gh`.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git openssh-client ca-certificates curl bash less; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --create-home --uid 10001 coder; \
    mkdir -p /workspace; \
    chown -R coder:coder /workspace /home/coder

# Bring in the pinned `gh` binary from the fetch stage (provenance fixed above).
COPY --from=gh-fetch /opt/gh/bin/gh /usr/local/bin/gh

# Build-time assertion: the minimum push toolchain is genuinely here. If either
# tool is missing the build fails, so a published base image is never broken.
RUN gh --version && git --version

USER coder
WORKDIR /workspace

# A bare workspace just idles; the agent drives it via OpenShell `exec`.
CMD ["bash", "-l"]
