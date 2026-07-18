# PrefectRunnerAgent image — a *least-privilege* executor for generated flows.
#
# It inherits the Python code-workspace image (interpreter + `uv`, and the
# git/gh toolchain which the runner's WAN-isolated envelope renders inert) and
# adds exactly what a generated `flow.py` needs to run and to dispatch work:
#   * `prefect` — the generated flow is a Prefect `@flow`/`@task` DAG;
#   * the `atrium` core package — provides the A2A transport the trusted
#     `atrium_dispatch` primitive wraps;
#   * `atrium_dispatch` — the ONLY way generated flow code reaches a subagent.
#
# Inheritance mirrors the Python class hierarchy:
#   Docker:  python_code_workspace_agent  ->  prefect_runner_agent
#   Python:  PythonCodeWorkspaceAgent      ->  PrefectRunnerAgent
#
# The runner runs WAN-isolated (NetworkMode.INTERNAL): no public egress, only the
# host-local control LAN for A2A dispatch. So everything a flow may import must be
# baked in here — there is no runtime package fetch. Built by the BuilderAgent with
# rootless Kaniko and pushed to local-registry/prefect_runner_agent:<version>.
#
# **Multi-stage** so the ~4 GB uv download cache stays in the throwaway builder and
# never lands in the final image: the runtime stage COPYs only the finished venv.
# (A single-stage build also duplicated that cache a second time via `chown -R`,
# taking the image to ~8.5 GB — see issue #32.)
#
#   docker build -t local-registry/prefect_runner_agent:0.1.0 \
#       --build-arg BASE_IMAGE=local-registry/python_code_workspace_agent:0.1.0 \
#       -f src/atrium/agents/prefect_runner_agent/sandbox/Dockerfile .

ARG BASE_IMAGE=local-registry/python_code_workspace_agent:0.1.0

# --------------------------------------------------------------------------- #
# Builder — install atrium (+ its prefect>=3 dep) into a self-contained venv.  #
# The uv cache lives here and is discarded with the stage.                     #
# --------------------------------------------------------------------------- #
FROM ${BASE_IMAGE} AS builder
USER root
# Cache in /tmp (thrown away with the stage), not under the runtime home.
ENV UV_CACHE_DIR=/tmp/uv-cache
# atrium is not on PyPI — install it from the build context, never a remote fetch.
COPY pyproject.toml README.md /opt/atrium/
COPY src /opt/atrium/src
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python /opt/atrium \
    && rm -rf /tmp/uv-cache

# --------------------------------------------------------------------------- #
# Runtime — only the finished venv + the dispatch primitive. No build caches.  #
# --------------------------------------------------------------------------- #
FROM ${BASE_IMAGE}
USER root

# The venv is first on PATH so `python flow.py` resolves the flow runtime, and the
# trusted dispatch primitive is importable as `atrium_dispatch`.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/usr/local/lib/atrium_dispatch \
    # Ephemeral/local Prefect: no server, no worker — the flow runs to completion
    # in-process inside this sandbox and reports via exit code / stdout.
    PREFECT_API_URL="" \
    PREFECT_HOME=/home/coder/.prefect \
    # Quiet the ephemeral event-service teardown noise (harmless ERROR tracebacks
    # emitted as the in-process events worker stops) so the executor's stderr — which
    # the review gate captures — stays clean; the flow's own prints still show.
    PREFECT_LOGGING_LEVEL=CRITICAL \
    # No outbound telemetry: the runner is WAN-isolated, so the ephemeral server's
    # telemetry service would only emit a DNS-failure ERROR. Disable it at the source.
    PREFECT_SERVER_ANALYTICS_ENABLED=false

COPY --from=builder /opt/venv /opt/venv
COPY src/atrium/agents/prefect_runner_agent/sandbox/dispatch.py /usr/local/lib/atrium_dispatch/atrium_dispatch.py

# Build-time assertion that the runtime materialised (fail a half-provisioned image
# early) and prepare the writable Prefect home for the non-root user (no recursive
# chown — `install -d` creates it already owned).
RUN python3 -c "import prefect, atrium_dispatch" \
    && install -d -o coder -g coder /home/coder/.prefect

USER coder
