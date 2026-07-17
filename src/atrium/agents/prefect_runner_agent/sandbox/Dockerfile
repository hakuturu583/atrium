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
#   docker build -t local-registry/prefect_runner_agent:0.1.0 \
#       --build-arg BASE_IMAGE=local-registry/python_code_workspace_agent:0.1.0 \
#       -f src/atrium/agents/prefect_runner_agent/sandbox/Dockerfile .

ARG BASE_IMAGE=local-registry/python_code_workspace_agent:0.1.0
FROM ${BASE_IMAGE}

# Toolchain install needs root; drop back to the inherited non-root user at the end.
USER root

ENV UV_CACHE_DIR=/home/coder/.cache/uv \
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

# The atrium core provides the A2A transport the trusted `atrium_dispatch` primitive
# wraps, and it already depends on `prefect>=3`, so installing it from the local
# source (uv is inherited from the Python base) brings the whole flow runtime. It is
# not on PyPI — install the package from the build context, never a remote fetch.
# Install into a dedicated venv (the base's system Python is PEP-668 externally
# managed) and put it first on PATH so `python flow.py` resolves the flow runtime.
COPY pyproject.toml README.md /opt/atrium/
COPY src /opt/atrium/src
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python /opt/atrium
ENV PATH=/opt/venv/bin:$PATH

# The trusted dispatch primitive, importable as `atrium_dispatch` from generated
# flow code (`from atrium_dispatch import atrium_dispatch`).
COPY src/atrium/agents/prefect_runner_agent/sandbox/dispatch.py /usr/local/lib/atrium_dispatch/atrium_dispatch.py
ENV PYTHONPATH=/usr/local/lib/atrium_dispatch

# Build-time assertion that the runtime materialised (fail a half-provisioned image
# early) and prepare the writable Prefect home for the non-root user.
RUN python3 -c "import prefect, atrium_dispatch" \
    && mkdir -p /home/coder/.prefect \
    && chown -R coder:coder /home/coder/.prefect /home/coder/.cache

USER coder
