"""Tests for the PrefectRunnerAgent — the least-privilege flow executor.

Sandbox-free: construction is exercised for the envelope re-checks, and the
``{files, commands}`` path is driven with ``start_sandbox``/``execute_in_sandbox``
monkeypatched so the run is captured, not executed. Mirrors
``test_code_workspace_agent.py``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from atrium.agents.prefect_runner_agent import PrefectRunnerAgent
from atrium.agents.prefect_runner_agent.sandbox import (
    IMAGE_REPOSITORY,
    build_sandbox_config,
)
from atrium.core.errors import PolicyViolationError
from atrium.core.types import ExecutionResult, NetworkMode, SandboxConfig
from atrium.protocol import Role, data_part, get_message_data, metadata_dict, text_message

_SANDBOX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "atrium", "agents", "prefect_runner_agent", "sandbox"
)


def _runner_with_capture(**kwargs):
    agent = PrefectRunnerAgent("runner-test", "0.1.0", **kwargs)
    commands: list[str] = []

    async def fake_start() -> None:
        return None

    async def fake_exec(command, *, timeout=None):
        commands.append(command)
        return ExecutionResult(command=command, exit_code=0, stdout="ok")

    agent.start_sandbox = fake_start  # type: ignore[assignment]
    agent.execute_in_sandbox = fake_exec  # type: ignore[assignment]
    return agent, commands


def _request(payload):
    return text_message("", role=Role.ROLE_USER, extra_parts=[data_part(payload)])


# --------------------------------------------------------------------------- #
# Security envelope                                                            #
# --------------------------------------------------------------------------- #
def test_default_envelope_is_wan_isolated_and_credential_free():
    cfg = build_sandbox_config("0.1.0")
    assert cfg.network == NetworkMode.INTERNAL
    assert cfg.internal is True
    assert cfg.image == f"{IMAGE_REPOSITORY}:0.1.0"
    # No GitHub token forwarded (the base forwards GH_TOKEN/GITHUB_TOKEN by ref).
    assert cfg.secret_env == {}
    # No dispatch allow-list unless the deployment supplies one.
    assert "ATRIUM_DISPATCH_ENDPOINTS" not in cfg.env


def test_dispatch_endpoints_injected_into_env():
    import json

    cfg = build_sandbox_config(
        "0.1.0", dispatch_endpoints={"coder:active": "http://coder.local", "rev:active": "http://rev.local"}
    )
    injected = json.loads(cfg.env["ATRIUM_DISPATCH_ENDPOINTS"])
    assert injected == {"coder:active": "http://coder.local", "rev:active": "http://rev.local"}
    # The agent forwards the allow-list through to its sandbox env too.
    agent = PrefectRunnerAgent("runner-e", "0.1.0", dispatch_endpoints={"coder:active": "http://coder.local"})
    assert "coder:active" in agent.sandbox_config.env["ATRIUM_DISPATCH_ENDPOINTS"]


def test_rejects_wan_bridge_config():
    bridge = SandboxConfig(network=NetworkMode.BRIDGE, internal=False)
    with pytest.raises(PolicyViolationError):
        PrefectRunnerAgent("runner-x", "0.1.0", sandbox_config=bridge)


def test_rejects_docker_socket_mount():
    bad = build_sandbox_config("0.1.0")
    bad.volumes["/var/run/docker.sock"] = "/var/run/docker.sock"
    with pytest.raises(PolicyViolationError):
        PrefectRunnerAgent("runner-x", "0.1.0", sandbox_config=bad)


# --------------------------------------------------------------------------- #
# Execution path                                                              #
# --------------------------------------------------------------------------- #
def test_runs_staged_flow_command():
    agent, commands = _runner_with_capture()
    reply = asyncio.run(
        agent.handle_task(_request({"files": {"flow.py": "print(1)"}, "commands": ["python flow.py"]}))
    )
    assert metadata_dict(reply).get("status") == "ok"
    # setup_commands() is empty (offline), so only the request command runs.
    assert any("python flow.py" in c for c in commands)
    assert not any("uv sync" in c for c in commands)


def test_refuses_git_push_request():
    agent, _ = _runner_with_capture()
    reply = asyncio.run(
        agent.handle_task(
            _request(
                {
                    "files": {"flow.py": "print(1)"},
                    "git": {"push": True, "branch": "b", "commit_message": "m"},
                }
            )
        )
    )
    # A push request is a category error for a runner → structured error reply.
    assert metadata_dict(reply).get("status") == "error"
    data = {}
    for part in get_message_data(reply):
        data.update(part)
    assert "does not push" in (data.get("reason") or "")


# --------------------------------------------------------------------------- #
# Image / policy assertions (on the shipped files)                            #
# --------------------------------------------------------------------------- #
def test_dockerfile_preinstalls_prefect_and_dispatch():
    with open(os.path.join(_SANDBOX_DIR, "Dockerfile"), encoding="utf-8") as f:
        dockerfile = f.read()
    assert "prefect" in dockerfile
    assert "atrium_dispatch" in dockerfile


def test_policy_has_no_wan_allow_list():
    with open(os.path.join(_SANDBOX_DIR, "policy.yaml"), encoding="utf-8") as f:
        policy = f.read()
    # The runner drops the code-workspace GitHub/PyPI egress entries entirely.
    assert "github.com" not in policy
    assert "pypi.org" not in policy
    assert "network_policies: {}" in policy
