"""Tests for the code-workspace agents.

GPU/sandbox-free: ``start_sandbox`` and ``execute_in_sandbox`` are monkeypatched
so commands are *captured* instead of run, and the Base Docker Image's push-tool
guarantee is checked by asserting on the Dockerfile itself.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from atrium.agents.code_workspace_agent import (
    CodeWorkSpaceAgent,
    PythonCodeWorkspaceAgent,
    WorkspaceConfig,
)
from atrium.agents.code_workspace_agent.sandbox import build_sandbox_config
from atrium.core.errors import PolicyViolationError
from atrium.core.types import ExecutionResult, GPURequest, NetworkMode, SandboxConfig
from atrium.protocol import Role, data_part, get_message_data, metadata_dict, text_message

_SANDBOX_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "atrium",
    "agents",
    "code_workspace_agent",
    "sandbox",
)


def _agent_with_capture(cls=CodeWorkSpaceAgent, **kwargs):
    """Build an agent that records every in-sandbox command instead of running it."""
    agent = cls("ws-test", "0.1.0", **kwargs)
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
def test_rejects_gpu_passthrough():
    cfg = build_sandbox_config("0.1.0")
    cfg.device_requests = [GPURequest()]
    with pytest.raises(PolicyViolationError):
        CodeWorkSpaceAgent("ws", "0.1.0", sandbox_config=cfg)


def test_rejects_docker_socket_mount():
    cfg = build_sandbox_config("0.1.0")
    cfg.volumes = {"/var/run/docker.sock": "/var/run/docker.sock"}
    with pytest.raises(PolicyViolationError):
        CodeWorkSpaceAgent("ws", "0.1.0", sandbox_config=cfg)


# --------------------------------------------------------------------------- #
# Image identity                                                              #
# --------------------------------------------------------------------------- #
def test_image_names_track_slugs():
    base, _ = _agent_with_capture(CodeWorkSpaceAgent)
    py, _ = _agent_with_capture(PythonCodeWorkspaceAgent)
    assert base.image_name == "local-registry/codeworkspace_base:0.1.0"
    assert py.image_name == "local-registry/python_code_workspace_agent:0.1.0"


# --------------------------------------------------------------------------- #
# SandboxConfig envelope                                                      #
# --------------------------------------------------------------------------- #
def test_base_sandbox_config_envelope():
    cfg = build_sandbox_config("0.1.0", gh_token="secret-token")
    assert cfg.image == "local-registry/codeworkspace_base:0.1.0"
    assert cfg.network is NetworkMode.BRIDGE and cfg.wan_allowed is True
    assert cfg.gpu_enabled is False
    assert cfg.volumes == {}
    assert cfg.env["GH_TOKEN"] == "secret-token"
    assert cfg.policy_path and cfg.policy_path.endswith("policy.yaml")


def test_internal_network_stays_wan_isolated():
    cfg = build_sandbox_config("0.1.0", network=NetworkMode.INTERNAL)
    assert cfg.internal is True and cfg.wan_allowed is False


# --------------------------------------------------------------------------- #
# handle_task pipeline                                                        #
# --------------------------------------------------------------------------- #
def test_handle_task_runs_pipeline_in_order():
    agent, commands = _agent_with_capture(CodeWorkSpaceAgent)
    reply = asyncio.run(
        agent.dispatch(
            _request(
                {
                    "repo": "https://github.com/acme/widget.git",
                    "ref": "main",
                    "files": {"src/app.py": "print('hi')\n"},
                    "commands": ["make test"],
                    "git": {
                        "push": True,
                        "branch": "feature/x",
                        "commit_message": "do work",
                        "pull_request": {"title": "PR", "body": "body", "base": "main"},
                    },
                }
            )
        )
    )

    assert metadata_dict(reply)["status"] == "ok"
    blob = " || ".join(commands)
    # Clone the repo into a subdir derived from its name. (shlex.quote omits
    # quotes for values with no shell metacharacters, like these.)
    assert "git clone https://github.com/acme/widget.git /workspace/widget" in blob
    assert "checkout main" in blob
    # Commands run inside the cloned project dir.
    assert "cd /workspace/widget && make test" in blob
    # Push + PR.
    assert "checkout -B feature/x" in blob
    assert "gh pr create --title PR --body body --base main --head feature/x" in blob

    # Ordering: clone -> command -> push -> pr.
    order = [next(i for i, c in enumerate(commands) if needle in c)
             for needle in ("git clone", "make test", "checkout -B", "gh pr create")]
    assert order == sorted(order)


def test_handle_task_stops_at_first_failure():
    agent, commands = _agent_with_capture(CodeWorkSpaceAgent)

    async def failing_exec(command, *, timeout=None):
        commands.append(command)
        # Fail the project command; clone/config before it succeed.
        code = 1 if "broken" in command else 0
        return ExecutionResult(command=command, exit_code=code, stderr="boom")

    agent.execute_in_sandbox = failing_exec  # type: ignore[assignment]

    reply = asyncio.run(
        agent.dispatch(_request({"commands": ["broken-cmd", "never-runs"]}))
    )
    meta = metadata_dict(reply)
    assert meta["status"] == "error"
    payload = get_message_data(reply)[0]
    assert payload["failed_step"] == "command[0]"
    assert not any("never-runs" in c for c in commands)


def test_empty_request_is_rejected():
    agent, _ = _agent_with_capture(CodeWorkSpaceAgent)
    reply = asyncio.run(agent.dispatch(_request({})))
    assert metadata_dict(reply)["status"] == "error"
    assert "invalid workspace request" in get_message_data(reply)[0]["reason"]


def test_path_traversal_in_files_is_rejected():
    agent, _ = _agent_with_capture(CodeWorkSpaceAgent)
    reply = asyncio.run(agent.dispatch(_request({"files": {"../escape": "x"}})))
    assert metadata_dict(reply)["status"] == "error"


def test_option_injection_in_repo_is_rejected():
    agent, _ = _agent_with_capture(CodeWorkSpaceAgent)
    reply = asyncio.run(agent.dispatch(_request({"repo": "--upload-pack=evil"})))
    assert metadata_dict(reply)["status"] == "error"


# --------------------------------------------------------------------------- #
# Python derivation                                                          #
# --------------------------------------------------------------------------- #
def test_python_setup_and_test_defaults():
    agent, commands = _agent_with_capture(PythonCodeWorkspaceAgent)
    assert agent.DEFAULT_TEST_COMMAND == "uv run --frozen pytest"
    assert any("uv sync" in c for c in agent.setup_commands())

    reply = asyncio.run(agent.dispatch(_request({"test": True})))
    assert metadata_dict(reply)["status"] == "ok"
    blob = " || ".join(commands)
    assert "uv sync" in blob  # setup ran
    assert "uv run --frozen pytest" in blob  # default test command ran


def test_config_override_is_used():
    cfg = WorkspaceConfig(git_user_name="Bot", git_user_email="bot@x.io")
    agent, commands = _agent_with_capture(CodeWorkSpaceAgent, config=cfg)
    asyncio.run(agent.dispatch(_request({"commands": ["echo hi"]})))
    blob = " || ".join(commands)
    assert "user.name Bot" in blob
    assert "user.email bot@x.io" in blob


# --------------------------------------------------------------------------- #
# Base Docker Image guarantees (the gh/git push toolchain)                    #
# --------------------------------------------------------------------------- #
def test_base_dockerfile_guarantees_push_toolchain():
    with open(os.path.join(_SANDBOX_DIR, "Dockerfile"), encoding="utf-8") as f:
        df = f.read()
    # Multi-stage: a dedicated gh-fetch stage whose binary is COPYed into final.
    assert "AS gh-fetch" in df
    assert "COPY --from=gh-fetch" in df
    # git is installed and a build-time assertion proves both tools are present.
    assert "git" in df
    assert "gh --version && git --version" in df
    # Runs unprivileged.
    assert "USER coder" in df


def test_python_dockerfile_derives_base_and_adds_toolchain():
    with open(os.path.join(_SANDBOX_DIR, "python", "Dockerfile"), encoding="utf-8") as f:
        df = f.read()
    # Inherits the Base Docker Image (so gh/git come for free).
    assert "ARG BASE_IMAGE=local-registry/codeworkspace_base" in df
    assert "FROM ${BASE_IMAGE}" in df
    # Compiler + package manager preinstalled, asserted at build time.
    assert "build-essential" in df
    assert "COPY --from=uv" in df
    assert "python3 --version && cc --version && uv --version" in df
