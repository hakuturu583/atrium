"""Tests for :class:`atrium.agents.builder_agent.BuilderAgent`.

Kaniko/sandbox-free: ``start_sandbox`` / ``write_files_to_sandbox`` /
``execute_in_sandbox`` are monkeypatched so the build "runs" without a sandbox.
These cover request validation (traversal/injection/semver), the rootless Kaniko
command line, the success/error reply envelope (incl. digest → image_ref), and
the immutable-version collision guard.
"""

from __future__ import annotations

import asyncio

import pytest

from atrium.agents.builder_agent import BuilderAgent
from atrium.agents.builder_agent.agent import STATUS_ERROR, STATUS_OK
from atrium.core.errors import PolicyViolationError
from atrium.core.types import ExecutionResult, GPURequest
from atrium.agents.builder_agent.sandbox import build_sandbox_config
from atrium.protocol import (
    Role,
    data_part,
    get_message_data,
    metadata_dict,
    text_message,
)

_DOCKERFILE = "FROM scratch\n"


def _builder(**kwargs):
    return BuilderAgent("builder-1", "0.1.0", **kwargs)


def _build_agent(kaniko_exit=0, digest="sha256:abcd"):
    """A BuilderAgent whose sandbox interactions are faked; records the build cmd."""
    agent = _builder()
    state = {"commands": [], "staged": None}

    async def fake_start():
        return None

    async def fake_write(files, dest, *, clean=False):
        state["staged"] = {"files": files, "dest": dest, "clean": clean}
        return ExecutionResult(command="stage", exit_code=0)

    async def fake_exec(command, *, timeout=None):
        state["commands"].append(command)
        if command.startswith("cat "):  # _read_digest
            return ExecutionResult(command=command, exit_code=0, stdout=digest + "\n")
        return ExecutionResult(command=command, exit_code=kaniko_exit, stdout="build log")

    agent.start_sandbox = fake_start  # type: ignore[assignment]
    agent.write_files_to_sandbox = fake_write  # type: ignore[assignment]
    agent.execute_in_sandbox = fake_exec  # type: ignore[assignment]
    return agent, state


def _request(payload):
    return text_message(
        "",
        role=Role.ROLE_USER,
        metadata={"kind": "build"},
        extra_parts=[data_part(payload)],
    )


def _ok_payload(**overrides):
    payload = {
        "target_name": "myagent",
        "target_version": "0.2.0",
        "files": {"Dockerfile": _DOCKERFILE, "app.py": "print(1)\n"},
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Security envelope                                                            #
# --------------------------------------------------------------------------- #
def test_rejects_gpu_request():
    cfg = build_sandbox_config("0.1.0")
    cfg.device_requests = [GPURequest()]
    with pytest.raises(PolicyViolationError):
        _builder(sandbox_config=cfg)


def test_rejects_docker_socket_mount():
    cfg = build_sandbox_config("0.1.0")
    cfg.volumes = {"/var/run/docker.sock": "/var/run/docker.sock"}
    with pytest.raises(PolicyViolationError):
        _builder(sandbox_config=cfg)


# --------------------------------------------------------------------------- #
# Request validation (returned as structured error messages, not raised)       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload, needle",
    [
        (_ok_payload(target_name="Bad Name!"), "target_name"),
        (_ok_payload(target_name="--evil"), "target_name"),
        (_ok_payload(target_version="not-semver"), "semver"),
        ({"target_name": "x", "target_version": "1.0.0", "files": {}}, "files"),
        (
            {"target_name": "x", "target_version": "1.0.0", "files": {"app.py": "x"}},
            "dockerfile",
        ),
        (_ok_payload(build_args={"1bad": "v"}), "build-arg"),
    ],
)
def test_invalid_requests_return_error(payload, needle):
    agent, _ = _build_agent()
    reply = asyncio.run(agent.dispatch(_request(payload)))
    assert metadata_dict(reply)["status"] == STATUS_ERROR
    assert needle in get_message_data(reply)[0]["reason"]


def test_dockerfile_traversal_rejected():
    agent, _ = _build_agent()
    payload = _ok_payload(dockerfile="../Dockerfile")
    reply = asyncio.run(agent.dispatch(_request(payload)))
    assert metadata_dict(reply)["status"] == STATUS_ERROR


# --------------------------------------------------------------------------- #
# Kaniko command line (rootless: no docker socket, no privilege)               #
# --------------------------------------------------------------------------- #
def test_build_command_is_rootless_kaniko():
    agent = _builder()
    cmd = agent._build_command("myagent", "0.2.0", "Dockerfile", {})
    assert "/kaniko/executor" in cmd
    assert "--destination=local-registry/myagent:0.2.0" in cmd
    assert "--digest-file=" in cmd
    assert "--no-push=false" in cmd
    # The rootless guarantee: never a daemon socket or privileged flag.
    assert "docker.sock" not in cmd
    assert "--privileged" not in cmd


def test_build_command_includes_build_args():
    agent = _builder()
    cmd = agent._build_command("myagent", "0.2.0", "Dockerfile", {"VERSION": "1.2.3"})
    assert "--build-arg=VERSION=1.2.3" in cmd


def test_image_tag_format():
    agent = _builder()
    assert agent._image_tag("foo", "1.0.0") == "local-registry/foo:1.0.0"


# --------------------------------------------------------------------------- #
# Build outcomes                                                               #
# --------------------------------------------------------------------------- #
def test_successful_build_reports_digest_and_image_ref():
    agent, state = _build_agent(kaniko_exit=0, digest="sha256:deadbeef")
    reply = asyncio.run(agent.dispatch(_request(_ok_payload())))
    meta = metadata_dict(reply)
    assert meta["status"] == STATUS_OK
    payload = get_message_data(reply)[0]
    assert payload["image"] == "local-registry/myagent:0.2.0"
    assert payload["digest"] == "sha256:deadbeef"
    assert payload["image_ref"] == "local-registry/myagent@sha256:deadbeef"
    # The context was staged clean before building.
    assert state["staged"]["clean"] is True
    assert state["staged"]["dest"]  # a workspace path


def test_failed_kaniko_build_returns_error():
    agent, _ = _build_agent(kaniko_exit=1)
    reply = asyncio.run(agent.dispatch(_request(_ok_payload())))
    meta = metadata_dict(reply)
    assert meta["status"] == STATUS_ERROR
    payload = get_message_data(reply)[0]
    assert "kaniko build failed" in payload["reason"]
    assert payload["exit_code"] == 1


# --------------------------------------------------------------------------- #
# Immutable-version collision guard                                            #
# --------------------------------------------------------------------------- #
def test_collision_guard_blocks_existing_version(monkeypatch):
    agent, _ = _build_agent()
    # Simulate the registry reporting the version already exists.
    monkeypatch.setattr(agent, "_version_exists", lambda name, version: True)
    reply = asyncio.run(agent.dispatch(_request(_ok_payload())))
    meta = metadata_dict(reply)
    assert meta["status"] == STATUS_ERROR
    assert "already exists" in get_message_data(reply)[0]["reason"]


def test_no_endpoint_means_no_collision_check():
    # Default agent has no registry_endpoint → guard is skipped (returns False).
    agent, _ = _build_agent()
    assert agent._version_exists("myagent", "0.2.0") is False
