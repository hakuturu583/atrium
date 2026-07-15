"""Tests for :class:`atrium.core.base_agent.BaseAgent` infrastructure.

Sandbox- and network-free: ``Sandbox.create`` is monkeypatched to a fake handle,
so the lifecycle, ``execute_in_sandbox``, ``write_files_to_sandbox`` staging,
the file/path validation guards, ``merge_data_parts`` and the ``dispatch`` trace
seam are all exercised without OpenShell, Docker or a GPU.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from atrium.core.base_agent import DEFAULT_REGISTRY, BaseAgent
from atrium.core.errors import PolicyViolationError, SandboxNotRunningError
from atrium.core.types import ExecutionResult, SandboxConfig
from atrium.protocol import Role, data_part, text_message


class DummyAgent(BaseAgent):
    """Concrete BaseAgent recording each dispatched message."""

    AGENT_SLUG = "dummyagent"

    def __init__(self, agent_id="a1", version="1.2.3", sandbox_config=None):
        super().__init__(agent_id, version, sandbox_config)
        self.handled: list = []

    async def handle_task(self, message):
        self.handled.append(message)
        return text_message("handled", role=Role.ROLE_AGENT)


class FakeSandbox:
    """Stand-in for an OpenShell sandbox handle."""

    def __init__(self):
        self.is_running = True
        self.deleted = False
        self.exec_calls: list[str] = []

    async def exec(self, command, *, timeout=None):
        self.exec_calls.append(command)
        return ExecutionResult(command=command, exit_code=0, stdout="ok")

    async def delete(self):
        self.deleted = True
        self.is_running = False


@pytest.fixture
def fake_sandbox(monkeypatch):
    created = {}

    async def fake_create(image, config, *, name=None):
        sb = FakeSandbox()
        created["image"] = image
        created["name"] = name
        created["sandbox"] = sb
        return sb

    monkeypatch.setattr("atrium.core.base_agent.Sandbox.create", staticmethod(fake_create))
    return created


# --------------------------------------------------------------------------- #
# Identity                                                                     #
# --------------------------------------------------------------------------- #
def test_slug_and_image_name():
    agent = DummyAgent()
    assert agent.agent_slug == "dummyagent"
    assert DummyAgent.slug_for() == "dummyagent"
    assert agent.image_name == f"{DEFAULT_REGISTRY}/dummyagent:1.2.3"


def test_version_parsed_from_string():
    agent = DummyAgent(version="0.4.1")
    assert str(agent.version) == "0.4.1"


# --------------------------------------------------------------------------- #
# Sandbox lifecycle                                                            #
# --------------------------------------------------------------------------- #
def test_start_sandbox_creates_and_marks_running(fake_sandbox):
    agent = DummyAgent()
    assert agent.is_running is False
    asyncio.run(agent.start_sandbox())
    assert agent.is_running is True
    # Falls back to the version-derived image when config.image is unset.
    assert fake_sandbox["image"] == agent.image_name
    assert fake_sandbox["name"] == "a1"


def test_start_sandbox_is_idempotent(fake_sandbox):
    agent = DummyAgent()

    async def go():
        await agent.start_sandbox()
        first = agent.current_sandbox
        await agent.start_sandbox()  # no-op, same handle
        return first is agent.current_sandbox

    assert asyncio.run(go()) is True


def test_start_sandbox_prefers_config_image(fake_sandbox):
    cfg = SandboxConfig(image="local-registry/dummyagent@sha256:pinned")
    agent = DummyAgent(sandbox_config=cfg)
    asyncio.run(agent.start_sandbox())
    assert fake_sandbox["image"] == "local-registry/dummyagent@sha256:pinned"


def test_stop_sandbox_deletes_and_clears(fake_sandbox):
    agent = DummyAgent()

    async def go():
        await agent.start_sandbox()
        sb = agent.current_sandbox
        await agent.stop_sandbox()
        return sb

    sb = asyncio.run(go())
    assert sb.deleted is True
    assert agent.current_sandbox is None
    assert agent.is_running is False


def test_stop_sandbox_without_start_is_noop(fake_sandbox):
    agent = DummyAgent()
    asyncio.run(agent.stop_sandbox())  # must not raise
    assert agent.current_sandbox is None


def test_async_context_manager_starts_and_stops(fake_sandbox):
    agent = DummyAgent()

    async def go():
        async with agent as a:
            assert a.is_running is True
        return agent.is_running

    assert asyncio.run(go()) is False


# --------------------------------------------------------------------------- #
# execute_in_sandbox                                                           #
# --------------------------------------------------------------------------- #
def test_execute_without_sandbox_raises():
    agent = DummyAgent()
    with pytest.raises(SandboxNotRunningError):
        asyncio.run(agent.execute_in_sandbox("echo hi"))


def test_execute_delegates_to_sandbox(fake_sandbox):
    agent = DummyAgent()

    async def go():
        await agent.start_sandbox()
        return await agent.execute_in_sandbox("echo hi")

    result = asyncio.run(go())
    assert result.succeeded
    assert fake_sandbox["sandbox"].exec_calls == ["echo hi"]


# --------------------------------------------------------------------------- #
# write_files_to_sandbox                                                       #
# --------------------------------------------------------------------------- #
def test_write_files_encodes_base64_and_mkdirs(fake_sandbox, monkeypatch):
    agent = DummyAgent()
    captured = {}

    async def capture(command, *, timeout=None):
        captured["cmd"] = command
        return ExecutionResult(command=command, exit_code=0)

    monkeypatch.setattr(agent, "execute_in_sandbox", capture)
    asyncio.run(
        agent.write_files_to_sandbox(
            {"src/app.py": b"print(1)\n"}, "/workspace", clean=True
        )
    )
    cmd = captured["cmd"]
    assert "set -eu" in cmd
    assert "rm -rf /workspace/*" in cmd  # clean=True
    assert "mkdir -p /workspace" in cmd
    assert "mkdir -p /workspace/src" in cmd  # nested parent created
    b64 = base64.b64encode(b"print(1)\n").decode()
    assert b64 in cmd
    assert "base64 -d > /workspace/src/app.py" in cmd


# --------------------------------------------------------------------------- #
# Validation guards                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["../escape", "/abs/path", "a/../b", ""])
def test_check_safe_relpath_rejects_bad(name):
    with pytest.raises(ValueError):
        BaseAgent.check_safe_relpath(name)


@pytest.mark.parametrize("name", ["a.py", "src/app.py", "deep/nested/file.txt"])
def test_check_safe_relpath_accepts_good(name):
    BaseAgent.check_safe_relpath(name)  # must not raise


def test_coerce_file_content_str_to_utf8():
    assert BaseAgent.coerce_file_content("a.txt", "héllo") == "héllo".encode("utf-8")


def test_coerce_file_content_base64_mapping():
    payload = {"encoding": "base64", "content": base64.b64encode(b"\x00\x01").decode()}
    assert BaseAgent.coerce_file_content("blob.bin", payload) == b"\x00\x01"


def test_coerce_file_content_rejects_bad_base64():
    with pytest.raises(ValueError, match="invalid base64"):
        BaseAgent.coerce_file_content("blob.bin", {"encoding": "base64", "content": "!!!"})


def test_coerce_file_content_rejects_unknown_type():
    with pytest.raises(ValueError, match="unsupported content"):
        BaseAgent.coerce_file_content("x", 123)


def test_coerce_file_content_rejects_traversal():
    with pytest.raises(ValueError):
        BaseAgent.coerce_file_content("../evil", "x")


def test_merge_data_parts_merges_later_wins():
    msg = text_message(
        "",
        role=Role.ROLE_USER,
        extra_parts=[data_part({"a": 1, "b": 2}), data_part({"b": 3, "c": 4})],
    )
    assert BaseAgent.merge_data_parts(msg) == {"a": 1, "b": 3, "c": 4}


def test_forbid_docker_socket_raises():
    cfg = SandboxConfig(volumes={"/var/run/docker.sock": "/var/run/docker.sock"})
    agent = DummyAgent(sandbox_config=cfg)
    with pytest.raises(PolicyViolationError):
        agent.forbid_docker_socket()


def test_forbid_docker_socket_allows_clean_config():
    agent = DummyAgent(sandbox_config=SandboxConfig(volumes={"/data": "/data"}))
    agent.forbid_docker_socket()  # must not raise


# --------------------------------------------------------------------------- #
# dispatch / A2A                                                               #
# --------------------------------------------------------------------------- #
def test_dispatch_invokes_handle_task():
    agent = DummyAgent()
    msg = text_message("do it", role=Role.ROLE_USER)
    reply = asyncio.run(agent.dispatch(msg))
    assert agent.handled == [msg]
    from atrium.protocol import get_message_text

    assert get_message_text(reply) == "handled"


def test_a2a_endpoint_derives_from_agent_id():
    assert DummyAgent(agent_id="coder-1").a2a_endpoint() == "http://coder-1.local"
