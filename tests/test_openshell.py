"""Tests for the OpenShell CLI wrapper (:mod:`atrium.sandbox.openshell`).

The subprocess boundary (`_run`) and the CLI-presence check (`_require_cli`) are
monkeypatched, so these assert on *how the CLI would be driven* — argument
assembly, policy application, secret forwarding, and error handling — without
the real ``openshell`` binary ever being invoked.
"""

from __future__ import annotations

import asyncio

import pytest

from atrium.core.errors import SandboxError
from atrium.core.types import ExecutionResult, GPURequest, NetworkMode, SandboxConfig
from atrium.sandbox import openshell
from atrium.sandbox.openshell import Sandbox, openshell_available


@pytest.fixture
def fake_run(monkeypatch):
    """Replace ``_run`` with a recorder; ``_require_cli`` becomes a no-op."""
    calls: list[dict] = []
    outcomes: dict[str, ExecutionResult] = {}

    async def _run(*args, timeout=None, env=None):
        calls.append({"args": list(args), "env": env, "timeout": timeout})
        key = " ".join(args[:2])  # e.g. "policy set", "sandbox create"
        return outcomes.get(
            key, ExecutionResult(command=" ".join(args), exit_code=0, stdout="ok")
        )

    monkeypatch.setattr(openshell, "_run", _run)
    monkeypatch.setattr(openshell, "_require_cli", lambda: "/usr/bin/openshell")
    return calls, outcomes


def _args_of(calls, first_two):
    for c in calls:
        if c["args"][:2] == first_two.split():
            return c
    raise AssertionError(f"no call matching {first_two!r} in {calls}")


# --------------------------------------------------------------------------- #
# CLI presence                                                                 #
# --------------------------------------------------------------------------- #
def test_openshell_available_reflects_path(monkeypatch):
    monkeypatch.setattr(openshell.shutil, "which", lambda _b: "/usr/bin/openshell")
    assert openshell_available() is True
    monkeypatch.setattr(openshell.shutil, "which", lambda _b: None)
    assert openshell_available() is False


def test_require_cli_raises_when_absent(monkeypatch):
    monkeypatch.setattr(openshell.shutil, "which", lambda _b: None)
    with pytest.raises(SandboxError, match="not found on PATH"):
        openshell._require_cli()


# --------------------------------------------------------------------------- #
# Sandbox.create — argument assembly                                           #
# --------------------------------------------------------------------------- #
def test_create_sets_policy_then_creates(fake_run):
    calls, _ = fake_run
    cfg = SandboxConfig(network=NetworkMode.INTERNAL)
    sb = asyncio.run(Sandbox.create("local-registry/foo:1.0.0", cfg, name="foo"))
    assert sb.is_running is True
    # policy applied before create.
    assert calls[0]["args"][:2] == ["policy", "set"]
    create = _args_of(calls, "sandbox create")
    assert "--from" in create["args"] and "local-registry/foo:1.0.0" in create["args"]
    assert "--name" in create["args"] and "foo" in create["args"]


def test_create_adds_gpu_cpu_memory_volume_env(fake_run):
    calls, _ = fake_run
    cfg = SandboxConfig(
        device_requests=[GPURequest()],
        cpus=4.0,
        memory="8g",
        volumes={"/host/data": "/data"},
        env={"FOO": "bar"},
    )
    asyncio.run(Sandbox.create("img", cfg, name="gpu1"))
    create = _args_of(calls, "sandbox create")
    a = create["args"]
    assert "--gpu" in a
    assert a[a.index("--cpus") + 1] == "4.0"
    assert a[a.index("--memory") + 1] == "8g"
    assert "/host/data:/data" in a
    assert "FOO=bar" in a


def test_create_forwards_secret_env_by_name_only(fake_run, monkeypatch):
    calls, _ = fake_run
    monkeypatch.setattr(openshell.os, "environ", {"HOST_TOKEN": "s3cr3t"})
    cfg = SandboxConfig(secret_env={"GH_TOKEN": "HOST_TOKEN"})
    asyncio.run(Sandbox.create("img", cfg, name="sec"))
    create = _args_of(calls, "sandbox create")
    a = create["args"]
    # The container var name is passed on the command line...
    assert "GH_TOKEN" in a
    # ...but the secret VALUE never appears in argv.
    assert "s3cr3t" not in a
    # ...it's handed to the child via env instead.
    assert create["env"]["GH_TOKEN"] == "s3cr3t"


def test_create_raises_when_policy_set_fails(fake_run):
    calls, outcomes = fake_run
    outcomes["policy set"] = ExecutionResult(
        command="policy set", exit_code=1, stderr="denied"
    )
    with pytest.raises(SandboxError, match="policy set failed"):
        asyncio.run(Sandbox.create("img", SandboxConfig(), name="bad"))


def test_create_raises_when_create_fails(fake_run):
    _calls, outcomes = fake_run
    outcomes["sandbox create"] = ExecutionResult(
        command="sandbox create", exit_code=1, stderr="boom"
    )
    with pytest.raises(SandboxError, match="create failed"):
        asyncio.run(Sandbox.create("img", SandboxConfig(), name="bad"))


# --------------------------------------------------------------------------- #
# exec / delete                                                                #
# --------------------------------------------------------------------------- #
def test_exec_requires_running():
    sb = Sandbox(name="x", image="img", config=SandboxConfig())
    with pytest.raises(SandboxError, match="not running"):
        asyncio.run(sb.exec("echo hi"))


def test_exec_uses_login_shell(fake_run):
    calls, _ = fake_run
    sb = Sandbox(name="x", image="img", config=SandboxConfig(), _running=True)
    asyncio.run(sb.exec("echo hi"))
    args = calls[-1]["args"]
    assert args == ["sandbox", "exec", "x", "--", "bash", "-lc", "echo hi"]


def test_delete_when_running(fake_run):
    calls, _ = fake_run
    sb = Sandbox(name="x", image="img", config=SandboxConfig(), _running=True)
    asyncio.run(sb.delete())
    assert calls[-1]["args"] == ["sandbox", "delete", "x"]
    assert sb.is_running is False


def test_delete_when_not_running_is_noop(fake_run):
    calls, _ = fake_run
    sb = Sandbox(name="x", image="img", config=SandboxConfig())
    asyncio.run(sb.delete())
    assert calls == []  # nothing invoked
