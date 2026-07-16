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
def test_create_applies_policy_at_create(fake_run):
    calls, _ = fake_run
    cfg = SandboxConfig(network=NetworkMode.INTERNAL)
    sb = asyncio.run(Sandbox.create("local-registry/foo:1.0.0", cfg, name="foo"))
    assert sb.is_running is True
    # The policy is applied *at* create via `--policy` (the CLI's `policy set`
    # targets an already-running sandbox), so there is no separate policy call.
    assert not any(c["args"][:2] == ["policy", "set"] for c in calls)
    create = _args_of(calls, "sandbox create")
    a = create["args"]
    assert "--from" in a and "local-registry/foo:1.0.0" in a
    assert "--name" in a and "foo" in a
    assert "--policy" in a
    # A no-op command + no PTY so create returns and leaves the sandbox running.
    assert "--no-tty" in a
    assert a[-2:] == ["--", "true"]


def test_create_adds_gpu_cpu_memory_env(fake_run):
    calls, _ = fake_run
    cfg = SandboxConfig(
        device_requests=[GPURequest()],
        cpus=4.0,
        memory="8Gi",
        env={"FOO": "bar"},
    )
    asyncio.run(Sandbox.create("img", cfg, name="gpu1"))
    create = _args_of(calls, "sandbox create")
    a = create["args"]
    assert "--gpu" in a
    assert a[a.index("--cpu") + 1] == "4.0"
    assert a[a.index("--memory") + 1] == "8Gi"
    assert "FOO=bar" in a


def test_create_rejects_volumes(fake_run):
    # The OpenShell CLI has no host bind-mount flag; requesting one must fail
    # loudly rather than be silently dropped.
    cfg = SandboxConfig(volumes={"/host/data": "/data"})
    with pytest.raises(SandboxError, match="volume mounts are not supported"):
        asyncio.run(Sandbox.create("img", cfg, name="vol"))


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
    assert args == ["sandbox", "exec", "--name", "x", "--", "bash", "-lc", "echo hi"]


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
