"""Real OpenShell CLI smoke test: sandbox create → exec → delete.

Runs the actual ``openshell`` binary against a real image, so it validates the
CLI argument spelling and lifecycle that the unit tests can only mock. Skips when
the CLI is not installed. The subcommand spellings vary across OpenShell
versions; if this fails on a version mismatch, adjust
``atrium/sandbox/openshell.py`` (the command templates are centralized there).

Env:
    ATRIUM_INTEGRATION=1         opt in (required)
    ATRIUM_IT_IMAGE              image to launch (default: docker.io/library/busybox:latest)
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from atrium.core.types import NetworkMode, SandboxConfig
from atrium.sandbox.openshell import Sandbox, openshell_available

pytestmark = pytest.mark.integration

_IMAGE = os.environ.get("ATRIUM_IT_IMAGE", "docker.io/library/busybox:latest")


@pytest.fixture
def require_openshell():
    if not openshell_available():
        pytest.skip("the 'openshell' CLI is not on PATH")


def test_sandbox_create_exec_delete(require_openshell):
    name = f"atrium-it-{uuid.uuid4().hex[:8]}"
    config = SandboxConfig(network=NetworkMode.INTERNAL, internal=True)

    async def scenario() -> str:
        sandbox = await Sandbox.create(_IMAGE, config, name=name)
        try:
            assert sandbox.is_running
            result = await sandbox.exec("echo atrium-ok")
            assert result.succeeded, result.stderr
            return result.stdout
        finally:
            await sandbox.delete()

    out = asyncio.run(scenario())
    assert "atrium-ok" in out
