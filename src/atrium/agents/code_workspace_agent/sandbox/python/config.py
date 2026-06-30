"""SandboxConfig factory for the PythonCodeWorkspaceAgent container.

The Python workspace image inherits the Base Docker Image (so the ``git`` + ``gh``
push toolchain is guaranteed) and adds the Python toolchain (interpreter, C
compiler, ``uv`` package manager). Its isolation envelope is identical to the
base workspace; only the image repository differs, so this is a thin wrapper over
:func:`atrium.agents.code_workspace_agent.sandbox.config.build_sandbox_config`.
"""

from __future__ import annotations

from typing import Mapping, Optional

from atrium.agents.code_workspace_agent.sandbox.config import (
    build_sandbox_config as _build_base_sandbox_config,
)
from atrium.core.types import NetworkMode, SandboxConfig

__all__ = ["build_sandbox_config", "IMAGE_REPOSITORY"]

#: Local registry repository for the Python workspace image (FROM the base).
IMAGE_REPOSITORY = "local-registry/python_code_workspace_agent"


def build_sandbox_config(
    version: str,
    *,
    network: NetworkMode = NetworkMode.BRIDGE,
    memory: Optional[str] = "8g",
    cpus: Optional[float] = 4.0,
    env: Optional[Mapping[str, str]] = None,
    gh_token: Optional[str] = None,
) -> SandboxConfig:
    """Build the Python-workspace SandboxConfig for ``version`` of this agent.

    Same envelope as the base workspace (see the base
    :func:`~atrium.agents.code_workspace_agent.sandbox.config.build_sandbox_config`)
    but pinned to the Python image and with a slightly larger resource default
    (compiling native wheels and running test suites is heavier).
    """
    return _build_base_sandbox_config(
        version,
        repository=IMAGE_REPOSITORY,
        network=network,
        memory=memory,
        cpus=cpus,
        env=env,
        gh_token=gh_token,
        agent_slug="python_code_workspace_agent",
    )
