"""Sandbox definition for code workspaces (Base Docker Image, policy, config).

``Dockerfile`` here is the **Base Docker Image** every coding-agent workspace
inherits from; it guarantees the ``git`` + ``gh`` push toolchain via a Docker
multi-stage build. Language-specific images live in sibling sub-packages (e.g.
``python/``) that ``FROM`` this base.
"""

from __future__ import annotations

from atrium.agents.code_workspace_agent.sandbox.config import (
    BASE_IMAGE_REPOSITORY,
    DEFAULT_TOKEN_ENV,
    WORKSPACE,
    build_sandbox_config,
)

__all__ = [
    "build_sandbox_config",
    "BASE_IMAGE_REPOSITORY",
    "DEFAULT_TOKEN_ENV",
    "WORKSPACE",
]
