"""Sandbox definition for BuilderAgent (container image, egress policy, config)."""

from __future__ import annotations

from atrium.agents.builder_agent.sandbox.config import (
    DEFAULT_REGISTRY,
    IMAGE_REPOSITORY,
    POLICY_PATH,
    WORKSPACE,
    build_sandbox_config,
)

__all__ = [
    "build_sandbox_config",
    "IMAGE_REPOSITORY",
    "POLICY_PATH",
    "DEFAULT_REGISTRY",
    "WORKSPACE",
]
