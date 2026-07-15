"""Atrium core: agent base class, value objects, telemetry and errors."""

from __future__ import annotations

from atrium.core.errors import (
    A2ATransportError,
    AgentError,
    AtriumError,
    ModelNotReadyError,
    PolicyViolationError,
    SandboxError,
    SandboxNotRunningError,
)
from atrium.core.telemetry import configure_tracing, shutdown_tracing
from atrium.core.types import (
    ExecutionResult,
    GPURequest,
    NetworkMode,
    SandboxConfig,
    VersionTag,
)

# NOTE: ``base_agent`` is intentionally NOT imported here to keep this module
# importable without the A2A stack. Import it explicitly:
#     from atrium.core.base_agent import BaseAgent

__all__ = [
    "VersionTag",
    "NetworkMode",
    "GPURequest",
    "SandboxConfig",
    "ExecutionResult",
    "AtriumError",
    "AgentError",
    "SandboxError",
    "SandboxNotRunningError",
    "PolicyViolationError",
    "A2ATransportError",
    "ModelNotReadyError",
    "configure_tracing",
    "shutdown_tracing",
]
