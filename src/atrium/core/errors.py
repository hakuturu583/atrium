"""Exception hierarchy for the Atrium runtime.

All Atrium-specific errors derive from :class:`AtriumError` so callers can catch
the whole family with a single ``except`` while still being able to discriminate
between sandbox, policy, transport and inference failures.
"""

from __future__ import annotations

__all__ = [
    "AtriumError",
    "AgentError",
    "SandboxError",
    "SandboxNotRunningError",
    "PolicyViolationError",
    "A2ATransportError",
    "ModelNotReadyError",
]


class AtriumError(RuntimeError):
    """Base class for every error raised by the Atrium runtime."""


class AgentError(AtriumError):
    """Generic agent-level failure."""


class SandboxError(AgentError):
    """An OpenShell sandbox lifecycle or in-sandbox execution failure."""


class SandboxNotRunningError(SandboxError):
    """Raised when an operation needs a running sandbox but none is active."""


class PolicyViolationError(AgentError):
    """Raised when a security/isolation invariant would be violated.

    Raised by an agent's construction-time policy check to refuse a
    configuration that would break its isolation envelope — e.g. exposing a
    GPU-bearing inference container to the WAN, or mounting the host Docker
    socket into a code workspace.
    """


class A2ATransportError(AgentError):
    """An A2A message send/receive failure."""


class ModelNotReadyError(AgentError):
    """The inference backend has no servable model yet.

    Typically raised while a model is still being quantized (EXL3) or loaded by
    tabbyAPI. Callers may retry after :meth:`wait_until_ready`.
    """
