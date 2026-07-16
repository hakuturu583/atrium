"""Core value objects and configuration types shared by all Atrium agents.

These types are intentionally free of any heavy third-party dependency so that
they can be imported on the host control plane without pulling in inference or
HTTP stacks. The only external dependency is :mod:`semver`, used verbatim as the
agent version Value Object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# The agent version Value Object is a third-party ``semver.Version`` used as-is.
from semver import Version as VersionTag

__all__ = [
    "VersionTag",
    "NetworkMode",
    "GPURequest",
    "ExecutionResult",
    "SandboxConfig",
    "wan_sandbox_config",
]


class NetworkMode(str, Enum):
    """Container network isolation policy.

    * ``NONE`` — no NIC at all (fully offline).
    * ``INTERNAL`` — only the host-local container<->container / control-plane
      LAN is reachable; the public internet (WAN) is blocked. This is the policy
      required for GPU-bearing inference containers.
    * ``BRIDGE`` — standard outbound bridge networking; WAN reachable.
    """

    NONE = "none"
    INTERNAL = "internal"
    BRIDGE = "bridge"


@dataclass(slots=True)
class GPURequest:
    """An NVIDIA Container Toolkit GPU passthrough request.

    Mirrors the information OpenShell needs to expose host GPUs to a sandbox via
    ``--gpu``. ``count == -1`` means "all visible GPUs".
    """

    count: int = -1
    capabilities: tuple[str, ...] = ("gpu", "utility", "compute")
    device_ids: Optional[tuple[str, ...]] = None

    def as_cli_flags(self) -> list[str]:
        """Render this request as OpenShell CLI flags."""
        flags: list[str] = ["--gpu"]
        if self.device_ids:
            flags += ["--gpu-devices", ",".join(self.device_ids)]
        elif self.count >= 0:
            flags += ["--gpu-count", str(self.count)]
        return flags


@dataclass(slots=True)
class ExecutionResult:
    """The result of running a single command inside a sandbox."""

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        """``True`` when the command exited with status 0."""
        return self.exit_code == 0


@dataclass(slots=True)
class SandboxConfig:
    """Isolation, resource and network configuration for an OpenShell sandbox.

    ``image`` defaults to ``None``; when unset, the agent falls back to its
    version-derived image name (``local-registry/<slug>:<version>``).
    """

    image: Optional[str] = None
    network: NetworkMode = NetworkMode.INTERNAL
    internal: bool = True
    device_requests: list[GPURequest] = field(default_factory=list)
    cpus: Optional[float] = None
    memory: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    #: Credentials forwarded *by reference* from the host: ``{container_var:
    #: host_var}``. At sandbox-create time each ``host_var`` is read from the
    #: OpenShell process environment and exposed inside the sandbox as
    #: ``container_var``. Unlike :attr:`env`, the secret value never appears on the
    #: OpenShell command line (only the variable *name* is passed), so tokens are
    #: not visible in ``ps``/process listings. Entries whose ``host_var`` is unset
    #: on the host are silently skipped.
    secret_env: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    workdir: str = "/workspace"
    policy_path: Optional[str] = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def gpu_enabled(self) -> bool:
        """``True`` when at least one GPU passthrough request is present."""
        return bool(self.device_requests)

    @property
    def wan_allowed(self) -> bool:
        """``True`` only when this config would permit public internet egress."""
        return self.network == NetworkMode.BRIDGE and not self.internal

    def render_policy_yaml(self) -> str:
        """Render an OpenShell sandbox policy derived from this config.

        Targets the OpenShell CLI policy schema (``version: 1``):
        ``filesystem_policy`` + ``landlock`` + ``process`` (the static isolation
        posture) and ``network_policies`` (the egress allow-list). Network is
        default-deny, so ``INTERNAL``/``NONE`` (WAN-isolated) render an empty
        allow-list while host-local/A2A stays reachable. ``BRIDGE`` (WAN) needs
        explicit per-endpoint allow-lists in this schema, which a rendered
        default cannot express safely, so a workspace that needs broad public
        egress should pin a hand-authored policy via ``policy_path``. The exact
        schema may differ between CLI versions; the templates are centralized
        here so they can be adjusted in one place.
        """
        deny_wan = self.network in (NetworkMode.INTERNAL, NetworkMode.NONE) or self.internal
        # The sandbox workdir must be writable for file-staging + runs. De-dup
        # while preserving order (the workdir may already be one of the scratch
        # dirs below).
        read_write, seen = [], set()
        for path in (self.workdir or "/workspace", "/sandbox", "/tmp", "/dev/null"):
            if path not in seen:
                seen.add(path)
                read_write.append(path)
        net_comment = (
            "  # WAN-isolated: empty egress allow-list"
            if deny_wan
            else "  # NOTE: BRIDGE needs explicit endpoint allow-lists; pin one via policy_path"
        )
        lines = [
            "# Auto-generated by atrium.core.types.SandboxConfig.render_policy_yaml",
            "version: 1",
            "filesystem_policy:",
            "  include_workdir: true",
            "  read_only: [/usr, /lib, /proc, /dev/urandom, /app, /etc, /var/log]",
            f"  read_write: [{', '.join(read_write)}]",
            "landlock:",
            "  compatibility: best_effort",
            "process:",
            "  run_as_user: sandbox",
            "  run_as_group: sandbox",
            "network_policies: {}" + net_comment,
        ]
        return "\n".join(lines) + "\n"


def wan_sandbox_config() -> SandboxConfig:
    """A WAN-capable sandbox envelope (bridge network, not internal).

    The shared default for agents that must reach an external service — a chat
    app, the Prefect server, the builder — rather than staying WAN-isolated. The
    Docker-socket refusal is a runtime guard (``BaseAgent.forbid_docker_socket``),
    not part of this config.
    """
    return SandboxConfig(network=NetworkMode.BRIDGE, internal=False)
