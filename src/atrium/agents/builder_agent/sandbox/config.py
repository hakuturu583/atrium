"""SandboxConfig factory for the BuilderAgent container — the build envelope.

This is the package-internal source of truth for the BuilderAgent's security
guarantees: a rootless, WAN-isolated, GPU-free sandbox pinned to the shipped
OpenShell egress policy. The host never has to configure isolation — it is
encoded here and re-checked at construction by
:meth:`~atrium.agents.builder_agent.agent.BuilderAgent._enforce_build_policy`.
"""

from __future__ import annotations

import os
from typing import Optional

from atrium.core.types import NetworkMode, SandboxConfig

__all__ = [
    "build_sandbox_config",
    "IMAGE_REPOSITORY",
    "POLICY_PATH",
    "DEFAULT_REGISTRY",
    "WORKSPACE",
]

#: Build-context directory inside the sandbox (single source of truth, also the
#: sandbox ``workdir``; the agent writes the build context here).
WORKSPACE = "/workspace"

#: Local registry repository for this agent's own per-version images.
IMAGE_REPOSITORY = "local-registry/builder_agent"

#: Internal registry the agent builds *into* (push destination), WAN-unreachable.
DEFAULT_REGISTRY = "local-registry"

#: The OpenShell egress policy shipped alongside this config.
POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.yaml")


def build_sandbox_config(
    version: str,
    *,
    registry: str = DEFAULT_REGISTRY,
    memory: Optional[str] = "4g",
    cpus: Optional[float] = 2.0,
) -> SandboxConfig:
    """Build the rootless, WAN-isolated, GPU-free SandboxConfig for ``version``.

    The envelope, by construction:

    * ``network=INTERNAL`` + ``internal=True`` — host-local LAN only; WAN egress
      is denied (so a hostile Dockerfile cannot exfiltrate during a build), while
      ``registry`` (the local-registry push target) stays reachable.
    * ``device_requests=[]`` — no GPU passthrough (a builder needs none; minimal
      privilege).
    * ``volumes={}`` — crucially, the host Docker socket is *never* mounted.
    * pinned to the shipped ``policy.yaml`` (no_new_privileges, drop ALL caps,
      read-only root).
    """
    return SandboxConfig(
        image=f"{IMAGE_REPOSITORY}:{version}",
        network=NetworkMode.INTERNAL,
        internal=True,
        device_requests=[],
        cpus=cpus,
        memory=memory,
        env={
            # Kaniko reads registry credentials from $DOCKER_CONFIG/config.json;
            # an insecure local registry needs none, but keep the path explicit.
            "DOCKER_CONFIG": "/kaniko/.docker",
            "ATRIUM_REGISTRY": registry,
        },
        workdir=WORKSPACE,
        policy_path=POLICY_PATH if os.path.exists(POLICY_PATH) else None,
        labels={
            "atrium.agent": "builder_agent",
            "atrium.version": str(version),
            # Marks this sandbox as fixed infrastructure (excluded from evolution).
            "atrium.immutable": "true",
        },
    )
