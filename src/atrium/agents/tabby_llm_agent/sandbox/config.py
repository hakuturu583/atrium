"""SandboxConfig factory for the TabbyLLMAgent container.

Encodes this agent's isolation envelope: GPU passthrough on, WAN egress off
(LAN/A2A only), pinned to the shipped OpenShell egress policy.
"""

from __future__ import annotations

import os
from typing import Optional

from atrium.core.types import GPURequest, NetworkMode, SandboxConfig

__all__ = ["build_sandbox_config", "IMAGE_REPOSITORY", "POLICY_PATH"]

#: Local registry repository for this agent's per-version images.
IMAGE_REPOSITORY = "local-registry/tabby_llm_agent"

#: The OpenShell egress policy shipped alongside this config.
POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.yaml")


def build_sandbox_config(
    version: str,
    *,
    gpu_device_ids: Optional[tuple[str, ...]] = None,
    memory: Optional[str] = "24g",
) -> SandboxConfig:
    """Build the GPU, WAN-isolated SandboxConfig for ``version`` of this agent."""
    return SandboxConfig(
        image=f"{IMAGE_REPOSITORY}:{version}",
        network=NetworkMode.INTERNAL,
        internal=True,
        device_requests=[GPURequest(device_ids=gpu_device_ids)],
        memory=memory,
        policy_path=POLICY_PATH if os.path.exists(POLICY_PATH) else None,
        labels={"atrium.agent": "tabby_llm_agent", "atrium.version": str(version)},
    )
