"""Atrium — a self-evolving, security-isolated multi-agent runtime.

Public surface:

* :class:`~atrium.core.base_agent.BaseAgent` — abstract base equipping agents
  with OpenShell sandbox lifecycle, A2A communication and OpenInference tracing.
* :class:`~atrium.agents.inference_agent.InferenceAgent` — WAN-isolated, GPU-only
  inference base.
* :class:`~atrium.agents.tabby_llm_agent.TabbyLLMAgent` — concrete inference
  agent backed by tabbyAPI / exllamav3, spoken to exclusively over A2A.
* :class:`~atrium.agents.builder_agent.BuilderAgent` — fixed-infrastructure agent
  that builds agent images with rootless Kaniko (no host Docker daemon).
* :func:`~atrium.core.registry.ensure_local_registry` — fixed-infrastructure
  bootstrap that brings up the local container registry (the generation ledger)
  via the host Docker daemon, for the trusted main process only.

Communication between agents is A2A throughout (via ``a2a-sdk``); the host
package never imports ``httpx`` (that lives in the agent container images).
"""

from __future__ import annotations

from atrium.agents.builder_agent import BuilderAgent
from atrium.agents.inference_agent import InferenceAgent
from atrium.agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent
from atrium.core.base_agent import BaseAgent
from atrium.core.factory import (
    create_agent,
    create_agent_by_slug,
    register_agent_type,
    resolve_active_ref,
)
from atrium.core.morpher import (
    Attestation,
    AttestationSigner,
    Morpher,
    generate_trust_root,
)
from atrium.core.registry import (
    AgentRef,
    RegistryClient,
    RegistryConfig,
    ensure_local_registry,
    next_version,
)
from atrium.core.types import (
    ExecutionResult,
    GPURequest,
    NetworkMode,
    SandboxConfig,
    VersionTag,
)

__all__ = [
    "BaseAgent",
    "InferenceAgent",
    "TabbyLLMAgent",
    "TabbyAgentConfig",
    "BuilderAgent",
    "RegistryConfig",
    "ensure_local_registry",
    "RegistryClient",
    "AgentRef",
    "next_version",
    "create_agent",
    "create_agent_by_slug",
    "register_agent_type",
    "resolve_active_ref",
    "Morpher",
    "Attestation",
    "AttestationSigner",
    "generate_trust_root",
    "SandboxConfig",
    "NetworkMode",
    "GPURequest",
    "ExecutionResult",
    "VersionTag",
]

# Register the built-in concrete agents so they can be launched from a bare slug
# (create_agent_by_slug) once the registry has an active generation for them.
register_agent_type(BuilderAgent)
register_agent_type(TabbyLLMAgent)
