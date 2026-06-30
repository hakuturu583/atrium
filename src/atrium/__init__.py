"""Atrium — a self-evolving, security-isolated multi-agent runtime.

Public surface:

* :class:`~atrium.core.base_agent.BaseAgent` — abstract base equipping agents
  with OpenShell sandbox lifecycle, A2A communication and OpenInference tracing.
* :class:`~atrium.agents.inference_agent.InferenceAgent` — WAN-isolated, GPU-only
  inference base.
* :class:`~atrium.agents.tabby_llm_agent.TabbyLLMAgent` — concrete inference
  agent backed by tabbyAPI / exllamav3, spoken to exclusively over A2A.

Communication between agents is A2A throughout (via ``a2a-sdk``); the host
package never imports ``httpx`` (that lives in the agent container images).
"""

from __future__ import annotations

from atrium.agents.inference_agent import InferenceAgent
from atrium.agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent
from atrium.core.base_agent import BaseAgent
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
    "SandboxConfig",
    "NetworkMode",
    "GPURequest",
    "ExecutionResult",
    "VersionTag",
]
