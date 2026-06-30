"""TabbyLLMAgent — a self-contained, independently versioned Atrium agent.

This package bundles everything that defines one generation of the agent:

* ``agent.py``   — the host-side :class:`TabbyLLMAgent` (A2A only).
* ``card.py``    — its A2A :class:`AgentCard`.
* ``bridge/``    — the in-sandbox A2A↔tabbyAPI bridge (container-side; the only
  place OpenAI-HTTP exists).
* ``sandbox/``   — the OpenShell container image (Dockerfile), dependencies,
  egress policy and :class:`SandboxConfig` factory for this agent.

``__version__`` below is the single source of truth for the agent version and
its image tag ``local-registry/tabby_llm_agent:<__version__>`` — bump it to ship
a new generation.

Note: ``bridge.server`` is intentionally *not* imported here; it depends on the
container-only ASGI/HTTP stack. Import it explicitly inside the sandbox.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.tabby_llm_agent.agent import TabbyAgentConfig, TabbyLLMAgent

__all__ = ["TabbyLLMAgent", "TabbyAgentConfig", "__version__"]
