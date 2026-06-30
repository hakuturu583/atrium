"""BuilderAgent — the fixed (non-evolving) rootless image builder.

This package bundles everything that defines one generation of the agent:

* ``agent.py``   — the :class:`BuilderAgent` (A2A in, image built + pushed).
* ``sandbox/``   — the OpenShell container image (``Dockerfile`` with Kaniko),
  the egress ``policy.yaml`` and the :class:`SandboxConfig` factory that together
  form this agent's security envelope.

Unlike the evolving agents, ``BuilderAgent`` is *fixed infrastructure*: it is the
trusted step that turns other agents' source into images, so it is excluded from
the self-evolution loop (see :data:`atrium.agents.builder_agent.agent.BuilderAgent.IMMUTABLE`)
and only ever changed by an explicit, human-approved edit.

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/builder_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.builder_agent.agent import BuilderAgent
from atrium.agents.builder_agent.sandbox import build_sandbox_config

__all__ = ["BuilderAgent", "build_sandbox_config", "__version__"]
