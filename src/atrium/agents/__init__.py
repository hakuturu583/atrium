"""Fixed Atrium agents — the evolution machinery that must not self-rewrite.

Only the agents that are part of the trusted, non-evolving infrastructure live
here:

* :mod:`~atrium.agents.builder_agent` — the rootless image builder (``IMMUTABLE``);
* :mod:`~atrium.agents.task_agent` — the self-evolution driver;
* :mod:`~atrium.agents.code_workspace_agent` — the sandboxed coding "hands".

Each is its own self-contained, independently versioned package bundling its
host-side A2A code together with its sandbox container definition and policy.

The *evolvable* worker agents (e.g. the tabby LLM inference agent) have been
split out into the separate ``atrium_agents`` distribution, which depends on this
package for the shared runtime — so the self-evolution loop can rewrite them
without ever touching the machinery above.
"""

from __future__ import annotations

__all__: list[str] = []
