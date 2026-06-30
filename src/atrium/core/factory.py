"""Agent factory — launch agents pinned to their *active* generation.

The runtime resolves which generation of an agent to run by reading the registry
(the ledger): ``<slug>:active`` names the live generation, and it resolves to an
immutable ``sha256:…`` digest. The factory turns that into a constructed agent
whose sandbox launches that exact image — replacing the package ``__version__``
default on the evolving path. This is the "reflection" step of the version
lifecycle (see ``docs/design/agent-versioning.md``).

Two seams:

* :func:`resolve_active_ref` / :func:`active_image` — pure resolution
  (``slug`` → pinned :class:`AgentRef` / ``<registry>/<slug>@<digest>``).
* :func:`create_agent` / :func:`create_agent_by_slug` — construct an agent and
  pin its sandbox image to the resolved generation.

``registry`` is the image-name prefix the sandbox addresses (e.g.
``"local-registry"``); ``registry_endpoint`` is the host-side ``host:port`` the
factory queries over HTTP. They differ because host and sandbox reach the
registry by different addresses.
"""

from __future__ import annotations

from typing import Any, TypeVar

from atrium.core.base_agent import DEFAULT_REGISTRY, BaseAgent
from atrium.core.errors import AtriumError
from atrium.core.registry import AgentRef, RegistryClient

__all__ = [
    "NoActiveGenerationError",
    "register_agent_type",
    "agent_type_for",
    "resolve_active_ref",
    "active_image",
    "create_agent",
    "create_agent_by_slug",
]

A = TypeVar("A", bound=BaseAgent)

#: slug → agent class, populated by :func:`register_agent_type` (used by
#: :func:`create_agent_by_slug`). Kept here so the factory imports no concrete
#: agent packages; registration is done by the owner (e.g. ``atrium/__init__``).
_AGENT_TYPES: dict[str, type[BaseAgent]] = {}


class NoActiveGenerationError(AtriumError):
    """Raised when an agent slug has no ``:active`` generation to launch."""


def register_agent_type(agent_cls: type[A]) -> type[A]:
    """Register ``agent_cls`` under its slug so it can be built from a bare slug.

    Usable as a decorator; returns the class unchanged.
    """
    _AGENT_TYPES[agent_cls.slug_for()] = agent_cls
    return agent_cls


def agent_type_for(slug: str) -> type[BaseAgent]:
    """The registered agent class for ``slug`` (raises ``KeyError`` if unknown)."""
    try:
        return _AGENT_TYPES[slug]
    except KeyError:
        raise KeyError(
            f"no agent type registered for slug {slug!r}; "
            f"known: {sorted(_AGENT_TYPES)}"
        ) from None


def resolve_active_ref(slug: str, *, registry_endpoint: str) -> AgentRef:
    """Resolve ``<slug>:active`` to its pinned :class:`AgentRef` (with digest).

    :raises NoActiveGenerationError: when ``slug`` has no active generation.
    """
    ref = RegistryClient(registry_endpoint).active(slug)
    if ref is None:
        raise NoActiveGenerationError(
            f"no active generation for {slug!r} in registry {registry_endpoint!r}"
        )
    return ref


def active_image(
    slug: str, *, registry_endpoint: str, registry: str = DEFAULT_REGISTRY
) -> str:
    """The immutable ``<registry>/<slug>@<digest>`` of ``slug``'s active generation."""
    return resolve_active_ref(slug, registry_endpoint=registry_endpoint).pull_ref(registry)


def create_agent(
    agent_cls: type[A],
    agent_id: str,
    *,
    registry_endpoint: str,
    registry: str = DEFAULT_REGISTRY,
    **kwargs: Any,
) -> A:
    """Construct ``agent_cls`` pinned to its active-generation image.

    Resolves the active digest and sets ``sandbox_config.image`` to the immutable
    ``<registry>/<slug>@<digest>``, so ``start_sandbox()`` launches the live
    generation rather than the package ``__version__`` default. Extra ``kwargs``
    pass through to the agent constructor.
    """
    image = active_image(
        agent_cls.slug_for(), registry_endpoint=registry_endpoint, registry=registry
    )
    agent = agent_cls(agent_id, **kwargs)
    # Pin only the image, after construction: the agent's __init__ builds its own
    # security envelope (network/GPU/policy/labels) into sandbox_config — passing a
    # fresh SandboxConfig here would discard it. The agent isn't started yet.
    agent.sandbox_config.image = image
    return agent


def create_agent_by_slug(
    slug: str,
    agent_id: str,
    *,
    registry_endpoint: str,
    registry: str = DEFAULT_REGISTRY,
    **kwargs: Any,
) -> BaseAgent:
    """Like :func:`create_agent` but selects the class from the slug registry."""
    return create_agent(
        agent_type_for(slug),
        agent_id,
        registry_endpoint=registry_endpoint,
        registry=registry,
        **kwargs,
    )
