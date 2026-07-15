"""Tests for the agent factory — the "Reflect" step of the version lifecycle.

The factory resolves ``<slug>:active`` -> pinned digest and pins the agent's
sandbox image to it, so ``start_sandbox()`` launches the live generation instead
of the ``__version__`` default. The registry is stubbed (the factory's
``RegistryClient`` is monkeypatched) so no registry is needed.
"""

from __future__ import annotations

import pytest

from atrium.core import factory
from atrium.core.base_agent import BaseAgent
from atrium.core.factory import (
    NoActiveGenerationError,
    active_image,
    agent_type_for,
    create_agent,
    create_agent_by_slug,
    register_agent_type,
    resolve_active_ref,
)
from atrium.core.registry import AgentRef


class DummyAgent(BaseAgent):
    """A concrete, do-nothing agent so the factory has something to construct."""

    AGENT_SLUG = "dummyagent"

    def __init__(self, agent_id, version="0.0.0", **kwargs):
        super().__init__(agent_id, version)
        self.kwargs = kwargs

    async def handle_task(self, message):  # pragma: no cover - never invoked here
        return message


class StubClient:
    """Stands in for ``RegistryClient``; returns a fixed active ref (or none)."""

    def __init__(self, endpoint, active_ref=None):
        self._active_ref = active_ref

    def active(self, slug):
        return self._active_ref


@pytest.fixture
def with_active(monkeypatch):
    """Patch the factory's RegistryClient so ``active(slug)`` returns ``ref``."""

    def _install(ref):
        monkeypatch.setattr(
            factory, "RegistryClient", lambda endpoint: StubClient(endpoint, ref)
        )

    return _install


def test_resolve_active_ref_returns_pinned_ref(with_active):
    with_active(AgentRef(slug="dummyagent", digest="sha256:abc", version="active"))
    ref = resolve_active_ref("dummyagent", registry_endpoint="127.0.0.1:5000")
    assert ref.digest == "sha256:abc"


def test_resolve_active_ref_raises_without_active(with_active):
    with_active(None)
    with pytest.raises(NoActiveGenerationError, match="no active generation"):
        resolve_active_ref("dummyagent", registry_endpoint="127.0.0.1:5000")


def test_active_image_is_content_addressed(with_active):
    with_active(AgentRef(slug="dummyagent", digest="sha256:abc", version="active"))
    image = active_image(
        "dummyagent", registry_endpoint="127.0.0.1:5000", registry="local-registry"
    )
    assert image == "local-registry/dummyagent@sha256:abc"


def test_create_agent_pins_active_image(with_active):
    with_active(AgentRef(slug="dummyagent", digest="sha256:abc", version="active"))
    agent = create_agent(
        DummyAgent, "dummy-1", registry_endpoint="127.0.0.1:5000"
    )
    assert isinstance(agent, DummyAgent)
    assert agent.sandbox_config.image == "local-registry/dummyagent@sha256:abc"
    assert not agent.is_running


def test_create_agent_forwards_kwargs(with_active):
    with_active(AgentRef(slug="dummyagent", digest="sha256:abc", version="active"))
    agent = create_agent(
        DummyAgent, "dummy-1", registry_endpoint="127.0.0.1:5000", foo="bar"
    )
    assert agent.kwargs == {"foo": "bar"}


def test_register_and_lookup_agent_type():
    register_agent_type(DummyAgent)
    assert agent_type_for("dummyagent") is DummyAgent


def test_agent_type_for_unknown_slug_raises():
    with pytest.raises(KeyError, match="no agent type registered"):
        agent_type_for("does-not-exist")


def test_create_agent_by_slug_uses_registry(with_active):
    register_agent_type(DummyAgent)
    with_active(AgentRef(slug="dummyagent", digest="sha256:abc", version="active"))
    agent = create_agent_by_slug(
        "dummyagent", "dummy-1", registry_endpoint="127.0.0.1:5000"
    )
    assert isinstance(agent, DummyAgent)
    assert agent.sandbox_config.image == "local-registry/dummyagent@sha256:abc"
