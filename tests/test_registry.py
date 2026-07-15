"""Tests for the version ledger primitives in :mod:`atrium.core.registry`.

The registry *is* the generation ledger, so these cover the pure ledger logic
without a real registry: ``next_version`` (semver bump + guard), the
``AgentRef`` typed view, ``image_ref_from_tag``, and ``RegistryClient``'s
tag/digest/active reads plus the Morpher-only ``set_active`` re-tag. HTTP is
faked by stubbing the client's single ``_open`` plumbing method, so no socket is
ever opened.
"""

from __future__ import annotations

import pytest

from atrium.core.registry import (
    ACTIVE_TAG,
    AgentRef,
    RegistryClient,
    image_ref_from_tag,
    next_version,
)
from atrium.core.types import VersionTag


# --------------------------------------------------------------------------- #
# next_version — the "Decide" step                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "current, level, expected",
    [
        ("1.2.3", "patch", "1.2.4"),
        ("1.2.3", "minor", "1.3.0"),
        ("1.2.3", "major", "2.0.0"),
        ("0.0.0", "patch", "0.0.1"),
    ],
)
def test_next_version_bumps(current, level, expected):
    assert str(next_version(current, level)) == expected


def test_next_version_defaults_to_patch():
    assert str(next_version("1.0.0")) == "1.0.1"


def test_next_version_accepts_versiontag():
    assert str(next_version(VersionTag.parse("2.4.9"), "minor")) == "2.5.0"


def test_next_version_rejects_unknown_level():
    with pytest.raises(ValueError, match="level must be one of"):
        next_version("1.0.0", "epoch")


# --------------------------------------------------------------------------- #
# image_ref_from_tag / AgentRef.pull_ref                                       #
# --------------------------------------------------------------------------- #
def test_image_ref_from_tag_strips_version_keeps_registry_prefix():
    ref = image_ref_from_tag("local-registry/tabbyllmagent:0.2.0", "sha256:abc")
    assert ref == "local-registry/tabbyllmagent@sha256:abc"


def test_image_ref_from_tag_handles_registry_host_port():
    ref = image_ref_from_tag("127.0.0.1:5000/builder_agent:1.0.0", "sha256:def")
    assert ref == "127.0.0.1:5000/builder_agent@sha256:def"


def test_agentref_pull_ref_prefers_digest():
    ref = AgentRef(slug="foo", digest="sha256:aaa", version="0.1.0")
    assert ref.pull_ref("local-registry") == "local-registry/foo@sha256:aaa"


def test_agentref_pull_ref_falls_back_to_version():
    ref = AgentRef(slug="foo", version="0.1.0")
    assert ref.pull_ref("local-registry") == "local-registry/foo:0.1.0"


def test_agentref_pull_ref_requires_digest_or_version():
    with pytest.raises(ValueError, match="neither digest nor version"):
        AgentRef(slug="foo").pull_ref("local-registry")


# --------------------------------------------------------------------------- #
# RegistryClient — faked HTTP via the single _open seam                        #
# --------------------------------------------------------------------------- #
class FakeRegistry:
    """A tiny in-memory stand-in for a registry, driving ``RegistryClient._open``.

    ``manifests`` maps ``(slug, reference)`` -> ``digest`` for HEAD/GET, and
    ``tags`` maps ``slug`` -> list of tag names for the tags/list endpoint.
    """

    def __init__(self, tags=None, manifests=None):
        self.tags = tags or {}
        self.manifests = manifests or {}
        self.put_calls: list[tuple[str, bytes]] = []

    def open(self, method, path, *, data=None, headers=None):
        # GET /v2/<slug>/tags/list
        if path.endswith("/tags/list"):
            slug = path[len("/v2/") : -len("/tags/list")]
            import json

            body = json.dumps({"name": slug, "tags": self.tags.get(slug)}).encode()
            return 200, {}, body
        # HEAD/GET/PUT /v2/<slug>/manifests/<ref>
        if "/manifests/" in path:
            rest = path[len("/v2/") :]
            slug, _, reference = rest.partition("/manifests/")
            if method == "PUT":
                self.put_calls.append((reference, data))
                return 201, {"Docker-Content-Digest": "sha256:put"}, b""
            digest = self.manifests.get((slug, reference))
            if digest is None:
                return 404, {}, None
            hdrs = {
                "Docker-Content-Digest": digest,
                "Content-Type": "application/vnd.oci.image.manifest.v1+json",
            }
            body = b'{"manifest": true}' if method == "GET" else None
            return 200, hdrs, body
        raise AssertionError(f"unexpected registry call: {method} {path}")


@pytest.fixture
def client_and_registry(monkeypatch):
    def _make(tags=None, manifests=None):
        fake = FakeRegistry(tags=tags, manifests=manifests)
        client = RegistryClient("127.0.0.1:5000")
        monkeypatch.setattr(client, "_open", fake.open)
        return client, fake

    return _make


def test_versions_sorted_semantically_excludes_active(client_and_registry):
    client, _ = client_and_registry(
        tags={"foo": ["0.10.0", "0.2.0", "0.1.0", ACTIVE_TAG]}
    )
    assert client.versions("foo") == ["0.1.0", "0.2.0", "0.10.0"]


def test_versions_drops_non_semver_by_default(client_and_registry):
    client, _ = client_and_registry(tags={"foo": ["0.1.0", "latest", "nightly"]})
    assert client.versions("foo") == ["0.1.0"]


def test_versions_empty_when_no_tags(client_and_registry):
    client, _ = client_and_registry(tags={"foo": None})
    assert client.versions("foo") == []


def test_digest_resolves_existing_reference(client_and_registry):
    client, _ = client_and_registry(manifests={("foo", "0.1.0"): "sha256:abc"})
    assert client.digest("foo", "0.1.0") == "sha256:abc"


def test_digest_none_when_absent(client_and_registry):
    client, _ = client_and_registry()
    assert client.digest("foo", "9.9.9") is None


def test_exists_reflects_presence(client_and_registry):
    client, _ = client_and_registry(manifests={("foo", "0.1.0"): "sha256:abc"})
    assert client.exists("foo", "0.1.0") is True
    assert client.exists("foo", "0.2.0") is False


def test_active_returns_ref_when_present(client_and_registry):
    client, _ = client_and_registry(manifests={("foo", ACTIVE_TAG): "sha256:live"})
    ref = client.active("foo")
    assert ref == AgentRef(slug="foo", digest="sha256:live", version=ACTIVE_TAG)


def test_active_none_when_no_pointer(client_and_registry):
    client, _ = client_and_registry()
    assert client.active("foo") is None


def test_set_active_retags_existing_digest(client_and_registry):
    client, fake = client_and_registry(
        manifests={("foo", "sha256:live"): "sha256:live"}
    )
    client.set_active("foo", "sha256:live")
    # It re-PUTs the fetched manifest under the mutable :active tag.
    assert fake.put_calls and fake.put_calls[0][0] == ACTIVE_TAG


def test_set_active_refuses_unknown_digest(client_and_registry):
    from atrium.core.registry import LocalRegistryError

    client, _ = client_and_registry()
    with pytest.raises(LocalRegistryError, match="not in registry"):
        client.set_active("foo", "sha256:ghost")
