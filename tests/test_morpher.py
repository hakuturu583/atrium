"""Tests for the Morpher — the attestation-gated ``:active`` mover.

These use a *real* Ed25519 trust root (via ``generate_trust_root`` /
``AttestationSigner``) so the signature path is genuinely exercised, and a tiny
in-memory registry stub so no HTTP is involved. The security-relevant refusals
(wrong slug, forged/unsigned verdict, failing verdict, digest not in registry)
are each asserted, plus the happy-path promote/rollback and audit-log append.
"""

from __future__ import annotations

import json

import pytest

from atrium.core.morpher import (
    Attestation,
    AttestationSigner,
    Morpher,
    MorpherError,
    generate_trust_root,
)
from atrium.core.registry import ACTIVE_TAG, AgentRef


class StubRegistry:
    """Minimal ``RegistryClient`` stand-in: known digests + a recorded active move."""

    def __init__(self, known):
        self._known = set(known)  # digests present in the registry
        self.active_moves: list[tuple[str, str]] = []

    def digest(self, slug, reference):
        return reference if reference in self._known else None

    def set_active(self, slug, digest):
        self.active_moves.append((slug, digest))


@pytest.fixture
def trust():
    private, public = generate_trust_root()
    return AttestationSigner(private), public


def test_promote_moves_active_on_valid_attestation(trust):
    signer, public = trust
    reg = StubRegistry(known={"sha256:good"})
    morpher = Morpher(reg, public)

    att = signer.sign("myagent", "sha256:good", ok=True)
    ref = morpher.promote("myagent", att)

    assert ref == AgentRef(slug="myagent", digest="sha256:good", version=ACTIVE_TAG)
    assert reg.active_moves == [("myagent", "sha256:good")]


def test_rollback_uses_same_gate(trust):
    signer, public = trust
    reg = StubRegistry(known={"sha256:prev"})
    morpher = Morpher(reg, public)

    att = signer.sign("myagent", "sha256:prev", ok=True)
    ref = morpher.rollback("myagent", att)

    assert ref.digest == "sha256:prev"
    assert reg.active_moves == [("myagent", "sha256:prev")]


def test_refuses_slug_mismatch(trust):
    signer, public = trust
    reg = StubRegistry(known={"sha256:good"})
    morpher = Morpher(reg, public)

    att = signer.sign("other", "sha256:good", ok=True)
    with pytest.raises(MorpherError, match="attestation slug"):
        morpher.promote("myagent", att)
    assert reg.active_moves == []


def test_refuses_failing_verdict(trust):
    signer, public = trust
    reg = StubRegistry(known={"sha256:good"})
    morpher = Morpher(reg, public)

    att = signer.sign("myagent", "sha256:good", ok=False)
    with pytest.raises(MorpherError, match="passing"):
        morpher.promote("myagent", att)
    assert reg.active_moves == []


def test_refuses_forged_signature(trust):
    _signer, public = trust
    reg = StubRegistry(known={"sha256:good"})
    morpher = Morpher(reg, public)

    # A verdict never signed by the trust root (attacker fabricates the hex).
    forged = Attestation(
        slug="myagent", digest="sha256:good", ok=True, signature="00" * 64
    )
    with pytest.raises(MorpherError, match="passing"):
        morpher.promote("myagent", forged)
    assert reg.active_moves == []


def test_refuses_attestation_from_a_different_trust_root(trust):
    _signer, public = trust
    # A different keypair (a compromised validator without the real key).
    other_private, _other_public = generate_trust_root()
    other_signer = AttestationSigner(other_private)
    reg = StubRegistry(known={"sha256:good"})
    morpher = Morpher(reg, public)

    att = other_signer.sign("myagent", "sha256:good", ok=True)
    with pytest.raises(MorpherError, match="passing"):
        morpher.promote("myagent", att)
    assert reg.active_moves == []


def test_refuses_digest_not_in_registry(trust):
    signer, public = trust
    reg = StubRegistry(known=set())  # digest is validly attested but absent
    morpher = Morpher(reg, public)

    att = signer.sign("myagent", "sha256:missing", ok=True)
    with pytest.raises(MorpherError, match="not in registry"):
        morpher.promote("myagent", att)
    assert reg.active_moves == []


def test_verify_true_only_for_passing_signed_verdict(trust):
    signer, public = trust
    morpher = Morpher(StubRegistry(known=set()), public)
    assert morpher.verify(signer.sign("a", "sha256:x", ok=True)) is True
    assert morpher.verify(signer.sign("a", "sha256:x", ok=False)) is False


def test_attestation_dict_roundtrip(trust):
    signer, _ = trust
    att = signer.sign("a", "sha256:x", ok=True)
    assert Attestation.from_dict(att.to_dict()) == att


def test_audit_log_records_promotion(trust, tmp_path):
    signer, public = trust
    reg = StubRegistry(known={"sha256:good"})
    log = tmp_path / "promotions.log"
    morpher = Morpher(reg, public, audit_log=str(log))

    att = signer.sign("myagent", "sha256:good", ok=True)
    morpher.promote("myagent", att)

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["action"] == "promote"
    assert entry["attestation"]["digest"] == "sha256:good"
    assert "ts" in entry
