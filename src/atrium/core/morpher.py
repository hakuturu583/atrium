"""Morpher — the only authority that moves an agent's ``:active`` pointer.

Promotion (and rollback) is the crown-jewel write of the version lifecycle:
whoever moves ``<slug>:active`` decides what code runs next. The Morpher gates it
on a **signed validation attestation** — a validator's (e.g. a TestAgent's)
pass/fail verdict over the *exact* image digest, signed with the trust-root
Ed25519 key. A compromised actor that can push a version tag still cannot promote
it: it cannot forge an attestation for its digest, so the Morpher refuses. Every
promote/rollback is appended — with its attestation — to a re-verifiable audit
log.

This is fixed infrastructure: the Morpher and its trust root are human-managed,
never evolved. See ``docs/design/agent-versioning.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from atrium.core.errors import AtriumError
from atrium.core.registry import ACTIVE_TAG, AgentRef, RegistryClient


def _signing_message(slug: str, digest: str, ok: bool) -> bytes:
    """Canonical bytes signed/verified for a verdict (binds slug + digest + verdict)."""
    return f"{slug}\n{digest}\n{int(ok)}".encode()

__all__ = [
    "Attestation",
    "AttestationSigner",
    "Morpher",
    "MorpherError",
    "generate_trust_root",
]


class MorpherError(AtriumError):
    """Raised when a promotion is refused (absent/invalid attestation, missing digest)."""


@dataclass(frozen=True, slots=True)
class Attestation:
    """A signed validation verdict over one image generation.

    ``ok`` is the validator's pass/fail; ``signature`` is Ed25519 (hex) over the
    canonical ``slug\\ndigest\\nok`` message, made with the trust-root key. The
    digest is the immutable identity — an attestation is bound to *that* image,
    so it cannot be replayed onto a different one.
    """

    slug: str
    digest: str
    ok: bool
    signature: str  # hex-encoded Ed25519 signature

    def signing_message(self) -> bytes:
        """The exact bytes that are signed/verified (binds slug + digest + verdict)."""
        return _signing_message(self.slug, self.digest, self.ok)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Attestation":
        return cls(
            slug=data["slug"],
            digest=data["digest"],
            ok=bool(data["ok"]),
            signature=data["signature"],
        )


def generate_trust_root() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a fresh Ed25519 trust-root keypair (private signs, public verifies)."""
    private = Ed25519PrivateKey.generate()
    return private, private.public_key()


class AttestationSigner:
    """Signs validation verdicts with the trust-root **private** key.

    Held by the validator (e.g. a TestAgent) — the side that decides a generation
    passed. Producing an :class:`Attestation` requires this key, which is exactly
    why a compromised agent without it cannot fabricate one.
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key

    def sign(self, slug: str, digest: str, ok: bool = True) -> Attestation:
        """Sign a verdict that ``slug``'s image ``digest`` did (``ok``) or did not pass."""
        signature = self._key.sign(_signing_message(slug, digest, ok))
        return Attestation(slug=slug, digest=digest, ok=ok, signature=signature.hex())


class Morpher:
    """Gates and performs ``promote``/``rollback`` of ``<slug>:active``.

    Holds the trust-root **public** key (it only verifies; it never signs
    verdicts) and a :class:`RegistryClient`. A move requires a passing
    :class:`Attestation` whose signature verifies against the trust root *and*
    whose digest is really present in the registry; then it moves ``:active`` and
    appends the decision to the audit log.
    """

    def __init__(
        self,
        client: RegistryClient,
        trust_root: Ed25519PublicKey,
        *,
        audit_log: Optional[str] = None,
    ) -> None:
        self._client = client
        self._trust_root = trust_root
        self._audit_log = audit_log

    def verify(self, attestation: Attestation) -> bool:
        """Whether ``attestation`` is a *passing* verdict signed by the trust root."""
        if not attestation.ok:
            return False
        try:
            self._trust_root.verify(
                bytes.fromhex(attestation.signature), attestation.signing_message()
            )
            return True
        except (InvalidSignature, ValueError):
            return False

    def promote(self, slug: str, attestation: Attestation) -> AgentRef:
        """Make ``attestation.digest`` the live generation of ``slug`` (gated)."""
        return self._move(slug, attestation, action="promote")

    def rollback(self, slug: str, attestation: Attestation) -> AgentRef:
        """Re-point ``:active`` at a previously-attested earlier digest (gated)."""
        return self._move(slug, attestation, action="rollback")

    def _move(self, slug: str, attestation: Attestation, *, action: str) -> AgentRef:
        if attestation.slug != slug:
            raise MorpherError(
                f"refusing {action}: attestation slug {attestation.slug!r} != {slug!r}"
            )
        if not self.verify(attestation):
            raise MorpherError(
                f"refusing {action}: attestation is not a passing, trust-root-signed verdict"
            )
        if self._client.digest(slug, attestation.digest) is None:
            raise MorpherError(
                f"refusing {action}: attested digest not in registry "
                f"({slug}@{attestation.digest})"
            )
        self._client.set_active(slug, attestation.digest)
        self._record(action, attestation)
        return AgentRef(slug=slug, digest=attestation.digest, version=ACTIVE_TAG)

    def _record(self, action: str, attestation: Attestation) -> None:
        """Append a re-verifiable ``{action, ts, attestation}`` line to the audit log."""
        if not self._audit_log:
            return
        entry = {"action": action, "ts": time.time(), "attestation": attestation.to_dict()}
        with open(self._audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
