"""The review gate — an independent verdict between a node's work and "done".

By default a workboard node's completion is the *doer agent's own* self-report
(``run_node`` reads its reply ``status``). A code-authoring agent is the highest
prompt-injection-exposure component in the system, so trusting its "I'm done" as
the achievement verdict is weak. This module carries the policy and the wire glue
for inserting a **mandatory, independent reviewer** in front of that verdict:

> the doer produces a deliverable → a *separate* reviewer agent (no shared
> context, no board access) judges it → the trusted ``run_node`` writes that
> verdict as the node's outcome → Prefect records it → the scheduler decides what
> runs next.

This is the same "agent proposes, trusted worker disposes" invariant the rest of
the workboard follows (``docs/design/orchestration-workboard.md``): the reviewer
only *proposes* a verdict in its A2A reply; the trusted runner is what turns it
into board state. Everything here is Prefect-free and unit-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from atrium.orchestration.types import WorkNode
from atrium.protocol import Message, Role, data_part, text_message

__all__ = [
    "REVIEW_REQUEST_TYPE",
    "ReviewPolicy",
    "build_review_request",
    "default_review_policy",
]

#: ``type`` tag on the structured part the runner sends to a reviewer agent (the
#: reviewer reads it; a single source of truth shared with ``ReviewerAgent``).
REVIEW_REQUEST_TYPE = "review_request"


@dataclass(slots=True)
class ReviewPolicy:
    """How the review gate behaves for a workboard run.

    Serialized across the Prefect parameter boundary (``to_dict``/``from_dict``).
    A run with a policy gates *every* node that is :attr:`WorkNode.reviewable`;
    passing no policy keeps the old self-report behavior untouched.
    """

    reviewer: str = ""
    """A2A target of the reviewer agent — a slug or base URL. May be empty when
    every node carries its own :attr:`WorkNode.reviewer` override."""
    enabled: bool = True
    max_attempts: int = 1
    """Doer attempts. ``>1`` enables the rework loop: a rejected node is sent back
    to the doer with the reviewer's feedback and re-reviewed, up to this many times."""
    review_kind: str = "review"
    """A2A ``kind`` stamped on the review request (routing hint for the reviewer)."""

    def reviewer_for(self, node: WorkNode) -> str:
        """The reviewer target for ``node`` — its override, else the policy default."""
        return node.reviewer or self.reviewer

    def applies_to(self, node: WorkNode) -> bool:
        """Whether the gate runs for ``node`` (enabled, reviewable, has a target)."""
        return self.enabled and node.reviewable and bool(self.reviewer_for(node))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "enabled": self.enabled,
            "max_attempts": self.max_attempts,
            "review_kind": self.review_kind,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["ReviewPolicy"]:
        """Build a policy from a mapping; ``None``/empty maps to no policy."""
        if not data:
            return None
        return cls(
            reviewer=str(data.get("reviewer", "")),
            enabled=bool(data.get("enabled", True)),
            max_attempts=int(data.get("max_attempts", 1)),
            review_kind=str(data.get("review_kind", "review")),
        )


def build_review_request(
    node: WorkNode,
    deliverable_text: str,
    deliverable_result: dict[str, Any],
    *,
    review_kind: str = "review",
    workboard_id: str = "",
    context_id: Optional[str] = None,
    attempt: int = 1,
) -> Message:
    """Build the A2A request that asks a reviewer to judge ``node``'s deliverable.

    Carries the task ``instruction`` and the doer's deliverable (text + structured
    result) as a :data:`REVIEW_REQUEST_TYPE` data part. It deliberately does **not**
    carry the doer's conversation/context — the reviewer judges only the artifact,
    which is what makes its verdict independent. The node id rides as ``task_id``
    so a board→agent cancel can still target this unit of work.
    """
    payload = {
        "type": REVIEW_REQUEST_TYPE,
        "instruction": node.instruction,
        "deliverable": deliverable_text,
        "result": dict(deliverable_result or {}),
        "attempt": attempt,
    }
    return text_message(
        node.instruction,
        role=Role.ROLE_USER,
        context_id=context_id,
        task_id=node.id,
        metadata={
            "kind": review_kind,
            "workboard.id": workboard_id,
            "workboard.node": node.id,
            "review.attempt": attempt,
        },
        extra_parts=[data_part(payload)],
    )


def default_review_policy() -> Optional[ReviewPolicy]:
    """A policy from the environment, so the gate can be **always-on** by deployment.

    ``ATRIUM_REVIEWER`` (the reviewer A2A target) turns the gate on for every job;
    ``ATRIUM_REVIEW_MAX_ATTEMPTS`` (optional, default 1) sets the rework budget.
    Returns ``None`` when ``ATRIUM_REVIEWER`` is unset, leaving self-report behavior.
    """
    reviewer = os.environ.get("ATRIUM_REVIEWER", "").strip()
    if not reviewer:
        return None
    try:
        max_attempts = int(os.environ.get("ATRIUM_REVIEW_MAX_ATTEMPTS", "1"))
    except ValueError:
        max_attempts = 1
    return ReviewPolicy(reviewer=reviewer, max_attempts=max(1, max_attempts))
