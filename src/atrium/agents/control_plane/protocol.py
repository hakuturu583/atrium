"""Shared A2A contract between an interface agent and the control plane.

An :class:`~atrium.agents.control_plane.agent.ControlPlaneAgent` is the trusted
seam that turns a human's chat turn (forwarded by an *interface agent* — see
``docs/design/interface-agent.md``) into a workboard run. The interface holds no
authority: its only egress is a ``workboard.submit`` message to the control
plane, which is the sole caller of :func:`atrium.orchestration.kick.submit_job`.

This module is the **contract both sides share**, defined in the trusted core so
the (evolvable, `atrium_agents`-tier) interface imports it rather than re-deriving
it. Everything here is pure wire glue — no orchestration, no I/O — so it is
serializable and unit-testable without a Prefect backend.

Two directions:

* :func:`build_submit_request` / :func:`parse_submit_request` — the interface's
  ``submit_work`` request into the control plane.
* :func:`build_submitted_reply` — the control plane's ack (the scheduled
  flow-run id), and :func:`build_job_update` — an async progress/terminal push
  back toward the originating thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from atrium.protocol import (
    Message,
    Role,
    data_part,
    get_message_data,
    get_message_text,
    text_message,
)

__all__ = [
    "KIND_SUBMIT",
    "KIND_UPDATE",
    "SUBMIT_TYPE",
    "SUBMITTED_TYPE",
    "JOB_UPDATE_TYPE",
    "SubmitRequest",
    "build_submit_request",
    "parse_submit_request",
    "build_submitted_reply",
    "build_job_update",
]

#: A2A ``metadata.kind`` routing hints for the two directions.
KIND_SUBMIT = "workboard.submit"
KIND_UPDATE = "workboard.update"

#: ``type`` tags on the structured data parts.
SUBMIT_TYPE = "workboard_submit"
SUBMITTED_TYPE = "workboard_submitted"
JOB_UPDATE_TYPE = "job_update"


@dataclass(slots=True)
class SubmitRequest:
    """A human turn, normalized by the interface, asking the board to do work.

    ``agent`` is the doer target (a slug or A2A URL). ``instruction`` is the
    natural-language ask. ``context_id`` ties the whole chat thread together
    (``f"{source}:{channel}:{thread}"``). ``payload`` carries ride-along reply
    coords and any submit-time ``steering``. ``review`` is an optional
    :class:`~atrium.orchestration.review.ReviewPolicy` mapping. ``feedback_for``,
    when set, marks this as a human reply relayed toward a waiting review rather
    than a new job (the mid-flight rework path).
    """

    agent: str
    instruction: str = ""
    context_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    review: Optional[dict[str, Any]] = None
    feedback_for: Optional[str] = None


def _envelope(
    text: str,
    *,
    role: "Role",
    kind: str,
    data: dict[str, Any],
    context_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Message:
    """A text-summary message carrying one structured ``data`` part.

    The shared shape of every message on this contract: a human-readable summary
    plus a typed data part, tagged with a ``metadata.kind`` (and optional
    ``status``). Centralizes the ``text_message(..., extra_parts=[data_part(...)])``
    scaffolding the three builders would otherwise repeat.
    """
    metadata = {"kind": kind} if status is None else {"kind": kind, "status": status}
    return text_message(
        text, role=role, context_id=context_id, metadata=metadata, extra_parts=[data_part(data)]
    )


def build_submit_request(
    agent: str,
    instruction: str = "",
    *,
    context_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    review: Optional[dict[str, Any]] = None,
    feedback_for: Optional[str] = None,
) -> Message:
    """Build the interface → control-plane ``workboard.submit`` message.

    The instruction rides as text (so a human-readable trace shows the ask) and
    the full structured request as a :data:`SUBMIT_TYPE` data part.
    """
    body: dict[str, Any] = {
        "type": SUBMIT_TYPE,
        "agent": agent,
        "instruction": instruction,
        "context_id": context_id,
        "payload": dict(payload or {}),
    }
    if review is not None:
        body["review"] = review
    if feedback_for is not None:
        body["feedback_for"] = feedback_for
    return _envelope(instruction, role=Role.ROLE_USER, kind=KIND_SUBMIT, data=body, context_id=context_id)


def parse_submit_request(message: Message) -> SubmitRequest:
    """Read a ``workboard.submit`` message back into a :class:`SubmitRequest`.

    Prefers the :data:`SUBMIT_TYPE` data part; falls back to the message text for
    the instruction so a bare text ask still resolves. Raises :class:`ValueError`
    when no doer ``agent`` is present (an unroutable request).
    """
    part: dict[str, Any] = {}
    for candidate in get_message_data(message):
        if candidate.get("type") == SUBMIT_TYPE:
            part = candidate
            break
        part = part or candidate  # tolerate a bare data part without the tag

    agent = str(part.get("agent") or "").strip()
    if not agent:
        raise ValueError("workboard.submit carried no doer agent")
    instruction = str(part.get("instruction") or get_message_text(message) or "")
    context_id = part.get("context_id") or (message.context_id or None)
    return SubmitRequest(
        agent=agent,
        instruction=instruction,
        context_id=str(context_id) if context_id else None,
        payload=dict(part.get("payload") or {}),
        review=part.get("review"),
        feedback_for=(str(part["feedback_for"]) if part.get("feedback_for") else None),
    )


def build_submitted_reply(
    job_id: str, *, request: Optional[Message] = None, status: str = "ok"
) -> Message:
    """Build the control-plane ack: the scheduled flow-run id (or an error)."""
    return _envelope(
        f"submitted job {job_id}" if status == "ok" else f"submit failed: {job_id}",
        role=Role.ROLE_AGENT,
        kind=KIND_SUBMIT,
        status=status,
        data={"type": SUBMITTED_TYPE, "status": status, "job_id": job_id},
        context_id=request.context_id if request else None,
    )


def build_job_update(
    job_id: str,
    *,
    status: str,
    coords: Optional[dict[str, Any]] = None,
    result: Optional[dict[str, Any]] = None,
) -> Message:
    """Build an async progress/terminal push toward the originating thread.

    ``coords`` are the ride-along reply coordinates (e.g. ``{channel, thread_ts}``)
    the interface uses to :meth:`deliver` into the right thread without a local
    session lookup.
    """
    return _envelope(
        f"job {job_id} {status}",
        role=Role.ROLE_AGENT,
        kind=KIND_UPDATE,
        status=status,
        data={
            "type": JOB_UPDATE_TYPE,
            "job_id": job_id,
            "status": status,
            "coords": dict(coords or {}),
            "result": dict(result or {}),
        },
    )
