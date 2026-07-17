"""Shared A2A contract between the control plane and a planning agent.

The control plane turns a human request into a runnable :class:`~atrium.orchestration.job.Job`
by asking a **plan agent** (an evolvable, LLM-backed specialist in the
``atrium_agents`` tier) to draft the job's Prefect ``flow.py`` + params. This
module is the **contract both sides share**, defined in the trusted core so the
plan agent imports it rather than re-deriving it — the exact same split as the
interface↔control-plane submit contract (:mod:`atrium.agents.control_plane.protocol`).

Everything here is pure wire glue — no orchestration, no I/O — so it is
serializable and unit-testable without a backend. Two directions:

* :func:`build_plan_request` / :func:`parse_plan_request` — the control plane's
  "plan this request" ask to the plan agent.
* :func:`build_plan_result` / :func:`parse_plan_result` — the plan agent's
  *proposed* ``{flow_source, params, requirements}`` (or an error). The plan agent
  only proposes; the control plane validates readiness and disposes.
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
    metadata_dict,
    text_message,
)

__all__ = [
    "KIND_PLAN",
    "PLAN_REQUEST_TYPE",
    "PLAN_RESULT_TYPE",
    "PlanRequest",
    "build_plan_request",
    "parse_plan_request",
    "build_plan_result",
    "parse_plan_result",
]

#: A2A ``metadata.kind`` routing hint for both directions of the plan exchange.
KIND_PLAN = "workboard.plan"

#: ``type`` tags on the structured data parts.
PLAN_REQUEST_TYPE = "plan_request"
PLAN_RESULT_TYPE = "plan_result"


@dataclass(slots=True)
class PlanRequest:
    """A human request handed to the plan agent to draft a job's flow + params.

    ``request`` is the normalized human ask (JSON). ``instruction`` is its
    natural-language summary (also rides as the message text so a human-readable
    trace shows the ask). ``context_id`` ties the chat thread together.
    ``constraints`` carries what the planner must plan *within* — e.g. the roster
    of dispatchable subagents (role name + capability), the allowed libraries, and
    size caps — so the model produces a flow that assigns work only to real,
    reachable agents and never assumes arbitrary network egress.
    """

    request: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    context_id: Optional[str] = None
    constraints: dict[str, Any] = field(default_factory=dict)


def _envelope(
    text: str,
    *,
    role: "Role",
    data: dict[str, Any],
    context_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Message:
    """A text-summary message carrying one structured ``data`` part, tagged ``KIND_PLAN``.

    Centralizes the ``text_message(..., extra_parts=[data_part(...)])`` scaffolding
    both builders share (mirrors ``control_plane.protocol._envelope``).
    """
    metadata = {"kind": KIND_PLAN} if status is None else {"kind": KIND_PLAN, "status": status}
    return text_message(
        text, role=role, context_id=context_id, metadata=metadata, extra_parts=[data_part(data)]
    )


def build_plan_request(
    request: Optional[dict[str, Any]] = None,
    instruction: str = "",
    *,
    context_id: Optional[str] = None,
    constraints: Optional[dict[str, Any]] = None,
) -> Message:
    """Build the control-plane → plan-agent "plan this request" message."""
    body: dict[str, Any] = {
        "type": PLAN_REQUEST_TYPE,
        "request": dict(request or {}),
        "instruction": instruction,
        "context_id": context_id,
        "constraints": dict(constraints or {}),
    }
    return _envelope(instruction, role=Role.ROLE_USER, data=body, context_id=context_id)


def parse_plan_request(message: Message) -> PlanRequest:
    """Read a ``plan_request`` message back into a :class:`PlanRequest`.

    Prefers the :data:`PLAN_REQUEST_TYPE` data part; falls back to the message text
    for the instruction so a bare text ask still resolves.
    """
    part: dict[str, Any] = {}
    for candidate in get_message_data(message):
        if candidate.get("type") == PLAN_REQUEST_TYPE:
            part = candidate
            break
        part = part or candidate  # tolerate a bare data part without the tag

    context_id = part.get("context_id") or (message.context_id or None)
    return PlanRequest(
        request=dict(part.get("request") or {}),
        instruction=str(part.get("instruction") or get_message_text(message) or ""),
        context_id=str(context_id) if context_id else None,
        constraints=dict(part.get("constraints") or {}),
    )


def build_plan_result(
    flow_source: str = "",
    params: Optional[dict[str, Any]] = None,
    *,
    requirements: Optional[list[str]] = None,
    reason: str = "",
    status: str = "ok",
    request: Optional[Message] = None,
) -> Message:
    """Build the plan agent's proposed flow + params (or an error).

    On success the generated ``flow.py`` rides as ``flow_source`` with its
    ``params`` and any declared ``requirements``; on failure ``status="error"``
    carries a ``reason`` and empty artifacts, so the control plane's readiness gate
    rejects it rather than trying to run a half-formed job (fail-closed).
    """
    summary = "planned flow" if status == "ok" else f"plan failed: {reason}"
    return _envelope(
        summary,
        role=Role.ROLE_AGENT,
        status=status,
        data={
            "type": PLAN_RESULT_TYPE,
            "status": status,
            "flow_source": flow_source,
            "params": dict(params or {}),
            "requirements": list(requirements or []),
            "reason": reason,
        },
        context_id=request.context_id if request else None,
    )


def parse_plan_result(message: Message) -> dict[str, Any]:
    """Read a ``plan_result`` into ``{status, flow_source, params, requirements, reason}``.

    Absent the tagged part, falls back to ``{status: <metadata|error>}`` with empty
    artifacts so a malformed reply reads as an error, never as a runnable plan.
    """
    for part in get_message_data(message):
        if part.get("type") == PLAN_RESULT_TYPE:
            return {
                "status": str(part.get("status", "ok")),
                "flow_source": str(part.get("flow_source") or ""),
                "params": dict(part.get("params") or {}),
                "requirements": [str(r) for r in (part.get("requirements") or [])],
                "reason": str(part.get("reason") or ""),
            }
    return {
        "status": str(metadata_dict(message).get("status", "error")),
        "flow_source": "",
        "params": {},
        "requirements": [],
        "reason": "no plan_result part in reply",
    }
