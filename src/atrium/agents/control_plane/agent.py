"""``ControlPlaneAgent`` — the trusted seam that turns a chat turn into a run.

An *interface agent* (Slack/Discord/…, in the evolvable ``atrium_agents`` tier)
holds no authority: its only egress is a ``workboard.submit`` A2A message. The
``ControlPlaneAgent`` is the fixed-infrastructure counterpart that receives it and
is the **sole caller of** :func:`atrium.orchestration.kick.submit_job` — so the
"agent proposes, trusted worker disposes" boundary the workboard already uses
extends cleanly to the human I/O edge (see ``docs/design/interface-agent.md``).

It deliberately does one thing: validate a submit and kick a workboard run,
returning the scheduled flow-run id. Steering (``payload["steering"]``) rides
through to the doer untouched; an optional ``review`` policy gates completion. The
mid-flight feedback-relay path (a human reply routed to a waiting reviewer) is
declared in the shared contract but not yet wired here — it is rejected explicitly
rather than mis-handled as a new job.

The ``submit_job`` seam is module-level so tests exercise the whole path with a
scripted kick and no Prefect backend.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from atrium.agents.control_plane.protocol import (
    SubmitRequest,
    build_job_update,
    build_submitted_reply,
    parse_submit_request,
)
from atrium.core.base_agent import BaseAgent
from atrium.core.types import SandboxConfig, VersionTag, wan_sandbox_config
from atrium.orchestration.kick import submit_job, workboard_state
from atrium.orchestration.review import ReviewPolicy
from atrium.protocol import Message
from atrium.protocol.a2a_transport import SendTarget

logger = logging.getLogger("atrium.agents.control_plane")

__all__ = ["ControlPlaneAgent"]

#: First version minted for the control plane with no ledger history.
DEFAULT_INITIAL_VERSION = "0.1.0"

#: Doer used when a submit names no explicit agent (D5 — the interface forwards a
#: turn and the control plane routes; a coding turn defaults to a code workspace).
DEFAULT_DOER = "python_code_workspace_agent:active"

#: ``job_update`` status that asks the interface to present a deliverable for human
#: review (動線2b, async ticket) rather than reporting a terminal outcome.
REVIEW_STATUS = "review"

#: Human verdict tokens that count as approval; anything else is request-changes.
_APPROVE_TOKENS = ("approve", "lgtm", "ok", "👍", ":+1:", "looks good", "ship it")


class ControlPlaneAgent(BaseAgent):
    """Receive ``workboard.submit`` and kick a workboard run over Prefect."""

    AGENT_SLUG = "control_plane"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        deployment: str = "default",
        default_agent: str = DEFAULT_DOER,
        interface: Optional[SendTarget] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        super().__init__(
            agent_id, version or DEFAULT_INITIAL_VERSION, sandbox_config or wan_sandbox_config()
        )
        #: Which ``atrium-workboard/<deployment>`` a kick targets.
        self.deployment = deployment
        #: The doer a turn routes to when it names no explicit agent override.
        self.default_agent = default_agent
        #: The interface to push a :func:`job_update <build_job_update>` to when a
        #: job finishes (D6). ``None`` disables the push (poll-only / no delivery).
        self.interface = interface
        #: job_id → job record (coords, context_id, instruction, doer, human_review)
        #: for the completion push. In-memory and reconstructable (the same coords
        #: ride in the flow-run's parameters).
        self._pending: dict[str, dict[str, Any]] = {}
        #: review token → open ticket, for the async human-review loop (動線2b).
        self._tickets: dict[str, dict[str, Any]] = {}

    async def handle_task(self, message: Message) -> Message:
        """Route an inbound submit: a human reply resolves a review, else kick a run."""
        req = parse_submit_request(message)
        if req.feedback_for is not None:
            # 動線2(b): a human reply resolving an open review ticket (async).
            return await self._resolve_ticket(req, message)
        target = self._route(req)
        if not target:
            return build_submitted_reply("no doer agent and no default configured", request=message, status="error")
        job_id = await self._kick(req, target)
        self._track(job_id, req, target)
        logger.info("kicked job %s (doer %s) for context %s", job_id, target, req.context_id)
        return build_submitted_reply(job_id, request=message)

    def _track(self, job_id: str, req: SubmitRequest, doer: str) -> None:
        """Record what a later completion needs: where to deliver, and how to rework."""
        self._pending[job_id] = {
            "coords": dict(req.payload.get("reply_coords") or {}),
            "context_id": req.context_id,
            "instruction": req.instruction,
            "doer": doer,
            "human_review": self._wants_human_review(req),
        }

    @staticmethod
    def _wants_human_review(req: SubmitRequest) -> bool:
        """Whether this job's deliverable should be gated on a human (``review.human``)."""
        return bool(req.review) and bool(req.review.get("human"))

    def _route(self, req: SubmitRequest) -> str:
        """Pick the doer for ``req``: its explicit override, else the default (D5).

        Routing is the control plane's job, not the interface's: a turn may carry
        an explicit ``@agent`` override (``req.agent``), otherwise it routes to
        :attr:`default_agent`. A richer intent/capability router slots in here
        behind the same seam without touching the interface.
        """
        return req.agent or self.default_agent

    async def _kick(self, req: SubmitRequest, target: str) -> str:
        """Kick a single-node workboard for ``req`` and return its flow-run id."""
        return await submit_job(
            target,
            req.instruction,
            payload=req.payload,
            context_id=req.context_id,
            review=ReviewPolicy.from_dict(req.review),
            deployment=self.deployment,
        )

    # ------------------------------------------------------------------ #
    # D6 — progress notifications (push, with poll as the trigger)        #
    # ------------------------------------------------------------------ #
    async def poll_once(self) -> list[str]:
        """Push a ``job_update`` for every tracked job that has finished.

        The poll *trigger* for the push-with-fallback-poll design (D6): a Prefect
        terminal-state hook would call :meth:`notify` directly; where no hook
        fires this sweep catches the completion instead. Returns the job ids
        notified this sweep (and stops tracking them).
        """
        notified: list[str] = []
        for job_id, rec in list(self._pending.items()):
            state = await workboard_state(job_id)
            if not state.get("done"):
                continue
            ok = state.get("state") == "COMPLETED"
            if ok and rec.get("human_review"):
                await self._open_ticket(job_id, rec)
            else:
                await self.notify(job_id, "ok" if ok else "error", coords=rec.get("coords"))
            notified.append(job_id)
        return notified

    async def notify(
        self,
        job_id: str,
        status: str,
        *,
        coords: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Push a terminal ``job_update`` to the interface and stop tracking ``job_id``.

        ``coords`` default to the ride-along coords recorded at submit; a Prefect
        hook may pass fresher ones. A no-op (beyond untracking) when no interface
        is configured.
        """
        coords = coords if coords is not None else self._coords_of(job_id)
        self._pending.pop(job_id, None)
        await self._push(build_job_update(job_id, status=status, coords=coords or {}, result=result))

    def _coords_of(self, job_id: str) -> dict[str, Any]:
        return dict(self._pending.get(job_id, {}).get("coords") or {})

    async def _push(self, message: Message) -> None:
        if self.interface is None:
            logger.info("no interface configured; job_update not pushed")
            return
        await self.send_a2a_message(self.interface, message)

    # ------------------------------------------------------------------ #
    # 動線2(b) — async human review (post-hoc ticket loop; no gate held)   #
    # ------------------------------------------------------------------ #
    async def _open_ticket(self, job_id: str, rec: dict[str, Any]) -> None:
        """Present a finished deliverable to the human and park a review ticket.

        Async by construction: the run already completed, so nothing is blocked.
        The interface posts the deliverable (status ``review``), marks its session
        ``pending_review``, and the human's later reply arrives as a ``feedback_for``
        submit that :meth:`_resolve_ticket` closes.
        """
        token = rec.get("context_id") or job_id
        self._pending.pop(job_id, None)
        self._tickets[token] = {**rec, "job_id": job_id}
        await self._push(
            build_job_update(
                job_id,
                status=REVIEW_STATUS,
                coords=rec.get("coords") or {},
                result={"token": token, "instruction": rec.get("instruction", "")},
            )
        )

    async def _resolve_ticket(self, req: SubmitRequest, message: Message) -> Message:
        """Close an open review with the human's verdict: approve → done, else rework."""
        token = req.feedback_for
        ticket = self._tickets.pop(token, None)
        if ticket is None:
            return build_submitted_reply(f"no open review {token}", request=message, status="error")
        if self._is_approval(req.instruction):
            await self.notify(str(ticket["job_id"]), "ok", coords=ticket.get("coords"))
            return build_submitted_reply(str(ticket["job_id"]), request=message)
        # Request-changes: kick a rework carrying the human feedback as steering.
        rework = SubmitRequest(
            agent=str(ticket.get("doer") or ""),
            instruction=str(ticket.get("instruction") or ""),
            context_id=token,
            payload={
                "reply_coords": dict(ticket.get("coords") or {}),
                "steering": {"review_feedback": req.instruction},
            },
            review={"human": True},
        )
        target = self._route(rework)
        job_id = await self._kick(rework, target)
        self._track(job_id, rework, target)
        logger.info("review %s requested changes; kicked rework %s", token, job_id)
        return build_submitted_reply(job_id, request=message)

    @staticmethod
    def _is_approval(text: str) -> bool:
        low = text.strip().lower()
        return any(tok in low for tok in _APPROVE_TOKENS)
