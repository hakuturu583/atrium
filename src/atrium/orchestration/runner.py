"""Execute a single workboard node over A2A (review-gated, cooperative-cancel aware).

:func:`run_node` is the leaf the Prefect adapter wraps in a ``@task``: it
dispatches the node to its agent, and — unless disabled — routes the deliverable
through an independent **reviewer** before the node may count as done (the review
gate, see :mod:`atrium.orchestration.review`). The reviewer's verdict, not the
doer's self-report, becomes the node's outcome; a rejected node can be sent back
to the doer with feedback and re-reviewed (the rework loop). If Prefect cancels
the surrounding task, a best-effort A2A cancel is forwarded to the agent before
unwinding (the board→agent half of cancellation; the agent-side half lives in
:mod:`atrium.orchestration.cancel`).

The outward A2A seam is injectable (``send``), so the whole thing — doer, review,
rework — is unit-testable in-process with a scripted send and no network.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Awaitable, Callable, Optional

from atrium.core import telemetry as tel
from atrium.orchestration.cancel import request_remote_cancel
from atrium.orchestration.protocol import build_node_request, extract_board_update
from atrium.orchestration.review import ReviewPolicy, build_review_request
from atrium.orchestration.types import BoardUpdate, NodeOutcome, NodeResult, WorkNode
from atrium.protocol import Message, get_message_text
from atrium.protocol.a2a_transport import SendTarget, send_message

logger = logging.getLogger("atrium.orchestration.runner")

__all__ = ["run_node", "SendFn"]

#: The outward A2A seam: ``(target, message) -> reply``. Defaults to the real
#: transport; tests and alternative transports substitute their own.
SendFn = Callable[[SendTarget, Message], Awaitable[Message]]


async def run_node(
    node: WorkNode,
    *,
    send: SendFn = send_message,
    workboard_id: str = "",
    context_id: Optional[str] = None,
    review: Optional[ReviewPolicy] = None,
) -> NodeResult:
    """Dispatch ``node`` to its agent, gate on review, return the :class:`NodeResult`.

    Without a ``review`` policy (or for a non-:attr:`~WorkNode.reviewable` node)
    this is the original behavior: the doer's reply *is* the outcome. With the
    gate on, a doer success is only provisional — an independent reviewer must
    approve it, else (within the rework budget) the doer is retried with the
    reviewer's feedback. A domain failure is data (a not-``ok`` ``NodeResult``),
    never an exception; a *transport* failure propagates. On cancellation, a
    best-effort A2A cancel is sent to the doer, then ``CancelledError`` re-raises.
    """
    gated = review is not None and review.applies_to(node)
    attempts = review.max_attempts if gated else 1

    with tel.start_span(
        "workboard.run_node",
        kind=tel.AGENT,
        attributes={"workboard.id": workboard_id, "workboard.node": node.id},
    ) as span:
        try:
            feedback: Optional[str] = None
            last_review_text = ""
            for attempt in range(1, attempts + 1):
                doer = _dispatch_node(
                    node, feedback, workboard_id=workboard_id, context_id=context_id
                )
                reply = await send(node.agent, doer)
                update = extract_board_update(reply)

                # Doer self-failure, or no gate: the doer's reply is the outcome
                # (unchanged behavior — its proposed subtasks/cancels pass through).
                if not update.outcome.ok or not gated:
                    return _leaf_result(node.id, update, get_message_text(reply))

                # Gate: an independent reviewer judges the deliverable.
                verdict = await _review(
                    node,
                    reply,
                    update,
                    send=send,
                    review=review,  # type: ignore[arg-type]  (gated ⇒ not None)
                    workboard_id=workboard_id,
                    context_id=context_id,
                    attempt=attempt,
                )
                last_review_text = verdict.reply_text
                span.set_attribute("workboard.review.verdict", verdict.status)
                span.set_attribute("workboard.review.attempts", attempt)

                if verdict.ok:  # APPROVED — the node is genuinely done.
                    return _approved_result(node.id, update, verdict, attempt, reply)

                # REJECTED — retry with feedback if the rework budget allows.
                feedback = verdict.reason
                logger.info(
                    "node %s failed review (attempt %d/%d): %s",
                    node.id, attempt, attempts, verdict.reason,
                )

            # Rework budget exhausted and still rejected: the node fails.
            return _rejected_result(node.id, review, attempts, feedback, last_review_text)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            # Board→agent cancel: tell the doer to stop the work we started, then
            # let the cancellation unwind. Best-effort — never masks the cancel.
            logger.info("node %s cancelled; forwarding A2A cancel", node.id)
            await request_remote_cancel(node.agent, node.id)
            raise


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #
def _dispatch_node(
    node: WorkNode,
    feedback: Optional[str],
    *,
    workboard_id: str,
    context_id: Optional[str],
) -> Message:
    """Build the doer request, threading review feedback in on a rework attempt."""
    if feedback:
        node = replace(node, payload={**node.payload, "review_feedback": feedback})
    return build_node_request(node, workboard_id=workboard_id, context_id=context_id)


class _Verdict:
    """A reviewer's parsed verdict (internal to the runner)."""

    __slots__ = ("status", "reason", "reply_text")

    def __init__(self, status: str, reason: Optional[str], reply_text: str) -> None:
        self.status = status
        self.reason = reason
        self.reply_text = reply_text

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def _review(
    node: WorkNode,
    doer_reply: Message,
    doer_update: BoardUpdate,
    *,
    send: SendFn,
    review: ReviewPolicy,
    workboard_id: str,
    context_id: Optional[str],
    attempt: int,
) -> _Verdict:
    """Dispatch the deliverable to the reviewer and parse its verdict."""
    target = review.reviewer_for(node)
    request = build_review_request(
        node,
        get_message_text(doer_reply),
        doer_update.outcome.result,
        review_kind=review.review_kind,
        workboard_id=workboard_id,
        context_id=context_id,
        attempt=attempt,
    )
    reply = await send(target, request)
    outcome = extract_board_update(reply).outcome
    return _Verdict(outcome.status, outcome.reason, get_message_text(reply))


def _leaf_result(node_id: str, update: BoardUpdate, reply_text: str) -> NodeResult:
    """A node whose outcome is the agent's own reply (ungated / doer failure)."""
    return NodeResult(
        node_id=node_id,
        outcome=update.outcome,
        add_subtasks=update.add_subtasks,
        cancel=update.cancel,
        reply_text=reply_text,
    )


def _approved_result(
    node_id: str, doer: BoardUpdate, verdict: _Verdict, attempts: int, doer_reply: Message
) -> NodeResult:
    """A reviewed-and-approved node: doer proposals graft; verdict recorded."""
    outcome = NodeOutcome(
        status="ok",
        reason=doer.outcome.reason,
        result={**doer.outcome.result, "review": _review_info(verdict, attempts)},
    )
    return NodeResult(
        node_id=node_id,
        outcome=outcome,
        add_subtasks=doer.add_subtasks,
        cancel=doer.cancel,
        reply_text=get_message_text(doer_reply),
    )


def _rejected_result(
    node_id: str,
    review: ReviewPolicy,
    attempts: int,
    feedback: Optional[str],
    review_text: str,
) -> NodeResult:
    """A node that never passed review: fails, and grafts none of the doer's proposals."""
    verdict = _Verdict("error", feedback, review_text)
    outcome = NodeOutcome(
        status="error",
        reason=f"review rejected after {attempts} attempt(s): {feedback or 'no reason given'}",
        result={"review": _review_info(verdict, attempts)},
    )
    return NodeResult(node_id=node_id, outcome=outcome, reply_text=review_text)


def _review_info(verdict: _Verdict, attempts: int) -> dict[str, object]:
    return {
        "verdict": "approve" if verdict.ok else "request-changes",
        "status": verdict.status,
        "reason": verdict.reason,
        "attempts": attempts,
    }
