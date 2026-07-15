"""Execute a single workboard node over A2A (cooperative-cancel aware).

:func:`run_node` is the leaf the Prefect adapter wraps in a ``@task``: it
dispatches the node to its agent, reads the reply into a :class:`NodeResult`, and
— if Prefect cancels the surrounding task — forwards a best-effort A2A cancel to
the agent before unwinding (the board→agent half of cancellation; the agent-side
half lives in :mod:`atrium.orchestration.cancel`).

The outward A2A seam is injectable (``send``), so the whole thing is unit-testable
in-process with a scripted send and no network.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from atrium.core import telemetry as tel
from atrium.orchestration.cancel import request_remote_cancel
from atrium.orchestration.protocol import build_node_request, extract_board_update
from atrium.orchestration.types import NodeResult, WorkNode
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
) -> NodeResult:
    """Dispatch ``node`` to its agent and return the parsed :class:`NodeResult`.

    A domain failure (the agent replies ``error``) is data — it comes back as a
    ``NodeResult`` whose outcome is not ``ok`` — not an exception; the scheduler
    uses it to fail dependents. A *transport* failure propagates (the node's
    Prefect task fails). On cancellation, a best-effort A2A cancel is sent to the
    agent, then :class:`asyncio.CancelledError` re-raises.
    """
    request = build_node_request(node, workboard_id=workboard_id, context_id=context_id)
    with tel.start_span(
        "workboard.run_node",
        kind=tel.AGENT,
        attributes={"workboard.id": workboard_id, "workboard.node": node.id},
    ):
        try:
            reply = await send(node.agent, request)
        except asyncio.CancelledError:
            # Board→agent cancel: tell the agent to stop the work we started, then
            # let the cancellation unwind. Best-effort — never masks the cancel.
            logger.info("node %s cancelled; forwarding A2A cancel", node.id)
            await request_remote_cancel(node.agent, node.id)
            raise
        update = extract_board_update(reply)
        return NodeResult(
            node_id=node.id,
            outcome=update.outcome,
            add_subtasks=update.add_subtasks,
            cancel=update.cancel,
            reply_text=get_message_text(reply),
        )
