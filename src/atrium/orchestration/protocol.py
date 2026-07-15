"""A2A wire glue for the workboard: node → request, reply → board update.

This is the boundary between the orchestrator (host, trusted) and an agent (in
its sandbox). Two directions:

* :func:`build_node_request` — turn a :class:`WorkNode` into the A2A request the
  agent receives. The node id rides as ``task_id`` so a later cancel can name it.
* :func:`extract_board_update` — read an agent's reply into a :class:`BoardUpdate`.
  Agents that speak the workboard protocol reply with a ``workboard_update`` data
  part (built via :func:`board_update_message`); agents that don't (e.g. a plain
  :class:`~atrium.agents.task_agent.TaskAgent` replying ``task_result``) are still
  understood as a leaf node via their ``status`` — so any Atrium agent can be a
  node with no changes.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from atrium.orchestration.types import (
    STATUS_ERROR,
    STATUS_OK,
    BoardUpdate,
    NodeOutcome,
    WorkNode,
)
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
    "WORKBOARD_UPDATE_TYPE",
    "build_node_request",
    "extract_board_update",
    "board_update_message",
]

#: ``type`` tag on the structured part carrying an agent's board proposal.
WORKBOARD_UPDATE_TYPE = "workboard_update"


def build_node_request(
    node: WorkNode, *, workboard_id: str = "", context_id: Optional[str] = None
) -> Message:
    """Build the A2A request message that dispatches ``node`` to its agent.

    The node id is stamped as ``task_id`` (so a board→agent cancel can target
    this exact unit of work) and echoed in metadata for trace/debugging. The
    node ``payload`` rides as a structured data part; the ``instruction`` as text.
    """
    extra_parts = [data_part(node.payload)] if node.payload else None
    return text_message(
        node.instruction,
        role=Role.ROLE_USER,
        context_id=context_id,
        task_id=node.id,
        metadata={
            "kind": node.kind,
            "workboard.id": workboard_id,
            "workboard.node": node.id,
        },
        extra_parts=extra_parts,
    )


def extract_board_update(message: Message) -> BoardUpdate:
    """Interpret an agent's reply as a :class:`BoardUpdate`.

    Prefers an explicit ``workboard_update`` part; otherwise falls back to the
    reply's ``status`` (metadata or any data part) so a non-workboard-aware agent
    still resolves to an ``ok``/``error`` leaf outcome carrying its result data.
    """
    parts = get_message_data(message)
    for part in parts:
        if part.get("type") == WORKBOARD_UPDATE_TYPE:
            return BoardUpdate.from_dict(part)

    # Fallback: treat the reply as a leaf outcome.
    status: Optional[str] = metadata_dict(message).get("status")
    reason: Optional[str] = None
    result: dict[str, Any] = {}
    for part in parts:
        result.update(part)
        status = status or part.get("status")
        reason = reason or part.get("reason")
    return BoardUpdate(
        outcome=NodeOutcome(status=str(status or STATUS_OK), result=result, reason=reason)
    )


def board_update_message(
    outcome: NodeOutcome,
    *,
    add_subtasks: Iterable[WorkNode] = (),
    cancel: Iterable[str] = (),
    text: str = "",
    request: Optional[Message] = None,
) -> Message:
    """Build an agent's reply that *proposes* a board update (the write path).

    Agents call this to return their outcome plus any subtasks to graft or nodes
    to cancel. It only proposes — the orchestrator decides whether to apply it —
    so an agent needs no Prefect credentials and no board write access. Correlation
    ids from ``request`` are echoed when provided.
    """
    update = BoardUpdate(outcome=outcome, add_subtasks=list(add_subtasks), cancel=list(cancel))
    payload = {"type": WORKBOARD_UPDATE_TYPE, **update.to_dict()}
    body = text or (outcome.reason if not outcome.ok else "") or "workboard node handled"
    return text_message(
        body,
        role=Role.ROLE_AGENT,
        context_id=(request.context_id or None) if request is not None else None,
        task_id=(request.task_id or None) if request is not None else None,
        metadata={"kind": "workboard", "status": outcome.status},
        extra_parts=[data_part(payload)],
    )
