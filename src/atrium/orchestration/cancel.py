"""Cooperative cancellation for workboard nodes — the board→agent control path.

Completion and failure need no special machinery: a node's Prefect task simply
returns or the agent replies ``error``. **Cancellation** is the one signal that
flows the other way — from the control plane *into* a running, isolated agent —
so it needs a protocol:

* Host side (orchestrator): when Prefect cancels a node's task, the runner sends
  an A2A cancel naming the node's ``task_id`` — :func:`request_remote_cancel`.
* Agent side (in the sandbox bridge): :class:`CancellableAgentExecutor` maps that
  cancel onto a :class:`CancelToken` the handler cooperatively polls via
  :func:`raise_if_cancelled`. Long-running handlers must check it; single-shot
  ones simply never see a cancel land mid-flight and finish normally.

Cancellation is *cooperative* by design: an agent is never force-killed by the
board (that authority stays with the sandbox lifecycle), it is asked to stop.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Optional

from atrium.core.errors import AgentError
from atrium.protocol.a2a_transport import AtriumAgentExecutor, SendTarget, cancel_task

logger = logging.getLogger("atrium.orchestration.cancel")

__all__ = [
    "CancelToken",
    "TaskCancelledError",
    "current_cancel_token",
    "raise_if_cancelled",
    "CancellableAgentExecutor",
    "request_remote_cancel",
]


class TaskCancelledError(AgentError):
    """Raised inside a handler when its node has been cancelled by the board."""


class CancelToken:
    """A one-shot, awaitable cancellation flag shared with a running handler."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`TaskCancelledError` if a cancel has landed (poll points)."""
        if self._event.is_set():
            raise TaskCancelledError("workboard node was cancelled")

    async def wait(self) -> None:
        """Await until cancelled (to race a token against in-flight work)."""
        await self._event.wait()


#: Set by :class:`CancellableAgentExecutor` around the handler so cooperative
#: code deep in the call stack can reach the active token without threading it.
_CANCEL_TOKEN: ContextVar[Optional[CancelToken]] = ContextVar(
    "atrium_cancel_token", default=None
)


def current_cancel_token() -> Optional[CancelToken]:
    """The :class:`CancelToken` for the node being handled, or ``None``."""
    return _CANCEL_TOKEN.get()


def raise_if_cancelled() -> None:
    """Convenience poll point: raise if the current node has been cancelled."""
    token = _CANCEL_TOKEN.get()
    if token is not None:
        token.raise_if_cancelled()


class CancellableAgentExecutor(AtriumAgentExecutor):
    """An :class:`AtriumAgentExecutor` that honours A2A ``cancel`` cooperatively.

    Keeps a per-``task_id`` :class:`CancelToken`, binds it into the context var
    around the handler, and on an inbound cancel sets the matching token. The
    handler observes it by calling :func:`raise_if_cancelled` (or awaiting
    ``token.wait()``); if it never checks, the cancel is simply a no-op for that
    single-shot call. Unlike the base class, ``cancel`` does not raise.
    """

    def __init__(self, handler: Any, *, name: str = "atrium-agent") -> None:
        super().__init__(handler, name=name)
        self._tokens: dict[str, CancelToken] = {}

    @staticmethod
    def _task_key(context: Any) -> str:
        task_id = getattr(context, "task_id", None)
        if not task_id:
            message = getattr(context, "message", None)
            task_id = getattr(message, "task_id", None)
        return str(task_id or "")

    async def execute(self, context: Any, event_queue: Any) -> None:
        key = self._task_key(context)
        token = CancelToken()
        if key:
            self._tokens[key] = token
        reset = _CANCEL_TOKEN.set(token)
        try:
            await super().execute(context, event_queue)
        finally:
            _CANCEL_TOKEN.reset(reset)
            self._tokens.pop(key, None)

    async def cancel(self, context: Any, event_queue: Any) -> None:
        key = self._task_key(context)
        token = self._tokens.get(key)
        if token is not None:
            token.cancel()
            logger.info("cancel signalled for task %s", key)
        else:
            logger.debug("cancel for unknown/finished task %s (no-op)", key)


async def request_remote_cancel(target: SendTarget, task_id: str) -> bool:
    """Ask ``target`` to stop the work for node ``task_id`` (the board→agent send).

    A thin orchestration seam over the transport's :func:`cancel_task`: this names
    the *intent* (cancel a running workboard node), while the A2A client lifecycle
    lives in the transport alongside :func:`send_message`. Best-effort; never raises.
    """
    return await cancel_task(target, task_id)
