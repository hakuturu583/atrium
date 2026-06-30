"""A2A transport: client send path and server-side execution adapter.

Built on the official A2A SDK (``a2a-sdk`` >= 1.x). This module is host-safe: it
never imports ``httpx`` directly (the SDK manages its own transport) and never
imports the ASGI server stack (``starlette``/``sse_starlette``), which lives in
the container image where the in-sandbox bridge actually serves A2A.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, Union

from a2a.types import AgentCard, Message, SendMessageRequest

from atrium.core import telemetry as tel
from atrium.core.errors import A2ATransportError

logger = logging.getLogger("atrium.protocol.transport")

# A handler turns an inbound A2A message into a reply message.
MessageHandler = Callable[[Message], Awaitable[Message]]
# A send target is either a base URL or a resolved AgentCard.
SendTarget = Union[str, AgentCard]

__all__ = ["send_message", "AtriumAgentExecutor", "build_request_handler"]


async def send_message(target: SendTarget, message: Message) -> Message:
    """Send ``message`` to ``target`` over A2A and return the aggregated reply.

    The current W3C trace context is injected into ``message.metadata`` before
    sending so the remote agent (even in another container) parents its work
    under this span. The SDK's ``send_message`` yields a stream; we drive it to
    completion and return the final reply message.
    """
    tel.inject_traceparent(message.metadata)

    # Imported lazily so importing this module never requires the client extras.
    from a2a.client import create_client

    with tel.start_span("a2a.send_message", kind=tel.CHAIN):
        try:
            client = await create_client(target)
        except Exception as exc:  # noqa: BLE001
            raise A2ATransportError(f"failed to create A2A client for {target!r}") from exc

        try:
            request = SendMessageRequest(message=message)
            reply: Optional[Message] = None
            async for response in client.send_message(request):
                # StreamResponse is a oneof of task/message/status_update/...
                if response.HasField("message"):
                    reply = response.message
                elif response.HasField("task") and reply is None:
                    task = response.task
                    # Fall back to the last agent turn recorded on the task.
                    for turn in reversed(list(task.history)):
                        reply = turn
                        break
            if reply is None:
                raise A2ATransportError("A2A response stream produced no message")
            return reply
        except A2ATransportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise A2ATransportError(f"A2A send to {target!r} failed") from exc
        finally:
            try:
                await client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("A2A client close failed", exc_info=True)


class AtriumAgentExecutor:
    """Server-side adapter: routes inbound A2A requests to a message handler.

    Implements the A2A SDK ``AgentExecutor`` protocol. On each request it
    restores the W3C trace context carried in the inbound message metadata so
    the handler's work stitches into the caller's trace, runs the handler under
    an AGENT span, and enqueues the reply.
    """

    def __init__(self, handler: MessageHandler, *, name: str = "atrium-agent") -> None:
        self._handler = handler
        self._name = name

    async def execute(self, context: Any, event_queue: Any) -> None:
        incoming: Message = context.message
        parent = tel.extract_context(dict(incoming.metadata)) if incoming.metadata else None
        with tel.start_span(f"{self._name}.handle_task", kind=tel.AGENT, context=parent):
            reply = await self._handler(incoming)
            await event_queue.enqueue_event(reply)

    async def cancel(self, context: Any, event_queue: Any) -> None:
        # Atrium tasks are short-lived, single-shot inference/exec calls; there
        # is nothing to cancel mid-flight. Subclasses may override.
        raise NotImplementedError("AtriumAgentExecutor does not support cancellation")


def build_request_handler(
    handler: MessageHandler,
    agent_card: "AgentCard",
    *,
    name: str = "atrium-agent",
    task_store: Optional[Any] = None,
) -> Any:
    """Build an A2A ``DefaultRequestHandler`` wrapping ``handler``.

    The returned handler can be mounted onto a Starlette/FastAPI app via the A2A
    SDK route builders (done container-side in the bridge server, where the ASGI
    extras are installed).
    """
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    return DefaultRequestHandler(
        agent_executor=AtriumAgentExecutor(handler, name=name),
        task_store=task_store or InMemoryTaskStore(),
        agent_card=agent_card,
    )
