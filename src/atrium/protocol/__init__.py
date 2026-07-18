"""A2A protocol layer for Atrium.

This package wraps the official A2A SDK (``a2a-sdk``, import name ``a2a``) so the
rest of Atrium speaks one clean idiom while the SDK's protobuf-generated types
(where a ``Part`` directly carries ``text``/``data`` and ``metadata`` is a
``google.protobuf.Struct``) stay encapsulated here.

All inter-agent communication in Atrium is A2A — there is deliberately no second
wire protocol at the agent boundary.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping, Optional

from a2a.types import Artifact, Message, Part, Role, Task
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct, Value

from atrium.protocol.a2a_transport import (
    AtriumAgentExecutor,
    build_request_handler,
    send_message,
)

__all__ = [
    # Re-exported A2A SDK types (Atrium namespace)
    "Message",
    "Part",
    "Role",
    "Artifact",
    "Task",
    # Construction / extraction helpers
    "text_part",
    "data_part",
    "text_message",
    "data_message",
    "get_message_text",
    "get_message_data",
    "merge_data_parts",
    "metadata_dict",
    "set_metadata",
    # Transport
    "send_message",
    "AtriumAgentExecutor",
    "build_request_handler",
]


def _to_value(data: Mapping[str, Any]) -> Value:
    """Wrap a JSON-like mapping into a protobuf ``Value`` (struct value)."""
    value = Value()
    value.struct_value.update(dict(data))
    return value


def text_part(text: str) -> Part:
    """Build a text :class:`~a2a.types.Part`."""
    return Part(text=text)


def data_part(data: Mapping[str, Any], *, media_type: str = "application/json") -> Part:
    """Build a structured-data :class:`~a2a.types.Part` from a JSON-like mapping.

    Used to carry machine-readable payloads (e.g. tool-call definitions and
    ``tool_calls`` results) over A2A instead of stuffing them into free text.
    """
    return Part(data=_to_value(data), media_type=media_type)


def text_message(
    text: str,
    *,
    role: "Role" = Role.ROLE_AGENT,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    extra_parts: Optional[Iterable[Part]] = None,
) -> Message:
    """Build an A2A :class:`~a2a.types.Message` carrying ``text``.

    ``context_id``/``task_id`` are propagated so multi-turn exchanges (e.g. a
    tool-call followed by its result) stay correlated.
    """
    parts = [text_part(text)]
    if extra_parts:
        parts.extend(extra_parts)
    return _build_message(parts, role, context_id, task_id, metadata)


def data_message(
    data: Mapping[str, Any],
    *,
    role: "Role" = Role.ROLE_AGENT,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Message:
    """Build an A2A message whose sole part is structured data."""
    return _build_message([data_part(data)], role, context_id, task_id, metadata)


def _build_message(
    parts: list[Part],
    role: "Role",
    context_id: Optional[str],
    task_id: Optional[str],
    metadata: Optional[Mapping[str, Any]],
) -> Message:
    # A2A (v0.3) requires every message to carry a message_id; a compliant server
    # rejects one without it (INVALID_PARAMS). Generate one so any atrium-built
    # message is wire-valid without every caller having to remember.
    message = Message(role=role, parts=parts, message_id=uuid.uuid4().hex)
    if context_id:
        message.context_id = context_id
    if task_id:
        message.task_id = task_id
    if metadata:
        set_metadata(message, metadata)
    return message


def get_message_text(message: Message) -> str:
    """Concatenate all text parts of a message."""
    return "".join(part.text for part in message.parts if part.text)


def get_message_data(message: Message) -> list[dict[str, Any]]:
    """Return every structured-data part of a message as plain dicts."""
    out: list[dict[str, Any]] = []
    for part in message.parts:
        if part.HasField("data"):
            decoded = MessageToDict(part.data)
            if isinstance(decoded, dict):
                out.append(decoded)
    return out


def merge_data_parts(message: Message) -> dict[str, Any]:
    """Merge every structured DataPart of ``message`` into one mapping.

    The common shape for a structured A2A request/reply: one logical payload spread
    across one or more DataParts. Later parts win on key clashes.
    """
    merged: dict[str, Any] = {}
    for part in get_message_data(message):
        if isinstance(part, dict):
            merged.update(part)
    return merged


def metadata_dict(message: Message) -> dict[str, Any]:
    """Return the message metadata (a protobuf ``Struct``) as a plain dict."""
    return dict(message.metadata)


def set_metadata(message: Message, values: Mapping[str, Any]) -> None:
    """Set scalar metadata keys on a message's protobuf ``Struct``."""
    for key, value in values.items():
        message.metadata[key] = value
