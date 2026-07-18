"""``atrium_dispatch`` — the trusted A2A primitive baked into the runner image.

A generated ``flow.py`` never gets a raw A2A client. Its *only* way to make a
subagent do work is this primitive, preinstalled in the runner sandbox: given an
agent (a slug or an A2A URL), an instruction and a payload, it sends one A2A
request and returns the subagent's reply as plain data. The generated
orchestration composes these calls into a DAG; it cannot reach anything the
primitive + the sandbox's WAN-isolation do not already allow.

Endpoint resolution, in order:

* a full URL (``http(s)://…``) is used as-is;
* otherwise the slug (optionally ``slug:generation``) is looked up in the
  ``ATRIUM_DISPATCH_ENDPOINTS`` env var (a JSON ``{slug: url}`` map the deployment
  injects — the allow-list of reachable subagents);
* failing that, the host-local convention ``http://<slug>.local`` is used (the
  same default as :meth:`BaseAgent.a2a_endpoint`), which only resolves on the
  control LAN the runner is confined to.

Both a sync (:func:`atrium_dispatch`) and an async (:func:`atrium_dispatch_async`)
entry point are exposed so the primitive is callable from ordinary Prefect tasks
whether or not they are ``async``.
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import Any, Mapping, Optional

from atrium.protocol import (
    get_message_text,
    merge_data_parts,
    metadata_dict,
    text_message,
)
from atrium.protocol import Role, data_part
from atrium.protocol.a2a_transport import send_message

__all__ = ["atrium_dispatch", "atrium_dispatch_async", "resolve_endpoint"]

#: Env var carrying the ``{slug: url}`` allow-list of dispatchable subagents.
ENDPOINTS_ENV = "ATRIUM_DISPATCH_ENDPOINTS"


@lru_cache(maxsize=1)
def _parse_endpoints(raw: str) -> Mapping[str, str]:
    """Parse the ``{slug: url}`` allow-list JSON (memoized — the env is process-fixed)."""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def resolve_endpoint(agent: str) -> str:
    """Resolve an agent slug/URL to an A2A base URL (see module docstring)."""
    if agent.startswith("http://") or agent.startswith("https://"):
        return agent
    endpoints = _parse_endpoints(os.environ.get(ENDPOINTS_ENV) or "")
    # Try the full slug, then the same without its generation (``coder:active`` → ``coder``).
    for key in (agent, agent.split(":", 1)[0]):
        if key in endpoints:
            return str(endpoints[key])
    return f"http://{agent.split(':', 1)[0]}.local"


async def atrium_dispatch_async(
    agent: str,
    instruction: str,
    payload: Optional[dict[str, Any]] = None,
    *,
    context_id: Optional[str] = None,
) -> dict[str, Any]:
    """Send one A2A request to ``agent`` and return ``{status, text, data}``.

    ``instruction`` rides as the message text (the subagent's user prompt) and
    ``payload`` as a structured data part. The reply is normalized to a plain dict
    so generated flow code never touches the A2A message types.
    """
    parts = [data_part(payload)] if payload else None
    message = text_message(
        instruction, role=Role.ROLE_USER, context_id=context_id, extra_parts=parts
    )
    reply = await send_message(resolve_endpoint(agent), message)
    data = merge_data_parts(reply)
    status = str(metadata_dict(reply).get("status") or data.get("status") or "ok")
    return {"status": status, "text": get_message_text(reply), "data": data}


def atrium_dispatch(
    agent: str,
    instruction: str,
    payload: Optional[dict[str, Any]] = None,
    *,
    context_id: Optional[str] = None,
) -> dict[str, Any]:
    """Synchronous wrapper over :func:`atrium_dispatch_async`.

    Runs the coroutine on a fresh event loop so it is callable from a plain
    (non-async) Prefect task. Inside an already-running loop, callers should await
    :func:`atrium_dispatch_async` directly instead.
    """
    return asyncio.run(atrium_dispatch_async(agent, instruction, payload, context_id=context_id))
