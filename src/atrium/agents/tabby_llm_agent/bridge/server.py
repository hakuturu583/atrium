"""A2A↔tabbyAPI bridge — the only place OpenAI-HTTP exists in Atrium.

Runs inside the TabbyLLMAgent sandbox, co-located with tabbyAPI. It exposes an
A2A server to the rest of Atrium and translates each request into a loopback
OpenAI-compatible call to tabbyAPI (exllamav3 backend):

* inference            → ``POST /v1/chat/completions``
* ``load`` / ``unload`` → ``POST /v1/model/{load,unload}`` (admin)
* ``ready?``           → ``GET  /v1/model``

Function/tool calling rides through transparently: ``tools``/``tool_choice`` are
forwarded to tabby, and any ``tool_calls`` the model returns come back as a
structured A2A data part so the caller can execute tools and continue the
conversation (``role:"tool"`` turns) over A2A.

This module imports the container-only HTTP/ASGI stack and must not be imported
from the Atrium host package.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from atrium.agents.tabby_llm_agent.agent import (
    KIND_LOAD,
    KIND_READY,
    KIND_UNLOAD,
    STATUS_NOT_READY,
    STATUS_OK,
    STATUS_TOOL_CALLS,
)
from atrium.agents.tabby_llm_agent.card import build_agent_card
from atrium.core import telemetry as tel
from atrium.protocol import (
    Message,
    data_message,
    get_message_data,
    metadata_dict,
    text_message,
)

logger = logging.getLogger("atrium.agents.tabby.bridge")

STATUS_ERROR = "error"


@dataclass(slots=True)
class TabbyConfig:
    """Container-side configuration for reaching the co-located tabbyAPI."""

    base_url: str = "http://127.0.0.1:5000"
    api_key: Optional[str] = None
    admin_key: Optional[str] = None
    model_name: Optional[str] = None
    model_dir: Optional[str] = None
    timeout: float = 300.0
    bridge_port: int = 8730
    version: str = "0.0.0"
    # 'native' = use tabby's tool calling; 'json_schema' = formatron-constrained
    # fallback for models whose chat template lacks tool support.
    tool_mode: str = "native"

    @classmethod
    def from_env(cls) -> "TabbyConfig":
        return cls(
            base_url=os.environ.get("TABBY_BASE_URL", "http://127.0.0.1:5000"),
            api_key=os.environ.get("TABBY_API_KEY"),
            admin_key=os.environ.get("TABBY_ADMIN_KEY"),
            model_name=os.environ.get("TABBY_MODEL_NAME"),
            model_dir=os.environ.get("TABBY_MODEL_DIR"),
            bridge_port=int(os.environ.get("BRIDGE_PORT", "8730")),
            version=os.environ.get("AGENT_VERSION", "0.0.0"),
            tool_mode=os.environ.get("TABBY_TOOL_MODE", "native"),
        )


class TabbyA2AExecutor:
    """A2A ``AgentExecutor`` translating A2A requests to tabbyAPI calls."""

    def __init__(self, config: TabbyConfig) -> None:
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url, timeout=self.config.timeout
            )
        return self._client

    def _headers(self, *, admin: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        if admin and self.config.admin_key:
            headers["x-admin-key"] = self.config.admin_key
        # Stitch the cross-container trace: carry the W3C parent to tabby.
        tel.inject_traceparent(headers)
        return headers

    # ---- A2A AgentExecutor protocol -------------------------------------- #
    async def execute(self, context: Any, event_queue: Any) -> None:
        incoming: Message = context.message
        parent = tel.extract_context(dict(incoming.metadata)) if incoming.metadata else None
        kind = metadata_dict(incoming).get("kind", "")
        with tel.start_span("tabby_bridge.execute", kind=tel.CHAIN, context=parent):
            reply = await self._route(kind, incoming)
            await event_queue.enqueue_event(reply)

    async def cancel(self, context: Any, event_queue: Any) -> None:
        raise NotImplementedError("tabby bridge does not support cancellation")

    # ---- routing --------------------------------------------------------- #
    async def _route(self, kind: str, message: Message) -> Message:
        payload = _first_data(message)
        try:
            if kind == KIND_READY or payload.get("type") == KIND_READY:
                return await self._ready()
            if kind == KIND_LOAD or payload.get("type") == KIND_LOAD:
                return await self._load(payload)
            if kind == KIND_UNLOAD or payload.get("type") == KIND_UNLOAD:
                return await self._unload()
            return await self._chat(payload, context_id=message.context_id, task_id=message.task_id)
        except httpx.HTTPError as exc:
            logger.warning("tabby call failed: %s", exc)
            return _status_reply(STATUS_ERROR, {"error": str(exc)})

    # ---- handlers -------------------------------------------------------- #
    async def _ready(self) -> Message:
        try:
            resp = await self._http().get("/v1/model", headers=self._headers())
        except httpx.HTTPError:
            return _status_reply(STATUS_NOT_READY, {"reason": "tabby unreachable"})
        if resp.status_code == 200 and (resp.json() or {}).get("id"):
            return _status_reply(STATUS_OK, {"model": resp.json().get("id")})
        return _status_reply(STATUS_NOT_READY, {"reason": "no model loaded"})

    async def _load(self, payload: dict[str, Any]) -> Message:
        body = {"model_name": payload.get("model_name") or self.config.model_name}
        body.update({k: v for k, v in payload.items() if k not in ("type", "model_name")})
        resp = await self._http().post(
            "/v1/model/load", json=body, headers=self._headers(admin=True)
        )
        if resp.status_code // 100 == 2:
            return _status_reply(STATUS_OK, {"loading": body["model_name"]})
        return _status_reply(STATUS_ERROR, {"error": resp.text, "code": resp.status_code})

    async def _unload(self) -> Message:
        resp = await self._http().post("/v1/model/unload", headers=self._headers(admin=True))
        ok = resp.status_code // 100 == 2
        return _status_reply(STATUS_OK if ok else STATUS_ERROR, {"unloaded": ok})

    async def _chat(
        self, payload: dict[str, Any], *, context_id: str = "", task_id: str = ""
    ) -> Message:
        request = _build_chat_request(payload, self.config)
        attributes = {
            tel.INPUT_VALUE: _safe_json(request.get("messages")),
            tel.LLM_INVOCATION_PARAMETERS: _safe_json(
                {k: request[k] for k in ("max_tokens", "temperature", "tool_choice") if k in request}
            ),
        }
        if request.get("model"):
            attributes[tel.LLM_MODEL_NAME] = request["model"]

        with tel.start_span("tabby.chat_completions", kind=tel.LLM, attributes=attributes) as span:
            resp = await self._http().post(
                "/v1/chat/completions", json=request, headers=self._headers()
            )
            if resp.status_code == 503 or _looks_unloaded(resp):
                return _status_reply(STATUS_NOT_READY, {"reason": "model loading/quantizing"})
            if resp.status_code // 100 != 2:
                return _status_reply(STATUS_ERROR, {"error": resp.text, "code": resp.status_code})

            data = resp.json()
            _record_usage(span, data.get("usage"))
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message", {}) or {}
            tool_calls = msg.get("tool_calls")
            if tool_calls or choice.get("finish_reason") == "tool_calls":
                span.set_attribute(tel.OUTPUT_VALUE, _safe_json(tool_calls))
                return data_message(
                    {"type": STATUS_TOOL_CALLS, "tool_calls": tool_calls or []},
                    context_id=context_id or None,
                    task_id=task_id or None,
                    metadata={"status": STATUS_TOOL_CALLS},
                )
            content = msg.get("content") or ""
            span.set_attribute(tel.OUTPUT_VALUE, content)
            return text_message(
                content,
                context_id=context_id or None,
                task_id=task_id or None,
                metadata={"status": STATUS_OK},
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _first_data(message: Message) -> dict[str, Any]:
    parts = get_message_data(message)
    return parts[0] if parts else {}


def _status_reply(status: str, data: dict[str, Any]) -> Message:
    return data_message({"status": status, **data}, metadata={"status": status})


def _build_chat_request(payload: dict[str, Any], config: TabbyConfig) -> dict[str, Any]:
    request: dict[str, Any] = {
        "messages": payload.get("messages", []),
        "max_tokens": payload.get("max_tokens", 512),
        "temperature": payload.get("temperature", 0.7),
        "stream": False,
    }
    model = payload.get("model") or config.model_name
    if model:
        request["model"] = model
    if payload.get("tools"):
        request["tools"] = payload["tools"]
        request["tool_choice"] = payload.get("tool_choice", "auto")
        if config.tool_mode == "json_schema":
            # Fallback for models whose chat template lacks tool support:
            # constrain output to the tool schema instead of native tool calling.
            request["response_format"] = {"type": "json_object"}
    for extra in ("top_p", "stop", "seed", "frequency_penalty", "presence_penalty"):
        if extra in payload:
            request[extra] = payload[extra]
    return request


def _looks_unloaded(resp: httpx.Response) -> bool:
    if resp.status_code // 100 != 2:
        text = resp.text.lower()
        return "model" in text and ("not" in text and "load" in text)
    return False


def _record_usage(span: Any, usage: Optional[dict[str, Any]]) -> None:
    if not usage:
        return
    if "prompt_tokens" in usage:
        span.set_attribute(tel.LLM_TOKEN_COUNT_PROMPT, usage["prompt_tokens"])
    if "completion_tokens" in usage:
        span.set_attribute(tel.LLM_TOKEN_COUNT_COMPLETION, usage["completion_tokens"])
    if "total_tokens" in usage:
        span.set_attribute(tel.LLM_TOKEN_COUNT_TOTAL, usage["total_tokens"])


def _safe_json(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# ASGI app + entrypoint (container-side only)                                  #
# --------------------------------------------------------------------------- #
def build_bridge_app(config: Optional[TabbyConfig] = None):
    """Build the Starlette ASGI app serving this bridge's A2A endpoint."""
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette

    config = config or TabbyConfig.from_env()
    card = build_agent_card(config.version)
    handler = DefaultRequestHandler(
        agent_executor=TabbyA2AExecutor(config),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_jsonrpc_routes(handler, "/")
    return Starlette(routes=routes)


def main() -> None:
    import uvicorn

    config = TabbyConfig.from_env()
    logging.basicConfig(level=logging.INFO)
    # Install the OTLP exporter before serving so the in-sandbox bridge's spans
    # (and the trace context carried in from the host) actually reach Phoenix.
    tel.configure_tracing(f"tabby-llm-agent@{config.version}")
    logger.info("starting tabby A2A bridge on :%d -> %s", config.bridge_port, config.base_url)
    try:
        uvicorn.run(build_bridge_app(config), host="0.0.0.0", port=config.bridge_port)
    finally:
        tel.shutdown_tracing()


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    main()
