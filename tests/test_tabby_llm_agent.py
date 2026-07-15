"""Tests for :class:`atrium.agents.tabby_llm_agent.TabbyLLMAgent` and the bridge's
OpenAI request builder.

No sandbox, GPU or network: the agent's single outward seam
(``send_a2a_message``) is replaced with a scripted stand-in for the in-sandbox
bridge, so ``infer`` / ``chat`` / the readiness+model controls / the
``not_ready`` retry / the ``tool_calls`` path are all exercised in-process. The
bridge's pure ``_build_chat_request`` is tested directly (skipped if the
container-only ``httpx`` dep is absent on the host).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from atrium.agents.tabby_llm_agent import (
    KVCacheConfig,
    TabbyLLMAgent,
    plan_cache_size,
)
from atrium.agents.tabby_llm_agent.agent import (
    STATUS_NOT_READY,
    STATUS_OK,
    STATUS_TOOL_CALLS,
)
from atrium.core.errors import ModelNotReadyError, PolicyViolationError
from atrium.protocol import data_message, get_message_data, text_message


def _agent():
    agent = TabbyLLMAgent("coder-1", "0.1.0")
    agent.config.retry_backoff_s = 0  # no real sleeping between retries
    return agent


def _script(agent, replies):
    """Drive ``send_a2a_message`` from a fixed list of replies; record requests."""
    sent: list = []
    counter = {"i": 0}

    async def fake_send(target, message):
        sent.append(message)
        i = min(counter["i"], len(replies) - 1)
        counter["i"] += 1
        return replies[i]

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    return sent


def _ok(text):
    return text_message(text, metadata={"status": STATUS_OK})


def _not_ready():
    return text_message("", metadata={"status": STATUS_NOT_READY})


def _tool_calls(calls):
    return data_message(
        {"type": STATUS_TOOL_CALLS, "tool_calls": calls},
        metadata={"status": STATUS_TOOL_CALLS},
    )


def _last_request(sent):
    return get_message_data(sent[-1])[0]


# --------------------------------------------------------------------------- #
# infer                                                                        #
# --------------------------------------------------------------------------- #
def test_infer_returns_text():
    agent = _agent()
    sent = _script(agent, [_ok("hello world")])
    out = asyncio.run(agent.infer("hi"))
    assert out == "hello world"
    req = _last_request(sent)
    assert req["type"] == "infer"
    # system prompt (coding-agent memory) + user prompt were composed.
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user"]
    assert req["messages"][-1]["content"] == "hi"


def test_infer_explicit_system_overrides_memory():
    agent = _agent()
    sent = _script(agent, [_ok("ok")])
    asyncio.run(agent.infer("hi", system="be terse"))
    req = _last_request(sent)
    assert req["messages"][0] == {"role": "system", "content": "be terse"}


def test_infer_forwards_tools_and_returns_tool_calls_json():
    agent = _agent()
    calls = [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]
    sent = _script(agent, [_tool_calls(calls)])
    tools = [{"type": "function", "function": {"name": "read"}}]
    out = asyncio.run(agent.infer("do", tools=tools))
    # tools ride along in the request...
    assert _last_request(sent)["tools"] == tools
    # ...and a tool-call reply comes back as JSON of the structured parts.
    decoded = json.loads(out)
    assert decoded[0]["tool_calls"] == calls


def test_infer_retries_on_not_ready_then_succeeds():
    agent = _agent()
    _script(agent, [_not_ready(), _not_ready(), _ok("finally")])
    out = asyncio.run(agent.infer("hi"))
    assert out == "finally"


def test_infer_raises_after_exhausting_retries():
    agent = _agent()
    agent.config.max_retries = 3
    _script(agent, [_not_ready()])  # always not ready
    with pytest.raises(ModelNotReadyError, match="not ready after 3 attempts"):
        asyncio.run(agent.infer("hi"))


def test_infer_passes_generation_params():
    agent = _agent()
    sent = _script(agent, [_ok("x")])
    asyncio.run(agent.infer("hi", max_tokens=128, temperature=0.0))
    req = _last_request(sent)
    assert req["max_tokens"] == 128
    assert req["temperature"] == 0.0  # explicit 0 respected, not overridden


# --------------------------------------------------------------------------- #
# chat                                                                         #
# --------------------------------------------------------------------------- #
def test_chat_prepends_system_when_absent():
    agent = _agent()
    sent = _script(agent, [_ok("reply")])
    asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
    req = _last_request(sent)
    assert req["messages"][0]["role"] == "system"


def test_chat_keeps_existing_system():
    agent = _agent()
    sent = _script(agent, [_ok("reply")])
    asyncio.run(
        agent.chat(
            [
                {"role": "system", "content": "mine"},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    req = _last_request(sent)
    systems = [m for m in req["messages"] if m["role"] == "system"]
    assert systems == [{"role": "system", "content": "mine"}]


def test_chat_returns_full_message():
    agent = _agent()
    _script(agent, [_ok("reply")])
    reply = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
    from atrium.protocol import get_message_text

    assert get_message_text(reply) == "reply"


# --------------------------------------------------------------------------- #
# Readiness / model control                                                    #
# --------------------------------------------------------------------------- #
def test_is_model_ready_reads_status():
    agent = _agent()
    _script(agent, [_ok("")])
    assert asyncio.run(agent.is_model_ready()) is True


def test_is_model_ready_false_when_not_ready():
    agent = _agent()
    _script(agent, [_not_ready()])
    assert asyncio.run(agent.is_model_ready()) is False


def test_load_model_sends_model_name():
    agent = _agent()
    sent = _script(agent, [_ok("")])
    asyncio.run(agent.load_model("Ornith-1.0-35B"))
    req = _last_request(sent)
    assert req["type"] == "load"
    assert req["model_name"] == "Ornith-1.0-35B"
    assert agent.config.model_name == "Ornith-1.0-35B"


def test_load_model_forwards_cache_options():
    agent = _agent()
    sent = _script(agent, [_ok("")])
    cache = KVCacheConfig.for_agents(8, max_seq_len=32768, cache_mode="Q6")
    asyncio.run(agent.load_model("Ornith-1.0-35B", cache=cache))
    req = _last_request(sent)
    assert req["cache_mode"] == "Q6"
    assert req["cache_size"] == 8 * 32768
    assert req["max_seq_len"] == 32768
    assert req["max_batch_size"] == 8


def test_load_model_explicit_option_overrides_cache():
    agent = _agent()
    sent = _script(agent, [_ok("")])
    cache = KVCacheConfig(cache_mode="Q6", cache_size=1000)
    asyncio.run(agent.load_model("m", cache=cache, cache_size=2000))
    req = _last_request(sent)
    assert req["cache_size"] == 2000  # explicit kwarg wins over cache


# --------------------------------------------------------------------------- #
# Shared backend / client mode                                                 #
# --------------------------------------------------------------------------- #
def test_connect_builds_client_mode_agent():
    client = TabbyLLMAgent.connect(
        "reviewer-1", "http://coder-1.local:8730", model_name="Ornith-1.0-35B"
    )
    assert client.is_client is True
    assert client.config.bridge_url == "http://coder-1.local:8730"
    assert client.config.model_name == "Ornith-1.0-35B"
    # A client owns no sandbox, so it declares no GPU requirement of its own.
    assert client._require_gpu is False


def test_backend_owner_is_not_client():
    agent = _agent()
    assert agent.is_client is False
    assert agent._require_gpu is True


def test_client_start_sandbox_is_noop():
    client = TabbyLLMAgent.connect("reviewer-1", "http://coder-1.local:8730")
    asyncio.run(client.start_sandbox())
    # No sandbox is created for a client; it only talks A2A to the shared bridge.
    assert client.is_running is False
    assert client.current_sandbox is None


def test_client_routes_inference_to_shared_bridge():
    client = TabbyLLMAgent.connect("reviewer-1", "http://coder-1.local:8730")
    client.config.retry_backoff_s = 0
    targets: list = []

    async def fake_send(target, message):
        targets.append(target)
        return _ok("shared reply")

    client.send_a2a_message = fake_send  # type: ignore[assignment]
    out = asyncio.run(client.infer("review this"))
    assert out == "shared reply"
    assert targets == ["http://coder-1.local:8730"]


def test_client_cannot_load_model():
    client = TabbyLLMAgent.connect("reviewer-1", "http://coder-1.local:8730")
    with pytest.raises(PolicyViolationError, match="backend-owner operation"):
        asyncio.run(client.load_model("Ornith-1.0-35B"))


def test_client_cannot_unload_model():
    client = TabbyLLMAgent.connect("reviewer-1", "http://coder-1.local:8730")
    with pytest.raises(PolicyViolationError, match="backend-owner operation"):
        asyncio.run(client.unload_model())


def test_client_may_check_readiness():
    # Read-only readiness is open to clients waiting on the shared model.
    client = TabbyLLMAgent.connect("reviewer-1", "http://coder-1.local:8730")
    _script(client, [_ok("")])
    assert asyncio.run(client.is_model_ready()) is True


def test_bridge_url_of_backend_owner_is_host_local():
    agent = _agent()  # id "coder-1", no bridge_url -> owns its backend
    assert agent.bridge_url() == "http://coder-1.local:8730"


def test_clients_infer_concurrently_against_one_backend():
    """Several client agents drive one shared backend *in parallel*.

    A ``Barrier`` sized to the fleet only releases once every client's request is
    in flight — so if ``infer`` calls were serialized, the first would block
    forever waiting for siblings that never started and ``wait_for`` would time
    out. Passing proves all N requests overlap on the one shared bridge, which is
    exactly the fan-in tabbyAPI then serves via continuous batching.
    """
    bridge = "http://coder-1.local:8730"
    clients = [TabbyLLMAgent.connect(f"agent-{i}", bridge) for i in range(4)]
    n = len(clients)
    barrier = asyncio.Barrier(n)
    targets: list = []

    async def shared_backend(target, message):
        targets.append(target)
        await barrier.wait()  # all N must arrive before any proceeds
        prompt = get_message_data(message)[0]["messages"][-1]["content"]
        return _ok(f"handled:{prompt}")

    for client in clients:
        client.send_a2a_message = shared_backend  # type: ignore[assignment]

    async def run():
        return await asyncio.gather(
            *(client.infer(f"req-{i}") for i, client in enumerate(clients))
        )

    outs = asyncio.run(asyncio.wait_for(run(), timeout=5))
    # gather preserves input order regardless of completion order.
    assert outs == [f"handled:req-{i}" for i in range(n)]
    # Every client fanned into the *same* shared backend.
    assert targets == [bridge] * n


# --------------------------------------------------------------------------- #
# KV cache sizing                                                              #
# --------------------------------------------------------------------------- #
def test_plan_cache_size_spans_the_fleet():
    assert plan_cache_size(8, 32768) == 8 * 32768


def test_plan_cache_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        plan_cache_size(0, 32768)


def test_kv_cache_config_rejects_bad_mode():
    with pytest.raises(ValueError, match="cache_mode"):
        KVCacheConfig(cache_mode="Q3")


def test_kv_cache_load_options_omits_unset():
    opts = KVCacheConfig(cache_mode="Q8").load_options()
    assert opts == {"cache_mode": "Q8"}  # None fields are dropped


def test_for_agents_preset_defaults():
    cache = KVCacheConfig.for_agents(4)
    assert cache.cache_mode == "Q6"
    assert cache.max_seq_len == 32768
    assert cache.cache_size == 4 * 32768
    assert cache.max_batch_size == 4


def test_unload_model_clears_ready_flag():
    agent = _agent()
    _script(agent, [_ok("")])
    agent._model_ready = True
    asyncio.run(agent.unload_model())
    assert agent._model_ready is False


# --------------------------------------------------------------------------- #
# Bridge OpenAI request builder (host-optional: needs httpx)                    #
# --------------------------------------------------------------------------- #
def test_bridge_build_chat_request():
    pytest.importorskip("httpx")
    from atrium.agents.tabby_llm_agent.bridge.server import (
        TabbyConfig,
        _build_chat_request,
    )

    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "temperature": 0.1,
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "top_p": 0.9,
    }
    req = _build_chat_request(payload, TabbyConfig(model_name="m"))
    assert req["stream"] is False
    assert req["model"] == "m"
    assert req["max_tokens"] == 256
    assert req["tool_choice"] == "auto"  # defaulted when tools present
    assert req["top_p"] == 0.9


def test_bridge_json_schema_tool_mode_sets_response_format():
    pytest.importorskip("httpx")
    from atrium.agents.tabby_llm_agent.bridge.server import (
        TabbyConfig,
        _build_chat_request,
    )

    payload = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    req = _build_chat_request(payload, TabbyConfig(tool_mode="json_schema"))
    assert req["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- #
# Bridge A2A surface: the socket path an a2a-sdk client actually walks.         #
# Regression coverage for issue #22 (card 404 + message/stream unregistered).   #
# --------------------------------------------------------------------------- #
def _bridge_test_client():
    pytest.importorskip("httpx")
    pytest.importorskip("starlette")
    pytest.importorskip("a2a.server.routes")
    from starlette.testclient import TestClient

    from atrium.agents.tabby_llm_agent.bridge.server import TabbyConfig, build_bridge_app

    return TestClient(build_bridge_app(TabbyConfig(version="9.9.9")))


def test_bridge_serves_well_known_agent_card():
    # Gap 1: create_client(url) resolves /.well-known/agent-card.json first; a
    # bare create_jsonrpc_routes mount 404s here and the client can't connect.
    client = _bridge_test_client()
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "tabby_llm_agent"
    assert card["version"] == "9.9.9"


def test_bridge_accepts_v0_3_message_methods():
    # Gap 2: the card advertises streaming, so a client may pick the v0.3
    # JSON-RPC names (message/send, message/stream). With v0.3 compat enabled the
    # bridge routes them instead of rejecting with -32601 (Method not found).
    client = _bridge_test_client()
    for method in ("message/send", "message/stream"):
        body = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": method,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "data", "data": {"type": "ready?"}}],
                    "messageId": "m1",
                    "kind": "message",
                }
            },
        }
        resp = client.post("/", json=body)
        # The method must be recognized — a routed request never returns -32601.
        # (tabby itself is absent in this unit test, so a downstream error is
        # fine; we only assert the method was found and dispatched.)
        error = resp.json().get("error") if "json" in resp.headers.get("content-type", "") else None
        assert not (error and error.get("code") == -32601), f"{method} not routed: {error}"
