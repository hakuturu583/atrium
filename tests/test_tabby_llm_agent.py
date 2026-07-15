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

from atrium.agents.tabby_llm_agent import TabbyLLMAgent
from atrium.agents.tabby_llm_agent.agent import (
    STATUS_NOT_READY,
    STATUS_OK,
    STATUS_TOOL_CALLS,
)
from atrium.core.errors import ModelNotReadyError
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
