"""Auto-wire tests: the composed ``prompt_memory`` reaches the request as the
system message, and an explicit ``system=`` always overrides it.

These construct a ``TabbyLLMAgent`` (no sandbox is started) and monkeypatch the
A2A send so nothing leaves the process.
"""

from __future__ import annotations

import asyncio

from atrium.agents.prompt_memory import PromptLayer, PromptMemory, tools_layer
from atrium.agents.tabby_llm_agent.agent import TabbyLLMAgent
from atrium.protocol import Role, text_message

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


def _agent_with_capture(prompt_memory=None):
    """Build an agent whose ``_infer_with_retry`` records the request instead of
    sending it, returning a canned OK reply."""
    agent = TabbyLLMAgent("coder-test", "0.1.0", prompt_memory=prompt_memory)
    captured: dict = {}

    async def fake_send(request):
        captured["request"] = request
        return text_message("OK", role=Role.ROLE_AGENT, metadata={"status": "ok"})

    agent._infer_with_retry = fake_send  # type: ignore[assignment]
    return agent, captured


def test_infer_injects_composed_system_prompt():
    mem = PromptMemory(order=("identity", "tools"))
    mem.record(PromptLayer("identity", order=10, content="You are a test agent."))
    mem.record(tools_layer(order=40))
    agent, captured = _agent_with_capture(mem)

    asyncio.run(agent.infer("hi", tools=[SAMPLE_TOOL]))

    messages = captured["request"]["messages"]
    assert messages[0]["role"] == "system"
    assert "You are a test agent." in messages[0]["content"]
    assert "<tools>" in messages[0]["content"]  # tools rendered into the layer
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_explicit_system_overrides_memory():
    mem = PromptMemory().record(PromptLayer("identity", content="composed"))
    agent, captured = _agent_with_capture(mem)

    asyncio.run(agent.infer("hi", system="explicit override"))

    messages = captured["request"]["messages"]
    assert messages[0] == {"role": "system", "content": "explicit override"}


def test_empty_memory_yields_no_system_message():
    agent, captured = _agent_with_capture()  # default empty PromptMemory
    asyncio.run(agent.infer("hi"))
    messages = captured["request"]["messages"]
    assert all(m["role"] != "system" for m in messages)


def test_chat_prepends_system_only_when_absent():
    mem = PromptMemory().record(PromptLayer("identity", content="composed"))
    agent, captured = _agent_with_capture(mem)

    # No system turn -> composed memory is prepended.
    asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
    assert captured["request"]["messages"][0] == {"role": "system", "content": "composed"}

    # Existing system turn -> left untouched.
    asyncio.run(
        agent.chat(
            [
                {"role": "system", "content": "caller system"},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    systems = [m for m in captured["request"]["messages"] if m["role"] == "system"]
    assert systems == [{"role": "system", "content": "caller system"}]
