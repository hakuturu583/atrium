"""Real tabbyAPI inference smoke test, driven end-to-end over A2A.

Talks to an already-running in-sandbox bridge (co-located with tabbyAPI on the
exllamav3 backend) at ``ATRIUM_IT_BRIDGE_URL`` and drives a real completion
through it. Because bringing up a GPU box + quantized model is heavy, this test
assumes the bridge is already serving (e.g. the TabbyLLMAgent sandbox is up); it
does not launch the GPU sandbox itself. Skips unless the bridge URL is provided.

Env:
    ATRIUM_INTEGRATION=1         opt in (required)
    ATRIUM_IT_BRIDGE_URL         A2A base URL of the running bridge (required),
                                 e.g. http://10.0.0.5:8730
    ATRIUM_IT_MODEL              model to ensure loaded first (optional)
    ATRIUM_IT_READY_TIMEOUT      seconds to wait for the model (default 300)
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from atrium.agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent

from .conftest import require_env

pytestmark = pytest.mark.integration


def _agent() -> TabbyLLMAgent:
    env = require_env("ATRIUM_IT_BRIDGE_URL")
    config = TabbyAgentConfig(
        bridge_url=env["ATRIUM_IT_BRIDGE_URL"],
        model_name=os.environ.get("ATRIUM_IT_MODEL"),
    )
    # No start_sandbox(): the bridge is assumed already serving.
    return TabbyLLMAgent("it-tabby", "0.1.0", config=config)


def test_bridge_reports_readiness():
    agent = _agent()
    # Should not raise; returns a bool regardless of whether a model is loaded.
    ready = asyncio.run(agent.is_model_ready())
    assert isinstance(ready, bool)


def test_infer_returns_text():
    agent = _agent()
    timeout = float(os.environ.get("ATRIUM_IT_READY_TIMEOUT", "300"))

    async def scenario() -> str:
        if agent.config.model_name:
            await agent.load_model(agent.config.model_name)
        await agent.wait_until_ready(timeout=timeout)
        return await agent.infer(
            "Reply with the single word: pong", max_tokens=16, temperature=0.0
        )

    out = asyncio.run(scenario())
    assert isinstance(out, str) and out.strip()


def test_parallel_infer_fans_into_one_backend():
    """Several client agents share one bridge/backend and infer concurrently.

    Exercises the shared-backend path on real hardware: N ``connect()`` clients
    fan into the one already-serving bridge and run in parallel via tabbyAPI's
    continuous batching. Asserts every client gets a real completion, and that
    concurrent wall-clock beats running the same requests sequentially (the
    batching win). Set ``ATRIUM_IT_PARALLEL`` to change the fleet size.
    """
    env = require_env("ATRIUM_IT_BRIDGE_URL")
    bridge_url = env["ATRIUM_IT_BRIDGE_URL"]
    model_name = os.environ.get("ATRIUM_IT_MODEL")
    timeout = float(os.environ.get("ATRIUM_IT_READY_TIMEOUT", "300"))
    n = int(os.environ.get("ATRIUM_IT_PARALLEL", "4"))

    clients = [
        TabbyLLMAgent.connect(f"it-tabby-{i}", bridge_url, model_name=model_name)
        for i in range(n)
    ]
    prompts = [f"Reply with the single word: number{i}" for i in range(n)]

    async def ask(agent: TabbyLLMAgent, prompt: str) -> str:
        return await agent.infer(prompt, max_tokens=16, temperature=0.0)

    async def scenario() -> tuple[list[str], float, float]:
        await clients[0].wait_until_ready(timeout=timeout)

        t0 = time.perf_counter()
        for agent, prompt in zip(clients, prompts):
            await ask(agent, prompt)
        seq_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        outs = await asyncio.gather(
            *(ask(agent, prompt) for agent, prompt in zip(clients, prompts))
        )
        conc_wall = time.perf_counter() - t0
        return outs, seq_wall, conc_wall

    outs, seq_wall, conc_wall = asyncio.run(scenario())

    assert all(c.is_client for c in clients)
    assert len(outs) == n
    assert all(isinstance(o, str) and o.strip() for o in outs)
    # Sharing one batched backend, concurrent beats sequential (allow slack for
    # short prompts / scheduler warmup).
    assert conc_wall < seq_wall
