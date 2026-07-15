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
