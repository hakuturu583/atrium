"""Cross-container trace-stitch check: one A→bridge→tabby timeline in Phoenix.

Drives a real inference over A2A with tracing configured, flushes the exporter,
then asks Phoenix whether the host-side span and the in-sandbox bridge span landed
under a **single trace id** — the observability guarantee from
``docs/design/observability.md`` (W3C ``traceparent`` carried across the
physically-isolated containers).

Phoenix's GraphQL schema shifts between releases, so the query is best-effort: on
any shape mismatch the test skips with a pointer to verify manually in the UI
(``open $ATRIUM_IT_PHOENIX_URL`` and confirm the trace is one timeline).

Env:
    ATRIUM_INTEGRATION=1                 opt in (required)
    ATRIUM_IT_BRIDGE_URL                 running bridge A2A URL (required)
    ATRIUM_IT_PHOENIX_URL                Phoenix base URL, e.g. http://localhost:6006 (required)
    OTEL_EXPORTER_OTLP_ENDPOINT          OTLP traces endpoint the bridge+host both ship to
    ATRIUM_IT_MODEL                      model to ensure loaded (optional)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request

import pytest

from atrium.agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent
from atrium.core import telemetry as tel

from .conftest import require_env

pytestmark = pytest.mark.integration


def _phoenix_graphql(base_url: str, query: str) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _spans_query() -> str:
    # Kept intentionally small; parsed defensively below.
    return (
        "{ projects(first:1){ edges{ node{ spans(first:200){ edges{ node{ "
        "name context{ traceId } } } } } } } }"
    )


def _collect_traces(payload: dict) -> dict[str, set[str]]:
    """Map trace_id -> set(span names) from a Phoenix GraphQL response.

    Raises ``KeyError``/``TypeError`` on an unexpected shape (caller skips).
    """
    traces: dict[str, set[str]] = {}
    projects = payload["data"]["projects"]["edges"]
    for proj in projects:
        for span_edge in proj["node"]["spans"]["edges"]:
            node = span_edge["node"]
            trace_id = node["context"]["traceId"]
            traces.setdefault(trace_id, set()).add(node["name"])
    return traces


def test_infer_produces_single_stitched_trace():
    env = require_env("ATRIUM_IT_BRIDGE_URL", "ATRIUM_IT_PHOENIX_URL")
    require_env("OTEL_EXPORTER_OTLP_ENDPOINT")

    tel.configure_tracing("atrium-integration-test")
    agent = TabbyLLMAgent(
        "it-trace",
        "0.1.0",
        config=TabbyAgentConfig(
            bridge_url=env["ATRIUM_IT_BRIDGE_URL"],
            model_name=os.environ.get("ATRIUM_IT_MODEL"),
        ),
    )

    async def scenario() -> None:
        await agent.wait_until_ready(timeout=float(os.environ.get("ATRIUM_IT_READY_TIMEOUT", "300")))
        await agent.infer("Reply with: pong", max_tokens=16, temperature=0.0)

    asyncio.run(scenario())
    tel.shutdown_tracing()  # flush the BatchSpanProcessor

    # Give Phoenix a moment to ingest, then query.
    time.sleep(3)
    try:
        payload = _phoenix_graphql(env["ATRIUM_IT_PHOENIX_URL"], _spans_query())
        traces = _collect_traces(payload)
    except (urllib.error.URLError, KeyError, TypeError, ValueError) as exc:
        pytest.skip(
            f"could not read Phoenix spans via GraphQL ({exc}); verify the stitched "
            f"trace manually at {env['ATRIUM_IT_PHOENIX_URL']}"
        )

    # A stitched trace contains BOTH a host-side send span and the in-sandbox
    # bridge/LLM span under one trace id.
    host_markers = {"a2a.send_message", "agent.send_a2a_message"}
    bridge_markers = {"tabby_bridge.execute", "tabby.chat_completions"}
    stitched = [
        tid
        for tid, names in traces.items()
        if names & host_markers and names & bridge_markers
    ]
    assert stitched, (
        "no single trace carried both host-side and in-sandbox bridge spans; "
        f"saw traces: { {t: sorted(n) for t, n in traces.items()} }"
    )
