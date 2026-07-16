"""Tests for the ControlPlaneAgent and the interface↔control-plane contract.

No Prefect: the ``submit_job`` kick seam is monkeypatched, so the submit path
(parse → validate → kick → ack) and the shared wire contract are exercised
in-process. Mirrors how a real interface agent would drive the control plane over
A2A, and how the control plane drives ``orchestration.kick``.
"""

from __future__ import annotations

import asyncio

import pytest

from atrium.agents.control_plane import ControlPlaneAgent
from atrium.agents.control_plane.protocol import (
    KIND_SUBMIT,
    SUBMITTED_TYPE,
    build_submit_request,
    build_submitted_reply,
    parse_submit_request,
)
from atrium.core.types import NetworkMode
from atrium.protocol import data_part, get_message_data, metadata_dict, text_message


def _agent(**kwargs):
    return ControlPlaneAgent("control-1", "0.1.0", **kwargs)


def _script_kick(monkeypatch, *, job_id="flow-run-123"):
    """Replace the kick seam; record the (args, kwargs) it was called with."""
    calls: list = []

    async def fake_submit_job(agent, instruction="", **kw):
        calls.append({"agent": agent, "instruction": instruction, **kw})
        return job_id

    monkeypatch.setattr(
        "atrium.agents.control_plane.agent.submit_job", fake_submit_job
    )
    return calls


# --------------------------------------------------------------------------- #
# Contract round-trip                                                          #
# --------------------------------------------------------------------------- #
def test_submit_request_roundtrip():
    msg = build_submit_request(
        "python_code_workspace_agent:active",
        "write hello world",
        context_id="slack:C1:1699999999.0001",
        payload={"slack": {"channel": "C1"}, "steering": {"style": "terse"}},
        review={"reviewer": "slack_reviewer:active"},
    )
    assert metadata_dict(msg)["kind"] == KIND_SUBMIT
    req = parse_submit_request(msg)
    assert req.agent == "python_code_workspace_agent:active"
    assert req.instruction == "write hello world"
    assert req.context_id == "slack:C1:1699999999.0001"
    assert req.payload["steering"] == {"style": "terse"}
    assert req.review == {"reviewer": "slack_reviewer:active"}
    assert req.feedback_for is None


def test_parse_allows_missing_agent():
    # An agent-less turn is valid: the control plane routes it (D5).
    req = parse_submit_request(build_submit_request("", "do a thing"))
    assert req.agent == ""
    assert req.instruction == "do a thing"


def test_parse_falls_back_to_message_text_for_instruction():
    # A bare data part with the agent but no instruction: text carries the ask.
    msg = text_message(
        "build a widget",
        extra_parts=[data_part({"type": "workboard_submit", "agent": "widget_agent:active"})],
    )
    req = parse_submit_request(msg)
    assert req.agent == "widget_agent:active"
    assert req.instruction == "build a widget"


# --------------------------------------------------------------------------- #
# Envelope                                                                     #
# --------------------------------------------------------------------------- #
def test_envelope_is_wan_capable():
    agent = _agent()
    assert agent.sandbox_config.network is NetworkMode.BRIDGE


# --------------------------------------------------------------------------- #
# Happy path: submit -> kick -> ack                                            #
# --------------------------------------------------------------------------- #
def test_handle_submit_kicks_and_acks(monkeypatch):
    calls = _script_kick(monkeypatch, job_id="flow-run-abc")
    agent = _agent(deployment="prod")
    msg = build_submit_request(
        "python_code_workspace_agent:active",
        "write hello world",
        context_id="slack:C1:42",
        payload={"steering": {"lang": "python"}},
        review={"reviewer": "slack_reviewer:active"},
    )
    reply = asyncio.run(agent.dispatch(msg))

    # It kicked exactly one workboard run with the request's fields.
    assert len(calls) == 1
    kick = calls[0]
    assert kick["agent"] == "python_code_workspace_agent:active"
    assert kick["instruction"] == "write hello world"
    assert kick["context_id"] == "slack:C1:42"
    assert kick["payload"] == {"steering": {"lang": "python"}}
    assert kick["deployment"] == "prod"
    # review dict -> ReviewPolicy with the right reviewer.
    assert kick["review"] is not None and kick["review"].reviewer == "slack_reviewer:active"

    # The ack carries the scheduled flow-run id.
    assert metadata_dict(reply)["status"] == "ok"
    data = get_message_data(reply)[0]
    assert data["type"] == SUBMITTED_TYPE
    assert data["job_id"] == "flow-run-abc"


def test_no_review_means_no_policy(monkeypatch):
    calls = _script_kick(monkeypatch)
    agent = _agent()
    msg = build_submit_request("widget_agent:active", "x", context_id="slack:C1:7")
    asyncio.run(agent.dispatch(msg))
    assert calls[0]["review"] is None


# --------------------------------------------------------------------------- #
# D5 — target-agent routing                                                    #
# --------------------------------------------------------------------------- #
def test_agentless_turn_routes_to_default(monkeypatch):
    calls = _script_kick(monkeypatch)
    agent = _agent(default_agent="python_code_workspace_agent:active")
    # A turn with no explicit doer — the control plane picks the default.
    msg = build_submit_request("", "write hello world", context_id="slack:C1:9")
    asyncio.run(agent.dispatch(msg))
    assert calls[0]["agent"] == "python_code_workspace_agent:active"


def test_explicit_agent_overrides_default(monkeypatch):
    calls = _script_kick(monkeypatch)
    agent = _agent(default_agent="python_code_workspace_agent:active")
    msg = build_submit_request("widget_agent:active", "x", context_id="slack:C1:9")
    asyncio.run(agent.dispatch(msg))
    assert calls[0]["agent"] == "widget_agent:active"


def test_unroutable_when_no_agent_and_no_default(monkeypatch):
    calls = _script_kick(monkeypatch)
    agent = _agent(default_agent="")
    msg = build_submit_request("", "x", context_id="slack:C1:9")
    reply = asyncio.run(agent.dispatch(msg))
    assert calls == []
    assert metadata_dict(reply)["status"] == "error"


def test_feedback_relay_is_refused_not_submitted(monkeypatch):
    calls = _script_kick(monkeypatch)
    agent = _agent()
    msg = build_submit_request(
        "widget_agent:active", "looks good", context_id="slack:C1:7", feedback_for="review-9"
    )
    reply = asyncio.run(agent.dispatch(msg))
    # It must NOT spawn a job for a feedback relay it can't yet handle.
    assert calls == []
    assert metadata_dict(reply)["status"] == "error"


# --------------------------------------------------------------------------- #
# Ack builder                                                                  #
# --------------------------------------------------------------------------- #
def test_submitted_reply_error_status():
    reply = build_submitted_reply("boom", status="error")
    assert metadata_dict(reply)["status"] == "error"
    assert get_message_data(reply)[0]["status"] == "error"
