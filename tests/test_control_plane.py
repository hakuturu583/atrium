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
    build_job_update,
    build_submit_request,
    build_submitted_reply,
    parse_job_update,
    parse_submit_request,
    parse_submitted_reply,
)
from atrium.agents.plan_agent_protocol import build_plan_result, parse_plan_request
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


# --------------------------------------------------------------------------- #
# D6 — progress notifications (poll → push job_update)                          #
# --------------------------------------------------------------------------- #
def _capture_pushes(agent):
    pushed: list = []

    async def fake_send(target, message):
        pushed.append((target, message))
        return message

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    return pushed


def _submit(agent, monkeypatch, *, job_id, coords):
    _script_kick(monkeypatch, job_id=job_id)
    msg = build_submit_request(
        "widget_agent:active", "x", context_id="slack:C1:9", payload={"reply_coords": coords}
    )
    asyncio.run(agent.dispatch(msg))


def test_submit_tracks_reply_coords(monkeypatch):
    agent = _agent()
    _submit(agent, monkeypatch, job_id="j1", coords={"channel": "C1", "thread": "9"})
    assert agent._pending["j1"]["coords"] == {"channel": "C1", "thread": "9"}


def test_poll_pushes_job_update_when_done(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit(agent, monkeypatch, job_id="j1", coords={"channel": "C1", "thread": "9"})
    pushed = _capture_pushes(agent)

    async def fake_state(job_id):
        return {"id": job_id, "state": "COMPLETED", "done": True}

    monkeypatch.setattr("atrium.agents.control_plane.agent.workboard_state", fake_state)
    notified = asyncio.run(agent.poll_once())

    assert notified == ["j1"]
    target, msg = pushed[0]
    assert target == "http://interface.local"
    upd = parse_job_update(msg)
    assert upd["job_id"] == "j1" and upd["status"] == "ok"
    assert upd["coords"] == {"channel": "C1", "thread": "9"}
    assert "j1" not in agent._pending  # stopped tracking


def test_poll_skips_unfinished_jobs(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit(agent, monkeypatch, job_id="j1", coords={})
    pushed = _capture_pushes(agent)

    async def fake_state(job_id):
        return {"id": job_id, "state": "RUNNING", "done": False}

    monkeypatch.setattr("atrium.agents.control_plane.agent.workboard_state", fake_state)
    assert asyncio.run(agent.poll_once()) == []
    assert pushed == [] and "j1" in agent._pending


def test_failed_job_pushes_error_status(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit(agent, monkeypatch, job_id="j1", coords={})
    pushed = _capture_pushes(agent)

    async def fake_state(job_id):
        return {"id": job_id, "state": "FAILED", "done": True}

    monkeypatch.setattr("atrium.agents.control_plane.agent.workboard_state", fake_state)
    asyncio.run(agent.poll_once())
    assert parse_job_update(pushed[0][1])["status"] == "error"


# --------------------------------------------------------------------------- #
# 動線2(b) — async human review (post-hoc ticket loop)                          #
# --------------------------------------------------------------------------- #
def _submit_human(agent, monkeypatch, *, job_id, ctx="slack:C1:9", coords=None):
    _script_kick(monkeypatch, job_id=job_id)
    msg = build_submit_request(
        "widget_agent:active", "build a widget", context_id=ctx,
        payload={"reply_coords": coords or {"channel": "C1", "thread": "9"}},
        review={"human": True},
    )
    asyncio.run(agent.dispatch(msg))


def _finish(agent, monkeypatch, *, completed=True):
    async def fake_state(job_id):
        return {"id": job_id, "state": "COMPLETED" if completed else "FAILED", "done": True}

    monkeypatch.setattr("atrium.agents.control_plane.agent.workboard_state", fake_state)


def test_completed_human_review_opens_ticket_not_done(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit_human(agent, monkeypatch, job_id="j1")
    pushed = _capture_pushes(agent)
    _finish(agent, monkeypatch, completed=True)
    asyncio.run(agent.poll_once())

    # It presents the deliverable for review (status=review) with a token, not "ok".
    upd = parse_job_update(pushed[0][1])
    assert upd["status"] == "review"
    assert upd["result"]["token"] == "slack:C1:9"
    assert "slack:C1:9" in agent._tickets  # ticket parked
    assert "j1" not in agent._pending


def test_approval_closes_ticket_as_done(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit_human(agent, monkeypatch, job_id="j1")
    _finish(agent, monkeypatch, completed=True)
    pushed = _capture_pushes(agent)
    asyncio.run(agent.poll_once())  # pushes the review presentation

    # Human approves via a feedback_for submit.
    reply = build_submit_request("", "LGTM", context_id="slack:C1:9", feedback_for="slack:C1:9")
    asyncio.run(agent.dispatch(reply))

    assert parse_job_update(pushed[-1][1])["status"] == "ok"  # last push is the done update
    assert "slack:C1:9" not in agent._tickets  # closed


def test_request_changes_kicks_rework(monkeypatch):
    agent = _agent(interface="http://interface.local")
    _submit_human(agent, monkeypatch, job_id="j1")
    _finish(agent, monkeypatch, completed=True)
    _capture_pushes(agent)
    asyncio.run(agent.poll_once())  # pushes the review presentation

    # Human asks for changes → a rework job is kicked with the feedback as steering.
    rework_calls = _script_kick(monkeypatch, job_id="j2")
    reply = build_submit_request("", "please add error handling", context_id="slack:C1:9", feedback_for="slack:C1:9")
    ack = asyncio.run(agent.dispatch(reply))

    assert rework_calls[0]["agent"] == "widget_agent:active"  # same doer
    assert rework_calls[0]["payload"]["steering"]["review_feedback"] == "please add error handling"
    assert get_message_data(ack)[0]["job_id"] == "j2"
    # The rework job is itself tracked for human review (the loop continues).
    assert agent._pending["j2"]["human_review"] is True


def test_feedback_for_unknown_review_errors(monkeypatch):
    agent = _agent()
    reply = build_submit_request("", "hi", feedback_for="nope")
    out = asyncio.run(agent.dispatch(reply))
    assert metadata_dict(out)["status"] == "error"


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


def test_submitted_reply_roundtrip():
    parsed = parse_submitted_reply(build_submitted_reply("flow-run-9"))
    assert parsed == {"status": "ok", "job_id": "flow-run-9"}


def test_job_update_roundtrip():
    msg = build_job_update(
        "flow-run-9", status="ok",
        coords={"channel": "C1", "thread_ts": "42"}, result={"digest": "sha256:abc"},
    )
    parsed = parse_job_update(msg)
    assert parsed["job_id"] == "flow-run-9"
    assert parsed["status"] == "ok"
    assert parsed["coords"] == {"channel": "C1", "thread_ts": "42"}
    assert parsed["result"] == {"digest": "sha256:abc"}
    # A non-update message parses to empty.
    assert parse_job_update(build_submitted_reply("x")) == {}


# --------------------------------------------------------------------------- #
# Plan path: submit(plan) -> PlanAgent proposes flow -> Job gate -> workboard  #
# --------------------------------------------------------------------------- #
_VALID_FLOW = "from prefect import flow\n\n@flow\ndef main():\n    return 1\n"


def _script_plan(monkeypatch, *, flow_source=_VALID_FLOW, params=None, requirements=None,
                 status="ok", reason="", run_id="flow-run-plan"):
    """Script the plan agent (via send_a2a_message) and the submit_workboard seam.

    Returns a dict with ``plan_requests`` (what the planner was asked) and ``kicks``
    (the workboards submitted), so a test can assert on both ends without a backend.
    """
    captured: dict = {"plan_requests": [], "kicks": []}

    async def fake_send(self, target, message):
        captured["plan_requests"].append({"target": target, "req": parse_plan_request(message)})
        return build_plan_result(
            flow_source, params or {"x": 1}, requirements=requirements, reason=reason, status=status
        )

    async def fake_submit_workboard(workboard, **kw):
        captured["kicks"].append({"workboard": workboard, **kw})
        return run_id

    monkeypatch.setattr(ControlPlaneAgent, "send_a2a_message", fake_send)
    monkeypatch.setattr(
        "atrium.agents.control_plane.agent.submit_workboard", fake_submit_workboard
    )
    return captured


def _plan_submit(instruction="build a report", **payload_extra):
    payload = {"plan": True, "request": {"instruction": instruction}}
    payload.update(payload_extra)
    return build_submit_request("", instruction, context_id="slack:C1:7", payload=payload)


def test_plan_turn_dispatches_and_kicks_execution_workboard(monkeypatch):
    cap = _script_plan(monkeypatch)
    agent = _agent(plan_agent="plan_agent:active", deployment="prod")
    reply = asyncio.run(agent.dispatch(_plan_submit()))

    # It asked the plan agent to plan the request...
    assert len(cap["plan_requests"]) == 1
    assert cap["plan_requests"][0]["target"] == "plan_agent:active"
    assert cap["plan_requests"][0]["req"].request == {"instruction": "build a report"}

    # ...and, the plan being ready, kicked exactly one execution workboard.
    assert len(cap["kicks"]) == 1
    kick = cap["kicks"][0]
    wb = kick["workboard"]
    node = wb.nodes[0]
    assert node.id == "run_flow"
    assert node.agent == "prefect_runner_agent:active"
    assert node.reviewable is True
    assert node.payload["files"]["flow.py"] == _VALID_FLOW
    assert node.payload["commands"] == ["python flow.py"]
    assert kick["deployment"] == "prod"
    assert kick["context_id"] == "slack:C1:7"

    # The ack carries the flow-run id.
    assert parse_submitted_reply(reply) == {"status": "ok", "job_id": "flow-run-plan"}


def test_plan_reviewer_adds_pre_execution_source_review(monkeypatch):
    cap = _script_plan(monkeypatch)
    agent = _agent(plan_agent="plan_agent:active", plan_reviewer="flow_reviewer:active")
    asyncio.run(agent.dispatch(_plan_submit()))

    wb = cap["kicks"][0]["workboard"]
    assert [n.id for n in wb.nodes] == ["review_source", "run_flow"]
    assert wb.nodes[0].agent == "flow_reviewer:active"
    assert wb.nodes[1].depends_on == ["review_source"]


def test_not_ready_plan_kicks_nothing(monkeypatch):
    # A fail-closed plan error (empty flow) must never reach a workboard.
    cap = _script_plan(monkeypatch, flow_source="", status="error", reason="no python block")
    agent = _agent(plan_agent="plan_agent:active")
    reply = asyncio.run(agent.dispatch(_plan_submit()))

    assert cap["kicks"] == []
    parsed = parse_submitted_reply(reply)
    assert parsed["status"] == "error"
    assert "not ready" in parsed["job_id"]


def test_plan_requiring_unavailable_library_is_rejected(monkeypatch):
    # Planner declares a dep outside the deployment's allow-list → rejected pre-run.
    cap = _script_plan(monkeypatch, requirements=["requests"])
    agent = _agent(
        plan_agent="plan_agent:active",
        plan_constraints={"allowed_libraries": ["prefect"]},
    )
    reply = asyncio.run(agent.dispatch(_plan_submit()))
    assert cap["kicks"] == []
    parsed = parse_submitted_reply(reply)
    assert parsed["status"] == "error"
    assert "unavailable" in parsed["job_id"]


def test_broken_flow_syntax_is_gated(monkeypatch):
    cap = _script_plan(monkeypatch, flow_source="def broken(:\n  pass")
    agent = _agent(plan_agent="plan_agent:active")
    reply = asyncio.run(agent.dispatch(_plan_submit()))
    assert cap["kicks"] == []
    assert parse_submitted_reply(reply)["status"] == "error"


def test_plan_flag_ignored_without_plan_agent(monkeypatch):
    # No plan agent configured → a plan-flagged turn falls through to the fast path.
    calls = _script_kick(monkeypatch, job_id="flow-run-fast")
    agent = _agent()  # no plan_agent
    reply = asyncio.run(agent.dispatch(_plan_submit()))
    assert len(calls) == 1  # single-node fast path ran
    assert parse_submitted_reply(reply) == {"status": "ok", "job_id": "flow-run-fast"}


def test_plan_agent_and_reviewer_read_from_env(monkeypatch):
    monkeypatch.setenv("ATRIUM_PLAN_AGENT", "plan_agent:active")
    monkeypatch.setenv("ATRIUM_PLAN_REVIEWER", "flow_reviewer:active")
    agent = _agent()  # no explicit plan_agent/plan_reviewer args
    assert agent.plan_agent == "plan_agent:active"
    assert agent.plan_reviewer == "flow_reviewer:active"


def test_explicit_plan_agent_overrides_env(monkeypatch):
    monkeypatch.setenv("ATRIUM_PLAN_AGENT", "env_agent:active")
    agent = _agent(plan_agent="explicit_agent:active")
    assert agent.plan_agent == "explicit_agent:active"


def test_non_plan_turn_keeps_fast_path_even_with_plan_agent(monkeypatch):
    # A turn WITHOUT payload["plan"] is untouched even when a plan agent exists.
    calls = _script_kick(monkeypatch, job_id="flow-run-direct")
    agent = _agent(plan_agent="plan_agent:active")
    msg = build_submit_request("coder:active", "just chat", context_id="slack:C1:9")
    reply = asyncio.run(agent.dispatch(msg))
    assert len(calls) == 1
    assert calls[0]["agent"] == "coder:active"
    assert parse_submitted_reply(reply) == {"status": "ok", "job_id": "flow-run-direct"}
