"""Tests for the control-plane ↔ plan-agent A2A contract (pure wire glue).

Round-trips the plan request/result envelopes and asserts the fail-closed
behaviour a malformed reply must have. No orchestration, no backend.
"""

from __future__ import annotations

from atrium.agents.plan_agent_protocol import (
    KIND_PLAN,
    PLAN_REQUEST_TYPE,
    PLAN_RESULT_TYPE,
    build_plan_request,
    build_plan_result,
    parse_plan_request,
    parse_plan_result,
)
from atrium.protocol import get_message_data, metadata_dict, text_message


def test_plan_request_round_trip():
    msg = build_plan_request(
        {"instruction": "build X"},
        "build X",
        context_id="slack:C1:1",
        constraints={"agents": ["coder"]},
    )
    assert metadata_dict(msg).get("kind") == KIND_PLAN
    assert any(p.get("type") == PLAN_REQUEST_TYPE for p in get_message_data(msg))
    req = parse_plan_request(msg)
    assert req.request == {"instruction": "build X"}
    assert req.instruction == "build X"
    assert req.context_id == "slack:C1:1"
    assert req.constraints == {"agents": ["coder"]}


def test_plan_request_text_fallback():
    # A bare text message (no data part) still yields the instruction.
    msg = text_message("just do it")
    req = parse_plan_request(msg)
    assert req.instruction == "just do it"
    assert req.request == {}


def test_plan_result_round_trip():
    msg = build_plan_result(
        "print('hi')", {"x": 1}, requirements=["prefect"], reason="ok then"
    )
    assert metadata_dict(msg).get("status") == "ok"
    res = parse_plan_result(msg)
    assert res == {
        "status": "ok",
        "flow_source": "print('hi')",
        "params": {"x": 1},
        "requirements": ["prefect"],
        "reason": "ok then",
    }


def test_plan_result_error_is_fail_closed():
    msg = build_plan_result(status="error", reason="no python block")
    res = parse_plan_result(msg)
    assert res["status"] == "error"
    assert res["flow_source"] == ""
    assert res["reason"] == "no python block"


def test_parse_plan_result_missing_part_reads_as_error():
    res = parse_plan_result(text_message("nonsense"))
    assert res["status"] == "error"
    assert res["flow_source"] == ""
    assert PLAN_RESULT_TYPE  # symbol exported
