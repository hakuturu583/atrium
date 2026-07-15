"""Tests for the workboard orchestration core (Prefect-free).

Everything here exercises the parts that must be correct without a backend: the
DAG value objects, the scheduler's dependency/failure/cancel/subtask logic, the
A2A protocol glue, the node runner (with a scripted send — no network), and
cooperative cancellation. The Prefect adapter is a thin driver over these and is
covered by the opt-in integration suite, not here.
"""

from __future__ import annotations

import asyncio

import pytest

from atrium.orchestration.bootstrap import BootstrapConfig
from atrium.orchestration import (
    BoardUpdate,
    CancellableAgentExecutor,
    CancelToken,
    NodeOutcome,
    NodeResult,
    TaskCancelledError,
    WorkNode,
    Workboard,
    WorkboardError,
    WorkboardScheduler,
    board_update_message,
    build_node_request,
    current_cancel_token,
    extract_board_update,
    run_node,
)
from atrium.orchestration import runner as runner_mod
from atrium.protocol import (
    Role,
    data_message,
    get_message_data,
    metadata_dict,
    text_message,
)


# --------------------------------------------------------------------------- #
# Workboard validation                                                         #
# --------------------------------------------------------------------------- #
def _wb(*nodes: WorkNode) -> Workboard:
    return Workboard(id="wb", nodes=list(nodes))


def _node(nid: str, deps: list[str] | None = None, agent: str = "http://a.local") -> WorkNode:
    return WorkNode(id=nid, agent=agent, instruction=f"do {nid}", depends_on=deps or [])


def test_validate_accepts_a_dag():
    _wb(_node("a"), _node("b", ["a"]), _node("c", ["a", "b"])).validate()


def test_validate_rejects_duplicate_ids():
    with pytest.raises(WorkboardError):
        _wb(_node("a"), _node("a")).validate()


def test_validate_rejects_dangling_dependency():
    with pytest.raises(WorkboardError):
        _wb(_node("a", ["ghost"])).validate()


def test_validate_rejects_self_dependency():
    with pytest.raises(WorkboardError):
        _wb(_node("a", ["a"])).validate()


def test_validate_rejects_a_cycle():
    with pytest.raises(WorkboardError):
        _wb(_node("a", ["b"]), _node("b", ["a"])).validate()


def test_workboard_dict_roundtrip():
    wb = _wb(_node("a"), _node("b", ["a"]))
    assert Workboard.from_dict(wb.to_dict()).to_dict() == wb.to_dict()


def test_bootstrap_config_derives_endpoints():
    cfg = BootstrapConfig()
    assert cfg.prefect_api_url == "http://127.0.0.1:4200/api"
    assert cfg.prefect_health_url == "http://127.0.0.1:4200/api/health"
    assert cfg.otlp_endpoint == "http://127.0.0.1:6006/v1/traces"


def test_single_builds_a_valid_one_node_job():
    wb = Workboard.single("agent-x", "do the thing", payload={"k": 1})
    wb.validate()
    assert [n.id for n in wb.nodes] == ["root"]
    node = wb.nodes[0]
    assert node.agent == "agent-x" and node.instruction == "do the thing"
    assert node.depends_on == [] and node.payload == {"k": 1}


# --------------------------------------------------------------------------- #
# Scheduler                                                                    #
# --------------------------------------------------------------------------- #
def _ok(nid: str, **kw) -> NodeResult:
    return NodeResult(node_id=nid, outcome=NodeOutcome(status="ok"), **kw)


def _err(nid: str) -> NodeResult:
    return NodeResult(node_id=nid, outcome=NodeOutcome(status="error", reason="boom"))


def test_scheduler_respects_dependencies():
    sched = WorkboardScheduler(_wb(_node("a"), _node("b", ["a"])))
    wave1 = [n.id for n in sched.ready()]
    assert wave1 == ["a"]  # b blocked on a
    assert sched.ready() == []  # a already handed out, not yet recorded
    sched.record(_ok("a"))
    assert [n.id for n in sched.ready()] == ["b"]
    sched.record(_ok("b"))
    assert sched.finished
    assert sched.summary()["done"] == ["a", "b"]


def test_scheduler_runs_independent_nodes_in_one_wave():
    sched = WorkboardScheduler(_wb(_node("a"), _node("b"), _node("c", ["a", "b"])))
    assert {n.id for n in sched.ready()} == {"a", "b"}
    sched.record(_ok("a"))
    sched.record(_ok("b"))
    assert [n.id for n in sched.ready()] == ["c"]


def test_failure_cascades_to_dependents_as_skipped():
    sched = WorkboardScheduler(_wb(_node("a"), _node("b", ["a"]), _node("c", ["b"])))
    sched.ready()
    sched.record(_err("a"))
    assert sched.finished  # nothing else can run
    assert sched.ready() == []
    assert sched.summary() == {
        "done": [],
        "failed": ["a"],
        "cancelled": [],
        "skipped": ["b", "c"],
    }


def test_agent_proposed_subtask_is_grafted():
    sched = WorkboardScheduler(_wb(_node("a")))
    sched.ready()
    graft = WorkNode(id="a.child", agent="http://a.local", depends_on=["a"])
    sched.record(_ok("a", add_subtasks=[graft]))
    assert [n.id for n in sched.ready()] == ["a.child"]
    sched.record(_ok("a.child"))
    assert sched.finished


def test_agent_proposed_cancel_skips_the_branch():
    sched = WorkboardScheduler(_wb(_node("a"), _node("b"), _node("c", ["b"])))
    sched.ready()
    sched.record(_ok("a", cancel=["b"]))
    # b cancelled → c (depends on b) skips; only a ran.
    assert sched.finished
    summary = sched.summary()
    assert summary["cancelled"] == ["b"]
    assert summary["skipped"] == ["c"]
    assert summary["done"] == ["a"]


# --------------------------------------------------------------------------- #
# A2A protocol glue                                                            #
# --------------------------------------------------------------------------- #
def test_build_node_request_stamps_identity_and_payload():
    node = WorkNode(id="n1", agent="http://a.local", instruction="hi", payload={"k": "v"})
    msg = build_node_request(node, workboard_id="wb1")
    assert msg.task_id == "n1"
    meta = metadata_dict(msg)
    assert meta["workboard.id"] == "wb1" and meta["workboard.node"] == "n1"
    assert {"k": "v"} in get_message_data(msg)


def test_extract_reads_explicit_workboard_update():
    graft = WorkNode(id="child", agent="http://a.local")
    reply = board_update_message(
        NodeOutcome(status="ok", result={"n": 1}), add_subtasks=[graft], cancel=["x"]
    )
    update = extract_board_update(reply)
    assert update.outcome.ok and update.outcome.result == {"n": 1}
    assert [s.id for s in update.add_subtasks] == ["child"]
    assert update.cancel == ["x"]


def test_extract_falls_back_to_status_for_plain_agents():
    # A non-workboard-aware agent (e.g. a TaskAgent) replies with a task_result.
    reply = data_message(
        {"type": "task_result", "status": "error", "reason": "nope"},
        role=Role.ROLE_AGENT,
        metadata={"status": "error"},
    )
    update = extract_board_update(reply)
    assert not update.outcome.ok
    assert update.outcome.reason == "nope"
    assert update.add_subtasks == [] and update.cancel == []


def test_board_update_message_roundtrips_through_extract():
    original = BoardUpdate(
        outcome=NodeOutcome(status="ok"),
        add_subtasks=[WorkNode(id="c", agent="http://a.local", depends_on=["p"])],
        cancel=["z"],
    )
    reply = board_update_message(
        original.outcome, add_subtasks=original.add_subtasks, cancel=original.cancel
    )
    assert extract_board_update(reply).to_dict() == original.to_dict()


# --------------------------------------------------------------------------- #
# Node runner (scripted send — no network)                                     #
# --------------------------------------------------------------------------- #
def test_run_node_returns_ok_outcome_and_proposals():
    graft = WorkNode(id="child", agent="http://a.local")

    async def send(target, message):
        assert target == "http://a.local"
        return board_update_message(NodeOutcome(status="ok"), add_subtasks=[graft])

    result = asyncio.run(run_node(_node("a"), send=send))
    assert result.ok
    assert [s.id for s in result.add_subtasks] == ["child"]


def test_run_node_surfaces_error_outcome_as_data_not_exception():
    async def send(target, message):
        return board_update_message(NodeOutcome(status="error", reason="boom"))

    result = asyncio.run(run_node(_node("a"), send=send))
    assert not result.ok and result.outcome.reason == "boom"


def test_run_node_forwards_remote_cancel_then_reraises(monkeypatch):
    cancels: list[tuple] = []

    async def fake_cancel(target, task_id):
        cancels.append((target, task_id))
        return True

    monkeypatch.setattr(runner_mod, "request_remote_cancel", fake_cancel)

    async def send(target, message):
        raise asyncio.CancelledError

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await run_node(_node("a"), send=send)

    asyncio.run(scenario())
    assert cancels == [("http://a.local", "a")]


# --------------------------------------------------------------------------- #
# Cooperative cancellation                                                     #
# --------------------------------------------------------------------------- #
def test_cancel_token_raises_only_after_cancel():
    token = CancelToken()
    token.raise_if_cancelled()  # no-op
    token.cancel()
    assert token.cancelled
    with pytest.raises(TaskCancelledError):
        token.raise_if_cancelled()


class _Ctx:
    def __init__(self, message):
        self.message = message
        self.task_id = message.task_id


class _Queue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def test_executor_binds_token_and_cancel_signals_the_handler():
    seen: dict = {}

    async def handler(message):
        token = current_cancel_token()
        seen["bound"] = token is not None
        await token.wait()  # block until the board cancels us
        seen["cancelled"] = token.cancelled
        return text_message("stopped")

    async def scenario():
        executor = CancellableAgentExecutor(handler)
        ctx = _Ctx(text_message("go", role=Role.ROLE_USER, task_id="n1"))
        queue = _Queue()
        exec_task = asyncio.create_task(executor.execute(ctx, queue))
        await asyncio.sleep(0)  # let execute register the token
        await executor.cancel(ctx, queue)
        await exec_task
        return queue

    queue = asyncio.run(scenario())
    assert seen == {"bound": True, "cancelled": True}
    assert len(queue.events) == 1  # handler still produced its reply
