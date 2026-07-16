"""Prefect adapter — the workboard DAG *as* a Prefect flow.

The flow is a thin driver over :class:`WorkboardScheduler`: each ready node runs
as a Prefect ``@task`` (so it shows up as its own state box in the Prefect UI —
the "状態可視化"), the resulting :class:`NodeResult` is folded back into the
scheduler (applying any grafted subtasks / proposed cancels), and the loop
repeats until every node is terminal. All the ordering logic lives in the
Prefect-free scheduler; this layer only maps it onto Prefect tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from prefect import flow, task

from atrium.orchestration.review import ReviewPolicy, default_review_policy
from atrium.orchestration.runner import run_node
from atrium.orchestration.scheduler import WorkboardScheduler
from atrium.orchestration.types import (
    STATUS_ERROR,
    NodeOutcome,
    NodeResult,
    WorkNode,
    Workboard,
)

logger = logging.getLogger("atrium.orchestration.flow")

__all__ = ["WORKBOARD_FLOW_NAME", "build_workboard_flow", "serve_workboards"]

#: Flow name; the ``<flow>/<deployment>`` a kick targets (see :mod:`.kick`).
WORKBOARD_FLOW_NAME = "atrium-workboard"

#: Backstop against a runaway self-grafting workboard (agents proposing subtasks
#: that propose subtasks…). Generous; a real DAG settles in far fewer waves.
_MAX_WAVES = 1000


def build_workboard_flow() -> Any:
    """Construct (and return) the Prefect ``@flow`` that runs a workboard.

    Built behind a function rather than at module scope so the flow/task objects
    are created lazily on first use (and re-created cleanly per serving process),
    while the ``prefect`` import itself is a plain top-level dependency.
    """

    @task(task_run_name="node:{node_dict[id]}")
    async def _node_task(
        node_dict: dict[str, Any],
        workboard_id: str,
        context_id: Optional[str],
        review_dict: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one node over A2A (review-gated); its Prefect state reflects the fate."""
        node = WorkNode.from_dict(node_dict)
        result = await run_node(
            node,
            workboard_id=workboard_id,
            context_id=context_id,
            review=ReviewPolicy.from_dict(review_dict),
        )
        return result.to_dict()

    @flow(name=WORKBOARD_FLOW_NAME)
    async def workboard_flow(
        workboard: dict[str, Any],
        context_id: Optional[str] = None,
        review: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        wb = Workboard.from_dict(workboard)
        sched = WorkboardScheduler(wb)

        # Explicit per-run policy wins; otherwise fall back to the deployment
        # default (``ATRIUM_REVIEWER`` env) so the gate can be always-on.
        policy = ReviewPolicy.from_dict(review) or default_review_policy()
        review_dict = policy.to_dict() if policy else None

        waves = 0
        while not sched.finished and waves < _MAX_WAVES:
            wave = sched.ready()
            if not wave:
                # Nothing runnable but not finished → the rest is blocked by
                # failed/cancelled upstreams; let the scheduler settle & stop.
                break
            waves += 1
            futures = {
                node.id: _node_task.submit(node.to_dict(), wb.id, context_id, review_dict)
                for node in wave
            }
            for node_id, future in futures.items():
                sched.record(_resolve(node_id, future))

        return {
            "workboard": wb.id,
            "summary": sched.summary(),
            "results": {nid: r.to_dict() for nid, r in sched.results.items()},
        }

    return workboard_flow


def _resolve(node_id: str, future: Any) -> NodeResult:
    """Await a node task's future, turning a task crash into an ``error`` outcome.

    A transport/infra failure surfaces as the Prefect task failing; we still fold
    it back as a failed :class:`NodeResult` so the scheduler cascades the skip to
    dependents instead of hanging.
    """
    try:
        return NodeResult.from_dict(future.result())
    except Exception as exc:  # noqa: BLE001 - map any task failure to a node failure
        logger.warning("node %s task failed: %s", node_id, exc)
        return NodeResult(
            node_id=node_id,
            outcome=NodeOutcome(status=STATUS_ERROR, reason=str(exc)),
        )


def serve_workboards(name: str = "default", **serve_kwargs: Any) -> None:
    """Serve the workboard flow as a Prefect deployment (blocking).

    This is what the ``prefect-worker`` service runs: it registers a deployment
    named ``atrium-workboard/<name>`` with the Prefect server and executes runs
    created against it (e.g. by :func:`atrium.orchestration.kick.submit_workboard`).
    """
    build_workboard_flow().serve(name=name, **serve_kwargs)
