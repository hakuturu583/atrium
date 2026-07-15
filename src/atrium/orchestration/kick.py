"""Kick a workboard from Atrium's control plane, and read its state back.

This is the API surface "Atrium本体" calls: hand a :class:`Workboard` to Prefect
and get a flow-run id whose progress is then visible in the Prefect UI and
queryable here. Two ways in:

* :func:`submit_workboard` — create a run against a registered deployment on the
  Prefect **server** (fire-and-forget; server-tracked, UI-visible). The
  production path.
* :func:`run_workboard_local` — run the flow in-process to completion and return
  its result. For dev / smoke tests without a server.

The server path needs a configured ``PREFECT_API_URL`` pointing at the Prefect
server (see the ``docker-compose`` stack).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from prefect.client.orchestration import get_client
from prefect.deployments import run_deployment

from atrium.orchestration.flow import WORKBOARD_FLOW_NAME, build_workboard_flow
from atrium.orchestration.types import Workboard

logger = logging.getLogger("atrium.orchestration.kick")

__all__ = ["submit_job", "submit_workboard", "workboard_state", "run_workboard_local"]


async def submit_job(
    agent: str,
    instruction: str = "",
    *,
    payload: Optional[dict[str, Any]] = None,
    job_id: Optional[str] = None,
    deployment: str = "default",
    context_id: Optional[str] = None,
) -> str:
    """Submit a single-agent job and return its flow-run id.

    The ergonomic entry point for the common "run one agent" case: it builds a
    one-node workboard (:meth:`Workboard.single`) and kicks it through the same
    path as any DAG, so a trivial job is server-tracked and UI-visible like the
    rest. For multi-step jobs, build a :class:`Workboard` and use
    :func:`submit_workboard`.
    """
    workboard = Workboard.single(
        agent, instruction, id=job_id or f"job-{uuid.uuid4().hex[:12]}", payload=payload
    )
    return await submit_workboard(workboard, deployment=deployment, context_id=context_id)


async def submit_workboard(
    workboard: Workboard,
    *,
    deployment: str = "default",
    context_id: Optional[str] = None,
) -> str:
    """Create a server-tracked run of ``workboard`` and return its flow-run id.

    Targets the deployment ``atrium-workboard/<deployment>`` served by the worker
    (:func:`atrium.orchestration.flow.serve_workboards`). Returns immediately
    after scheduling; poll :func:`workboard_state` for progress or watch the UI.
    """
    workboard.validate()
    run = await run_deployment(
        name=f"{WORKBOARD_FLOW_NAME}/{deployment}",
        parameters={"workboard": workboard.to_dict(), "context_id": context_id},
        timeout=0,  # fire-and-forget: return the scheduled run, don't await it
    )
    logger.info("submitted workboard %s as flow run %s", workboard.id, run.id)
    return str(run.id)


async def workboard_state(flow_run_id: str) -> dict[str, Any]:
    """Read a run's coarse state: ``{id, name, state, done}`` (for polling)."""
    async with get_client() as client:
        run = await client.read_flow_run(flow_run_id)
    state = run.state
    return {
        "id": str(run.id),
        "name": run.name,
        "state": getattr(state, "type", None).value if state and state.type else None,
        "done": bool(state and state.is_final()),
    }


async def run_workboard_local(
    workboard: Workboard, *, context_id: Optional[str] = None
) -> dict[str, Any]:
    """Run ``workboard`` to completion in-process and return the flow result.

    No Prefect server required — the flow executes locally (still recorded by any
    configured Prefect backend). Handy for a smoke run of a DAG end to end.
    """
    workboard.validate()
    flow = build_workboard_flow()
    return await flow(workboard.to_dict(), context_id=context_id)
