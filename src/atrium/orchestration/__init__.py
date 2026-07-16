"""Orchestration — the workboard: a task-dependency DAG run on Prefect.

Atrium's agents are deliberately isolated and stateless between requests; there
is no shared task board in the core runtime. This package adds one *as fixed
infrastructure* (the same trust tier as the registry and the Morpher): a
workboard is a DAG of :class:`WorkNode` s, each an A2A dispatch to an agent, run
by Prefect so runs are kicked over an API and their state is visible in the UI.

Design (see ``docs/design/orchestration-workboard.md``):

* **The orchestrator is the sole writer of board state.** Agents never mutate the
  board directly (they get no Prefect credentials — code-authoring agents have the
  highest prompt-injection exposure). An agent *proposes* changes — new subtasks,
  cancellations — in its A2A reply (:func:`board_update_message`); the trusted
  worker disposes (:class:`WorkboardScheduler`). This mirrors "TaskAgent authors,
  Morpher promotes".
* **Completion/failure are free**: a node's Prefect task returning / the agent
  replying ``error`` *is* the state transition. **Cancellation** is the one signal
  that flows board→agent, so it has an explicit cooperative protocol
  (:mod:`atrium.orchestration.cancel`).

Prefect is the job-execution entry point and a **mandatory** dependency: a job
runs as a workboard via :func:`submit_workboard` / :func:`submit_job`, and its
state is visible in the Prefect UI. Mandatory-to-install is not the same as
eagerly-imported, though: the Prefect-backed names (``submit_job``, ``flow``,
``serve``, …) are imported lazily on first access so that ``import
atrium.orchestration`` for the Prefect-free core (types, protocol, runner,
scheduler, cancel) stays cheap and the DAG/ordering logic remains unit-testable
without loading Prefect.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from atrium.orchestration.bootstrap import (
    BootstrapConfig,
    OrchestrationBootstrapError,
    OrchestrationEndpoints,
    ensure_orchestration_services,
    stop_orchestration_services,
)
from atrium.orchestration.cancel import (
    CancellableAgentExecutor,
    CancelToken,
    TaskCancelledError,
    current_cancel_token,
    raise_if_cancelled,
    request_remote_cancel,
)
from atrium.orchestration.protocol import (
    WORKBOARD_UPDATE_TYPE,
    board_update_message,
    build_node_request,
    extract_board_update,
)
from atrium.orchestration.review import (
    REVIEW_REQUEST_TYPE,
    ReviewPolicy,
    build_review_request,
    default_review_policy,
)
from atrium.orchestration.runner import run_node
from atrium.orchestration.scheduler import WorkboardScheduler
from atrium.orchestration.types import (
    STATUS_ERROR,
    STATUS_OK,
    BoardUpdate,
    NodeOutcome,
    NodeResult,
    WorkNode,
    Workboard,
    WorkboardError,
)

__all__ = [
    # Value objects
    "WorkNode",
    "Workboard",
    "WorkboardError",
    "NodeOutcome",
    "NodeResult",
    "BoardUpdate",
    "STATUS_OK",
    "STATUS_ERROR",
    # A2A protocol glue
    "WORKBOARD_UPDATE_TYPE",
    "build_node_request",
    "extract_board_update",
    "board_update_message",
    # Review gate
    "REVIEW_REQUEST_TYPE",
    "ReviewPolicy",
    "build_review_request",
    "default_review_policy",
    # Execution
    "run_node",
    "WorkboardScheduler",
    # Cancellation
    "CancelToken",
    "TaskCancelledError",
    "current_cancel_token",
    "raise_if_cancelled",
    "CancellableAgentExecutor",
    "request_remote_cancel",
    # Dependency-service bootstrap (drives docker-compose; Prefect-free)
    "ensure_orchestration_services",
    "stop_orchestration_services",
    "BootstrapConfig",
    "OrchestrationEndpoints",
    "OrchestrationBootstrapError",
    # Prefect-backed job entry point (imported lazily — see module docstring)
    "WORKBOARD_FLOW_NAME",
    "build_workboard_flow",
    "serve_workboards",
    "submit_job",
    "submit_workboard",
    "workboard_state",
    "run_workboard_local",
]

# Prefect is mandatory to install but heavy to import, so the Prefect-backed
# entry points load on first access — keeping the Prefect-free core cheap to
# import and testable without a backend.
_LAZY = {
    "WORKBOARD_FLOW_NAME": "atrium.orchestration.flow",
    "build_workboard_flow": "atrium.orchestration.flow",
    "serve_workboards": "atrium.orchestration.flow",
    "submit_job": "atrium.orchestration.kick",
    "submit_workboard": "atrium.orchestration.kick",
    "workboard_state": "atrium.orchestration.kick",
    "run_workboard_local": "atrium.orchestration.kick",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
