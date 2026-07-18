"""The :class:`Job` — a request paired with the generated flow that runs it.

A *job* is the unit that becomes runnable once a **request JSON + a generated
Prefect ``flow.py``** are both present and valid. The ``flow.py`` is not the work
itself: it is an *agent-dispatch orchestration* — a Prefect DAG whose tasks each
assign a piece of work to a subagent (a role-bearing inference agent) via the
trusted ``atrium_dispatch`` primitive. Submitting the job runs that flow inside a
minimal-privilege executor sandbox, and the subagents do the actual work.

This module is pure, Prefect-free data (mirroring :mod:`atrium.orchestration.types`),
so a job is trivially serializable and unit-testable without any backend:

* :class:`Job` bundles the request, the generated flow source and its params, and
  gates readiness (:meth:`Job.is_ready`) behind *both* artifacts being present and
  the flow source being syntactically valid.
* :func:`build_execution_workboard` turns a ready job into the degenerate DAG the
  trusted ``atrium-workboard`` flow runs: one review-gated node that executes the
  generated ``flow.py`` on the minimal-privilege runner. The generated flow never
  runs in the trusted Prefect worker — only as a node's sandboxed work.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from atrium.orchestration.types import WorkNode, Workboard

__all__ = [
    "Job",
    "JobNotReadyError",
    "build_execution_workboard",
    "unsupported_requirements",
    "DEFAULT_EXECUTOR_AGENT",
]

#: The minimal-privilege executor a generated flow runs on (its ``:active`` slug).
DEFAULT_EXECUTOR_AGENT = "prefect_runner_agent:active"

#: Filenames staged into the executor sandbox for a run.
FLOW_FILENAME = "flow.py"
PARAMS_FILENAME = "params.json"

#: The command the executor runs to start the generated flow (reads ``params.json``).
RUN_COMMAND = f"python {FLOW_FILENAME}"

#: The entrypoint the planner is contracted to define (see the ``planner`` profile).
ENTRYPOINT_NAME = "main"

#: A generous upper bound on generated flow source (chars) — a real plan is far
#: smaller; the cap only stops a runaway/degenerate generation from being staged.
_MAX_FLOW_SOURCE_CHARS = 200_000


class JobNotReadyError(ValueError):
    """Raised when a :class:`Job` cannot be run (missing/invalid artifact)."""


@dataclass(slots=True)
class Job:
    """A human request paired with the generated flow that will fulfil it.

    ``request`` is the normalized human ask (JSON). ``flow_source`` is the
    LLM-generated Prefect ``flow.py`` (an agent-dispatch DAG), and ``params`` its
    inputs — together the "JSON + Python-script pair" that must both be present
    before the job is ready. ``requirements`` lists any extra libraries the flow
    declares it needs (all of which must already be available in the executor
    image — the runner is WAN-isolated). ``plan_reason`` is the planner's
    free-text rationale, kept for tracing/debugging.
    """

    id: str
    request: dict[str, Any] = field(default_factory=dict)
    flow_source: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)
    plan_reason: str = ""

    def is_ready(self) -> bool:
        """Whether both artifacts are present and the flow source is well-formed.

        The readiness gate: a job never runs until the request + generated flow
        pair is complete *and* the flow parses. Deliberately does not attempt to
        judge what the flow *does* — that is the review gate's and the sandbox's
        job, not a static check's (see :mod:`atrium.orchestration.review`).
        """
        try:
            self.static_check()
        except JobNotReadyError:
            return False
        return True

    def static_check(self) -> None:
        """Validate the pair, raising :class:`JobNotReadyError` on any problem.

        Checks are intentionally light: the request is present, the flow source is
        present, within a sane size, *syntactically valid Python* (parsed with
        :func:`ast.parse` — **never executed**), and defines the contracted
        ``main`` entrypoint the executor runs. The real safety is the
        minimal-privilege sandbox plus the review gate, not this check — so it does
        not attempt to judge what the flow *does*.
        """
        if not self.request:
            raise JobNotReadyError("job has no request")
        source = self.flow_source or ""
        if not source.strip():
            raise JobNotReadyError("job has no flow_source")
        if len(source) > _MAX_FLOW_SOURCE_CHARS:
            raise JobNotReadyError(
                f"flow_source too large ({len(source)} > {_MAX_FLOW_SOURCE_CHARS} chars)"
            )
        try:
            tree = ast.parse(source, filename=f"<{FLOW_FILENAME}>")  # parse only, never exec
        except SyntaxError as exc:
            raise JobNotReadyError(f"flow_source is not valid Python: {exc}") from exc
        if not _defines_entrypoint(tree):
            raise JobNotReadyError(
                f"flow_source defines no {ENTRYPOINT_NAME!r} entrypoint (a top-level "
                f"'def {ENTRYPOINT_NAME}' the runner can invoke)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": dict(self.request),
            "flow_source": self.flow_source,
            "params": dict(self.params),
            "requirements": list(self.requirements),
            "plan_reason": self.plan_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(
            id=str(data["id"]),
            request=dict(data.get("request") or {}),
            flow_source=str(data.get("flow_source") or ""),
            params=dict(data.get("params") or {}),
            requirements=[str(r) for r in (data.get("requirements") or [])],
            plan_reason=str(data.get("plan_reason") or ""),
        )


def _defines_entrypoint(tree: ast.Module) -> bool:
    """Whether ``tree`` defines a top-level ``main`` (sync or async) function.

    A decorator (``@flow``) leaves the function name unchanged, so a plain scan of
    the module body for a ``def``/``async def`` named :data:`ENTRYPOINT_NAME` is
    enough — the executor runs it via ``python flow.py``.
    """
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == ENTRYPOINT_NAME:
            return True
    return False


#: Id of the optional pre-execution source-review node (Phase 2).
REVIEW_NODE_ID = "review_source"
#: Id of the node that executes the generated flow.
RUN_NODE_ID = "run_flow"


#: First PEP-508 version/extra/marker delimiter — everything after it is stripped.
_REQ_DELIM_RE = re.compile(r"[\[=<>~!;\s]")


def _requirement_name(spec: str) -> str:
    """Reduce a requirement spec to its bare package name for allow-list matching.

    ``prefect==3.7.8`` / ``prefect[extra]`` / ``prefect>=3`` → ``prefect``: split on
    the first version/extra/marker delimiter (order-independent), lowercased.
    """
    return _REQ_DELIM_RE.split(spec.strip(), 1)[0].lower()


def unsupported_requirements(requirements: list[str], allowed: Optional[list[str]]) -> list[str]:
    """Return the requirements not covered by ``allowed`` (the runner's preinstalled set).

    The runner is WAN-isolated, so a generated flow can only import what the image
    already ships. When a deployment declares that allow-list (via the planner
    constraints), this flags any declared requirement outside it so the job is
    rejected *before* it runs rather than failing with an import error mid-flight.
    An empty/omitted ``allowed`` means "no allow-list configured" → nothing is
    flagged (the runtime import remains the backstop).
    """
    if not allowed:
        return []
    allowed_names = {_requirement_name(a) for a in allowed}
    return [r for r in requirements if _requirement_name(r) not in allowed_names]


def build_execution_workboard(
    job: Job,
    *,
    executor_agent: str = DEFAULT_EXECUTOR_AGENT,
    reviewer_agent: Optional[str] = None,
) -> Workboard:
    """Turn a ready ``job`` into the DAG the trusted ``atrium-workboard`` flow runs.

    The generated flow always runs as the ``run_flow`` node — the minimal-privilege
    executor stages ``flow.py`` + ``params.json`` into its sandbox and runs it (the
    ``{files, commands}`` shape ``CodeWorkSpaceAgent`` — which
    :class:`PrefectRunnerAgent` extends — already handles), so the flow executes
    only as sandboxed node work, never in the trusted worker. ``run_flow`` is
    ``reviewable`` so ``run_node``'s run-level gate reviews its *result* after it
    runs.

    When ``reviewer_agent`` is given, a **pre-execution source review** is prepended
    (Phase 2): a ``review_source`` node dispatches the generated ``flow.py`` to the
    reviewer, whose verdict is that node's outcome; ``run_flow`` ``depends_on`` it,
    so a rejected flow is never executed (the scheduler cascades the skip). Without
    a ``reviewer_agent`` the board is just the single ``run_flow`` node.

    Raises :class:`JobNotReadyError` if ``job`` is not ready — a not-ready job must
    never reach a workboard.
    """
    job.static_check()
    depends_on: list[str] = []
    nodes: list[WorkNode] = []

    if reviewer_agent:
        # The reviewer *is* this node's doer: its verdict (an ok/error reply the
        # scheduler reads via extract_board_update) becomes the outcome, so the node
        # is not itself reviewable. reviewer_role reads instruction + deliverable.
        nodes.append(
            WorkNode(
                id=REVIEW_NODE_ID,
                agent=reviewer_agent,
                instruction="Review the generated Prefect flow before it runs.",
                payload={
                    "instruction": (
                        "Review this generated Prefect flow for correctness and safety "
                        "before it is executed. It must define a single 'main' entrypoint, "
                        "dispatch to subagents only via atrium_dispatch, and attempt no "
                        "network egress or unsafe operations."
                    ),
                    "deliverable": job.flow_source,
                },
                kind="review",
                reviewable=False,
            )
        )
        depends_on = [REVIEW_NODE_ID]

    nodes.append(
        WorkNode(
            id=RUN_NODE_ID,
            agent=executor_agent,
            instruction=job.plan_reason or "run the generated Prefect flow",
            payload={
                "files": {
                    FLOW_FILENAME: job.flow_source,
                    PARAMS_FILENAME: json.dumps(job.params),
                },
                "commands": [RUN_COMMAND],
            },
            depends_on=depends_on,
            kind="task",
            reviewable=True,
        )
    )
    return Workboard(id=job.id, nodes=nodes).validate()
