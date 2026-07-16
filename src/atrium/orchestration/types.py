"""Value objects for the workboard — the task-dependency DAG.

A *workboard* is a DAG of :class:`WorkNode` s: each node is one unit of work
dispatched to an Atrium agent over A2A, and ``depends_on`` wires the edges. The
workboard is the **input** an operator (Atrium's trusted control plane) hands to
the orchestrator; the *live state* of a run lives in Prefect (see
``docs/design/orchestration-workboard.md``).

Everything here is pure data — no Prefect, no A2A, no I/O — so the shapes are
trivially serializable (``to_dict`` / ``from_dict``, used at the Prefect
parameter boundary) and unit-testable without any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "WorkNode",
    "Workboard",
    "NodeOutcome",
    "BoardUpdate",
    "NodeResult",
    "STATUS_OK",
    "STATUS_ERROR",
    "WorkboardError",
]

#: Node/outcome statuses, matching the rest of the runtime's ``ok``/``error``.
STATUS_OK = "ok"
STATUS_ERROR = "error"


class WorkboardError(ValueError):
    """Raised when a workboard is structurally invalid (bad ids, cycle, dangling dep)."""


@dataclass(slots=True)
class WorkNode:
    """One unit of work in a workboard: an A2A dispatch to a single agent.

    ``agent`` is the dispatch target — an A2A base URL / ``AgentCard`` URL, or a
    bare agent *slug* the runner resolves to the slug's ``:active`` endpoint.
    ``depends_on`` lists node ids that must finish successfully first. A node
    authored dynamically at runtime (an agent's proposed subtask) is the very
    same shape, so there is one node type, not two.
    """

    id: str
    agent: str
    instruction: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    #: A2A ``kind`` stamped on the request metadata (routing hint for the agent).
    kind: str = "task"
    #: Whether this node's completion must pass the review gate (see
    #: :mod:`atrium.orchestration.review`). Set False for a node with no reviewable
    #: deliverable (e.g. a pure fetch/setup step); it then completes on self-report.
    reviewable: bool = True
    #: Optional per-node reviewer A2A target, overriding the run's
    #: :attr:`ReviewPolicy.reviewer` for this node.
    reviewer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "instruction": self.instruction,
            "payload": dict(self.payload),
            "depends_on": list(self.depends_on),
            "kind": self.kind,
            "reviewable": self.reviewable,
            "reviewer": self.reviewer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkNode":
        reviewer = data.get("reviewer")
        return cls(
            id=str(data["id"]),
            agent=str(data["agent"]),
            instruction=str(data.get("instruction", "")),
            payload=dict(data.get("payload") or {}),
            depends_on=[str(d) for d in (data.get("depends_on") or [])],
            kind=str(data.get("kind", "task")),
            reviewable=bool(data.get("reviewable", True)),
            reviewer=str(reviewer) if reviewer else None,
        )


@dataclass(slots=True)
class Workboard:
    """A named DAG of :class:`WorkNode` s handed to the orchestrator to run."""

    id: str
    nodes: list[WorkNode] = field(default_factory=list)

    @classmethod
    def single(
        cls,
        agent: str,
        instruction: str = "",
        *,
        id: str = "job",
        payload: Optional[dict[str, Any]] = None,
    ) -> "Workboard":
        """A one-node workboard: run one agent, no dependencies.

        The degenerate DAG — the canonical shape of a *single-step* job. Routing
        it through the same board machinery as a multi-node one means even a
        trivial "call one agent" job gets a flow-run id, UI visibility, and the
        shared retry / cancel / trace behaviour, with no separate code path.
        """
        node = WorkNode(id="root", agent=agent, instruction=instruction, payload=dict(payload or {}))
        return cls(id=id, nodes=[node])

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def validate(self) -> "Workboard":
        """Check the DAG is well-formed: unique ids, no dangling deps, acyclic.

        Returns ``self`` for chaining; raises :class:`WorkboardError` otherwise.
        """
        seen: set[str] = set()
        for node in self.nodes:
            if not node.id:
                raise WorkboardError("work node has an empty id")
            if node.id in seen:
                raise WorkboardError(f"duplicate node id {node.id!r}")
            seen.add(node.id)
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in seen:
                    raise WorkboardError(
                        f"node {node.id!r} depends on unknown node {dep!r}"
                    )
                if dep == node.id:
                    raise WorkboardError(f"node {node.id!r} depends on itself")
        self._assert_acyclic()
        return self

    def _assert_acyclic(self) -> None:
        """DFS cycle check (raises with the offending node on a back edge)."""
        deps = {n.id: n.depends_on for n in self.nodes}
        WHITE, GREY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in deps}

        def visit(nid: str) -> None:
            color[nid] = GREY
            for dep in deps[nid]:
                if color[dep] == GREY:
                    raise WorkboardError(f"workboard has a cycle through node {dep!r}")
                if color[dep] == WHITE:
                    visit(dep)
            color[nid] = BLACK

        for nid in deps:
            if color[nid] == WHITE:
                visit(nid)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workboard":
        return cls(
            id=str(data["id"]),
            nodes=[WorkNode.from_dict(n) for n in (data.get("nodes") or [])],
        )


@dataclass(slots=True)
class NodeOutcome:
    """An agent's verdict on one node: a status plus its (opaque) result data."""

    status: str = STATUS_OK
    result: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "result": dict(self.result), "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeOutcome":
        return cls(
            status=str(data.get("status", STATUS_OK)),
            result=dict(data.get("result") or {}),
            reason=data.get("reason"),
        )


@dataclass(slots=True)
class BoardUpdate:
    """What an agent *proposes* back to the board after handling a node.

    The heart of the "agent proposes, trusted worker disposes" model: an agent
    never writes the board directly. It returns an outcome plus, optionally, new
    subtasks to graft into the DAG (``add_subtasks``) and/or node ids it thinks
    should be cancelled (``cancel``). The orchestrator — the sole writer — is
    what actually applies these. See ``docs/design/orchestration-workboard.md``.
    """

    outcome: NodeOutcome = field(default_factory=NodeOutcome)
    add_subtasks: list[WorkNode] = field(default_factory=list)
    cancel: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.to_dict(),
            "add_subtasks": [n.to_dict() for n in self.add_subtasks],
            "cancel": list(self.cancel),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoardUpdate":
        return cls(
            outcome=NodeOutcome.from_dict(data.get("outcome") or {}),
            add_subtasks=[WorkNode.from_dict(n) for n in (data.get("add_subtasks") or [])],
            cancel=[str(c) for c in (data.get("cancel") or [])],
        )


@dataclass(slots=True)
class NodeResult:
    """The runner's record of executing one node: outcome + proposed mutations."""

    node_id: str
    outcome: NodeOutcome = field(default_factory=NodeOutcome)
    add_subtasks: list[WorkNode] = field(default_factory=list)
    cancel: list[str] = field(default_factory=list)
    reply_text: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "outcome": self.outcome.to_dict(),
            "add_subtasks": [n.to_dict() for n in self.add_subtasks],
            "cancel": list(self.cancel),
            "reply_text": self.reply_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeResult":
        return cls(
            node_id=str(data["node_id"]),
            outcome=NodeOutcome.from_dict(data.get("outcome") or {}),
            add_subtasks=[WorkNode.from_dict(n) for n in (data.get("add_subtasks") or [])],
            cancel=[str(c) for c in (data.get("cancel") or [])],
            reply_text=str(data.get("reply_text", "")),
        )
