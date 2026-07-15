"""The workboard scheduler — a pure DAG state machine (no Prefect, no I/O).

Deliberately backend-free so the *ordering* logic — dependency readiness,
dynamic subtask grafting, failure/cancel cascades — is unit-testable on its own.
The Prefect adapter (:mod:`atrium.orchestration.flow`) is a thin driver: ask
:meth:`ready` for the nodes runnable now, execute each as a Prefect task, feed
each :class:`NodeResult` back via :meth:`record`, repeat until :attr:`finished`.

State a node can be in: pending → running (caller's concern) → one of
``done`` (ok) / ``failed`` (agent said error) / ``cancelled`` (proposed cancel) /
``skipped`` (an upstream dep failed/cancelled/was skipped — it can never run).
"""

from __future__ import annotations

import logging

from atrium.orchestration.types import NodeResult, WorkNode, Workboard

logger = logging.getLogger("atrium.orchestration.scheduler")

__all__ = ["WorkboardScheduler"]


class WorkboardScheduler:
    """Drives a :class:`Workboard` to completion, one wave of ready nodes at a time.

    Not thread-safe and not concerned with *how* nodes run — only with what may
    run next given what has finished. The driver owns concurrency.
    """

    def __init__(self, workboard: Workboard) -> None:
        workboard.validate()
        self._nodes: dict[str, WorkNode] = {n.id: n for n in workboard.nodes}
        self.results: dict[str, NodeResult] = {}
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.cancelled: set[str] = set()
        self.skipped: set[str] = set()
        self._dispatched: set[str] = set()
        # Set whenever a terminal set changes; guards _settle from re-running the
        # skip cascade on every finished/ready call when nothing has moved.
        self._dirty = True

    # ------------------------------------------------------------------ #
    # Terminal bookkeeping                                                #
    # ------------------------------------------------------------------ #
    @property
    def _terminal(self) -> set[str]:
        return self.done | self.failed | self.cancelled | self.skipped

    @property
    def finished(self) -> bool:
        """True when every known node has reached a terminal state."""
        self._settle()
        return self._terminal >= set(self._nodes)

    def _settle(self) -> None:
        """Cascade: mark as skipped any pending node whose dep is unrunnable.

        Repeated to a fixed point so skips propagate transitively down the DAG.
        A no-op unless a terminal set has changed since the last settle.
        """
        if not self._dirty:
            return
        changed = True
        while changed:
            changed = False
            blocked = self.failed | self.cancelled | self.skipped
            terminal = self.done | blocked
            for nid, node in self._nodes.items():
                if nid in terminal:
                    continue
                if any(dep in blocked for dep in node.depends_on):
                    self.skipped.add(nid)
                    logger.info("skipping node %s: an upstream dependency did not pass", nid)
                    changed = True
        self._dirty = False

    # ------------------------------------------------------------------ #
    # Driver interface                                                    #
    # ------------------------------------------------------------------ #
    def ready(self) -> list[WorkNode]:
        """Nodes runnable right now: all deps done-ok, not already dispatched/terminal.

        Idempotent per node — a node is handed out at most once (tracked in
        ``_dispatched``); call :meth:`record` with its result before it can affect
        downstream readiness.
        """
        self._settle()
        terminal = self._terminal
        out = [
            node
            for nid, node in self._nodes.items()
            if nid not in self._dispatched
            and nid not in terminal
            and all(dep in self.done for dep in node.depends_on)
        ]
        for node in out:
            self._dispatched.add(node.id)
        return out

    def record(self, result: NodeResult) -> None:
        """Fold a finished node's :class:`NodeResult` back into the DAG.

        Applies the node's proposed board mutations — grafting new subtasks and
        marking proposed cancellations — which is where "agent proposes, scheduler
        disposes" actually happens. Unknown-node and duplicate grafts are ignored.
        """
        nid = result.node_id
        self.results[nid] = result
        self._dirty = True
        if result.ok:
            self.done.add(nid)
        else:
            self.failed.add(nid)
            logger.warning("node %s failed: %s", nid, result.outcome.reason)

        for sub in result.add_subtasks:
            if sub.id in self._nodes:
                logger.warning("ignoring proposed subtask %s: id already exists", sub.id)
                continue
            self._nodes[sub.id] = sub
            logger.info("grafted subtask %s (deps=%s) from node %s", sub.id, sub.depends_on, nid)

        for target in result.cancel:
            self.cancel(target)

    def cancel(self, node_id: str) -> None:
        """Mark a not-yet-terminal node cancelled (its dependents then skip)."""
        if node_id in self._nodes and node_id not in self._terminal:
            self.cancelled.add(node_id)
            self._dirty = True
            logger.info("cancelled node %s", node_id)

    def summary(self) -> dict[str, list[str]]:
        """A snapshot of terminal buckets (handy for the flow's return value)."""
        return {
            "done": sorted(self.done),
            "failed": sorted(self.failed),
            "cancelled": sorted(self.cancelled),
            "skipped": sorted(self.skipped),
        }
