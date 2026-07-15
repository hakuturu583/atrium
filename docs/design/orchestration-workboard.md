# Design: Workboard — a Task-Dependency DAG on Prefect

> Status: **IMPLEMENTED (core) / INTEGRATION-PENDING (deployment)** — the
> Prefect-free core (`atrium.orchestration.types` / `.protocol` / `.runner` /
> `.scheduler` / `.cancel`) is in code and unit-tested (`tests/test_orchestration.py`);
> the Prefect adapter (`.flow` / `.kick` / `.serve`) runs against a live Prefect
> (validated with a patched runner) and is wired into the `docker-compose` stack.
> **Prefect is a core dependency and the job-execution entry point** — not an
> optional add-on. Per-piece status at the end.

## Problem
Atrium's agents are 1:1 with a throwaway sandbox, isolated, and stateless between
A2A requests — by design. There is deliberately **no shared task board** in the
core: multi-agent work is choreographed by A2A message passing, not a central
tracker. That is the right default for isolation, but it leaves three gaps once
work spans *many dependent* agent calls:

1. **Dependencies** — express "run B and C after A, D after both" as a DAG.
2. **A kickable, observable run** — start a unit of work over an API and *see*
   its state (which nodes ran, failed, are pending) without bespoke plumbing.
3. **A place for dynamic structure** — an agent that discovers subtasks mid-run
   needs somewhere to add them; a run that goes moot needs cancellation.

## Approach: the workboard is a Prefect flow
A **workboard** is a DAG of `WorkNode`s; each node is one A2A dispatch to an
agent, and `depends_on` wires the edges. We run it as a **Prefect flow**: each
node is a Prefect `@task`, so the whole run is kicked over the Prefect API and
its state renders in the Prefect UI for free. Prefect (server + a worker) is
**fixed infrastructure** — the same trust tier as the registry ledger and the
Morpher, never an evolvable agent.

```
[Atrium control plane] --REST--> [Prefect Server] <---> [Prefect Worker]   ← trusted / fixed infra
   submit_workboard()             (workboard state,       │ each node = 1 A2A dispatch
   → flow run id; poll/UI          Postgres-backed)       ▼
                                                    [Atrium agents]         ← isolated sandboxes (A2A only)
```

Why Prefect rather than a bespoke store: the "record dependencies + kick over an
API + visualize state + retry/cancel" surface is exactly a workflow engine's job,
and Prefect's task/flow state model *is* the board state — no second source of
truth to keep consistent.

## The one hard rule: the orchestrator is the sole writer of board state
Agents **never** mutate the board directly. They get no Prefect credentials and
no board write access — code-authoring/task agents have the highest
prompt-injection exposure in the system, so handing them authority over "what
runs next" would be the same mistake the Morpher exists to prevent. Instead:

> **Agent proposes, trusted worker disposes.** An agent returns its outcome — and,
> optionally, *proposed* board mutations — as structured data in its A2A reply
> (`board_update_message`). The trusted worker (`WorkboardScheduler`) is what
> actually applies them.

This is the exact shape of "TaskAgent authors a generation but only the Morpher
promotes it" (`agent-versioning.md`), reused for board state.

### The three board operations, by direction
| Operation | Direction | Mechanism | New machinery |
|---|---|---|---|
| **Completion** | agent → board | node's Prefect task returns → Prefect marks it `Completed` | none — free |
| **Failure** | agent → board | agent replies `error`; runner returns a not-`ok` `NodeResult`; dependents cascade to `skipped` | none — free |
| **Add subtask** | agent → board | reply carries `add_subtasks: [WorkNode]`; scheduler grafts them | proposal schema |
| **Cancel others** | agent → board | reply carries `cancel: [node_id]`; scheduler marks them cancelled | proposal schema (same) |
| **Cancel a *running* node** | **board → agent** | cooperative A2A cancel (below) | the only real protocol addition |

Completion and failure need **no** new machinery — they *are* the Prefect task
lifecycle. The proposal channel (`workboard_update` data part) covers subtask
grafting and cancel-requests uniformly. The genuinely new thing is cancelling a
node that is already executing inside an isolated agent.

## Cooperative cancellation (board → agent)
Cancelling in-flight work is the one signal that flows *into* a running agent, so
it gets an explicit, cooperative protocol (an agent is asked to stop; it is never
force-killed by the board — that authority stays with the sandbox lifecycle):

- **Host side** (`runner.run_node`): when Prefect cancels the node's task, the
  runner catches `CancelledError`, sends a best-effort A2A cancel naming the
  node's `task_id` (`request_remote_cancel`), then re-raises so the cancel unwinds.
- **Agent side** (`cancel.CancellableAgentExecutor`, in the in-sandbox bridge):
  maps that A2A cancel onto a `CancelToken` bound into a context var around the
  handler. A long-running handler polls `raise_if_cancelled()` (or awaits
  `token.wait()`); a single-shot handler that never checks simply finishes — the
  cancel is a no-op for it, never a crash.

Race handling: **Prefect is authoritative.** If an agent's completion and a
cancel cross, the flow's recorded state wins; a late outcome for an
already-cancelled node is discarded by the scheduler (the node is terminal).

## Execution model
`WorkboardScheduler` is a **pure, backend-free state machine** — all the ordering
logic lives here so it is unit-testable without Prefect. A node moves
pending → running (the driver's concern) → one terminal state: `done` (ok),
`failed` (agent said error), `cancelled` (proposed cancel), or `skipped` (an
upstream dep did not pass — cascades transitively). The Prefect adapter
(`flow.build_workboard_flow`) is a thin driver: ask `ready()` for the runnable
nodes, run each as a Prefect task, fold each `NodeResult` back via `record()`
(which applies grafts/cancels), repeat until `finished`. Execution is wave-based;
finer-grained pipelining across waves is a possible refinement.

Any existing agent is a valid node with **no changes**: if it doesn't speak the
`workboard_update` protocol, `extract_board_update` falls back to its reply
`status` (e.g. a `TaskAgent`'s `task_result`) and treats it as a leaf outcome.

## Kicking a run ("API経由でキック")
Prefect is *the* job-execution entry point — every top-level job runs as a
workboard, so there is one submission path, not a "single call vs DAG" split:
- `submit_job(agent, instruction)` → the single-agent case: builds a **one-node
  workboard** (`Workboard.single`, the degenerate DAG) and kicks it, so a trivial
  job is server-tracked and UI-visible like any other. Thin sugar over the below.
- `submit_workboard(workboard, deployment=...)` → `run_deployment(...)` on the
  Prefect **server**: fire-and-forget, returns a flow-run id, server-tracked and
  UI-visible. The production path for Atrium's control plane.
- `workboard_state(flow_run_id)` → coarse `{state, done}` for polling.
- `run_workboard_local(workboard)` → run the flow in-process to completion
  (no server) for dev/smoke.

The `prefect-worker` compose service runs `python -m atrium.orchestration.serve`,
which registers the `atrium-workboard` flow as a deployment and executes its runs.

## Trust & isolation notes
- Prefect server/worker/DB are fixed infra; agents reach the worker only as
  *callees* over A2A (the worker holds the A2A reach, on the host-local control
  LAN — agents on `INTERNAL` can be dispatched to but hold no board access).
- The worker is the privileged component (board writes, agent dispatch, and — if
  a flow ever promotes — the Morpher call). Keep Prefect API credentials out of
  agent sandboxes.
- Board runs ship OpenTelemetry spans to the same Phoenix (W3C `traceparent`
  injected into each node's A2A message), so a workboard run and the cross-agent
  A2A chain it drives render as one stitched timeline (`observability.md`).

## Per-piece status
- **DONE** — `types` (DAG value objects + validation), `protocol` (A2A request /
  reply glue, back-compat fallback), `scheduler` (deps / failure / cancel /
  subtask-graft state machine), `runner` (A2A node execution + cooperative-cancel
  forwarding), `cancel` (token, `CancellableAgentExecutor`, best-effort remote
  cancel). Unit-tested in `tests/test_orchestration.py`.
- **DONE** — `flow` (Prefect adapter), `kick` (submit/state/local), `serve`
  (worker entrypoint); `docker-compose` Prefect stack (server + Postgres +
  worker); `prefect` is a core dependency.
- **TODO** — finalize the exact A2A cancel wire call against the deployed bridge
  server (`request_remote_cancel` resolves the SDK cancel method dynamically
  today); optional cross-wave pipelining; a `WorkboardAgent` A2A front door if
  agents should be able to *submit* whole workboards (still via proposal → worker).
```
