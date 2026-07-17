# Design: PlanAgent — request JSON → a concrete Prefect DAG, as review-gated jobs

> Status: **IMPLEMENTED (core + evolvable) / DEPLOYMENT-PENDING (image registration)**.
> The Prefect-free core (`atrium.orchestration.job`, `atrium.agents.plan_agent_protocol`),
> the minimal-privilege executor (`atrium.agents.prefect_runner_agent`), the
> control-plane plan path (`ControlPlaneAgent`), and the evolvable planner
> (`atrium_agents.PlanAgent` + `planner`/`flow_reviewer` roles) are in code and
> unit-tested. The runner image builds and runs a generated flow offline (validated;
> see `scripts/build-prefect-runner.sh`). What remains is environment wiring:
> registering the agents as active generations and building the GPU-bound
> `plan_agent` image. Per-piece status at the end.

## Problem

Before this, a human turn became a **single-node** workboard: the control plane
routed the turn to one doer and kicked it (`submit_job`). There was no planning
step and no notion of a request-specific "program" — a node was just an A2A
dispatch. We want a request (JSON) to be turned into a **concrete Prefect DAG**
that expresses *which subagent does what, in what order*, and for that DAG to
become a runnable **job** only once a JSON + Python pair is complete.

## Approach: an LLM planner drafts a flow; a minimal-privilege runner executes it

```
Human turn ──▶ ControlPlaneAgent           [core, trusted]
                 │  (payload["plan"])  build_plan_request
                 ▼
              PlanAgent                     [atrium_agents, evolvable, LLM]  ── proposes ──▶ flow.py + params
                 │  plan_result
                 ▼
              Job.is_ready()  (both artifacts present, flow parses, main() defined, deps allowed)
                 │  build_execution_workboard
                 ▼
              atrium-workboard flow          [the ONE trusted Prefect flow]
                 │  run_node (review-gated)
                 ▼
              PrefectRunnerAgent             [core, trusted, least privilege, WAN-isolated]
                 │  python flow.py  →  atrium_dispatch(agent, instruction, payload)
                 ▼
              role-bearing InferenceAgents   [the subagents — WAN-isolated, GPU-only, side-effect-free]
```

The generated `flow.py` is **not** the work — it is an *agent-dispatch
orchestration*: each Prefect task assigns a piece of work to a subagent and waits
on its result; the DAG edges are the dependencies. Submitting the job runs that
flow inside the runner's sandbox, and the subagents do the actual work.

## Decisions

### D1 — The job's "Python script" is a generated Prefect flow (not a Workboard dict)

The planner emits a real `flow.py` (an `@flow`-decorated `main`), so the DAG is
expressed as code the way a human would author it. The alternative — emit a
`Workboard` JSON that the trusted worker runs directly, with *no* generated code
executing — is strictly safer but was deliberately not chosen; it remains a
possible future variant.

### D2 — Generated code never runs in the trusted worker

The one hard rule of the workboard (`orchestration-workboard.md`) extends here:
agents propose, the trusted worker disposes. The generated `flow.py` runs only as
**sandboxed node work** on `PrefectRunnerAgent`; the trusted Prefect worker only
ever executes the fixed `atrium-workboard` flow. The planner holds no Prefect
credentials and no board write access, and the generated flow can reach nothing
beyond the `atrium_dispatch` primitive + the runner's WAN isolation.

### D3 — The executor is a dedicated least-privilege agent, not the general code workspace

`PrefectRunnerAgent` subclasses `PythonCodeWorkspaceAgent` to inherit the
`{files, commands}` execution machinery, but tightens the envelope to least
privilege (re-asserted at construction, `_enforce_runner_policy`):

| Privilege | Code workspace | PrefectRunnerAgent |
| --- | --- | --- |
| WAN egress | allowed (GitHub + PyPI) | **none** (`NetworkMode.INTERNAL` — control LAN only, for A2A dispatch) |
| GitHub credentials | forwarded | **none** (`forward_github_token=False`) |
| git push / PR | supported | **refused** (`_parse_request` rejects it) |
| deps at run time | `uv sync` from registries | **none** (prefect + `atrium_dispatch` preinstalled; offline) |

The generated flow therefore runs in a WAN-isolated, credential-free sandbox whose
only outward capability is dispatching to a bounded set of subagents.

### D4 — Subagents are role-bearing InferenceAgents (minimal by construction)

A "subagent" a flow dispatches to is a role-bearing `InferenceAgent`
(coder / reviewer / domain role) — the strictest envelope in the system
(WAN-isolated, GPU-only, LAN/A2A only, token-producing, no side effects). So a
malicious generated orchestration can at worst waste generation; it cannot
exfiltrate or cause side effects. Broad-privilege side-effecting agents are an
explicit, opt-in exception in the dispatch roster, never the default.

### D5 — `atrium_dispatch`: the only egress a generated flow gets

Generated flow code never receives a raw A2A client. It reaches subagents solely
through the trusted `atrium_dispatch(agent, instruction, payload) -> {status, text,
data}` primitive baked into the runner image. Endpoint resolution: a full URL is
used as-is; else the slug is looked up in the `ATRIUM_DISPATCH_ENDPOINTS`
allow-list (a `{slug: url}` map the deployment injects into the sandbox env, see
`prefect_runner_agent.sandbox.build_sandbox_config(dispatch_endpoints=…)`); else
the host-local `http://<slug>.local` convention, which only resolves on the
control LAN the runner is confined to.

### D6 — Readiness gate: JSON + Python pair, both valid

A `Job` becomes runnable only when `is_ready()`: the request is present, the
`flow_source` is present, parses (`ast.parse`, **never executed**), defines a
`main` entrypoint, and — when the deployment declares an allow-list — needs no
library outside it (`unsupported_requirements`; the runner is offline, so an
undeclared dep would otherwise fail mid-flight). A not-ready plan (including a
fail-closed planner error) never reaches a workboard.

### D7 — Two-stage review: source before, result after

`build_execution_workboard(job, reviewer_agent=…)` prepends a `review_source` node:
the generated `flow.py` is dispatched to a reviewer whose verdict *is* that node's
outcome, and `run_flow` `depends_on` it — so a rejected flow is never executed
(the scheduler cascades the skip). After it runs, `run_flow` is itself `reviewable`,
so the run-level `ReviewPolicy` reviews the *result*. The dedicated `flow_reviewer`
profile judges a flow safety-first (single `main`, dispatch-only, no egress, no
unbounded loops). Without a `reviewer_agent`, the board is just `run_flow` and only
the post-execution gate applies.

## What lives where

| Repo (tier) | Adds |
| --- | --- |
| **core `atrium`** (trusted) | `orchestration.job` (`Job`, `is_ready`, `build_execution_workboard`, `unsupported_requirements`); `agents.plan_agent_protocol` (the plan A2A contract); `agents.prefect_runner_agent` (the executor + its sandbox + the `atrium_dispatch` primitive); the `ControlPlaneAgent` plan path |
| **`atrium_agents`** (evolvable) | `planner_profile` / `PlannerRole` / `PlanAgent`; `flow_reviewer_profile` / `flow_reviewer_role` |

Dependency direction stays correct (`atrium_agents` → `atrium`): the planner imports
the plan contract and the `Job`/protocol surface from core.

## Configuration

- **Enable the plan path**: construct `ControlPlaneAgent(plan_agent=…)` or set
  `ATRIUM_PLAN_AGENT` (mirrors `ATRIUM_REVIEWER`). A turn opts in with
  `payload["plan"]`; every other turn keeps the single-node fast path unchanged.
- **Pre-execution review**: `plan_reviewer=…` or `ATRIUM_PLAN_REVIEWER`.
- **Dispatch roster / allow-list**: `plan_constraints` (rides in every
  `plan_request`, tells the planner which subagents + libraries it may use) and
  `dispatch_endpoints` on the runner sandbox (the enforced `{slug: url}` reach).

## Trust & isolation notes

- The runner holds no Prefect credentials, no board write access, no GitHub creds,
  and no WAN NIC. Its only capability is A2A dispatch to the injected allow-list.
- The generated flow is reviewed before it runs (D7) and can only orchestrate
  dispatches to minimal, side-effect-free subagents (D4) — layered least privilege
  end to end (planner → runner → subagents).
- The runner image ships no telemetry egress (`PREFECT_SERVER_ANALYTICS_ENABLED=false`)
  and runs Prefect in ephemeral/local mode (`PREFECT_API_URL=""`), so a flow runs
  to completion in-process, offline, reporting via exit code / stdout.

## Per-piece status

- **DONE** — `orchestration.job` (Job + readiness + execution DAG + requirements
  gate), `plan_agent_protocol` (the A2A contract), `prefect_runner_agent` (agent +
  sandbox config/policy/Dockerfile + `atrium_dispatch`), `ControlPlaneAgent` plan
  path (dispatch → Job → readiness/requirements gate → review-gated workboard;
  env config). Unit-tested (`tests/test_job.py`, `tests/test_plan_agent_protocol.py`,
  `tests/test_prefect_runner_agent.py`, `tests/test_control_plane.py`).
- **DONE** — `atrium_agents`: `planner`/`flow_reviewer` profiles + roles, `PlanAgent`.
  Unit-tested (`tests/test_role.py`, `tests/test_prompt_profiles.py`,
  `tests/test_plan_agent.py`).
- **DONE** — runner image builds + runs a generated flow offline (`--network none`,
  non-root, exit 0); reproducible via `scripts/build-prefect-runner.sh`.
- **TODO** — register `prefect_runner_agent:active` (and any dispatchable subagent
  slugs) as active generations the OpenShell gateway serves; build the GPU-bound
  `plan_agent` image (a `TabbyLLMAgent` derivative); slim the runner image
  (~8.5 GB → 1–2 GB, [#32](https://github.com/hakuturu583/atrium/issues/32)); the
  data-DAG variant (D1 alternative) if generated-code execution is ever undesired.
