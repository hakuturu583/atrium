# Design: Interface Agent — a channel-agnostic human I/O boundary

> Status: **DESIGN / NOT-YET-IMPLEMENTED**. This captures the agreed shape; no
> `InterfaceAgent` / `ControlPlaneAgent` code exists yet. A precursor refactor —
> peeling the author→build orchestration out of the old `SlackTaskAgent` so it is
> a pure I/O gateway (`atrium.agents.task_agent.slack`, with the engine now in
> `DelegatingTaskAgent`) — is in the working tree and unit-tested. This doc is the
> plan for turning that gateway into a reusable, correctly-placed abstraction.
> Per-piece status at the end.

## Problem

A human talks to Atrium through a chat app (Slack today; Discord/Teams/… later).
The old `SlackTaskAgent` subclassed `TaskAgent`, so the Slack ingress was fused
with the self-evolution author→build engine — the "Task" in its name is a vestige
of that inheritance, not its job. Its actual job is **user I/O**: take a chat
turn, get work done, put a reply back in the thread.

Three things follow:

1. **Wrong altitude / wrong name.** An I/O boundary should not carry (or inherit)
   authoring/build authority. Renamed to **`SlackInterfaceAgent`**.
2. **Every chat app repeats the same "when".** *When* to hand a turn off, *how* a
   thread maps to a conversation context, *how* a job's result gets back to its
   thread — identical across apps. Only *how to parse/render/post* differs. That
   shared "when" wants a **base class**.
3. **Wrong repository / wrong trust tier.** Once the interface only *proposes*
   (holds no authority), it is an evolvable, WAN-exposed worker — it belongs in
   the `atrium_agents` distribution, not the trusted core.

## Approach: `InterfaceAgent` base + per-app concretes, talking only to a control plane

```
BaseAgent (abstract, core)
├── TaskAgent → DelegatingTaskAgent          (core: authoring/build engine)
├── InferenceAgent → TabbyLLMAgent           (atrium_agents)
├── BuilderAgent                             (core)
└── InterfaceAgent (abstract)                (atrium_agents)   ← NEW
     ├── SlackInterfaceAgent                 (atrium_agents)   ← renamed + moved
     └── DiscordInterfaceAgent, …            (future)
```

The interface's **only** egress is an A2A `submit_work` to a trusted
**`ControlPlaneAgent`** (in core). The control plane is the sole caller of
`orchestration.kick.submit_job`; it, not the interface, decides new-job vs
steering vs review-reply and fans out to the workboard / runner / reviewer.

```
Chat app ──▶ SlackInterfaceAgent (parse ▸ derive context_id ▸ forward)   [atrium_agents, propose-only]
                     │  A2A: submit_work  (single egress; steering in payload)
                     ▼
              ControlPlaneAgent            [core, trusted — sole kick.submit_job caller]
                     │  submit_job / workboard_state / cancel / relay-feedback
                     ▼
              Prefect WorkBoard ──▶ runner.run_node (review gate + rework loop)
                     │  job_update (terminal/progress, carries thread coords)
                     ▼
              ControlPlaneAgent ──A2A──▶ SlackInterfaceAgent.deliver() ──▶ chat thread
```

This is the same "**agent proposes, trusted worker disposes**" model the workboard
already uses (`docs/design/orchestration-workboard.md`): the interface proposes,
the control plane disposes.

## Decisions

### D1 — Single egress (option A)

The interface talks **only** to the control plane. One method, `submit_work`;
human steering is folded into its payload. It gets **no** direct line to the
inference/coding agent and **no** `orchestration.kick` access. Even mid-flight
human feedback (§動線2b) goes out as a forward-to-control-plane — the control
plane relays it to the waiting reviewer/runner. Rejected alternative (B): a
second direct interface→inference channel; it would hand a WAN-exposed edge
inference-reachability for no authority gain.

### D2 — Thread = session, keyed by a derived `context_id`

A chat thread is a conversation session. Its key is **derived, not stored**:

```
context_id = f"{SOURCE}:{channel}:{thread}"      # e.g. "slack:C1:1699999999.000100"
```

`context_id` already threads end-to-end through the runtime
(`kick.submit_job(context_id=…)`, `protocol.build_node_request(context_id=…)`,
`review.build_review_request(context_id=…)`) and through tracing. Same thread →
same `context_id` → the doer/inference side accumulates conversation context/KV
under that key. The **conversation's system of record is the chat thread itself
plus the inference-side KV**, not the interface.

### D3 — Stateful but non-authoritative, reconstructable

The interface is **not** stateless — thread management needs real state — but it
holds only a **reconstructable coordination/presentation cache**, never the
system of record:

| Kind | Example | Where |
| --- | --- | --- |
| Derived | thread → `context_id` | pure function (no state) |
| Ride-along | reply coords `{channel,thread,user}` echoed via payload | in the job, not local |
| **Authoritative** | which job is in-flight, its state | **control plane** (core) |
| **Interface-local cache** | edit-target message ts, event dedup ids, pending-review pointer, cached `active_job_id` | interface `SessionStore` (atrium_agents) |

Because the cache is keyed by `context_id`, it is rebuildable after a restart from
the chat thread + `workboard_state(context_id)`. The interface *manages threads*
but is neither the authority nor durable for job state.

```python
@dataclass
class Session:                  # common: the routing brain
    context_id: str
    active_job_id: str | None    # cache of control-plane truth
    pending_review: str | None   # a review awaiting a human reply

class SlackSession(Session):     # channel-specific: presentation state
    working_msg_ts: str | None   # message to edit as the job progresses
    seen_event_ids: set[str]     # Slack retry de-duplication
```

### D4 — Placement & trust

Option A makes the interface **propose-only, zero-authority**. A compromised
interface can at worst submit junk jobs, which the control plane + mandatory
review gate still govern. That profile — sandboxed, WAN-exposed, no authority — is
an **evolvable worker**, so:

| Repo (tier) | Holds |
| --- | --- |
| **core `atrium`** (trusted / fixed infra) | `orchestration` (kick/flow/runner/review), **`ControlPlaneAgent`**, `BaseAgent`, `BuilderAgent`, `DelegatingTaskAgent`, **the interface↔control-plane A2A message contract** |
| **`atrium_agents`** (evolvable / untrusted) | `InferenceAgent` (existing), **`InterfaceAgent` base + `SlackInterfaceAgent` + `SessionStore`** |

Dependency direction stays correct (`atrium_agents` → `atrium`): the interface
imports `BaseAgent`, the A2A protocol, and the shared submit contract from core.
Chat credentials + external-ingress exposure remain, but are an orthogonal
secrets-management concern (`openshell_secrets`), not an authority one.

## The base class (shared "when")

```python
class InterfaceAgent(BaseAgent, abc.ABC):
    SOURCE: str                                   # "slack" / "discord" — used in context_id & payload.source

    def __init__(self, agent_id, version=None, *, control_plane: SendTarget, sandbox_config=None):
        super().__init__(...)                     # WAN-capable, no Docker socket
        self.control_plane = control_plane        # the ONLY egress target
        self.sessions = SessionStore()

    async def handle_task(self, inbound: Message) -> Message:
        turn = self.parse_turn(inbound)           # ← concrete
        ctx  = self.context_id_for(turn)          # common: f"{SOURCE}:{channel}:{thread}"
        sess = self.sessions.get_or_create(ctx)
        if self.is_duplicate(turn, sess):
            return self.render_ack(turn, None)
        ack = await self.send_a2a_message(         # single egress → control plane
            self.control_plane, self.forward_request(turn, ctx, sess)
        )
        return self.render_ack(turn, ack)

    async def on_job_update(self, update) -> None: # egress trigger: control plane pushed a result
        text = self.render_result(update) if update.ok else self.render_error(update)
        await self.deliver(update.channel, update.thread, text)   # ← concrete

    def context_id_for(self, turn) -> str:
        channel, thread = self.thread_key(turn)    # ← concrete
        return f"{self.SOURCE}:{channel}:{thread}"

    # ---- channel-specific seams ----
    @abc.abstractmethod
    def parse_turn(self, inbound) -> "Turn": ...
    @abc.abstractmethod
    def thread_key(self, turn) -> tuple[str, str]: ...
    @abc.abstractmethod
    def render_ack(self, turn, ack) -> Message: ...
    @abc.abstractmethod
    def render_result(self, update) -> str: ...
    @abc.abstractmethod
    def render_error(self, update) -> str: ...
    @abc.abstractmethod
    async def deliver(self, channel, thread, text) -> None: ...
```

`Turn` is the shared normalized value object: `{source, instruction, user,
channel, thread, raw}`. Adding a chat app = one concrete class (parse / thread_key
/ render* / deliver / `SOURCE`) — no dispatch timing, no `context_id` logic, no
A2A contract, no result-routing rewritten.

## The two flows (both ride existing orchestration seams)

### 動線1 — register work on the WorkBoard

`submit_work` → control plane → `kick.submit_job(agent, instruction,
payload=…, context_id=ctx, review=…)` → Prefect flow-run id. Fire-and-forget: the
interface acks "received (job `xxx`)"; the terminal state comes back as a
`job_update` carrying the ride-along thread coords, posted into the thread.

### 動線2 — intervene in the inference agent's prompt

- **(a) submit-time steering** — human context folded into `payload["steering"]`;
  rides `WorkNode.payload` → `build_node_request` data part → the doer's prompt.
  No orchestration change; a payload convention.
- **(b) mid-flight feedback (rework)** — modeled as the **review gate**:
  `ReviewPolicy.reviewer` is a Slack-backed human reviewer. The runner sends the
  deliverable to it (`review.build_review_request`); the human's in-thread reply
  becomes the verdict, which the runner threads into the next attempt as
  `payload["review_feedback"]` (`runner.py`, existing). Under option A the human's
  reply still enters via the interface → control plane → (relay to waiting
  reviewer/runner), so the single-egress rule holds.

**Nothing in `orchestration` core changes**: both flows ride `submit_job`,
`context_id`, `ReviewPolicy.reviewer`, and the `review_feedback` payload seam that
already exist.

## Message contract (interface ↔ control plane, defined in core)

```jsonc
// interface → control plane   metadata.kind = "workboard.submit"
{ "type": "workboard_submit",
  "agent": "python_code_workspace_agent:active",   // who should do the work
  "instruction": "<normalized turn text>",
  "context_id": "slack:C1:1699999999.000100",
  "payload": {
    "slack": { "channel": "C1", "thread_ts": "…", "user": "U9" },  // ride-along reply coords
    "steering": { … }                                              // 動線2(a)
  },
  "review": { "reviewer": "slack_reviewer:active", … },            // 動線2(b), optional
  "feedback_for": "<review-token>"                                 // set when relaying a human reply
}
// control plane → interface
{ "type": "workboard_submitted", "status": "ok", "job_id": "<flow-run-id>" }

// control plane → interface (async)   metadata.kind = "workboard.update"
{ "type": "job_update", "job_id": "…", "status": "ok|error",
  "slack": { "channel": "C1", "thread_ts": "…" }, "result": { … } }
```

## What changes where

- **core `atrium`**: add `ControlPlaneAgent` + the shared submit/update contract;
  **remove** `agents/task_agent/slack.py` (moves out); keep `DelegatingTaskAgent`.
- **`atrium_agents`**: add `InterfaceAgent` base + `Turn` + `SessionStore`, and
  `SlackInterfaceAgent` (parse/render/deliver + `SOURCE="slack"`); register as
  `slack_interface_agent`. A future `SlackReviewerAgent` (for 動線2b) lives here
  too and may share the same Slack transport helpers.
- The rename/move **cannot complete in the core repo alone** — the destination is
  `atrium_agents`.

## Per-piece status

| Piece | Status |
| --- | --- |
| Peel orchestration out of `SlackTaskAgent` (gateway + `DelegatingTaskAgent`) | **DONE (working tree)**, unit-tested |
| `InterfaceAgent` base + `Turn` + `SessionStore` | **DESIGN** |
| `SlackInterfaceAgent` (rename + move to `atrium_agents`) | **DESIGN** |
| `ControlPlaneAgent` + shared submit/update contract (core) | **DONE (core)** — `atrium.agents.control_plane`, unit-tested; submit path kicks `submit_job`, feedback-relay refused pending 動線2(b) |
| 動線2(b) `SlackReviewerAgent` on the review gate | **DESIGN** |
| `orchestration` core changes | **NONE NEEDED** (rides existing seams) |

## Deferred / open

- **Progress notifications**: push a `job_update` from a Prefect terminal-state
  hook vs. the control plane polling `workboard_state`. Lean push, fallback poll.
- **Target-agent routing**: how the control plane picks the doer from a turn —
  fixed `python_code_workspace_agent:active` first, routing later.
- **Human-reviewer waiting model**: block the review A2A request vs. ticket it
  asynchronously. Long human waits favor async.
- **Session cache durability**: in-memory + rebuild-on-miss vs. a small durable
  store; rebuild-on-miss is the default given D3.
