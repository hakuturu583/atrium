# Atrium

**A self-evolving, security-isolated multi-agent runtime.**

Atrium runs a fleet of autonomous agents, each living 1:1 inside its own
physically-isolated [OpenShell](https://github.com/hakuturu583/openshell) sandbox
container. Agents talk to one another over a **single wire protocol**
([A2A](https://github.com/a2aproject/A2A)), every hop is a distributed-tracing
span stitched into one timeline, and GPU inference runs behind a **WAN cut-off**.
The runtime is built to *evolve itself*: one agent authors a new generation of
another, a fixed-infrastructure builder turns it into a rootless container image,
and an attestation-gated promotion makes it live.

> Status: early but functional. The core runtime — sandbox lifecycle, A2A
> transport, distributed tracing, the version/generation ledger, the rootless
> image builder and the attestation-gated Morpher — is implemented and unit
> tested. The self-evolution *loop* (TaskAgent → BuilderAgent → validate →
> promote) is being assembled from these pieces.

---

## Why it looks the way it does

Four principles drive every design decision:

1. **Physical isolation, not process isolation.** Each agent is its own
   throwaway container. Compromising one agent yields a sandbox, not the host.
2. **One protocol at the agent boundary.** All inter-agent communication is A2A —
   there is deliberately no second wire protocol. Foreign protocols (OpenAI-style
   inference, Slack, …) are terminated at the edge and never leak inward.
3. **The registry *is* the ledger.** Every agent generation is an immutable,
   content-addressed OCI image. History, digests, the active-generation pointer
   and signed validation attestations all live in the container registry — no
   bespoke metadata store.
4. **Rootless agents, one trusted core.** Agents never touch the host Docker
   daemon; they build with rootless
   [Kaniko](https://github.com/GoogleContainerTools/kaniko) and only speak HTTP
   to the internal registry. The host daemon is used in exactly one place — the
   trusted main process bringing up that registry.

## Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │  Trusted main process (host)                   │
                         │   • ensure_local_registry()  ── host Docker ──▶│  registry:2
                         │   • factory: slug → :active → pinned digest    │  (the ledger)
                         │   • Morpher: attestation-gated promote/rollback│
                         └───────────────────┬───────────────────────────┘
                                             │  A2A (traceparent in Message.metadata)
        ┌───────────────┬────────────────────┼────────────────────┬───────────────────┐
        ▼               ▼                     ▼                    ▼                   ▼
  ┌───────────┐  ┌──────────────┐     ┌──────────────┐    ┌────────────────┐  ┌──────────────┐
  │ TaskAgent │  │ BuilderAgent │     │ TabbyLLMAgent│    │ CodeWorkspace  │  │  … agents    │
  │ (evolves) │  │ (fixed infra)│     │(GPU, WAN-cut)│    │ Agent          │  │              │
  └─────┬─────┘  └──────┬───────┘     └──────┬───────┘    └───────┬────────┘  └──────────────┘
        │  each agent ⇄ its own OpenShell sandbox container (version-pinned, throwaway)
        ▼               ▼                     ▼                    ▼
        └───────────── OTLP/HTTP spans ──────────────────────────────▶  Arize Phoenix (:6006)
                       (one stitched A→B→C timeline)
```

### Layers

| Layer | Modules | Responsibility |
| --- | --- | --- |
| **Agent base** | `core/base_agent.py` | Sandbox lifecycle, A2A send/receive, tracing, file-staging + path/traversal guards — every agent inherits this. |
| **A2A protocol** | `protocol/` | The single agent-boundary wire protocol, wrapping `a2a-sdk`. Message construction, data parts, W3C trace-context propagation. Host-safe (never imports `httpx`). |
| **Sandbox** | `sandbox/openshell.py` | Async wrapper over the OpenShell CLI: create / exec / delete version-pinned isolated containers; renders network/GPU/secret policy. |
| **Versioning / ledger** | `core/registry.py`, `core/factory.py`, `core/morpher.py` | The registry-as-ledger: version bump, digest resolution, `<slug>:active` pointer, attestation-gated promote/rollback, and the factory that launches the live generation. |
| **Telemetry** | `core/telemetry.py` | OpenInference/OpenTelemetry spans + OTLP export to Phoenix, with `traceparent` propagated across containers. |
| **Agents** | `agents/` | Concrete agents (below). |

### Agents

| Agent | Kind | Role |
| --- | --- | --- |
| `BuilderAgent` | fixed infra | Turns a `{filename: content}` build request into a container image with **rootless Kaniko**, pushes it to the internal registry, replies with the immutable `sha256:…` digest. Never mounts the Docker socket, never gets WAN or GPU. Excluded from the evolution loop. |
| `InferenceAgent` → `TabbyLLMAgent` | GPU inference | LLM inference on [tabbyAPI](https://github.com/theroyallab/tabbyAPI) / exllamav3, run `NetworkMode.INTERNAL` (WAN blocked). An in-sandbox bridge translates the OpenAI-style API to/from A2A so that protocol never crosses the agent boundary. |
| `CodeWorkspaceAgent` → `PythonCodeWorkspaceAgent` | code workspace | Operates a code-execution workspace sandbox (clone → stage → run → commit/push/PR). Allowed WAN (for GitHub); no GPU, no Docker socket. |
| `TaskAgent` → `SlackTaskAgent` | evolution *(in progress)* | The start of the self-evolution loop: takes a task, authors a new agent generation (source + Dockerfile), and drives BuilderAgent over A2A. |

Class hierarchy (`core/base_agent.py`):

```
BaseAgent (abstract)
├── TaskAgent        → SlackTaskAgent, …        (self-evolution loop)
├── InferenceAgent   → TabbyLLMAgent, …         (GPU, WAN-cut inference)
├── CodeWorkspaceAgent → PythonCodeWorkspaceAgent (code workspace)
└── BuilderAgent                                (fixed-infra image builder)
```

### The self-evolution lifecycle

```
TaskAgent          author new generation (edit package + next_version(level))
   └─ A2A build request {target_name, target_version, files} ─▶ BuilderAgent
BuilderAgent       reject if version exists → rootless Kaniko build+push
   └─ reply {image, digest}   (builds only; holds no authority over :active)
Morpher (fixed)    validate the exact digest → require a signed attestation
   ├ pass → set <slug>:active = digest     ← the only write that changes what runs
   └ fail → :active unchanged (never auto-activate an unvalidated build)
factory            slug → :active → pinned digest → start the live generation
```

The crown-jewel write — moving `<slug>:active` — is held solely by the Morpher
and gated on an Ed25519-signed validation attestation over the exact image
digest. A compromised agent can at most push an inert new *version* tag; it
cannot forge an attestation, so it cannot make a backdoored generation run. See
[`docs/design/agent-versioning.md`](docs/design/agent-versioning.md) and
[`docs/design/observability.md`](docs/design/observability.md).

---

## Development setup

Atrium uses [`uv`](https://docs.astral.sh/uv/) and targets **Python ≥ 3.12**.

```bash
# Install dependencies (incl. the dev group) into a local .venv
uv sync

# Run the test suite
uv run pytest            # or: uv run pytest -q
```

The unit tests are fully hermetic — no Docker daemon, GPU, OpenShell CLI or
network is required; the sandbox and subprocess boundaries are mocked.
Real-hardware integration checks (OpenShell CLI, tabbyAPI) are kept separate and
opt-in so they never run in the default `pytest` invocation or in CI.

### Observability stack (optional)

Bring up Arize Phoenix (trace UI + OTLP collector) with Docker Compose:

```bash
cp .env.example .env                          # standard OpenTelemetry env vars
docker compose up -d phoenix                  # UI + collector on :6006
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces
# start agents, then open http://localhost:6006 to see stitched traces
```

> Cross-container caveat: GPU sandboxes run WAN-blocked, so `localhost` inside a
> sandbox is *not* the host. Point `OTEL_EXPORTER_OTLP_ENDPOINT` at a
> host-local / control-plane LAN address reachable from the container (e.g.
> `http://host.docker.internal:6006/v1/traces`). Details in
> [`docs/design/observability.md`](docs/design/observability.md).

## Repository layout

```
src/atrium/
  core/            base_agent, registry (ledger), factory, morpher, telemetry, types, errors
  protocol/        A2A transport + message helpers (the single agent-boundary protocol)
  sandbox/         OpenShell CLI wrapper
  agents/
    builder_agent/         rootless Kaniko image builder (fixed infra)
    tabby_llm_agent/       GPU LLM inference (tabbyAPI) + in-sandbox A2A bridge
    code_workspace_agent/  code-execution workspace agent (+ Python specialization)
    inference_agent.py     WAN-isolated, GPU-only inference base
    prompt_memory.py       layered system-prompt composition
docs/design/       agent-versioning.md, observability.md
tests/             hermetic unit tests
```

## License

See the repository for license details.
