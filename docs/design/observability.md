# Design: Observability (distributed tracing → Arize Phoenix)

> Status: **IMPLEMENTED**. Atrium emits OpenInference-annotated OpenTelemetry
> spans for every cross-agent hop, in-sandbox command, and LLM call, and exports
> them over vendor-neutral OTLP to Arize Phoenix.

## Goal
Make the physically-isolated, multi-container agent runtime observable: a single
A→B→C task — host control plane, A2A bus, and GPU sandbox containers — should
render as **one stitched timeline** in Phoenix, with LLM calls showing token
counts and tool/exec calls showing inputs/outputs.

## Architecture
```
┌─────────────────────────────┐         A2A (traceparent in           ┌──────────────────────────┐
│ Host process (BaseAgent)     │         Message.metadata)             │ Sandbox container         │
│  send_a2a_message  AGENT ────┼──────────────────────────────────────▶  bridge.execute    AGENT  │
│  a2a.send_message  CHAIN     │                                       │  tabby.chat_comp.  LLM    │
│  execute_in_sandbox TOOL     │   OTEL_* env forwarded into sandbox    │   (token counts)          │
└──────────────┬──────────────┘   via --env (openshell)               └─────────────┬─────────────┘
               │ OTLP/HTTP                                                            │ OTLP/HTTP
               └──────────────────────────────▶  Arize Phoenix  ◀────────────────────┘
                                                  (:6006 UI + collector)
```

Two halves, both already wired in code:
- **Instrumentation** (`core/telemetry.py` helpers used across `a2a_transport.py`,
  `base_agent.py`, `builder_agent`, `tabby_llm_agent/bridge/server.py`): creates
  spans and propagates the W3C `traceparent` through `Message.metadata` so the
  remote agent parents its work under the caller's span.
- **Export** (`core/telemetry.py:configure_tracing`): installs a `TracerProvider`
  with a `BatchSpanProcessor` + OTLP/HTTP `OTLPSpanExporter`. **Without this the
  spans are dropped by OTel's default no-op provider** — it is the "元栓".

## Why vendor-neutral OTLP (not the Phoenix helper)
Phoenix speaks standard OTLP, so we export with `opentelemetry-exporter-otlp-proto-http`
and treat Phoenix as one interchangeable backend (Jaeger/Tempo/Grafana would work
unchanged by repointing `OTEL_EXPORTER_OTLP_ENDPOINT`). This keeps the dependency
on real upstream OTel libraries and isolates any backend-specific concern at the
endpoint, consistent with Atrium's "one unified protocol, foreign protocols at the
edge" principle.

## Configuration (standard OTel env vars)
`configure_tracing()` is **opt-in** and defers to standard env vars; it only fills
defaults when unset.

| Variable | Effect |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | OTLP target. Presence auto-enables tracing. Default `http://localhost:6006/v1/traces`. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes (e.g. `service.namespace=atrium`). |
| `OTEL_EXPORTER_OTLP_HEADERS` | Exporter headers (auth for hosted collectors). |
| `ATRIUM_TRACING_DISABLED=1` | Hard kill switch — overrides everything; no exporter, no connection. |

In code: call `configure_tracing(service_name)` once at process startup, and
`shutdown_tracing()` at exit to flush. The tabby bridge entrypoint
(`bridge/server.py:main()`) already does both. A future host launcher/orchestrator
must call `configure_tracing(...)` too (re-exported as
`from atrium.core import configure_tracing`).

## Network reachability (important)
GPU inference sandboxes run with `NetworkMode.INTERNAL` — WAN blocked, only
`127.0.0.0/8` and `10.0.0.0/8` reachable (`core/types.py: SandboxConfig.render_policy_yaml`).
Inside a sandbox, `localhost` is the container, **not** the host, so
`http://localhost:6006` won't reach a host-run Phoenix. Place Phoenix on a
host-local / control-plane LAN address and set `OTEL_EXPORTER_OTLP_ENDPOINT` to
that reachable address (e.g. `http://10.0.0.1:6006/v1/traces` or the docker
host-gateway). The host forwards its OTLP env vars into the sandbox automatically
(`base_agent._otel_env_passthrough` → `openshell --env`), so set it once on the host.

## Run it
```bash
docker compose up -d phoenix                 # UI + collector on :6006
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces
# ... start agents; open http://localhost:6006 to see traces
```

## Out of scope
- OpenInference auto-instrumentation libraries (manual spans suffice).
- Production Phoenix persistence/auth, multi-tier collector topology.
