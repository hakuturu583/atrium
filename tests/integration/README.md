# Integration tests (opt-in, real hardware)

These validate the parts a hermetic unit run can't: the real **OpenShell CLI**,
a real **tabbyAPI** inference over the A2A bridge, and the **cross-container
trace stitch** in Arize Phoenix. They are **excluded from the default `pytest`
run and from CI**, and only execute when explicitly opted in.

## Running

```bash
export ATRIUM_INTEGRATION=1        # required — without it every test here skips
uv run pytest tests/integration -v
```

Each test additionally self-skips when the specific resource it needs is
unavailable, so you can run just the parts your box supports.

### OpenShell CLI smoke — `test_openshell_smoke.py`
Needs the `openshell` binary on `PATH`. Creates a WAN-isolated sandbox, execs a
command, and deletes it — exercising the real CLI argument spelling.

```bash
export ATRIUM_IT_IMAGE=docker.io/library/busybox:latest   # optional
```

> If a call fails on a version mismatch, the OpenShell subcommand spellings vary
> across releases; adjust the centralized command templates in
> `src/atrium/sandbox/openshell.py`.

### tabbyAPI inference smoke — `test_tabby_smoke.py`
Talks to an **already-running** bridge (co-located with tabbyAPI on a GPU box);
it does not launch the GPU sandbox itself.

```bash
export ATRIUM_IT_BRIDGE_URL=http://10.0.0.5:8730   # required
export ATRIUM_IT_MODEL=Ornith-1.0-35B              # optional: ensure loaded first
export ATRIUM_IT_READY_TIMEOUT=300                 # optional
```

### Cross-container trace stitch — `test_trace_stitch.py`
Drives a real inference with tracing on and asks Phoenix whether the host span
and the in-sandbox bridge span share **one trace id**.

```bash
export ATRIUM_IT_BRIDGE_URL=http://10.0.0.5:8730
export ATRIUM_IT_PHOENIX_URL=http://localhost:6006
export OTEL_EXPORTER_OTLP_ENDPOINT=http://10.0.0.1:6006/v1/traces
```

Phoenix's GraphQL schema shifts between releases; if the query shape doesn't
match, the test skips and points you to verify manually in the UI
(`open $ATRIUM_IT_PHOENIX_URL`). Bring Phoenix up with
`docker compose up -d phoenix` (see `docs/design/observability.md`).

## Why opt-in (not in CI)
Real GPUs, a container runtime, the OpenShell binary and a live Phoenix are not
present in a unit CI environment. Keeping these behind `ATRIUM_INTEGRATION`
means the default `pytest` stays fast, hermetic and green, while the same suite
becomes a real end-to-end check on a properly provisioned host.
