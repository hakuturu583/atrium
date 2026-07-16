# Local OpenShell gateway (Docker driver)

atrium's agents run inside [OpenShell](https://github.com/NVIDIA/OpenShell)
sandboxes. The `openshell` **CLI** that `atrium.sandbox` drives is only a client —
it talks to an OpenShell **gateway** (a control-plane server that actually
launches sandbox containers). This directory brings up that gateway locally with
the Docker compute driver, so `atrium.sandbox.openshell.Sandbox` (and every agent
built on it) has something to talk to.

This is the single-player, trusted-local-dev setup: plaintext HTTP on loopback,
unauthenticated CLI, sandbox containers created as siblings on the host Docker
daemon (docker-outside-of-docker). It is **not** a production deployment — see
OpenShell's Helm chart for that.

## Prerequisites

- Docker Engine + the Compose plugin (`openshell doctor check` should pass).
- The `openshell` CLI on `PATH` (`uv tool install openshell`, or built from
  source via [`scripts/build-openshell.sh`](../../scripts/build-openshell.sh)).

## Bring-up

```bash
# 1. Provision the sandbox-JWT signing bundle (required for Docker sandboxes).
bash deploy/openshell/init-jwt.sh

# 2. Start the gateway.
docker compose -f deploy/openshell/docker-compose.yml up -d

# 3. Register + select it with the CLI (one-time).
openshell gateway add http://localhost:8080 --name atrium-local
openshell gateway select atrium-local

# 4. Verify.
openshell status                       # -> Status: Connected
openshell sandbox create --name smoke --from base --no-tty -- true
openshell sandbox exec --name smoke -- bash -lc 'python --version'
openshell sandbox delete smoke
```

Then atrium's real-hardware smoke test exercises the same path through atrium's
own wrapper:

```bash
ATRIUM_INTEGRATION=1 \
ATRIUM_IT_IMAGE=ghcr.io/nvidia/openshell-community/sandboxes/base:latest \
  uv run pytest tests/integration/test_openshell_smoke.py -v
```

Tear down with `docker compose -f deploy/openshell/docker-compose.yml down`.

## Why the non-obvious bits

The stock OpenShell compose targets Docker Desktop; three changes make it work on
Linux Docker Engine and satisfy the Docker driver's auth requirement:

- **`extra_hosts`** on the gateway service — Linux Docker does not auto-add
  `host.docker.internal` / `host.openshell.internal`, which sandbox containers use
  to call back to the gateway.
- **gateway published on all interfaces** (`8080:8080`, not `127.0.0.1:8080:8080`)
  — sandbox containers reach the gateway via the host bridge IP
  (`host.openshell.internal` -> `host-gateway`), which a loopback-only publish
  does not expose.
- **sandbox-JWT bundle + `allow_unauthenticated_users`** — the Docker driver
  requires gateway-minted sandbox JWTs (`init-jwt.sh` provisions them and points
  `OPENSHELL_LOCAL_TLS_DIR` at them). Enabling JWT auth otherwise flips the
  gateway into authenticated mode and rejects the plaintext CLI, so
  `[openshell.gateway.auth] allow_unauthenticated_users = true` keeps the local
  CLI trusted while sandbox JWTs still mint for callbacks.

Exact flags/schema track a specific OpenShell version; if a call fails on a
version mismatch, adjust the centralized command templates in
[`src/atrium/sandbox/openshell.py`](../../src/atrium/sandbox/openshell.py) and the
gateway settings here.
