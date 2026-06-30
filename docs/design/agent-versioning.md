# Design: Agent Version Management & Generational Swap

> Status: **DESIGN** — the *registry bootstrap* (below) is now implemented
> (`atrium.core.registry`); the ledger client, version bump, factory and Morpher
> remain design. Captures the mechanism by which a newly built agent generation
> is versioned, recorded, validated and made live.

## Problem
Atrium agents are 1:1 with their sandbox and independently versioned
(`__version__`, image tag `local-registry/<slug>:<version>`). When a TaskAgent
authors a new generation and `BuilderAgent` builds it, three things are missing:

1. **Decide** — who computes the next version (and guards against clobbering).
2. **Record** — an immutable ledger of which generations exist (by digest).
3. **Reflect** — an *active-generation pointer* the runtime resolves to start the
   right version; plus atomic **promote / rollback** (the generational swap).

Today none of these exist: no bump logic, no ledger, no active pointer, no
factory/orchestrator, and BuilderAgent returns only a (mutable) tag, no digest.

## The three concerns

### 1. Decide (version bump)
- Source of truth stays the package `__version__` (semver). Next version is
  computed with `semver`'s `bump_patch/minor/major` by change scope (TaskAgent
  chooses the level).
- Guard: reject a build whose `target_version` already exists in the ledger —
  **generations are immutable; tags are never overwritten.**

### 2. Record (immutable ledger) — *use the container registry itself*
BuilderAgent's whole job is producing OCI images, so the **registry IS the
ledger** — no second store for history. It already gives, for free:
- **History**: `GET /v2/<slug>/tags/list` → every version ever pushed.
- **Immutable identity**: each tag resolves to a content-addressed manifest
  `sha256:…`; pull-by-digest is immutable. (Enable **tag immutability** so a
  version tag can never be overwritten → generations are immutable *at the
  storage layer*, stronger than an app-level guard.)
- **Per-generation metadata**: OCI image-config **labels/annotations** carry
  `parent_version`, `built_at`, `source_hash`, etc.
- **Attestations**: a TestAgent's signed validation result is attached to the
  digest as a **signed OCI referrer artifact** (cosign/sigstore-style) — not a
  field in a side file.
- **Digest capture**: add `--digest-file` to the Kaniko command so BuilderAgent
  returns the exact `sha256:…` (also resolvable later via
  `HEAD /v2/<slug>/manifests/<tag>` → `Docker-Content-Digest`).

So `AgentRef(slug, version, digest)` is just a typed view over registry data; no
custom manifest store for history.

### 3. Reflect (generational swap) — a moving tag in the same registry
- **Active pointer = a mutable tag** `local-registry/<slug>:active` that points at
  the active generation's digest (exactly how `:stable`/`:prod` work). The runtime
  factory pulls `<slug>:active` (resolving to a pinned digest) instead of today's
  `__version__` defaulting. **That resolution is "reflection."**
- **Promote = re-tag** `:active` → new digest; **rollback = re-tag** `:active` →
  the previous digest. Atomic (single manifest PUT-by-tag). Both are **separate
  from build** and **gated**: build → validate → on pass move `:active`; on fail,
  leave `:active` (**never auto-activate an unvalidated build**).
- Caveat: a registry only shows the *current* `:active` target, not the history of
  promotions — keep a **thin append-only promotion log** (or derive it from signed
  promotion referrer artifacts) for the "who promoted what, when" audit trail.

## Registry bootstrap (the host-Docker decision) — IMPLEMENTED
The registry is **fixed infrastructure** (like BuilderAgent): the **trusted Atrium
main process brings it up via the host Docker daemon at startup**. Implemented in
`atrium/core/registry.py` (`ensure_local_registry`).

**Trust boundary (why host Docker here doesn't break "rootless agents").** Atrium's
rootless guarantee is *"agents never get the host Docker daemon."* Host Docker is
used **only** by the trusted, non-evolving main process to run the registry — it is
never exposed to any agent. Agents (incl. BuilderAgent) still build with rootless
Kaniko and only ever speak **HTTP** to the registry; `BuilderAgent._enforce_build_policy`
keeps rejecting any `docker.sock` mount. So:

- main process → `docker.sock` → run `registry:2` … OK (inside the trust boundary)
- any agent → `docker.sock` … **forbidden** (rootless build, HTTP push only)

**`ensure_local_registry` (idempotent startup step):**
- Uses the **Docker SDK for Python** (`docker.from_env()`) — a real client, not
  shelling out.
- Idempotent: `containers.get(name)` → start if stopped, create if absent, reuse if
  already running. Optionally reconcile drifted config.
- **Persistent named volume is mandatory** — the registry *is* the ledger, so its
  data must survive restarts (`atrium-registry-data:/var/lib/registry`); without it
  every generation is lost on restart.
- `restart_policy=always`; labelled as fixed infra (`atrium.component=registry`,
  `atrium.immutable=true`); published on `host_port` (default 5000).
- **Health-gate**: poll `GET /v2/` (returns 200) via stdlib `urllib` — the host
  package must not import `httpx` — before agents are allowed to start.
- Returns the pushable endpoint (`host:port`) so callers wire `--destination` /
  `--from` consistently.

**Honest gaps vs the security model above (single-host first cut):**
- `registry:2` (Distribution) has **no per-role auth/ACL** and **no native tag
  immutability**. The scoped-token model (BuilderAgent push-version-only; Morpher
  alone moves `:active`) needs either a **token-auth** front (registry supports a
  token auth service) or a richer registry (**zot**/**Harbor**, which also do tag
  immutability). The collision/overwrite guard therefore lives **app-side**
  (`tags/list` check) until then.
- **insecure HTTP + name resolution**: `local-registry` must resolve (host-IP:port
  or hosts/DNS) and HTTP must be permitted symmetrically — Kaniko already passes
  `--insecure`; OpenShell's pull side must match (or terminate TLS).
- The registry container's lifecycle is owned by the main process; it is **excluded
  from the evolution loop**.

## Lifecycle
```
TaskAgent: edit package + next_version(level)
   └─A2A build req {target_name, target_version, files} → BuilderAgent
BuilderAgent: reject if version exists → rootless Kaniko build+push
   → read --digest-file → reply {image, digest, version}   (records nothing; no active authority)
Morpher (orchestrator, fixed infra): ledger.record(ref) → validate new image (TestAgent/smoke)
   ├ pass → manifest.promote(slug, version)     ← the only write that changes what runs
   └ fail → status=failed, active unchanged
runtime factory: slug → active → start; rollback to revert
```

## Security: who may move the active pointer? (privilege-escalation analysis)

**The active pointer is the crown jewel.** Whoever can call `promote()` controls
what code runs next — they can activate a back-doored generation (persistence +
escalation), roll back a security fix, or repoint an agent at an attacker image.
This risk exists for **any** storage choice (a file is just as dangerous if any
agent can write it). The danger is *authority over the write path*, not the
transport.

**Why an A2A registry *service* (option 2) amplifies it.** A2A is the universal
bus; if `promote` is "just another agent endpoint," reachability becomes ambient
authority — every agent that can send a message becomes a candidate promoter. A
**compromised TaskAgent is the likely vector** (it authors code → high prompt-
injection exposure); if it can reach `promote`, it can self-promote a malicious
generation. So option 2 is viable for distributed self-improvement **only** with
an explicit authorization model — never "any caller who can connect."

**The registry-as-ledger *resolves* the escalation worry — reuse registry auth.**
Because history, digests and the `:active` pointer all live in the registry, the
authorization boundary becomes **registry push/tag ACLs** (scoped pull/push
tokens) — a battle-tested mechanism — instead of a bespoke promote RPC on the A2A
bus. There is then **no write path on the bus at all**:
1. **Scoped credentials per role.**
   - *BuilderAgent token*: may `push` new immutable version tags
     `<slug>:<version>` — but **cannot move `:active`** and (with tag immutability)
     cannot overwrite an existing version. It builds; it cannot decide what runs.
   - *Morpher token* (fixed infra): the **only** credential allowed to move
     `<slug>:active`.
   - *Agents / factory*: **pull-only** (read `:active`, pull by digest).
2. **Promote requires a verified attestation.** Before moving `:active`, the
   Morpher checks a valid **signed validation referrer** exists for that exact
   digest (cosign-style, against a fixed trust root). A compromised actor cannot
   promote a backdoored digest it has no attestation for.
3. **Tag immutability** at the registry = generations immutable at the storage
   layer; only `:active`/`:stable` aliases are mutable, and only by the Morpher.
4. **Immutable set never auto-promoted.** BuilderAgent / Registry / Morpher images
   move only by explicit human approval (two-person rule).
5. **Audit.** Promotions are a thin append-only log + signed promotion referrers
   (who moved `:active` to which digest, when, citing which attestation).

Net: a **compromised TaskAgent can at most push a new *version* tag** (inert until
promoted) — it holds no credential to move `:active` and can forge no attestation,
so it cannot make its malicious generation run. The crown-jewel write is held
solely by fixed infrastructure and gated by supply-chain proof.

## Recommendation (registry-native, security-first)
- **Ledger = the container registry.** History = version tags + digests; metadata
  = OCI labels/annotations; attestations = signed referrer artifacts. No custom
  manifest store.
- **Active pointer = a mutable `<slug>:active` tag** resolving to a digest; the
  factory pulls `:active`. Promote/rollback = re-tag.
- **Authz = registry scoped tokens**, not a custom A2A write RPC: BuilderAgent
  push-new-version-only; **Morpher alone** moves `:active`, and only after a
  verified attestation; everyone else pull-only. Enable **tag immutability**.
- **BuilderAgent stays minimal:** builds + returns the digest; **no** authority
  over `:active` (tiny fixed-infra blast radius).
- A2A's role shrinks to **read-only resolution** (or agents pull `:active`
  directly) + the build request/result — no promote on the bus.
- Keep a **thin promotion audit log** (registry shows only the current `:active`).

## Concrete pieces (status)
1. **Registry bootstrap** — ✅ implemented (`atrium/core/registry.py`,
   `ensure_local_registry`): host-Docker brings up `registry:2` as fixed infra.
2. **BuilderAgent digest** — ✅ implemented: Kaniko runs with `--digest-file`, and a
   successful build reply carries the immutable `digest` (`sha256:…`) plus the
   content-addressed `image_ref` (`<registry>/<name>@<digest>`) alongside the tag.
   Push is to `<slug>:<version>` only (BuilderAgent never moves `:active`).
3. **`next_version` + collision guard** — ✅ implemented. `next_version(current, level)`
   does the semver bump; the app-side guard is wired into BuilderAgent — when a
   `registry_endpoint` is configured it `RegistryClient.exists(name, version)`-checks
   before building and refuses to rebuild an existing version (best-effort: skipped
   when no endpoint is set, fail-open if the registry is unreachable).
4. **Registry client** — ✅ implemented in `atrium/core/registry.py`:
   `RegistryClient.versions(slug)` / `digest(slug, ref)` / `exists(slug, version)` /
   `active(slug)` / `set_active(slug, digest)` (Morpher-only re-tag), plus the
   `AgentRef(slug, digest, version)` typed view. Speaks the registry v2 HTTP API
   over stdlib urllib.
5. Agent **factory** `create_agent(slug)`: resolve `:active` → start at that digest
   (replaces `__version__` defaulting on the evolving path). — TODO
6. Attestation verify + `set_active` + audit log, owned by a (future) Morpher. — TODO
