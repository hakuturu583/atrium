"""Local container registry bootstrap — fixed infrastructure.

The container registry is Atrium's **generation ledger**: every agent image is
pushed here by version tag (immutable identity = its manifest digest), and the
active generation is a moving ``<slug>:active`` tag. This module brings that
registry up and health-gates it.

**Trust boundary.** The registry is started by the *trusted Atrium main process*
via the host Docker daemon. That daemon is **never** exposed to agents — agents
(including ``BuilderAgent``) build rootless with Kaniko and only ever speak HTTP
to the registry (``BuilderAgent._enforce_build_policy`` keeps rejecting any
``docker.sock`` mount). Running the registry here is the *one* place host Docker
is used, and only by fixed infrastructure, so the "agents are rootless"
guarantee is preserved. The registry itself is excluded from the evolution loop.

The Docker SDK and daemon are required only for this bootstrap; importing this
module does not require either (the SDK is imported lazily inside the calls).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from atrium.core.errors import AtriumError
from atrium.core.types import VersionTag

logger = logging.getLogger("atrium.registry")

__all__ = [
    "RegistryConfig",
    "LocalRegistryError",
    "ensure_local_registry",
    "registry_healthy",
    "stop_local_registry",
    "image_ref_from_tag",
    "AgentRef",
    "next_version",
    "ACTIVE_TAG",
    "RegistryClient",
]

#: The mutable tag that names the live generation of an agent (the active pointer).
ACTIVE_TAG = "active"

#: Manifest media types we accept when resolving a tag/digest to a manifest.
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)

#: Bind to /var/lib/registry inside the registry:2 container.
_REGISTRY_DATA_DIR = "/var/lib/registry"


def image_ref_from_tag(tag: str, digest: str) -> str:
    """Turn a tag ref into the immutable, content-addressed ref pinned by the ledger.

    ``<repo>:<version>`` + ``sha256:…`` → ``<repo>@<digest>`` (the repo is the tag
    minus its rightmost ``:<version>``; a registry ``host:port`` prefix is kept).
    Shared so BuilderAgent, the registry client and the agent factory all build
    ``repo@digest`` references identically.
    """
    return f"{tag.rsplit(':', 1)[0]}@{digest}"


@dataclass(frozen=True, slots=True)
class AgentRef:
    """A pinned reference to one agent generation in the registry (ledger).

    ``slug`` is the agent's repository; ``version`` is the (mutable) tag it was
    resolved through — a semver like ``"0.2.0"`` or the ``"active"`` pointer, or
    ``None`` when only the digest is known; ``digest`` is the immutable
    ``sha256:…`` identity. Pull/launch by digest, never by the bare tag.
    """

    slug: str
    digest: Optional[str] = None
    version: Optional[str] = None

    def pull_ref(self, registry: str) -> str:
        """``<registry>/<slug>@<digest>`` (immutable) when a digest is known,
        else the tag ref ``<registry>/<slug>:<version>``."""
        if self.digest:
            return f"{registry}/{self.slug}@{self.digest}"
        if self.version:
            return f"{registry}/{self.slug}:{self.version}"
        raise ValueError(f"AgentRef for {self.slug!r} has neither digest nor version")


def next_version(current: "str | VersionTag", level: str = "patch") -> VersionTag:
    """Compute the next semantic version from ``current`` by bumping ``level``.

    ``level`` is ``"major"``/``"minor"``/``"patch"`` (default patch). Thin wrapper
    over :mod:`semver`'s bump methods so the whole runtime computes generations
    the same way.
    """
    version = current if isinstance(current, VersionTag) else VersionTag.parse(str(current))
    bumps = {"major": version.bump_major, "minor": version.bump_minor, "patch": version.bump_patch}
    if level not in bumps:
        raise ValueError(f"level must be one of {sorted(bumps)}, got {level!r}")
    return bumps[level]()


class LocalRegistryError(AtriumError):
    """Raised when the local registry cannot be bootstrapped or reached."""


@dataclass(slots=True)
class RegistryConfig:
    """How the main process brings up the local registry (fixed infrastructure).

    Defaults bind to loopback for safe single-host development. For a deployment
    where in-sandbox agents (BuilderAgent / OpenShell) must reach the registry,
    set ``bind_host`` to ``"0.0.0.0"`` (or the host bridge IP) and point pushers
    at a name/IP that resolves from inside the sandboxes.
    """

    name: str = "local-registry"
    """Container name — also the hostname agents use to address the registry."""
    image: str = "registry:2"
    bind_host: str = "127.0.0.1"
    """Host interface the registry is published on (``0.0.0.0`` for cross-sandbox)."""
    host: str = "127.0.0.1"
    """Host used to build the endpoint + health URL (for the main process)."""
    host_port: int = 5000
    container_port: int = 5000
    data_volume: str = "atrium-registry-data"
    """Persistent named volume — the registry IS the ledger, so its data MUST
    survive restarts; without this, every built generation is lost on restart."""
    restart_policy: str = "always"
    health_timeout_s: float = 30.0
    labels: dict[str, str] = field(
        default_factory=lambda: {
            "atrium.component": "registry",
            # Fixed infrastructure: excluded from the self-evolution loop.
            "atrium.immutable": "true",
        }
    )
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """The pushable ``host:port`` (e.g. for Kaniko ``--destination`` / OpenShell ``--from``)."""
        return f"{self.host}:{self.host_port}"

    @property
    def health_url(self) -> str:
        """The registry readiness endpoint (``/v2/`` returns 200, or 401 with auth)."""
        return f"http://{self.host}:{self.host_port}/v2/"


def ensure_local_registry(config: Optional[RegistryConfig] = None) -> str:
    """Idempotently ensure the local registry container is running; return its
    endpoint (``host:port``).

    Reuses a running container, starts a stopped one, or creates one (with the
    persistent ledger volume) if absent — then health-gates on ``GET /v2/``
    before returning. Uses the host Docker daemon, so it must run only in the
    trusted main process, never inside an agent.

    :raises LocalRegistryError: if the Docker SDK/daemon is unavailable or the
        registry does not become healthy in time.
    """
    config = config or RegistryConfig()
    client = _docker_client()
    from docker.errors import APIError, NotFound

    try:
        container = client.containers.get(config.name)
    except NotFound:
        container = None
    except APIError as exc:
        raise LocalRegistryError(f"docker error inspecting registry: {exc}") from exc

    try:
        if container is None:
            logger.info("creating registry container %s from %s", config.name, config.image)
            client.containers.run(
                config.image,
                name=config.name,
                detach=True,
                restart_policy={"Name": config.restart_policy},
                ports={f"{config.container_port}/tcp": (config.bind_host, config.host_port)},
                volumes={config.data_volume: {"bind": _REGISTRY_DATA_DIR, "mode": "rw"}},
                environment=config.environment,
                labels=config.labels,
            )
        elif container.status != "running":
            logger.info("starting existing registry container %s", config.name)
            container.start()
        else:
            logger.debug("registry container %s already running", config.name)
    except APIError as exc:
        raise LocalRegistryError(f"failed to start registry container: {exc}") from exc

    _wait_until_healthy(config)
    logger.info("local registry ready at %s", config.endpoint)
    return config.endpoint


def registry_healthy(config: RegistryConfig) -> bool:
    """``True`` when ``GET /v2/`` is reachable (registry is up).

    Uses the stdlib ``urllib`` so the host package never imports ``httpx`` (that
    lives only in the agent container images). A ``401`` counts as "up" — the
    registry answers but has auth enabled.
    """
    try:
        with urllib.request.urlopen(config.health_url, timeout=2.0) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        return exc.code == 401
    except (urllib.error.URLError, OSError):
        return False


def stop_local_registry(
    config: Optional[RegistryConfig] = None, *, remove: bool = False
) -> None:
    """Stop (and optionally remove) the registry container.

    The persistent data volume — the ledger — is **never** touched, so the
    generation history survives a stop/remove and a later
    :func:`ensure_local_registry`.
    """
    config = config or RegistryConfig()
    try:
        client = _docker_client()
    except LocalRegistryError:
        return  # no SDK/daemon → nothing to stop
    from docker.errors import APIError, NotFound

    try:
        container = client.containers.get(config.name)
    except NotFound:
        return
    try:
        container.stop()
        if remove:
            container.remove()
    except APIError as exc:
        raise LocalRegistryError(f"failed to stop registry container: {exc}") from exc


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #
def _docker_client():
    """Return a Docker SDK client bound to the host daemon, or raise a clear error.

    Both the import and the connection are deferred to here so the module imports
    fine on hosts without the SDK/daemon (only the bootstrap path needs them).
    """
    try:
        import docker
    except ImportError as exc:
        raise LocalRegistryError(
            "the 'docker' SDK is required to bootstrap the local registry "
            "(`pip install docker`); this runs only in the trusted main process"
        ) from exc
    try:
        return docker.from_env()
    except docker.errors.DockerException as exc:
        raise LocalRegistryError(
            "cannot reach the host Docker daemon to manage the local registry"
        ) from exc


def _wait_until_healthy(config: RegistryConfig) -> None:
    """Poll ``/v2/`` until healthy or ``health_timeout_s`` elapses (capped backoff)."""
    deadline = time.monotonic() + config.health_timeout_s
    delay = 0.2
    while True:
        if registry_healthy(config):
            return
        if time.monotonic() >= deadline:
            raise LocalRegistryError(
                f"registry at {config.endpoint} did not become healthy within "
                f"{config.health_timeout_s}s"
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)


class RegistryClient:
    """Read / limited-write client for the local registry — the generation ledger.

    Speaks the registry **v2 HTTP API** over stdlib ``urllib`` (the host package
    never imports ``httpx``). Reads — :meth:`versions`, :meth:`digest`,
    :meth:`active`, :meth:`exists` — are open to any caller; :meth:`set_active`
    moves the ``<slug>:active`` pointer and is intended for the trusted Morpher
    only (the runtime restricts the credential, not this object).
    """

    def __init__(self, endpoint: str, *, scheme: str = "http", timeout: float = 5.0) -> None:
        self.endpoint = endpoint  # host:port the registry serves on
        self._base = f"{scheme}://{endpoint}"
        self._timeout = timeout

    @classmethod
    def from_config(cls, config: RegistryConfig, *, timeout: float = 5.0) -> "RegistryClient":
        """Build a client for the registry described by ``config`` (its host:port, HTTP)."""
        return cls(config.endpoint, scheme="http", timeout=timeout)

    # ---- reads ---------------------------------------------------------- #
    def versions(self, slug: str, *, semver_only: bool = True) -> list[str]:
        """Return ``slug``'s pushed version tags (its history), ascending.

        Excludes the mutable ``active`` pointer. With ``semver_only`` (default),
        non-semver tags are dropped and the rest are sorted semantically.
        """
        _, _, body = self._open("GET", f"/v2/{slug}/tags/list")
        if body is None:
            return []
        tags = [t for t in (json.loads(body).get("tags") or []) if t != ACTIVE_TAG]
        if not semver_only:
            return sorted(tags)
        parsed: list[VersionTag] = []
        for tag in tags:
            try:
                parsed.append(VersionTag.parse(tag))
            except ValueError:
                continue
        return [str(v) for v in sorted(parsed)]

    def digest(self, slug: str, reference: str) -> Optional[str]:
        """Resolve ``<slug>:<reference>`` (a tag or digest) to its immutable
        manifest digest (``sha256:…``), or ``None`` when it does not exist."""
        status, headers, _ = self._open(
            "HEAD", f"/v2/{slug}/manifests/{reference}", headers={"Accept": _MANIFEST_ACCEPT}
        )
        if status == 404:
            return None
        return headers.get("Docker-Content-Digest")

    def exists(self, slug: str, version: str) -> bool:
        """Whether a version tag already exists — the app-side collision guard
        (generations are immutable; never overwrite an existing version tag)."""
        return self.digest(slug, version) is not None

    def active(self, slug: str) -> Optional[AgentRef]:
        """The live generation (``<slug>:active``) as an :class:`AgentRef`, or None."""
        dig = self.digest(slug, ACTIVE_TAG)
        return AgentRef(slug=slug, digest=dig, version=ACTIVE_TAG) if dig else None

    # ---- write (Morpher-only) ------------------------------------------ #
    def set_active(self, slug: str, digest: str) -> None:
        """Point ``<slug>:active`` at an existing ``digest`` (promote / rollback).

        Re-tags by re-PUTting the existing manifest under the ``active`` tag —
        no blobs move. Raises if the digest is not already in the registry.
        """
        _, headers, manifest = self._open(
            "GET", f"/v2/{slug}/manifests/{digest}", headers={"Accept": _MANIFEST_ACCEPT}
        )
        if manifest is None:  # _open returns body=None only on 404
            raise LocalRegistryError(f"cannot set active: {slug}@{digest} not in registry")
        self._open(
            "PUT",
            f"/v2/{slug}/manifests/{ACTIVE_TAG}",
            data=manifest,
            headers={"Content-Type": headers.get("Content-Type", _MANIFEST_ACCEPT)},
        )

    # ---- HTTP plumbing -------------------------------------------------- #
    def _open(self, method: str, path: str, *, data=None, headers=None):
        """One registry HTTP call → ``(status, headers, body)``.

        A ``404`` is returned as ``(404, headers, None)`` (an expected "absent"
        answer for reads); any other transport/HTTP failure raises
        :class:`LocalRegistryError`.
        """
        req = urllib.request.Request(
            f"{self._base}{path}", method=method, data=data, headers=headers or {}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, exc.headers, None
            raise LocalRegistryError(f"registry {method} {path} -> HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LocalRegistryError(f"registry {method} {path} unreachable: {exc}") from exc
