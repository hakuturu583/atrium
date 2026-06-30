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

import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from atrium.core.errors import AtriumError

logger = logging.getLogger("atrium.registry")

__all__ = [
    "RegistryConfig",
    "LocalRegistryError",
    "ensure_local_registry",
    "registry_healthy",
    "stop_local_registry",
    "image_ref_from_tag",
]

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
