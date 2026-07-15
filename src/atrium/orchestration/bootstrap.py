"""Bring up the orchestration dependency services by driving ``docker-compose``.

Running a workboard needs a Prefect **server** (its API + UI) backed by a
**Postgres** database, and — for tracing — an **Arize Phoenix** collector. These
are declared once in the project's ``docker-compose.yaml``; this module starts
them programmatically via `python-on-whales
<https://gabrieldemarmiesse.github.io/python-on-whales/>`_ (a Pythonic wrapper
over the ``docker``/``docker compose`` CLI), so the service definitions live in
**one** place — the compose file — instead of being duplicated in Python. It runs
``docker compose up --wait`` for just the dependency services and health-gates on
the Prefect API.

Scope: these are the external services Atrium *depends on*. The workboard worker
is Atrium's own code and is served in-process (see
:func:`atrium.orchestration.flow.serve_workboards`), so it is deliberately **not**
brought up here — only ``prefect-db``, ``prefect-server`` and ``phoenix`` are.

Run only by the trusted Atrium main process (it drives the host Docker daemon,
never exposed to agents). The docker CLI + compose plugin must be installed;
``python-on-whales`` is imported lazily so importing this module needs neither.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from atrium.core.errors import AtriumError

logger = logging.getLogger("atrium.orchestration.bootstrap")

__all__ = [
    "OrchestrationBootstrapError",
    "BootstrapConfig",
    "OrchestrationEndpoints",
    "PREFECT_SERVICES",
    "PHOENIX_SERVICE",
    "ensure_orchestration_services",
    "stop_orchestration_services",
]

#: Compose service names for the Prefect dependency stack (server + its DB). The
#: workboard worker is intentionally excluded — Atrium serves the flow in-process.
PREFECT_SERVICES = ("prefect-db", "prefect-server")
#: Compose service name for the trace collector.
PHOENIX_SERVICE = "phoenix"


class OrchestrationBootstrapError(AtriumError):
    """Raised when the dependency services cannot be started or do not become healthy."""


@dataclass(slots=True)
class BootstrapConfig:
    """Where the compose file is and how the host reaches the started services.

    The service *definitions* live in the compose file; this only carries what the
    bootstrap needs on top: which file to drive and the host-side ports to build
    endpoint/health URLs from (kept in sync with the compose port mappings).
    """

    compose_file: str = os.environ.get("ATRIUM_COMPOSE_FILE", "docker-compose.yaml")
    project_name: Optional[str] = None
    host: str = "127.0.0.1"
    prefect_port: int = 4200
    phoenix_ui_port: int = 6006
    health_timeout_s: float = 120.0

    @property
    def prefect_api_url(self) -> str:
        """``PREFECT_API_URL`` the control plane / worker point at."""
        return f"http://{self.host}:{self.prefect_port}/api"

    @property
    def prefect_health_url(self) -> str:
        return f"http://{self.host}:{self.prefect_port}/api/health"

    @property
    def otlp_endpoint(self) -> str:
        """``OTEL_EXPORTER_OTLP_ENDPOINT`` for the host process' span exporter."""
        return f"http://{self.host}:{self.phoenix_ui_port}/v1/traces"


@dataclass(slots=True)
class OrchestrationEndpoints:
    """What the bootstrap resolved — hand these to the process' env/config."""

    prefect_api_url: str
    otlp_endpoint: Optional[str] = None


def ensure_orchestration_services(
    *, with_phoenix: bool = True, config: Optional[BootstrapConfig] = None
) -> OrchestrationEndpoints:
    """Start the workboard dependency services from the compose file; return endpoints.

    Runs ``docker compose up --wait`` for ``prefect-db`` + ``prefect-server`` (and
    ``phoenix`` unless ``with_phoenix=False``), so already-running services are a
    no-op and the call blocks until they are healthy, then confirms the Prefect
    API answers. The caller sets the returned endpoints in the process environment
    (``PREFECT_API_URL`` / ``OTEL_EXPORTER_OTLP_ENDPOINT``) before serving or
    kicking jobs.

    :raises OrchestrationBootstrapError: on compose/daemon trouble or health timeout.
    """
    config = config or BootstrapConfig()
    services = list(PREFECT_SERVICES) + ([PHOENIX_SERVICE] if with_phoenix else [])
    docker = _compose_client(config)
    from python_on_whales.exceptions import DockerException

    logger.info("docker compose up --wait %s (%s)", " ".join(services), config.compose_file)
    try:
        docker.compose.up(services=services, detach=True, wait=True)
    except DockerException as exc:
        raise OrchestrationBootstrapError(f"`docker compose up` failed: {exc}") from exc

    _confirm_healthy(config)
    logger.info("orchestration services ready: api=%s", config.prefect_api_url)
    return OrchestrationEndpoints(
        prefect_api_url=config.prefect_api_url,
        otlp_endpoint=config.otlp_endpoint if with_phoenix else None,
    )


def stop_orchestration_services(
    *, config: Optional[BootstrapConfig] = None, remove_volumes: bool = False
) -> None:
    """Stop the dependency services (``docker compose down``). Volumes kept by default."""
    config = config or BootstrapConfig()
    try:
        docker = _compose_client(config)
    except OrchestrationBootstrapError:
        return  # no CLI/daemon → nothing to stop
    from python_on_whales.exceptions import DockerException

    try:
        docker.compose.down(volumes=remove_volumes)
    except DockerException as exc:
        raise OrchestrationBootstrapError(f"`docker compose down` failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #
def _compose_client(config: BootstrapConfig):
    """A python-on-whales client bound to the project's compose file, or a clear error.

    Import is deferred here so the module imports fine without the library; only
    the bootstrap path needs it (trusted main process only).
    """
    try:
        from python_on_whales import DockerClient
    except ImportError as exc:
        raise OrchestrationBootstrapError(
            "'python-on-whales' is required to bootstrap orchestration services "
            "(`pip install python-on-whales`); this runs only in the trusted main process"
        ) from exc
    if not os.path.exists(config.compose_file):
        raise OrchestrationBootstrapError(
            f"compose file not found: {config.compose_file!r} "
            "(set ATRIUM_COMPOSE_FILE or pass BootstrapConfig(compose_file=...))"
        )
    return DockerClient(
        compose_files=[config.compose_file], compose_project_name=config.project_name
    )


def _confirm_healthy(config: BootstrapConfig) -> None:
    """Confirm the Prefect API answers (``--wait`` already gated on health; a final
    probe turns any lingering not-ready into a clear error)."""
    deadline = time.monotonic() + config.health_timeout_s
    delay = 0.3
    while True:
        if _http_ok(config.prefect_health_url):
            return
        if time.monotonic() >= deadline:
            raise OrchestrationBootstrapError(
                f"prefect API {config.prefect_api_url} did not answer within "
                f"{config.health_timeout_s:.0f}s"
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 3.0)


def _http_ok(url: str, *, ok_below: int = 300) -> bool:
    """Whether ``GET url`` answers with a status below ``ok_below`` (readiness probe).

    Uses stdlib ``urllib`` so the host package never imports ``httpx``.
    """
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.status < ok_below
    except urllib.error.HTTPError as exc:
        return exc.code < ok_below
    except (urllib.error.URLError, OSError):
        return False
