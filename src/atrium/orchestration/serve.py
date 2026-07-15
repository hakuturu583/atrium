"""Entrypoint that starts the workboard runtime: bootstrap deps, then serve.

By default this *is* Atrium's orchestration startup: it brings up the dependency
services (Postgres + Prefect server + Phoenix) via the host Docker SDK — no
separate ``docker compose up`` — then serves the ``atrium-workboard`` flow
in-process and executes runs created against it (e.g. by
:func:`atrium.orchestration.kick.submit_workboard`)::

    python -m atrium.orchestration.serve            # bootstrap deps + serve

Pass ``--no-bootstrap`` when the dependency services are already up and only
``PREFECT_API_URL`` should be honoured — e.g. the ``prefect-worker`` container in
``docker-compose``, which must not (and should not) reach the host Docker daemon::

    PREFECT_API_URL=http://prefect-server:4200/api \\
        python -m atrium.orchestration.serve --no-bootstrap
"""

from __future__ import annotations

import argparse
import os

from atrium.core.telemetry import configure_tracing, shutdown_tracing
from atrium.orchestration.bootstrap import ensure_orchestration_services
from atrium.orchestration.flow import serve_workboards


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Atrium workboard runtime.")
    parser.add_argument(
        "--name",
        default="default",
        help="Deployment name under the atrium-workboard flow (default: 'default').",
    )
    parser.add_argument(
        "--no-bootstrap",
        dest="bootstrap",
        action="store_false",
        help="Skip auto-starting dependency services (they are already running).",
    )
    parser.add_argument(
        "--no-phoenix",
        dest="phoenix",
        action="store_false",
        help="When bootstrapping, skip the Phoenix trace collector.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        # Auto-start the dependency services via the host Docker SDK and point
        # this process at them (only PREFECT_API_URL not already set is honoured).
        endpoints = ensure_orchestration_services(with_phoenix=args.phoenix)
        os.environ.setdefault("PREFECT_API_URL", endpoints.prefect_api_url)
        if endpoints.otlp_endpoint:
            os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", endpoints.otlp_endpoint)

    # One tracer for the flow process so board runs ship spans to Phoenix.
    configure_tracing(service_name="atrium-workboard")
    try:
        serve_workboards(name=args.name)
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
