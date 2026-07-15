"""Entrypoint that serves the workboard flow (the ``prefect-worker`` service).

Registers the ``atrium-workboard`` flow as a Prefect deployment and executes runs
created against it — e.g. by :func:`atrium.orchestration.kick.submit_workboard`.
Requires the ``orchestration`` extra and a reachable Prefect API::

    PREFECT_API_URL=http://prefect-server:4200/api \\
        python -m atrium.orchestration.serve --name default

Run from the compose ``orchestration`` profile, which wires those in.
"""

from __future__ import annotations

import argparse

from atrium.core.telemetry import configure_tracing, shutdown_tracing
from atrium.orchestration.flow import serve_workboards


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Atrium workboard flow.")
    parser.add_argument(
        "--name",
        default="default",
        help="Deployment name under the atrium-workboard flow (default: 'default').",
    )
    args = parser.parse_args()

    # One tracer for the flow process so board runs ship spans to Phoenix.
    configure_tracing(service_name="atrium-workboard")
    try:
        serve_workboards(name=args.name)
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
