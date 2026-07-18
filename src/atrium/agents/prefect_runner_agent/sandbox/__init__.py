"""Sandbox definition for the PrefectRunnerAgent (derived image, tight policy).

The ``Dockerfile`` here ``FROM``s the Python code-workspace image and preinstalls
``prefect`` + the trusted ``atrium_dispatch`` primitive; :func:`build_sandbox_config`
pins a sandbox to the resulting ``local-registry/prefect_runner_agent`` image with
a WAN-isolated, no-credentials envelope (see ``config.py`` and ``policy.yaml``).
"""

from __future__ import annotations

from atrium.agents.prefect_runner_agent.sandbox.config import (
    IMAGE_REPOSITORY,
    POLICY_PATH,
    build_sandbox_config,
)

__all__ = ["build_sandbox_config", "IMAGE_REPOSITORY", "POLICY_PATH"]
