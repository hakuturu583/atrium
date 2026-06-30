"""Python code-workspace sandbox (derived image + config).

The ``Dockerfile`` here ``FROM``s the Base Docker Image and preinstalls the
Python toolchain (interpreter, C compiler, ``uv``); :func:`build_sandbox_config`
pins a sandbox to the resulting ``local-registry/python_code_workspace_agent``
image.
"""

from __future__ import annotations

from atrium.agents.code_workspace_agent.sandbox.python.config import (
    IMAGE_REPOSITORY,
    build_sandbox_config,
)

__all__ = ["build_sandbox_config", "IMAGE_REPOSITORY"]
