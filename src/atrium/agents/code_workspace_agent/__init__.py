"""Code-workspace agents — sandboxes where coding agents write & test code.

This package bundles one generation of the code-workspace family:

* ``agent.py``        — :class:`CodeWorkSpaceAgent`, the language-agnostic base.
* ``python_agent.py`` — :class:`PythonCodeWorkspaceAgent`, a Python derivation.
* ``sandbox/``        — the **Base Docker Image** (``Dockerfile``, a multi-stage
  build guaranteeing the ``git`` + ``gh`` push toolchain), the egress
  ``policy.yaml`` and the :class:`SandboxConfig` factory; plus ``sandbox/python/``
  with the *derived* image (``FROM`` the base) preinstalling the Python toolchain.

The two inheritance axes are kept in lockstep — the Python agent subclasses the
base agent *and* its image derives from the Base Docker Image — so adding a new
language workspace is "subclass + a ``FROM``-the-base Dockerfile".

``__version__`` is the single source of truth for the agent version and the image
tags ``local-registry/codeworkspace_base:<__version__>`` and
``local-registry/python_code_workspace_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.code_workspace_agent.agent import (
    CodeWorkSpaceAgent,
    WorkspaceConfig,
)
from atrium.agents.code_workspace_agent.python_agent import PythonCodeWorkspaceAgent

__all__ = [
    "CodeWorkSpaceAgent",
    "PythonCodeWorkspaceAgent",
    "WorkspaceConfig",
    "__version__",
]
