"""``PythonCodeWorkspaceAgent`` — a Python-specialised code workspace.

This is the concrete derivation the package is built to demonstrate: by
subclassing :class:`~atrium.agents.code_workspace_agent.agent.CodeWorkSpaceAgent`
and pointing at the *derived* image (which ``FROM``s the Base Docker Image and
preinstalls the Python interpreter, a C compiler and the ``uv`` package manager),
a coding agent gets a ready-to-use Python toolchain while keeping the inherited
``git`` + ``gh`` push guarantee.

    BaseAgent → CodeWorkSpaceAgent → PythonCodeWorkspaceAgent
"""

from __future__ import annotations

from typing import Optional

from atrium.agents.code_workspace_agent.agent import CodeWorkSpaceAgent, WorkspaceConfig
from atrium.agents.code_workspace_agent.sandbox.python import build_sandbox_config
from atrium.core.types import SandboxConfig, VersionTag

__all__ = ["PythonCodeWorkspaceAgent"]


class PythonCodeWorkspaceAgent(CodeWorkSpaceAgent):
    """Code workspace with a preinstalled Python toolchain (interpreter + ``uv``)."""

    #: Image slug → ``local-registry/python_code_workspace_agent:<version>``.
    AGENT_SLUG = "python_code_workspace_agent"

    #: Run the project's test suite with pytest when a request asks to test.
    DEFAULT_TEST_COMMAND = "uv run --frozen pytest"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        config: Optional[WorkspaceConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        from atrium.agents.code_workspace_agent import __version__

        version = version or __version__
        # Pin to the derived Python image rather than the base workspace image.
        sandbox_config = sandbox_config or build_sandbox_config(str(version))
        super().__init__(agent_id, version, config=config, sandbox_config=sandbox_config)

    def setup_commands(self) -> list[str]:
        """Sync the project's locked dependencies before running its commands.

        ``uv sync`` is a no-op-cheap when there is nothing to install, so it is
        safe to run for any Python project; ``|| true`` keeps a repo without a
        ``uv`` lockfile from failing the whole task at setup time.
        """
        return ["uv sync --frozen || uv sync || true"]
