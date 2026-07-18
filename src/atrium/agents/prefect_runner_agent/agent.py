"""``PrefectRunnerAgent`` — run a generated flow under least privilege.

See the package docstring for the role. This module is the agent class: it
subclasses :class:`~atrium.agents.code_workspace_agent.python_agent.PythonCodeWorkspaceAgent`
to inherit the ``{files, commands}`` execution machinery (stage files → run
commands → structured ``workspace_result`` reply, all in an isolated sandbox), and
re-asserts the tightened envelope at construction (:meth:`_enforce_runner_policy`)
so a caller-supplied ``sandbox_config`` can never re-open WAN egress or hand the
runner GitHub push access.
"""

from __future__ import annotations

from typing import Mapping, Optional

from atrium.agents.code_workspace_agent.agent import WorkspaceConfig
from atrium.agents.code_workspace_agent.python_agent import PythonCodeWorkspaceAgent
from atrium.agents.prefect_runner_agent.sandbox import build_sandbox_config
from atrium.core.errors import PolicyViolationError
from atrium.core.types import NetworkMode, SandboxConfig, VersionTag

__all__ = ["PrefectRunnerAgent"]


class PrefectRunnerAgent(PythonCodeWorkspaceAgent):
    """Minimal-privilege executor for a generated Prefect ``flow.py``.

    The flow is staged and run exactly like any code-workspace task, but the
    envelope is WAN-isolated (only the control LAN, for A2A dispatch to subagents),
    holds no GitHub credentials, and refuses git *push* requests — the two
    privileges the inherited workspace has that a flow runner must not.
    """

    #: Image slug → ``local-registry/prefect_runner_agent:<version>``.
    AGENT_SLUG = "prefect_runner_agent"

    #: A flow runner never runs a test suite; it runs the generated flow directly.
    DEFAULT_TEST_COMMAND = None

    #: The runner holds no GitHub credentials and its job is to run a flow, not to
    #: publish code — so a git push/PR request is refused at parse time (base gate).
    ALLOWS_GIT = False

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        config: Optional[WorkspaceConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        dispatch_endpoints: Optional[Mapping[str, str]] = None,
    ) -> None:
        from atrium.agents.prefect_runner_agent import __version__

        version = version or __version__
        # Pin the least-privilege runner envelope unless the caller overrides it
        # (any override is still re-checked by _enforce_runner_policy below).
        # ``dispatch_endpoints`` (the {slug: url} subagent allow-list) is injected
        # into the sandbox env for the trusted atrium_dispatch primitive.
        sandbox_config = sandbox_config or build_sandbox_config(
            str(version), dispatch_endpoints=dispatch_endpoints
        )
        super().__init__(agent_id, version, config=config, sandbox_config=sandbox_config)
        self._enforce_runner_policy()

    # ------------------------------------------------------------------ #
    # Security envelope (re-checked at construction)                     #
    # ------------------------------------------------------------------ #
    def _enforce_runner_policy(self) -> None:
        """Refuse any config that breaks the least-privilege runner envelope.

        Extends the inherited workspace checks (no GPU, no Docker socket) with the
        two the runner adds: it must be **WAN-isolated** (only the control LAN, for
        A2A dispatch), never on the WAN bridge.
        """
        cfg = self.sandbox_config
        if cfg.network == NetworkMode.BRIDGE or not cfg.internal:
            raise PolicyViolationError(
                f"{type(self).__name__} must be WAN-isolated "
                f"(network={cfg.network.value}, internal={cfg.internal}); "
                "a flow runner only needs the control LAN for A2A dispatch"
            )

    # ------------------------------------------------------------------ #
    # Language hooks                                                     #
    # ------------------------------------------------------------------ #
    def setup_commands(self) -> list[str]:
        """No dependency sync: prefect + ``atrium_dispatch`` are in the image.

        The runner is WAN-isolated, so there is no package registry to sync
        against; everything a generated flow may import is baked into the image.
        """
        return []
