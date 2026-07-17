"""SandboxConfig factory for the PrefectRunnerAgent container — least privilege.

The runner image inherits the Python code-workspace image (so a Python interpreter
and ``uv`` are present) and adds ``prefect`` + the trusted ``atrium_dispatch``
primitive. Its isolation envelope, however, is deliberately *tighter* than a code
workspace: it is **WAN-isolated** (``NetworkMode.INTERNAL`` — only the host-local
control LAN, enough to dispatch to subagents over A2A but no public egress) and it
forwards **no GitHub credentials**. Those are the two privileges a code workspace
holds that a flow runner must not.

This is a thin wrapper over the base code-workspace
:func:`~atrium.agents.code_workspace_agent.sandbox.config.build_sandbox_config`
that flips those knobs and pins the runner's own image + egress ``policy.yaml``.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Mapping, Optional

from atrium.agents.code_workspace_agent.sandbox.config import (
    build_sandbox_config as _build_base_sandbox_config,
)
from atrium.agents.prefect_runner_agent.sandbox.dispatch import ENDPOINTS_ENV
from atrium.core.types import NetworkMode, SandboxConfig

__all__ = ["build_sandbox_config", "IMAGE_REPOSITORY", "POLICY_PATH"]

#: Local registry repository for the runner image (FROM the Python workspace image).
IMAGE_REPOSITORY = "local-registry/prefect_runner_agent"

#: The runner's own OpenShell egress/permission policy (WAN default-deny; the
#: control-LAN A2A reach comes from NetworkMode.INTERNAL, not this file).
POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.yaml")


def build_sandbox_config(
    version: str,
    *,
    memory: Optional[str] = "4Gi",
    cpus: Optional[float] = 2.0,
    env: Optional[Mapping[str, str]] = None,
    secret_env: Optional[Mapping[str, str]] = None,
    dispatch_endpoints: Optional[Mapping[str, str]] = None,
) -> SandboxConfig:
    """Build the least-privilege runner SandboxConfig for ``version``.

    Differences from the code-workspace envelope, by construction:

    * ``network=NetworkMode.INTERNAL`` (→ ``internal=True``) — WAN-isolated; only
      the host-local control LAN is reachable, which is all a flow needs to
      dispatch to subagents over A2A.
    * ``forward_github_token=False`` — the runner holds no GitHub credentials.
    * pinned to the runner image and its own ``policy.yaml``.

    ``dispatch_endpoints`` is the ``{slug: url}`` allow-list of subagents a
    generated flow may reach; it is injected as the ``ATRIUM_DISPATCH_ENDPOINTS``
    env the trusted :func:`atrium_dispatch` primitive resolves against, so the
    deployment declares reachable subagents explicitly rather than relying only on
    the ``http://<slug>.local`` convention. ``secret_env`` is still accepted for any
    non-GitHub credential a deployment must inject by reference.
    """
    sandbox_env: dict[str, str] = dict(env or {})
    if dispatch_endpoints:
        # Sorted for a stable, reproducible env value across builds.
        sandbox_env[ENDPOINTS_ENV] = json.dumps(dict(sorted(dispatch_endpoints.items())))

    cfg = _build_base_sandbox_config(
        version,
        repository=IMAGE_REPOSITORY,
        network=NetworkMode.INTERNAL,
        memory=memory,
        cpus=cpus,
        env=sandbox_env,
        forward_github_token=False,
        secret_env=secret_env,
        agent_slug="prefect_runner_agent",
    )
    # Pin the runner's own egress policy rather than the base workspace's.
    return replace(cfg, policy_path=POLICY_PATH if os.path.exists(POLICY_PATH) else None)
