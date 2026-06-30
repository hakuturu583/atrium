"""SandboxConfig factory for code-workspace containers — the workspace envelope.

This is the package-internal source of truth for the code-workspace security
posture. Unlike the inference/builder envelopes (which are WAN-isolated), a code
workspace is allowed public-internet egress because its job is to fetch
dependencies, clone repositories and **push code to GitHub via ``gh``**. The
isolation it keeps instead — non-root, no privilege escalation, all capabilities
dropped, read-only root with explicit writable scratch, no Docker socket, no GPU
— is encoded in the shipped ``policy.yaml`` and re-checked at construction by
:meth:`~atrium.agents.code_workspace_agent.agent.CodeWorkSpaceAgent._enforce_workspace_policy`.

Language-specific images (e.g. the PythonCodeWorkspaceAgent image) inherit the
Base Docker Image and reuse this factory with a different ``IMAGE_REPOSITORY``
via :func:`build_sandbox_config`'s ``repository`` argument.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from atrium.core.types import NetworkMode, SandboxConfig

__all__ = [
    "build_sandbox_config",
    "WORKSPACE",
    "BASE_IMAGE_REPOSITORY",
    "DEFAULT_TOKEN_ENV",
    "POLICY_PATH",
]

#: Project tree inside the sandbox (single source of truth; also the workdir).
WORKSPACE = "/workspace"

#: Local registry repository for the Base Docker Image (codeworkspace_base).
BASE_IMAGE_REPOSITORY = "local-registry/codeworkspace_base"

#: Default name of the env var carrying the GitHub token used by ``gh``/``git``.
DEFAULT_TOKEN_ENV = "GH_TOKEN"

#: The OpenShell egress/permission policy shipped alongside this config.
POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.yaml")


def build_sandbox_config(
    version: str,
    *,
    repository: str = BASE_IMAGE_REPOSITORY,
    network: NetworkMode = NetworkMode.BRIDGE,
    memory: Optional[str] = "4g",
    cpus: Optional[float] = 2.0,
    env: Optional[Mapping[str, str]] = None,
    gh_token: Optional[str] = None,
    token_env: str = DEFAULT_TOKEN_ENV,
    agent_slug: str = "codeworkspace_base",
) -> SandboxConfig:
    """Build the code-workspace SandboxConfig for ``version`` of ``repository``.

    The envelope, by construction:

    * ``network=BRIDGE`` + ``internal=False`` (default) — WAN egress permitted, so
      the workspace can reach GitHub (``gh``/``git push``) and package registries.
      Pass ``network=NetworkMode.INTERNAL`` to run a workspace offline (e.g.
      against a local mirror) with no public egress.
    * ``device_requests=[]`` — no GPU passthrough (a coding workspace needs none).
    * ``volumes={}`` — the host Docker socket is *never* mounted.
    * pinned to the shipped ``policy.yaml`` (read-only root, no_new_privileges,
      drop ALL caps) — the non-egress half of the isolation.

    Parameters
    ----------
    repository:
        Image repository (``<repository>:<version>``). Derived language images
        (e.g. ``local-registry/python_code_workspace_agent``) pass their own.
    gh_token:
        Optional GitHub token; when given it is injected into the sandbox as
        ``token_env`` so ``gh``/``git`` can authenticate for clone/push.
    """
    sandbox_env: dict[str, str] = {"ATRIUM_WORKSPACE": WORKSPACE}
    if env:
        sandbox_env.update(env)
    if gh_token:
        sandbox_env[token_env] = gh_token

    return SandboxConfig(
        image=f"{repository}:{version}",
        network=network,
        # WAN reachable only in BRIDGE mode; INTERNAL/NONE stay WAN-isolated.
        internal=network is not NetworkMode.BRIDGE,
        device_requests=[],
        cpus=cpus,
        memory=memory,
        env=sandbox_env,
        workdir=WORKSPACE,
        policy_path=POLICY_PATH if os.path.exists(POLICY_PATH) else None,
        labels={"atrium.agent": agent_slug, "atrium.version": str(version)},
    )
