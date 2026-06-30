"""Thin async adapter over the NVIDIA OpenShell **CLI**.

`OpenShell <https://github.com/NVIDIA/openshell>`_ is a Rust/CLI tool — "the
safe, private runtime for autonomous AI agents" — not a Python library. It is
driven by subcommands::

    openshell policy set <name> --policy <yaml>
    openshell sandbox create --from <image> [--gpu] --name <name>
    openshell sandbox exec   <name> -- <cmd...>
    openshell sandbox delete <name>
    openshell sandbox list

This module keeps the Python-level ``Sandbox.create()`` API the rest of Atrium
expects while driving the CLI underneath via :func:`asyncio.create_subprocess_exec`.

The CLI is optional at import time: if ``openshell`` is not on ``PATH`` the
module still imports, and any lifecycle/exec call raises :class:`SandboxError`
with guidance. Exact subcommand spellings vary across OpenShell versions, so the
command templates are centralized below for easy adjustment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Mapping, Optional

from atrium.core.errors import SandboxError
from atrium.core.types import ExecutionResult, SandboxConfig

logger = logging.getLogger("atrium.sandbox")

OPENSHELL_BIN = "openshell"

__all__ = ["Sandbox", "openshell_available", "OPENSHELL_BIN"]


def openshell_available() -> bool:
    """``True`` when the OpenShell CLI is discoverable on ``PATH``."""
    return shutil.which(OPENSHELL_BIN) is not None


def _require_cli() -> str:
    path = shutil.which(OPENSHELL_BIN)
    if path is None:
        raise SandboxError(
            f"'{OPENSHELL_BIN}' CLI not found on PATH. Install NVIDIA OpenShell "
            "(https://github.com/NVIDIA/openshell) to run agent sandboxes."
        )
    return path


def _resolve_secret_env(
    secret_env: Mapping[str, str], environ: Mapping[str, str]
) -> dict[str, str]:
    """Resolve ``{container_var: host_var}`` against ``environ``.

    Returns ``{container_var: value}`` for every mapping whose ``host_var`` is set
    on the host; mappings with an unset ``host_var`` are skipped. Kept pure (no
    process state) so it is unit-testable without the OpenShell CLI.
    """
    return {
        container_var: environ[host_var]
        for container_var, host_var in secret_env.items()
        if host_var in environ
    }


async def _run(
    *args: str,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ExecutionResult:
    """Run an ``openshell`` subcommand and capture its result.

    ``env`` overrides the child process environment (used to hand secret values to
    OpenShell out-of-band, so they never appear in ``args``/the command line). It
    is passed straight through; ``create_subprocess_exec`` does not mutate it.
    """
    _require_cli()
    command = f"{OPENSHELL_BIN} {' '.join(shlex.quote(a) for a in args)}"
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            OPENSHELL_BIN,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise SandboxError(f"openshell command timed out: {command}") from exc
    except OSError as exc:
        raise SandboxError(f"failed to spawn openshell: {command}") from exc
    return ExecutionResult(
        command=command,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=(out or b"").decode("utf-8", "replace"),
        stderr=(err or b"").decode("utf-8", "replace"),
        duration_s=time.monotonic() - start,
    )


@dataclass
class Sandbox:
    """A handle to a single OpenShell sandbox container (1:1 with an agent)."""

    name: str
    image: str
    config: SandboxConfig
    _running: bool = field(default=False, repr=False)

    @property
    def is_running(self) -> bool:
        """Whether this sandbox is believed to be running."""
        return self._running

    @classmethod
    async def create(
        cls,
        image: str,
        config: SandboxConfig,
        *,
        name: Optional[str] = None,
    ) -> "Sandbox":
        """Create and start an OpenShell sandbox for ``image`` under ``config``.

        Applies the network/permission policy (``config.policy_path`` if pinned,
        otherwise a freshly-rendered one) then launches the container with GPU
        passthrough when requested.
        """
        _require_cli()
        name = name or f"atrium-{uuid.uuid4().hex[:12]}"
        sandbox = cls(name=name, image=image, config=config)

        # 1) Apply the egress/permission policy.
        policy_path = config.policy_path
        tmp_path: Optional[str] = None
        if policy_path is None:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", prefix=f"{name}-policy-", delete=False
            )
            handle.write(config.render_policy_yaml())
            handle.flush()
            handle.close()
            policy_path = tmp_path = handle.name
        try:
            result = await _run("policy", "set", name, "--policy", policy_path)
            if not result.succeeded:
                raise SandboxError(
                    f"openshell policy set failed for {name}: {result.stderr.strip()}"
                )

            # 2) Create the sandbox container.
            args = ["sandbox", "create", "--from", image, "--name", name]
            for gpu in config.device_requests:
                args.extend(gpu.as_cli_flags())
            if config.cpus is not None:
                args.extend(["--cpus", str(config.cpus)])
            if config.memory is not None:
                args.extend(["--memory", config.memory])
            for host, container in config.volumes.items():
                args.extend(["--volume", f"{host}:{container}"])
            for key, value in config.env.items():
                args.extend(["--env", f"{key}={value}"])

            # Credentials forwarded by reference: resolve the host values from this
            # process's environment and pass each as a *name-only* `--env NAME`
            # flag, so OpenShell reads the value from the child environment below
            # rather than from the (process-visible) command line.
            secrets = _resolve_secret_env(config.secret_env, os.environ)
            for container_var in secrets:
                args.extend(["--env", container_var])
            child_env = {**os.environ, **secrets} if secrets else None
            if secrets:
                logger.debug(
                    "forwarding %d secret env var(s) to sandbox %s: %s",
                    len(secrets), name, sorted(secrets),  # names only, never values
                )

            result = await _run(*args, env=child_env)
            if not result.succeeded:
                raise SandboxError(
                    f"openshell sandbox create failed for {name}: {result.stderr.strip()}"
                )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:  # pragma: no cover
                    logger.debug("could not remove temp policy %s", tmp_path)

        sandbox._running = True
        logger.info("started OpenShell sandbox %s from %s", name, image)
        return sandbox

    async def exec(self, command: str, *, timeout: Optional[float] = None) -> ExecutionResult:
        """Execute ``command`` inside the running sandbox via a login shell."""
        if not self._running:
            raise SandboxError(f"sandbox {self.name} is not running")
        return await _run(
            "sandbox", "exec", self.name, "--", "bash", "-lc", command, timeout=timeout
        )

    async def delete(self) -> None:
        """Destroy the sandbox container and mark this handle stopped."""
        if not self._running:
            return
        result = await _run("sandbox", "delete", self.name)
        self._running = False
        if not result.succeeded:
            raise SandboxError(
                f"openshell sandbox delete failed for {self.name}: {result.stderr.strip()}"
            )
        logger.info("deleted OpenShell sandbox %s", self.name)
