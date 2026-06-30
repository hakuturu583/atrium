"""``BuilderAgent`` — the fixed, non-evolving image builder for Atrium.

When a TaskAgent produces a new generation of an agent (source + Dockerfile),
something has to turn it into a container image. ``BuilderAgent`` is that step,
and it is the runtime's most important security chokepoint:

* **Immutability (fixed infrastructure).** This agent is explicitly excluded from
  the self-evolution loop (:data:`BuilderAgent.IMMUTABLE`); its code and its
  sandbox definition are only ever changed by a human, never auto-rewritten.
* **Fully rootless build.** It never mounts the host ``/var/run/docker.sock`` and
  never runs ``--privileged``. Builds run with **Kaniko** in user space, inside
  the agent's own OpenShell sandbox.
* **Isolated, independent build + push.** The build happens in a WAN-isolated
  sandbox and pushes only to the internal ``local-registry``.

The security envelope is guaranteed by the package-internal sandbox definition
(:mod:`atrium.agents.builder_agent.sandbox` — ``config.py`` + ``policy.yaml`` +
``Dockerfile``) and re-asserted at construction by :meth:`_enforce_build_policy`.

The whole exchange is A2A: a build request arrives as a :class:`Message` carrying
a structured ``DataPart`` (target name/version + a ``{filename: content}`` map),
and the reply is a :class:`Message` reporting success (with the image tag) or
failure (with the Kaniko logs, so the TaskAgent can re-fix and retry). Every step
is wrapped in OpenTelemetry spans inherited from :class:`BaseAgent`, so the build
shows up stitched into the same Phoenix trace as the request.
"""

from __future__ import annotations

import base64
import re
import shlex
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional

from atrium.agents.builder_agent.sandbox import (
    DEFAULT_REGISTRY,
    WORKSPACE,
    build_sandbox_config,
)
from atrium.core import telemetry as tel
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import PolicyViolationError
from atrium.core.types import ExecutionResult, NetworkMode, SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role,
    data_part,
    get_message_data,
    text_message,
)

__all__ = ["BuilderAgent"]

# ---- A2A vocabulary (metadata "kind" + DataPart "status"/"type") ----------- #
KIND_BUILD = "build"
STATUS_OK = "ok"
STATUS_ERROR = "error"
RESULT_TYPE = "build_result"

#: Default Dockerfile name within the build context.
DEFAULT_DOCKERFILE = "Dockerfile"
#: Default wall-clock cap for a single Kaniko build (seconds).
DEFAULT_BUILD_TIMEOUT_S = 1800.0
#: How much of the build log (chars) to echo back to the requester.
_LOG_TAIL = 8192

#: A valid image-repository path component (lowercase, no command-injection chars).
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
#: A conservative build-arg key (env-var-like); values are shell-quoted regardless.
_BUILD_ARG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class BuilderAgent(BaseAgent):
    """Fixed-infrastructure agent that builds container images with rootless Kaniko.

    Parameters
    ----------
    agent_id:
        Unique id (also the sandbox name).
    version:
        Agent version; defaults to the package ``__version__`` (drives the image
        tag ``local-registry/builder_agent:<version>``).
    sandbox_config:
        Override the build envelope. Defaults to
        :func:`~atrium.agents.builder_agent.sandbox.build_sandbox_config`; any
        override is still validated by :meth:`_enforce_build_policy`.
    registry:
        Internal registry the agent pushes built images to.
    build_timeout_s:
        Wall-clock cap for a single Kaniko invocation.
    """

    #: Image slug → ``local-registry/builder_agent:<version>``.
    AGENT_SLUG = "builder_agent"

    #: Fixed infrastructure: this agent is excluded from the self-evolution loop.
    #: It is only ever modified by an explicit, human-approved change — never by
    #: another agent rewriting it. The evolution engine must honor this flag.
    IMMUTABLE = True

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        sandbox_config: Optional[SandboxConfig] = None,
        registry: str = DEFAULT_REGISTRY,
        build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    ) -> None:
        from atrium.agents.builder_agent import __version__

        version = version or __version__
        sandbox_config = sandbox_config or build_sandbox_config(str(version), registry=registry)
        super().__init__(agent_id, version, sandbox_config)

        self.registry = registry
        self.build_timeout_s = build_timeout_s
        self._enforce_build_policy()

    # ------------------------------------------------------------------ #
    # Security envelope (re-checked at construction)                     #
    # ------------------------------------------------------------------ #
    def _enforce_build_policy(self) -> None:
        """Refuse any sandbox config that breaks the rootless build envelope.

        Mirrors :meth:`InferenceAgent._enforce_isolation_policy`: the package
        defaults are already safe, but a caller-supplied ``sandbox_config`` must
        not re-open WAN access, request a GPU, or — most importantly — mount the
        host Docker socket.
        """
        cfg = self.sandbox_config
        if cfg.network == NetworkMode.BRIDGE or not cfg.internal:
            raise PolicyViolationError(
                f"{type(self).__name__} must not have WAN access "
                f"(network={cfg.network.value}, internal={cfg.internal})"
            )
        if cfg.gpu_enabled:
            raise PolicyViolationError(
                f"{type(self).__name__} must not request GPU passthrough "
                "(a rootless image build needs none)"
            )
        for path in (*cfg.volumes.keys(), *cfg.volumes.values()):
            if "docker.sock" in path:
                raise PolicyViolationError(
                    f"{type(self).__name__} must never mount the Docker socket "
                    f"(found volume referencing {path!r}); builds are rootless"
                )

    # ------------------------------------------------------------------ #
    # A2A entry point                                                    #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Build an image from an inbound A2A build request and report the result.

        Steps: parse + validate → stage the build context into ``/workspace`` →
        run rootless Kaniko → return a success or failure :class:`Message`.
        Validation/build failures are returned as structured error messages (so a
        TaskAgent can react) rather than raised.
        """
        request = self._merge_data(message)
        with tel.start_span(
            "builder.build",
            kind=tel.TOOL,
            attributes={"atrium.agent_id": self.agent_id},
        ) as span:
            try:
                name, version, files, dockerfile, build_args = self._parse_request(request)
            except ValueError as exc:
                return self._error_message(message, f"invalid build request: {exc}")

            tag = self._image_tag(name, version)
            span.set_attribute("atrium.build.image", tag)

            # Ensure the rootless build sandbox is up (idempotent).
            await self.start_sandbox()

            # 1) Stage the build context (filenames already traversal-checked).
            staged = await self.write_files_to_sandbox(files, WORKSPACE, clean=True)
            if not staged.succeeded:
                return self._error_message(
                    message, f"failed to stage build context for {tag}", result=staged
                )

            # 2) Rootless Kaniko build + push to the internal registry.
            command = self._build_command(name, version, dockerfile, build_args)
            result = await self.execute_in_sandbox(command, timeout=self.build_timeout_s)
            span.set_attribute("atrium.exit_code", result.exit_code)

            if result.succeeded:
                return self._success_message(message, tag, result)
            return self._error_message(
                message, f"kaniko build failed for {tag}", result=result
            )

    # ------------------------------------------------------------------ #
    # Request parsing / validation                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_data(message: Message) -> dict[str, Any]:
        """Merge every structured ``DataPart`` of the message into one mapping."""
        merged: dict[str, Any] = {}
        for part in get_message_data(message):
            if isinstance(part, dict):
                merged.update(part)
        return merged

    def _parse_request(
        self, data: Mapping[str, Any]
    ) -> tuple[str, str, dict[str, bytes], str, dict[str, str]]:
        """Extract and validate the build request, raising ``ValueError`` on any
        malformed/unsafe field. Returns ``(name, version, files, dockerfile,
        build_args)`` with file contents decoded to ``bytes``."""
        name = data.get("target_name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"missing/invalid target_name: {name!r}")

        version = data.get("target_version")
        try:
            VersionTag.parse(str(version))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"target_version must be semver, got {version!r}") from exc

        raw_files = data.get("files")
        if not isinstance(raw_files, Mapping) or not raw_files:
            raise ValueError("files must be a non-empty {filename: content} map")
        files = {fname: self._coerce_content(fname, content) for fname, content in raw_files.items()}

        dockerfile = data.get("dockerfile", DEFAULT_DOCKERFILE)
        self._check_safe_path(dockerfile)
        if dockerfile not in files:
            raise ValueError(
                f"dockerfile {dockerfile!r} not present in files {sorted(files)}"
            )

        build_args = self._coerce_build_args(data.get("build_args"))
        return name, str(version), files, dockerfile, build_args

    @staticmethod
    def _check_safe_path(name: Any) -> None:
        """Reject absolute paths and ``..`` traversal in a context-relative name."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid filename: {name!r}")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe filename (absolute or traversal): {name!r}")

    @classmethod
    def _coerce_content(cls, fname: Any, content: Any) -> bytes:
        """Validate ``fname`` and decode ``content`` to bytes.

        Accepts ``str`` (UTF-8 text — the common case for source/Dockerfiles) or a
        ``{"encoding": "base64", "content": "..."}`` mapping for binary blobs.
        """
        cls._check_safe_path(fname)
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, Mapping) and content.get("encoding") == "base64":
            try:
                return base64.b64decode(str(content.get("content", "")), validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid base64 content for {fname!r}") from exc
        raise ValueError(
            f"unsupported content for {fname!r}: expected str or base64 mapping"
        )

    @staticmethod
    def _coerce_build_args(value: Any) -> dict[str, str]:
        """Validate optional ``--build-arg`` keys (values are shell-quoted later)."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("build_args must be a mapping")
        out: dict[str, str] = {}
        for key, val in value.items():
            if not isinstance(key, str) or not _BUILD_ARG_KEY_RE.match(key):
                raise ValueError(f"invalid build-arg key: {key!r}")
            out[key] = str(val)
        return out

    # ------------------------------------------------------------------ #
    # Command assembly                                                   #
    # ------------------------------------------------------------------ #
    def _image_tag(self, name: str, version: str) -> str:
        """``<registry>/<name>:<version>`` — the build destination + reply tag."""
        return f"{self.registry}/{name}:{version}"

    def _build_command(
        self, name: str, version: str, dockerfile: str, build_args: Mapping[str, str]
    ) -> str:
        """Assemble the rootless Kaniko executor command line.

        Note the deliberate absence of any privileged/daemon flag: Kaniko builds
        in user space and pushes (``--no-push=false``) to the insecure, host-local
        ``local-registry``.
        """
        argv = [
            "/kaniko/executor",
            f"--context=dir://{WORKSPACE}",
            f"--dockerfile={WORKSPACE}/{dockerfile}",
            f"--destination={self._image_tag(name, version)}",
            "--no-push=false",
            "--force",  # permit running outside kaniko's own scratch image
            "--cache=true",
            f"--cache-repo={self.registry}/cache",
            "--insecure",  # local-registry speaks plain HTTP
            "--insecure-pull",
            "--skip-tls-verify",
        ]
        argv += [f"--build-arg={key}={val}" for key, val in build_args.items()]
        return " ".join(shlex.quote(arg) for arg in argv)

    # ------------------------------------------------------------------ #
    # Reply assembly                                                     #
    # ------------------------------------------------------------------ #
    def _reply(self, request: Message, text: str, status: str, payload: Mapping[str, Any]) -> Message:
        """Build a build-result reply: ``text`` + a ``DataPart`` carrying ``payload``.

        Correlates with ``request`` (context/task ids) and stamps the shared
        ``kind``/``status``/``type`` envelope so success and error replies stay
        structurally identical.
        """
        return text_message(
            text,
            role=Role.ROLE_AGENT,
            context_id=request.context_id or None,
            task_id=request.task_id or None,
            metadata={"kind": KIND_BUILD, "status": status},
            extra_parts=[data_part({"type": RESULT_TYPE, "status": status, **payload})],
        )

    def _success_message(self, request: Message, tag: str, result: ExecutionResult) -> Message:
        return self._reply(
            request,
            f"Build succeeded: {tag}",
            STATUS_OK,
            {
                "image": tag,
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "log_tail": (result.stdout or "")[-_LOG_TAIL:],
            },
        )

    def _error_message(
        self, request: Message, reason: str, *, result: Optional[ExecutionResult] = None
    ) -> Message:
        payload: dict[str, Any] = {"reason": reason}
        if result is not None:
            payload["exit_code"] = result.exit_code
            payload["stdout_tail"] = (result.stdout or "")[-_LOG_TAIL:]
            payload["stderr_tail"] = (result.stderr or "")[-_LOG_TAIL:]
        return self._reply(request, f"Build failed: {reason}", STATUS_ERROR, payload)
