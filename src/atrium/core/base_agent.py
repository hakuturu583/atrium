"""``BaseAgent`` — the abstract base class for every Atrium agent.

``BaseAgent`` standard-equips each agent with the runtime's infrastructure so
that concrete subclasses only implement their domain logic:

* **Sandbox lifecycle** — version-pinned, clean, throwaway OpenShell containers.
* **A2A communication** — send/receive over the official A2A SDK, the single
  wire protocol between agents.
* **Distributed tracing** — every send, receive and in-sandbox command is an
  OpenInference span, and W3C ``traceparent`` is propagated across the
  physically-isolated containers so Arize Phoenix shows one stitched timeline.

The type hierarchy is::

    BaseAgent (abstract)
    ├── TaskAgent       → DelegatingTaskAgent, ...
    ├── InferenceAgent  → TabbyLLMAgent, ...
    ├── BuilderAgent
    └── SlackTaskAgent   (Slack I/O gateway; forwards to a TaskAgent over A2A)
"""

from __future__ import annotations

import abc
import asyncio
import base64
import logging
import shlex
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Union

from atrium.core import telemetry as tel
from atrium.core.errors import AgentError, PolicyViolationError, SandboxError, SandboxNotRunningError
from atrium.core.types import ExecutionResult, SandboxConfig, VersionTag
from atrium.protocol import Message, merge_data_parts
from atrium.protocol.a2a_transport import SendTarget, send_message
from atrium.sandbox import Sandbox

logger = logging.getLogger("atrium.agent")

__all__ = ["BaseAgent"]

#: Default local registry that holds per-agent, per-version images.
DEFAULT_REGISTRY = "local-registry"


class BaseAgent(abc.ABC):
    """Abstract base class equipping agents with sandbox, A2A and tracing.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent instance (also used as the sandbox
        name, since agent and sandbox are 1:1).
    version:
        The agent's semantic version (:class:`~semver.Version`). A bare string
        is parsed. This drives the version-pinned image tag.
    sandbox_config:
        Isolation/resource/network configuration. Defaults to an empty
        :class:`SandboxConfig` (overridden by subclasses such as
        :class:`InferenceAgent`).
    """

    #: Override in subclasses to fix the image slug; defaults to the class name.
    AGENT_SLUG: Optional[str] = None

    def __init__(
        self,
        agent_id: str,
        version: Union[str, VersionTag],
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        self.agent_id = agent_id
        self.version: VersionTag = (
            version if isinstance(version, VersionTag) else VersionTag.parse(str(version))
        )
        self.sandbox_config: SandboxConfig = sandbox_config or SandboxConfig()
        self.current_sandbox: Optional[Sandbox] = None
        self._inbox: "asyncio.Queue[Message]" = asyncio.Queue()
        self._tracer = tel.get_tracer()

    # ------------------------------------------------------------------ #
    # Identity                                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def slug_for(cls) -> str:
        """The image slug for this agent *class* (class name unless overridden).

        The class-level counterpart to :attr:`agent_slug`, so a factory can derive
        the slug without an instance.
        """
        return cls.AGENT_SLUG or cls.__name__.lower()

    @property
    def agent_slug(self) -> str:
        """The image slug for this agent (see :meth:`slug_for`)."""
        return type(self).slug_for()

    @property
    def image_name(self) -> str:
        """Version-pinned image, e.g. ``local-registry/tabbyllmagent:0.1.0``."""
        return f"{DEFAULT_REGISTRY}/{self.agent_slug}:{self.version}"

    @property
    def is_running(self) -> bool:
        """Whether this agent currently owns a running sandbox."""
        return self.current_sandbox is not None and self.current_sandbox.is_running

    # ------------------------------------------------------------------ #
    # Sandbox lifecycle                                                  #
    # ------------------------------------------------------------------ #
    async def start_sandbox(self) -> None:
        """Start a clean, version-pinned OpenShell sandbox for this agent.

        Idempotent: a no-op when a sandbox is already running. Uses the
        configured image when set, otherwise the version-derived
        :attr:`image_name`.
        """
        if self.is_running:
            logger.debug("sandbox for %s already running", self.agent_id)
            return

        image = self.sandbox_config.image or self.image_name
        # Forward the host's OTLP settings so the in-container bridge ships its
        # spans to the same Phoenix; the trace then spans both processes.
        tel.apply_sandbox_env(self.sandbox_config.env)
        attributes = {
            "atrium.agent_id": self.agent_id,
            "atrium.image": image,
            "atrium.gpu": self.sandbox_config.gpu_enabled,
            "atrium.network": self.sandbox_config.network.value,
        }
        with tel.start_span("agent.start_sandbox", kind=tel.TOOL, attributes=attributes):
            try:
                self.current_sandbox = await Sandbox.create(
                    image, self.sandbox_config, name=self.agent_id
                )
            except SandboxError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise AgentError(f"failed to start sandbox for {self.agent_id}") from exc

    async def stop_sandbox(self) -> None:
        """Safely destroy the running sandbox and clear the reference."""
        sandbox = self.current_sandbox
        if sandbox is None:
            return
        with tel.start_span(
            "agent.stop_sandbox",
            kind=tel.TOOL,
            attributes={"atrium.agent_id": self.agent_id},
        ):
            try:
                await sandbox.delete()
            finally:
                self.current_sandbox = None

    async def execute_in_sandbox(
        self, command: str, *, timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Run ``command`` inside this agent's running sandbox.

        The call is wrapped in a TOOL span recording the command, exit code and
        captured output for visibility in Phoenix.
        """
        if self.current_sandbox is None or not self.current_sandbox.is_running:
            raise SandboxNotRunningError(
                f"agent {self.agent_id} has no running sandbox; call start_sandbox() first"
            )
        with tel.start_span(
            "agent.execute_in_sandbox",
            kind=tel.TOOL,
            attributes={"atrium.agent_id": self.agent_id, tel.INPUT_VALUE: command},
        ) as span:
            result = await self.current_sandbox.exec(command, timeout=timeout)
            span.set_attribute("atrium.exit_code", result.exit_code)
            span.set_attribute(tel.OUTPUT_VALUE, result.stdout[-4096:])
            return result

    async def write_files_to_sandbox(
        self, files: Mapping[str, bytes], dest: str, *, clean: bool = False
    ) -> ExecutionResult:
        """Stage ``{relative_path: content}`` files under ``dest`` in the sandbox.

        OpenShell exposes only command execution (no host→container file copy),
        so each file is base64-encoded host-side and decoded in-container — safe
        for binary content and shell metacharacters alike. Each parent directory
        is created once; ``clean=True`` empties ``dest`` first. Callers are
        responsible for validating ``relative_path`` (no traversal) beforehand.
        """
        dest_q = shlex.quote(dest)
        lines = ["set -eu"]
        if clean:
            lines.append(f"rm -rf {dest_q}/* {dest_q}/.[!.]* 2>/dev/null || true")
        lines.append(f"mkdir -p {dest_q}")
        made_dirs = {dest}
        for relpath, content in files.items():
            path = f"{dest}/{relpath}"
            parent = path.rsplit("/", 1)[0]
            if parent not in made_dirs:
                made_dirs.add(parent)
                lines.append(f"mkdir -p {shlex.quote(parent)}")
            b64 = base64.b64encode(content).decode("ascii")
            lines.append(f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}")
        return await self.execute_in_sandbox("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Shared request / file validation helpers                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_data_parts(message: Message) -> dict[str, Any]:
        """Merge every structured ``DataPart`` of ``message`` into one mapping.

        Thin instance-facing alias for :func:`atrium.protocol.merge_data_parts`
        (the shared merge rule: later parts win on key clashes).
        """
        return merge_data_parts(message)

    @staticmethod
    def check_safe_relpath(name: Any) -> None:
        """Reject absolute paths and ``..`` traversal in a context-relative name.

        The single audited guard for any caller staging caller-supplied files into
        the sandbox; raises ``ValueError`` on an unsafe name.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid filename: {name!r}")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe filename (absolute or traversal): {name!r}")

    @classmethod
    def coerce_file_content(cls, fname: Any, content: Any) -> bytes:
        """Validate ``fname`` and decode ``content`` to bytes.

        Accepts ``str`` (UTF-8 text — the common case for source/config files) or
        a ``{"encoding": "base64", "content": "..."}`` mapping for binary blobs.
        """
        cls.check_safe_relpath(fname)
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

    def forbid_docker_socket(self) -> None:
        """Refuse a sandbox config that mounts the host Docker socket.

        The single audited guard against the most dangerous escape an autonomous
        agent could be handed (the host daemon); used by every agent whose
        envelope must never reach it. Raises :class:`PolicyViolationError`.
        """
        cfg = self.sandbox_config
        for path in (*cfg.volumes.keys(), *cfg.volumes.values()):
            if "docker.sock" in path:
                raise PolicyViolationError(
                    f"{type(self).__name__} must never mount the Docker socket "
                    f"(found volume referencing {path!r})"
                )

    # ------------------------------------------------------------------ #
    # A2A communication                                                  #
    # ------------------------------------------------------------------ #
    async def send_a2a_message(self, target: Union["BaseAgent", SendTarget], message: Message) -> Message:
        """Send an A2A ``message`` to ``target`` and return the reply.

        ``target`` may be another :class:`BaseAgent` (resolved to its A2A
        endpoint), a base URL, or an ``AgentCard``. The W3C trace context is
        injected into the message metadata by the transport so the trace spans
        both agents.
        """
        resolved = target.a2a_endpoint() if isinstance(target, BaseAgent) else target
        with tel.start_span(
            "agent.send_a2a_message",
            kind=tel.AGENT,
            attributes={"atrium.agent_id": self.agent_id},
        ):
            return await send_message(resolved, message)

    async def receive_a2a_message(self) -> Message:
        """Await the next inbound A2A message delivered to this agent's inbox.

        The inbox is fed by the agent's A2A server executor (see
        :func:`atrium.protocol.build_request_handler`). This is the pull-style
        counterpart to :meth:`dispatch`.
        """
        return await self._inbox.get()

    async def dispatch(self, message: Message) -> Message:
        """Entry point for an inbound request: restore trace context, handle it.

        Restores the W3C parent carried in ``message.metadata`` so that, across
        an ``A → B → C`` chain, every hop reports under one trace, then invokes
        the subclass :meth:`handle_task`.
        """
        parent = tel.extract_context(dict(message.metadata)) if message.metadata else None
        with tel.start_span(
            f"{self.agent_id}.dispatch",
            kind=tel.AGENT,
            attributes={"atrium.agent_id": self.agent_id},
            context=parent,
        ):
            return await self.handle_task(message)

    def a2a_endpoint(self) -> str:
        """Return the A2A base URL other agents use to reach this one.

        The default derives a host-local address from the agent id; concrete
        deployments may override to return a resolved sandbox address or card
        URL.
        """
        return f"http://{self.agent_id}.local"

    # ------------------------------------------------------------------ #
    # Domain logic (subclass responsibility)                             #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    async def handle_task(self, message: Message) -> Message:
        """Agent-specific main logic. Must be implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Async context-manager sugar around the sandbox lifecycle           #
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "BaseAgent":
        await self.start_sandbox()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.stop_sandbox()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "running" if self.is_running else "stopped"
        return f"<{type(self).__name__} {self.agent_id} v{self.version} ({state})>"
