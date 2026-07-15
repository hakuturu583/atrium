"""``TaskAgent`` — the start of Atrium's self-evolution loop.

A ``TaskAgent`` takes a *task* (e.g. a Slack request), authors a **new generation**
of an agent (its source plus a ``Dockerfile``), and drives :class:`BuilderAgent`
over A2A to turn that into a container image. When a build fails, it inspects the
returned Kaniko logs, re-authors, and retries — the "author → build → (fix) →
build" inner loop of self-improvement.

Crucially, a ``TaskAgent`` holds **no authority over what runs**. It authors code
(so it has the highest prompt-injection exposure of any agent) and can therefore
only ask the builder to *push a new version tag*; that image is inert until the
fixed-infrastructure :class:`~atrium.core.morpher.Morpher` promotes it after a
signed validation. So this class deliberately stops at "a new version was built":
it never moves ``<slug>:active``. See ``docs/design/agent-versioning.md``.

The inheritance chain is preserved::

    BaseAgent → TaskAgent → SlackTaskAgent, ...
"""

from __future__ import annotations

import abc
import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

from atrium.agents.builder_agent.agent import (
    KIND_BUILD,
    RESULT_TYPE,
    STATUS_ERROR,
    STATUS_OK,
)
from atrium.core import telemetry as tel
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import AgentError
from atrium.core.registry import LocalRegistryError, RegistryClient, next_version
from atrium.core.types import NetworkMode, SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role,
    data_part,
    metadata_dict,
    text_message,
)
from atrium.protocol.a2a_transport import SendTarget

logger = logging.getLogger("atrium.agents.task")

__all__ = [
    "TaskAgent",
    "GenerationRequest",
    "BuildOutcome",
    "BuildFailedError",
    "DEFAULT_INITIAL_VERSION",
]

#: First version minted for an agent with no history in the ledger.
DEFAULT_INITIAL_VERSION = "0.1.0"

#: A2A ``type`` stamped on the structured result this agent replies with.
TASK_RESULT_TYPE = "task_result"


class BuildFailedError(AgentError):
    """Raised when every build attempt for a generation was rejected."""


@dataclass(slots=True)
class GenerationRequest:
    """A newly-authored agent generation, ready to hand to :class:`BuilderAgent`.

    ``files`` is the build context ``{relative_path: content}`` and must include
    ``dockerfile``; values are ``str`` (text) or ``bytes`` (binary — base64-encoded
    into the A2A payload). The target version is normally *decided* by the
    :class:`TaskAgent` from the ledger (bump ``version_bump`` off the latest tag);
    set ``version`` to pin one explicitly and skip that.
    """

    target_name: str
    files: dict[str, Union[str, bytes]]
    dockerfile: str = "Dockerfile"
    build_args: dict[str, str] = field(default_factory=dict)
    version_bump: str = "patch"
    version: Optional[str] = None

    def build_payload(self, version: str) -> dict[str, Any]:
        """Render the BuilderAgent build-request payload for ``version``.

        ``bytes`` file contents are base64-wrapped (JSON/A2A can't carry raw
        bytes); ``str`` contents pass through unchanged.
        """
        files_payload: dict[str, Any] = {}
        for name, content in self.files.items():
            if isinstance(content, (bytes, bytearray)):
                files_payload[name] = {
                    "encoding": "base64",
                    "content": base64.b64encode(bytes(content)).decode("ascii"),
                }
            else:
                files_payload[name] = content
        payload: dict[str, Any] = {
            "target_name": self.target_name,
            "target_version": version,
            "files": files_payload,
            "dockerfile": self.dockerfile,
        }
        if self.build_args:
            payload["build_args"] = self.build_args
        return payload


@dataclass(slots=True)
class BuildOutcome:
    """The result of one build round-trip with :class:`BuilderAgent`."""

    ok: bool
    target_name: str
    version: str
    image: Optional[str] = None
    digest: Optional[str] = None
    image_ref: Optional[str] = None
    reason: Optional[str] = None
    logs: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


class TaskAgent(BaseAgent, abc.ABC):
    """Abstract self-evolution driver: task → author generation → build (retry).

    Subclasses implement :meth:`author_generation` (turn a task into a
    :class:`GenerationRequest`, optionally reacting to a previous failure). The
    version-decide, the A2A build round-trip with :class:`BuilderAgent`, and the
    fix-and-retry loop are handled here.

    Parameters
    ----------
    agent_id, version:
        Standard :class:`BaseAgent` identity.
    builder:
        The build target — a :class:`BuilderAgent` (resolved to its A2A endpoint)
        or a base URL / ``AgentCard``.
    registry_endpoint:
        Host-side ``host:port`` of the registry ledger, used to *decide* the next
        version (bump off the latest existing tag). When ``None``, versions start
        at :data:`DEFAULT_INITIAL_VERSION` unless the generation pins one.
    max_build_attempts:
        How many times to (re-)author and build before giving up.
    """

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        builder: Union[BaseAgent, SendTarget],
        sandbox_config: Optional[SandboxConfig] = None,
        registry_endpoint: Optional[str] = None,
        max_build_attempts: int = 3,
    ) -> None:
        super().__init__(agent_id, version or DEFAULT_INITIAL_VERSION, sandbox_config or _task_defaults())
        self.builder = builder
        self.registry_endpoint = registry_endpoint
        self.max_build_attempts = max(1, max_build_attempts)
        self._enforce_task_policy()

    # ------------------------------------------------------------------ #
    # Security envelope                                                  #
    # ------------------------------------------------------------------ #
    def _enforce_task_policy(self) -> None:
        """A code-authoring agent must never be handed the host Docker socket.

        TaskAgents legitimately need WAN (to reach Slack, etc.), so — unlike the
        inference/builder envelopes — network is not constrained here; but the
        one escape that would let a compromised author own the host is refused.
        """
        self.forbid_docker_socket()

    # ------------------------------------------------------------------ #
    # Subclass responsibility                                            #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    async def author_generation(
        self,
        task: Mapping[str, Any],
        *,
        attempt: int,
        last_outcome: Optional[BuildOutcome],
    ) -> GenerationRequest:
        """Author the next generation for ``task``.

        Called once per build attempt. ``attempt`` is 1-based; ``last_outcome`` is
        the previous failed :class:`BuildOutcome` (with its Kaniko ``logs``) on a
        retry, or ``None`` on the first attempt — so the implementation can fix
        the source in response to the build error.
        """
        raise NotImplementedError

    def parse_task(self, message: Message) -> dict[str, Any]:
        """Normalize an inbound A2A request into a task mapping.

        Default: merge the message's structured data parts. Subclasses (e.g.
        :class:`SlackTaskAgent`) override to unpack a domain-specific envelope.
        """
        return self.merge_data_parts(message)

    # ------------------------------------------------------------------ #
    # The author → build → (fix) → build loop                            #
    # ------------------------------------------------------------------ #
    async def build_generation(self, task: Mapping[str, Any]) -> BuildOutcome:
        """Author and build a generation for ``task``, retrying on build failure.

        Returns the successful :class:`BuildOutcome`, or raises
        :class:`BuildFailedError` after exhausting :attr:`max_build_attempts`.
        """
        last: Optional[BuildOutcome] = None
        with tel.start_span(
            "task.build_generation",
            kind=tel.AGENT,
            attributes={"atrium.agent_id": self.agent_id},
        ) as span:
            for attempt in range(1, self.max_build_attempts + 1):
                gen = await self.author_generation(task, attempt=attempt, last_outcome=last)
                version = self._decide_version(gen)
                span.set_attribute("atrium.task.attempt", attempt)
                span.set_attribute("atrium.build.image", f"{gen.target_name}:{version}")
                reply = await self._request_build(gen, version)
                last = self._parse_build_result(reply, gen.target_name, version)
                if last.ok:
                    logger.info(
                        "built %s:%s on attempt %d (%s)",
                        gen.target_name, version, attempt, last.digest or "no-digest",
                    )
                    return last
                logger.warning(
                    "build attempt %d/%d for %s:%s failed: %s",
                    attempt, self.max_build_attempts, gen.target_name, version, last.reason,
                )
            raise BuildFailedError(
                f"failed to build {last.target_name if last else task!r} after "
                f"{self.max_build_attempts} attempt(s): {last.reason if last else 'no attempt'}"
            )

    # ------------------------------------------------------------------ #
    # Decide (version bump off the ledger)                               #
    # ------------------------------------------------------------------ #
    def _decide_version(self, gen: GenerationRequest) -> str:
        """Pick the target version: an explicit pin, else bump off the ledger.

        With a ``registry_endpoint`` configured, the latest existing tag for the
        target is bumped by ``gen.version_bump``; with no endpoint (or an empty/
        unreachable ledger) the first version is :data:`DEFAULT_INITIAL_VERSION`.
        """
        if gen.version is not None:
            VersionTag.parse(gen.version)  # validate; raises on non-semver
            return gen.version
        latest = self._latest_version(gen.target_name)
        if latest is None:
            return DEFAULT_INITIAL_VERSION
        return str(next_version(latest, gen.version_bump))

    def _latest_version(self, target_name: str) -> Optional[str]:
        """The newest semver tag for ``target_name`` in the ledger, or ``None``.

        Best-effort: returns ``None`` when no endpoint is configured or the
        registry is unreachable (so the caller falls back to the initial version).
        """
        if not self.registry_endpoint:
            return None
        try:
            versions = RegistryClient(self.registry_endpoint).versions(target_name)
        except LocalRegistryError as exc:
            logger.warning("version decide: ledger unreachable (%s); starting fresh", exc)
            return None
        return versions[-1] if versions else None

    # ------------------------------------------------------------------ #
    # A2A build round-trip with BuilderAgent                             #
    # ------------------------------------------------------------------ #
    async def _request_build(self, gen: GenerationRequest, version: str) -> Message:
        """Send a build request to the builder and return its reply message."""
        message = text_message(
            f"build {gen.target_name}:{version}",
            role=Role.ROLE_USER,
            metadata={"kind": KIND_BUILD},
            extra_parts=[data_part(gen.build_payload(version))],
        )
        return await self.send_a2a_message(self.builder, message)

    def _parse_build_result(
        self, reply: Message, target_name: str, version: str
    ) -> BuildOutcome:
        """Interpret a BuilderAgent reply as a :class:`BuildOutcome`."""
        data = self.merge_data_parts(reply)
        status = metadata_dict(reply).get("status") or data.get("status")
        ok = status == STATUS_OK and data.get("type", RESULT_TYPE) == RESULT_TYPE
        return BuildOutcome(
            ok=ok,
            target_name=target_name,
            version=version,
            image=data.get("image"),
            digest=data.get("digest"),
            image_ref=data.get("image_ref"),
            reason=data.get("reason") if not ok else None,
            logs=data.get("log_tail") or data.get("stdout_tail") or data.get("stderr_tail"),
            raw=data,
        )

    # ------------------------------------------------------------------ #
    # A2A entry point                                                    #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Drive the full loop for an inbound task and reply with the outcome.

        A build failure (after all retries) is reported as a structured error
        message rather than raised, so the requester (e.g. Slack) gets an answer.
        """
        task = self.parse_task(message)
        try:
            outcome = await self.build_generation(task)
        except BuildFailedError as exc:
            return self._result_message(message, task, None, error=str(exc))
        return self._result_message(message, task, outcome)

    # ------------------------------------------------------------------ #
    # Reply assembly                                                     #
    # ------------------------------------------------------------------ #
    def _result_message(
        self,
        request: Message,
        task: Mapping[str, Any],
        outcome: Optional[BuildOutcome],
        *,
        error: Optional[str] = None,
    ) -> Message:
        """Build the structured task-result reply (overridable for formatting)."""
        if outcome is not None and outcome.ok:
            status = STATUS_OK
            payload: dict[str, Any] = {
                "type": TASK_RESULT_TYPE,
                "status": status,
                "target_name": outcome.target_name,
                "version": outcome.version,
                "image": outcome.image,
                "digest": outcome.digest,
                "image_ref": outcome.image_ref,
            }
            text = self.format_reply(task, outcome)
        else:
            status = STATUS_ERROR
            reason = error or (outcome.reason if outcome else "build failed")
            payload = {"type": TASK_RESULT_TYPE, "status": status, "reason": reason}
            text = self.format_error(task, reason)
        return text_message(
            text,
            role=Role.ROLE_AGENT,
            context_id=request.context_id or None,
            task_id=request.task_id or None,
            metadata={"kind": "task", "status": status},
            extra_parts=[data_part(payload)],
        )

    def format_reply(self, task: Mapping[str, Any], outcome: BuildOutcome) -> str:
        """Human-facing success text (subclasses tailor for their channel)."""
        return (
            f"Built new generation {outcome.target_name}:{outcome.version} "
            f"({outcome.digest or 'digest unknown'}). It is inert until validated "
            f"and promoted."
        )

    def format_error(self, task: Mapping[str, Any], reason: str) -> str:
        """Human-facing failure text (subclasses tailor for their channel)."""
        return f"Could not build the requested generation: {reason}"


def _task_defaults() -> SandboxConfig:
    """Default TaskAgent sandbox envelope: WAN-capable (Slack), no Docker socket."""
    return SandboxConfig(network=NetworkMode.BRIDGE, internal=False)
