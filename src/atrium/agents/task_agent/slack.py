"""``SlackTaskAgent`` — a Slack ingress/egress gateway for task requests.

Its whole responsibility is **Slack I/O**: it normalizes an inbound Slack
envelope (Events API message/app_mention or a slash command) into a task,
forwards that task over A2A to a ``downstream`` task agent — the thing that does
the actual author → build work, e.g. a
:class:`~atrium.agents.task_agent.agent.DelegatingTaskAgent` — and renders the
downstream's result back as Slack ``mrkdwn``. It authors no code and drives no
build itself; the orchestration engine lives behind the A2A seam, not in this
class. That keeps the channel adapter a thin, swappable gateway::

    Slack ──▶ SlackTaskAgent (parse ▸ forward ▸ format) ──A2A──▶ DelegatingTaskAgent ──▶ BuilderAgent
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from atrium.agents.task_agent.agent import DEFAULT_INITIAL_VERSION, TASK_RESULT_TYPE
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import AgentError
from atrium.core.types import NetworkMode, SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role,
    data_part,
    metadata_dict,
    text_message,
)
from atrium.protocol.a2a_transport import SendTarget

__all__ = ["SlackTaskAgent"]

#: Node/result statuses on the task protocol (match the rest of the runtime).
STATUS_OK = "ok"
STATUS_ERROR = "error"

#: Leading Slack bot mention, e.g. ``<@U0123ABCD> build me a foo`` -> ``build me a foo``.
_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*")


class SlackTaskAgent(BaseAgent):
    """A Slack I/O gateway that forwards tasks to a downstream task agent over A2A."""

    AGENT_SLUG = "slack_task_agent"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        downstream: SendTarget,
        sandbox_config: "SandboxConfig | None" = None,
    ) -> None:
        super().__init__(agent_id, version or DEFAULT_INITIAL_VERSION, sandbox_config or _slack_defaults())
        #: A2A target that actually handles the task (author → build). Any agent
        #: speaking the task protocol (a ``task_result`` reply) works here.
        self.downstream = downstream
        # A gateway needs WAN (to reach Slack) but never the host Docker socket.
        self.forbid_docker_socket()

    # ------------------------------------------------------------------ #
    # A2A entry point: parse Slack ▸ forward ▸ format reply              #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Normalize the Slack request, forward it downstream, format the reply."""
        task = self.parse_task(message)
        reply = await self.send_a2a_message(self.downstream, self._forward_request(task, message))
        return self._slack_reply(message, task, reply)

    def _forward_request(self, task: Mapping[str, Any], request: Message) -> Message:
        """Build the A2A task request handed to the downstream agent.

        The instruction rides as text and the whole normalized task as a data
        part, so a downstream :class:`TaskAgent` (whose default ``parse_task``
        merges data parts) sees ``{instruction, user, channel, source, raw}``.
        """
        return text_message(
            str(task.get("instruction", "")),
            role=Role.ROLE_USER,
            context_id=request.context_id or None,
            task_id=request.task_id or None,
            metadata={"kind": "task"},
            extra_parts=[data_part(dict(task))],
        )

    def _slack_reply(
        self, request: Message, task: Mapping[str, Any], downstream_reply: Message
    ) -> Message:
        """Render the downstream ``task_result`` as a Slack ``mrkdwn`` reply.

        The downstream's structured result is echoed as a data part so any
        programmatic consumer still gets the machine-readable outcome alongside
        the Slack-formatted text.
        """
        data = self.merge_data_parts(downstream_reply)
        status = metadata_dict(downstream_reply).get("status") or data.get("status")
        if status == STATUS_OK:
            text = self.format_reply(task, data)
        else:
            text = self.format_error(task, str(data.get("reason") or "task failed"))
        return text_message(
            text,
            role=Role.ROLE_AGENT,
            context_id=request.context_id or None,
            task_id=request.task_id or None,
            metadata={"kind": "slack", "status": status or STATUS_ERROR},
            extra_parts=[data_part({"type": TASK_RESULT_TYPE, **data})] if data else None,
        )

    # ------------------------------------------------------------------ #
    # Slack envelope -> normalized task (ingress)                        #
    # ------------------------------------------------------------------ #
    def parse_task(self, message: Message) -> dict[str, Any]:
        """Unpack a Slack Events API / slash-command payload into a task.

        Produces ``{"instruction", "user", "channel", "source": "slack", "raw"}``.
        The bot mention prefix on an ``app_mention`` is stripped from the
        instruction. Raises :class:`AgentError` when no instruction text is found.
        """
        payload = self.merge_data_parts(message)
        return self.normalize_slack(payload)

    @staticmethod
    def normalize_slack(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a raw Slack payload (pure; unit-testable without A2A)."""
        event = payload.get("event") if isinstance(payload.get("event"), Mapping) else None
        if event is not None:  # Events API (message / app_mention)
            text = str(event.get("text", ""))
            user = event.get("user")
            channel = event.get("channel")
        elif "command" in payload:  # slash command
            text = str(payload.get("text", ""))
            user = payload.get("user_id")
            channel = payload.get("channel_id")
        else:  # already-normalized / direct invocation
            text = str(payload.get("instruction") or payload.get("text") or "")
            user = payload.get("user")
            channel = payload.get("channel")

        instruction = _MENTION_RE.sub("", text).strip()
        if not instruction:
            raise AgentError("Slack task carried no instruction text")
        return {
            "instruction": instruction,
            "user": user,
            "channel": channel,
            "source": "slack",
            "raw": dict(payload),
        }

    # ------------------------------------------------------------------ #
    # Slack-flavored replies (egress)                                    #
    # ------------------------------------------------------------------ #
    def format_reply(self, task: Mapping[str, Any], result: Mapping[str, Any]) -> str:
        target = result.get("target_name", "the requested generation")
        version = result.get("version", "?")
        digest = result.get("digest") or "unknown digest"
        return (
            f":white_check_mark: Built *{target}:{version}* (`{digest}`).\n"
            "It is *inert* until a validator attests it and the Morpher promotes it "
            "— I can't make it live myself."
        )

    def format_error(self, task: Mapping[str, Any], reason: str) -> str:
        return f":x: Couldn't build that: {reason}"


def _slack_defaults() -> SandboxConfig:
    """Default gateway sandbox envelope: WAN-capable (reach Slack), no Docker socket."""
    return SandboxConfig(network=NetworkMode.BRIDGE, internal=False)
