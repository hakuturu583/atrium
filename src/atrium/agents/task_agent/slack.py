"""``SlackTaskAgent`` — a :class:`TaskAgent` fed by Slack requests.

It normalizes a Slack envelope (Events API message/app_mention or a slash
command) into a task, delegates the actual source authoring to a pluggable
:data:`CodeAuthor` strategy (in a real deployment, an LLM-backed coding agent
reached over A2A), and formats the outcome as Slack ``mrkdwn``. The
generation → build → retry machinery and the "never promotes" guarantee are
inherited from :class:`TaskAgent`.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from atrium.agents.task_agent.agent import (
    BuildOutcome,
    GenerationRequest,
    TaskAgent,
)
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import AgentError
from atrium.core.types import SandboxConfig, VersionTag
from atrium.protocol import Message
from atrium.protocol.a2a_transport import SendTarget

__all__ = ["SlackTaskAgent", "CodeAuthor"]

#: Strategy that turns a normalized task into a generation. ``(task, attempt,
#: last_outcome) -> GenerationRequest`` — async so it can call out (e.g. to a
#: coding agent over A2A). ``attempt`` is 1-based; ``last_outcome`` is the prior
#: failed build (with Kaniko logs) on a retry, else ``None``.
CodeAuthor = Callable[
    [Mapping[str, Any], int, Optional[BuildOutcome]], Awaitable[GenerationRequest]
]

#: Leading Slack bot mention, e.g. ``<@U0123ABCD> build me a foo`` -> ``build me a foo``.
_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*")


class SlackTaskAgent(TaskAgent):
    """A TaskAgent whose tasks arrive as Slack messages / slash commands."""

    AGENT_SLUG = "slack_task_agent"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        builder: Union[BaseAgent, SendTarget],
        author: Optional[CodeAuthor] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        registry_endpoint: Optional[str] = None,
        max_build_attempts: int = 3,
    ) -> None:
        super().__init__(
            agent_id,
            version,
            builder=builder,
            sandbox_config=sandbox_config,
            registry_endpoint=registry_endpoint,
            max_build_attempts=max_build_attempts,
        )
        self._author = author

    # ------------------------------------------------------------------ #
    # Slack envelope -> normalized task                                  #
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
    # Authoring (delegated to the injected strategy)                     #
    # ------------------------------------------------------------------ #
    async def author_generation(
        self,
        task: Mapping[str, Any],
        *,
        attempt: int,
        last_outcome: Optional[BuildOutcome],
    ) -> GenerationRequest:
        """Delegate to the configured :data:`CodeAuthor`.

        Subclasses may instead override this method directly and leave ``author``
        unset; the default requires one so a bare agent fails loudly rather than
        silently doing nothing.
        """
        if self._author is None:
            raise AgentError(
                f"{type(self).__name__} has no code author configured; pass "
                "author=... or override author_generation()"
            )
        return await self._author(task, attempt, last_outcome)

    # ------------------------------------------------------------------ #
    # Slack-flavored replies                                             #
    # ------------------------------------------------------------------ #
    def format_reply(self, task: Mapping[str, Any], outcome: BuildOutcome) -> str:
        digest = outcome.digest or "unknown digest"
        return (
            f":white_check_mark: Built *{outcome.target_name}:{outcome.version}* "
            f"(`{digest}`).\n"
            "It is *inert* until a validator attests it and the Morpher promotes it "
            "— I can't make it live myself."
        )

    def format_error(self, task: Mapping[str, Any], reason: str) -> str:
        return f":x: Couldn't build that: {reason}"
