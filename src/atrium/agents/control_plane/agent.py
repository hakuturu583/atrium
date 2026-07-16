"""``ControlPlaneAgent`` — the trusted seam that turns a chat turn into a run.

An *interface agent* (Slack/Discord/…, in the evolvable ``atrium_agents`` tier)
holds no authority: its only egress is a ``workboard.submit`` A2A message. The
``ControlPlaneAgent`` is the fixed-infrastructure counterpart that receives it and
is the **sole caller of** :func:`atrium.orchestration.kick.submit_job` — so the
"agent proposes, trusted worker disposes" boundary the workboard already uses
extends cleanly to the human I/O edge (see ``docs/design/interface-agent.md``).

It deliberately does one thing: validate a submit and kick a workboard run,
returning the scheduled flow-run id. Steering (``payload["steering"]``) rides
through to the doer untouched; an optional ``review`` policy gates completion. The
mid-flight feedback-relay path (a human reply routed to a waiting reviewer) is
declared in the shared contract but not yet wired here — it is rejected explicitly
rather than mis-handled as a new job.

The ``submit_job`` seam is module-level so tests exercise the whole path with a
scripted kick and no Prefect backend.
"""

from __future__ import annotations

import logging
from typing import Optional

from atrium.agents.control_plane.protocol import (
    SubmitRequest,
    build_submitted_reply,
    parse_submit_request,
)
from atrium.core.base_agent import BaseAgent
from atrium.core.types import NetworkMode, SandboxConfig, VersionTag
from atrium.orchestration.kick import submit_job
from atrium.orchestration.review import ReviewPolicy
from atrium.protocol import Message

logger = logging.getLogger("atrium.agents.control_plane")

__all__ = ["ControlPlaneAgent"]

#: First version minted for the control plane with no ledger history.
DEFAULT_INITIAL_VERSION = "0.1.0"


class ControlPlaneAgent(BaseAgent):
    """Receive ``workboard.submit`` and kick a workboard run over Prefect."""

    AGENT_SLUG = "control_plane"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        deployment: str = "default",
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        super().__init__(
            agent_id, version or DEFAULT_INITIAL_VERSION, sandbox_config or _control_plane_defaults()
        )
        #: Which ``atrium-workboard/<deployment>`` a kick targets.
        self.deployment = deployment

    async def handle_task(self, message: Message) -> Message:
        """Validate the submit and kick a run; reply with the flow-run id."""
        req = parse_submit_request(message)
        if req.feedback_for is not None:
            # 動線2(b): relaying a human reply into a waiting review. The reviewer
            # waiting model is still open (see the design doc); refuse rather than
            # silently spawn a duplicate job.
            logger.info("feedback relay for %s not yet supported", req.feedback_for)
            return build_submitted_reply(
                f"feedback relay not yet implemented (review {req.feedback_for})",
                request=message,
                status="error",
            )
        job_id = await self._kick(req)
        logger.info("kicked job %s for context %s", job_id, req.context_id)
        return build_submitted_reply(job_id, request=message)

    async def _kick(self, req: SubmitRequest) -> str:
        """Kick a single-node workboard for ``req`` and return its flow-run id."""
        return await submit_job(
            req.agent,
            req.instruction,
            payload=req.payload,
            context_id=req.context_id,
            review=ReviewPolicy.from_dict(req.review),
            deployment=self.deployment,
        )


def _control_plane_defaults() -> SandboxConfig:
    """Trusted host-side envelope: WAN-capable (reach the Prefect server)."""
    return SandboxConfig(network=NetworkMode.BRIDGE, internal=False)
