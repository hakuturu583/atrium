"""TaskAgent — the start of Atrium's self-evolution loop.

* ``agent.py`` — the abstract :class:`TaskAgent` (task → author generation →
  drive :class:`~atrium.agents.builder_agent.BuilderAgent` over A2A → retry),
  plus the :class:`GenerationRequest` / :class:`BuildOutcome` value objects.
* ``slack.py`` — the concrete :class:`SlackTaskAgent` fed by Slack requests.

A TaskAgent authors code but holds **no authority over what runs**: it can only
have a new *version* built (inert until validated + promoted by the Morpher).

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/slack_task_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.task_agent.agent import (
    BuildFailedError,
    BuildOutcome,
    GenerationRequest,
    TaskAgent,
)
from atrium.agents.task_agent.slack import CodeAuthor, SlackTaskAgent

__all__ = [
    "TaskAgent",
    "SlackTaskAgent",
    "GenerationRequest",
    "BuildOutcome",
    "BuildFailedError",
    "CodeAuthor",
    "__version__",
]
