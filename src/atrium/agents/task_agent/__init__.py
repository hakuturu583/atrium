"""TaskAgent — the start of Atrium's self-evolution loop.

* ``agent.py`` — the abstract :class:`TaskAgent` (task → author generation →
  drive :class:`~atrium.agents.builder_agent.BuilderAgent` over A2A → retry), its
  concrete :class:`DelegatingTaskAgent` (authors via an injected
  :data:`CodeAuthor`), plus the :class:`GenerationRequest` / :class:`BuildOutcome`
  value objects.
* ``slack.py`` — the :class:`SlackTaskAgent` **gateway**: pure Slack ingress/egress
  that forwards normalized tasks to a downstream task agent over A2A and formats
  the reply. It does not author or build — that lives behind the A2A seam.

A TaskAgent authors code but holds **no authority over what runs**: it can only
have a new *version* built (inert until validated + promoted by the Morpher).

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/<slug>:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.task_agent.agent import (
    BuildFailedError,
    BuildOutcome,
    CodeAuthor,
    DelegatingTaskAgent,
    GenerationRequest,
    TaskAgent,
)
from atrium.agents.task_agent.slack import SlackTaskAgent

__all__ = [
    "TaskAgent",
    "DelegatingTaskAgent",
    "SlackTaskAgent",
    "GenerationRequest",
    "BuildOutcome",
    "BuildFailedError",
    "CodeAuthor",
    "__version__",
]
