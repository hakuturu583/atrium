"""``PrefectRunnerAgent`` — the minimal-privilege executor for generated flows.

A :class:`~atrium.orchestration.job.Job`'s generated Prefect ``flow.py`` is an
*agent-dispatch orchestration*: its tasks assign work to role-bearing inference
agents. Something has to *run* that flow — but the generated code is untrusted, so
it must run in the strictest sandbox that can still reach subagents over A2A, and
never in the trusted Prefect worker.

``PrefectRunnerAgent`` is that executor. It reuses the code-workspace machinery
(``{files, commands}`` → staged into a sandbox → run → structured reply) by
subclassing :class:`~atrium.agents.code_workspace_agent.python_agent.PythonCodeWorkspaceAgent`,
but tightens the envelope to least privilege:

* **WAN-isolated** (``NetworkMode.INTERNAL``): no GitHub push, no arbitrary WAN
  egress — only the host-local control LAN, so the flow can dispatch to subagents
  over A2A (via the preinstalled trusted ``atrium_dispatch`` primitive) and nothing
  else.
* **No GitHub credentials** forwarded, and git/PR *push* requests are refused.
* **prefect + ``atrium_dispatch`` preinstalled** in the image, so the flow runs
  with no runtime package fetch.

    BaseAgent → CodeWorkSpaceAgent → PythonCodeWorkspaceAgent → PrefectRunnerAgent

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/prefect_runner_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.prefect_runner_agent.agent import PrefectRunnerAgent

__all__ = ["PrefectRunnerAgent", "__version__"]
